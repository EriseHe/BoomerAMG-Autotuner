"""Alternative setup retry selectors for development comparisons."""
from __future__ import annotations

import numpy as np

from experiments.paper_final.recovery.observations import (
    ObservedSelection,
    ObservedFailureAware,
    observed_model as observed_failure_model,
)
from experiments.paper_final.recovery.setup_feedback import (
    DeferredCostLinUCB,
    FailureAwareRetryLinUCB,
)


RETRY_RULES = {
    "conservative_suffix": "Conservative retries + full remaining-attempt cost",
    "conservative_adjacent": "Conservative retries + adjacent-attempt cost",
    "failure_model_suffix": "Failure-model retries + full remaining-attempt cost",
    "failure_model_adjacent": "Failure-model retries + adjacent-attempt cost",
    "failure_model_alpha_adjacent": "Failure-model retries with runtime alpha + adjacent-attempt cost",
}


class AlphaFailureRetryLinUCB(FailureAwareRetryLinUCB):
    @property
    def failure_beta(self):
        """Use the runtime exploration coefficient at the same observation count."""
        return self._effective_alpha()


class ConservativeRetryLinUCB(DeferredCostLinUCB):
    def _recovery_best_locations(
        self, x, *, candidates, candidate_features, runtime_score,
        runtime_mean, runtime_uncert,
    ):
        score = runtime_mean + self._effective_alpha() * runtime_uncert
        return np.flatnonzero(np.isclose(score, np.min(score), atol=1e-12, rtol=0))


def use_retry_rule(model, rule):
    if rule not in RETRY_RULES:
        raise ValueError("Unknown retry rule")
    if rule == "failure_model_alpha_adjacent":
        cls = AlphaFailureRetryLinUCB
    else:
        cls = ConservativeRetryLinUCB if rule.startswith("conservative_") else FailureAwareRetryLinUCB
    return cls.from_runtime_model(model)


class ObservedAlphaFailureAware(AlphaFailureRetryLinUCB, ObservedFailureAware):
    pass


class ObservedConservative(ObservedSelection, ConservativeRetryLinUCB):
    def _recovery_best_locations(self, x, **kwargs):
        score = kwargs["runtime_mean"] + self._effective_alpha() * kwargs["runtime_uncert"]
        self._capture["retry_score"] = score
        return np.flatnonzero(np.isclose(score, np.min(score), atol=1e-12, rtol=0))

    def _record_choice(self, retry):
        if not retry:
            return super()._record_choice(retry)
        data = self._capture
        location = int(np.flatnonzero(data["arms"] == self._last_arm)[0])
        objective = data["retry_score"]
        ranked = np.lexsort((data["score"], objective))

        def candidate(i):
            return dict(arm=int(data["arms"][i]), mean=float(data["mean"][i]),
                width=float(data["width"][i]), runtime_lcb=float(data["score"][i]),
                runtime_ucb=float(objective[i]))

        self.selection_diagnostics.append(dict(strategy="runtime_ucb",
            observations=self.t, alpha=float(data["alpha"]), selected=candidate(location),
            best_two_gap=float(objective[ranked[1]]-objective[ranked[0]]) if len(ranked)>1 else None,
            candidate_count=len(ranked), candidate_arms=data["arms"].tolist(),
            top_candidates=[candidate(int(i)) for i in ranked[:8]]))
        self._capture = None


def observed_model(model):
    if isinstance(model, AlphaFailureRetryLinUCB):
        cls = ObservedAlphaFailureAware
    elif isinstance(model, ConservativeRetryLinUCB):
        cls = ObservedConservative
    else:
        return observed_failure_model(model)
    observed = object.__new__(cls)
    observed.__dict__ = model.__dict__.copy()
    observed.selection_diagnostics, observed._capture = [], None
    return observed
