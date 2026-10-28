"""Deferred setup-model updates and retry-dependent cost targets."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .learner import SharedLinUCB


@dataclass(frozen=True)
class DeferredCost:
    phi: np.ndarray
    arm: int
    history_index: int
    provisional_cost: float


class DeferredCostLinUCB(SharedLinUCB):
    @classmethod
    def from_runtime_model(cls, model):
        """Adopt a fresh runtime learner without rebuilding its candidate inputs."""
        if model.t or model.history or model._last_phi is not None or np.any(model.b):
            raise ValueError("Setup feedback requires a fresh learner")
        if not np.array_equal(model.A_inv, np.eye(model.d_phi) / model.l2_reg):
            raise ValueError("Expected untrained setup precision")
        adapted = object.__new__(cls)
        adapted.__dict__ = model.__dict__.copy()
        return adapted

    def defer_cost(self, succeeded):
        """Count the observation while preserving runtime means until case completion."""
        if self._last_phi is None or self._last_arm is None:
            raise RuntimeError("Deferred feedback requires a selected setup")
        phi = self._last_phi.copy()
        observation = DeferredCost(phi, int(self._last_arm), len(self.history)-1,
                                   float(phi @ (self.A_inv @ self.b)))
        self._rank_one_inverse_update(self.A_inv, phi)
        self._after_runtime_precision_update(phi)
        self.b += observation.provisional_cost * phi
        self.t += 1
        self._last_phi = self._last_arm = None
        return observation

    def resolve_cost(self, observation, loss):
        """Replace the provisional target without counting the observation twice."""
        loss = float(loss)
        if not np.isfinite(loss) or loss < 0:
            raise ValueError("Deferred runtime cost must be finite and nonnegative")
        if np.isfinite(self.history[observation.history_index].loss):
            raise RuntimeError("Deferred runtime cost was already resolved")
        self.b += (loss-observation.provisional_cost) * observation.phi
        self._cand.observe(observation.arm, loss)
        self.history[observation.history_index] = replace(
            self.history[observation.history_index], loss=loss)


class FailureAwareRetryLinUCB(DeferredCostLinUCB):
    """One binary target vector, sharing the runtime model's precision matrix."""

    failure_beta = 2.0  # Setup failure-score coefficient, not RL beta.

    @classmethod
    def from_runtime_model(cls, model):
        adapted = super().from_runtime_model(model)
        adapted.failure_b = np.zeros(model.d_phi, dtype=float)
        return adapted

    def _recovery_best_locations(
        self, x, *, candidates, candidate_features, runtime_score,
        runtime_mean, runtime_uncert,
    ):
        prediction = self._linear_prediction_subset(
            x, theta=self.A_inv @ self.failure_b, arms=candidates,
            action_features=candidate_features,
        )
        upper = prediction + self.failure_beta * runtime_uncert
        best = int(np.lexsort((runtime_score, upper))[0])
        return np.flatnonzero(
            np.isclose(upper, upper[best], atol=1e-12, rtol=0)
            & np.isclose(runtime_score, runtime_score[best], atol=1e-12, rtol=0)
        )

    def defer_cost(self, succeeded):
        observation = super().defer_cost(succeeded)
        self.failure_b += float(not succeeded) * observation.phi
        return observation

    def _mutable_state_extra_arrays(self):
        return {"retry_failure_b": self.failure_b}

    def _load_mutable_state_extra_arrays(self, payload, *, checkpoint_version):
        vector = np.asarray(payload["retry_failure_b"], dtype=float)
        if vector.shape != self.failure_b.shape or not np.all(np.isfinite(vector)):
            raise ValueError("Invalid retry failure-model checkpoint")
        self.failure_b[:] = vector


def setup_cost_target(attempts, costs, index, feedback_mode):
    """Assign one setup observation its own cost and the required recovery costs.

    ``costs`` includes the fixed-default fallback when one was executed.
    Failed attempts in adjacent mode extend through the next constructed
    hierarchy, or through the last executed attempt if construction never succeeds.
    """
    attempt = attempts[index]
    stop = index+2
    if feedback_mode == "adjacent_delayed" and not attempt.succeeded:
        while stop <= len(attempts) and attempts[stop-1].failure_stage == "setup":
            stop += 1
    return costs[index] if attempt.succeeded else sum(costs[index:stop]
        if feedback_mode == "adjacent_delayed" else costs[index:])
