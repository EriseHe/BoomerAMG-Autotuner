"""Development comparisons retain their selectors without entering official runs."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from development.retry_comparison import observed_model, use_retry_rule
from development.tests.test_fallback_feedback import CONTEXT, learner
from experiments.paper_final.recovery.observations import observed_model as official_observed_model
from experiments.paper_final.recovery.setup_feedback import FailureAwareRetryLinUCB


class RetryComparisonTests(unittest.TestCase):
    def test_conservative_comparison_uses_runtime_upper_score(self):
        model = use_retry_rule(learner(rule=None), "conservative_adjacent")
        mean = np.array([1., 2.])
        width = np.array([2., .1])
        model.alpha = 1.
        model.alpha_decay = False
        selected = model._recovery_best_locations(CONTEXT, candidates=np.array([0, 1]),
            candidate_features=None, runtime_score=mean-width,
            runtime_mean=mean, runtime_uncert=width)
        self.assertEqual(np.argmin(mean-width), 0)
        np.testing.assert_array_equal(selected, [1])
        self.assertFalse(hasattr(model, "failure_b"))
        with self.assertRaises(TypeError):
            official_observed_model(model)

    def test_comparison_diagnostics_preserve_initial_and_retry_scores(self):
        for rule, strategy in (("conservative_adjacent", "runtime_ucb"),
                               ("failure_model_alpha_adjacent", "failure_upper"),
                               ("failure_model_adjacent", "failure_upper")):
            with self.subTest(rule=rule):
                model = observed_model(use_retry_rule(learner(rule=None), rule))
                model.predict(CONTEXT)
                arm = model._last_arm
                failed = model.defer_cost(False)
                model.select_recovery(CONTEXT, excluded_arms=[arm])
                self.assertNotEqual(model._last_arm, arm)
                self.assertEqual(model.selection_diagnostics[0]["strategy"], "initial_guess")
                retry = model.selection_diagnostics[1]
                self.assertEqual(retry["strategy"], strategy)
                score = "runtime_ucb" if strategy == "runtime_ucb" else "failure_upper"
                self.assertEqual(retry["selected"][score], retry["top_candidates"][0][score])
                succeeded = model.defer_cost(True)
                model.resolve_cost(failed, 3.)
                model.resolve_cost(succeeded, 1.)
                self.assertEqual(model.t, 2)

    def test_failure_alpha_matches_the_runtime_schedule_after_each_observation(self):
        model = observed_model(use_retry_rule(learner(rule=None), "failure_model_alpha_adjacent"))
        model.alpha_decay = True
        model.alpha_decay_burnin_rounds = 0
        model.predict(CONTEXT)
        first_arm = model._last_arm
        failed = model.defer_cost(False)
        self.assertAlmostEqual(model.failure_beta, 1/np.sqrt(2))
        model.select_recovery(CONTEXT, excluded_arms=[first_arm])
        selected = model.selection_diagnostics[-1]["selected"]
        self.assertAlmostEqual(selected["failure_upper"],
                               selected["failure_mean"] + model.failure_beta*selected["width"])
        succeeded = model.defer_cost(True)
        self.assertAlmostEqual(model.failure_beta, 1/np.sqrt(3))
        model.resolve_cost(failed, 3.)
        model.resolve_cost(succeeded, 1.)

    def test_runtime_comparison_checkpoint_has_no_failure_model(self):
        model = use_retry_rule(learner(rule=None), "conservative_adjacent")
        model.predict(CONTEXT)
        observation = model.defer_cost(False)
        model.resolve_cost(observation, 2.)
        with TemporaryDirectory() as directory:
            path = Path(directory)/"model.npz"
            model.save_mutable_state(path)
            with np.load(path, allow_pickle=False) as checkpoint:
                self.assertFalse(any("failure" in key for key in checkpoint.files))

    def test_adaptation_rejects_pending_or_trained_models(self):
        for trained in (False, True):
            model = learner(rule=None)
            model.predict(CONTEXT)
            if trained:
                model.update(1.)
            with self.assertRaises(ValueError):
                FailureAwareRetryLinUCB.from_runtime_model(model)


if __name__ == "__main__":
    unittest.main()
