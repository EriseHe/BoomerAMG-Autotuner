"""Successful Default fallbacks supply one setup observation without another solve."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final.online.case_loop import report_online_outcome, validate_recovery_stream
from experiments.paper_final.recovery.audit import audit
from experiments.paper_final.recovery.metrics import VARIANT
from development.retry_comparison import use_retry_rule
from experiments.paper_final.recovery.setup_feedback import run_deferred_case
from experiments.paper_final.recovery import model_audit
from setup.learners.common.action_features import GenericActionFeatureEncoder, ParameterSpec, ParameterSpaceSpec
from setup.learners.common.aot_candidates import AOTCandidateSchedule, CompactActionCatalog, FactorizedActionFeatureCache
from setup.learners.linucb.learner import SharedLinUCB
from setup.learners.linucb.setup_reselection import run_same_context_setup_reselection
from setup.space import DEFAULT_SETUP_PARAMS


CONTEXT = np.array([1., .7])
SPEC = ParameterSpaceSpec((ParameterSpec(
    name="strong_threshold", kind="continuous", values=(.5, .75, 1.),
    default=.25, center=.25, scale=.75,
), ParameterSpec(name="coarsen_type", kind="categorical", values=(10, 8), default=10)))


def learner(*, rule="failure_model_adjacent", directory=None, compact=False, default_first=False):
    catalog = CompactActionCatalog(SPEC, fixed_params=DEFAULT_SETUP_PARAMS)
    kwargs = {}
    if directory is not None:
        cache = FactorizedActionFeatureCache(catalog, GenericActionFeatureEncoder(SPEC))
        schedule = AOTCandidateSchedule(directory=directory, catalog=catalog,
            rounds=9, pool_size=2, seed=3, factorized_cache=cache)
        kwargs.update(candidate_schedule=schedule, action_feature_cache=cache,
                      candidate_pool_size=2)
    model = SharedLinUCB(actions=catalog if compact or directory else list(catalog),
        context_dim=2, parameter_spec=SPEC, context_interaction_indices=(1,),
        initial_guess=DEFAULT_SETUP_PARAMS if default_first else catalog[0],
        initial_guess_rounds=1, elite_cache_size=2, seed=11, **kwargs)
    return model if rule is None else use_retry_rule(model, rule)


def policy(model):
    return SimpleNamespace(model=model,
        select=lambda *, context, parameter_space: (model.predict(context), {}))


def outcome(success, cost=1., stage="solve"):
    return dict(runtime=cost, setup_runtime=.2*cost, solve_runtime=.8*cost,
        infer_runtime=0., failed=not success, failure_stage="" if success else stage,
        failure_reason="" if success else "synthetic_failure",
        iterations=2, residual_norm=1e-7 if success else 1.)


class FallbackFeedbackTests(unittest.TestCase):
    def execute(self, *, rule="failure_model_adjacent", fallback_success=True,
                fallback_cost=2., success_on=None, learn=True, default_first=False,
                joint=False):
        model = learner(rule=rule, default_first=default_first)
        attempts = []
        fallback_calls = []
        before = []

        def solve(params):
            index = len(attempts)
            attempts.append(dict(params))
            return outcome(index == success_on, 1. + index)

        def fallback():
            fallback_calls.append(True)
            return outcome(fallback_success, fallback_cost)

        recorder = None
        if joint:
            recorder = SimpleNamespace(episodes=[], snapshot=None,
                finalize=lambda attempts, fallback, include_recovery: ([0.]*len(attempts), 0))
        feedback = "suffix_delayed" if rule.endswith("suffix") else "adjacent_delayed"
        result = run_deferred_case(policy=policy(model), parameter_space={}, context=CONTEXT,
            solver=solve, fallback_solver=fallback, variant=VARIANT,
            previous_update=.125, before_attempt=before.append,
            feedback_mode=feedback, learn=learn, recorder=recorder)
        reported = report_online_outcome(result.outcome, bandit_timing=result.timing)
        audit([dict(problem_index=1, outcome=reported)], VARIANT,
              feedback_mode=feedback, learn=learn)
        self.assertEqual(before, list(range(len(attempts))))
        self.assertEqual(len(fallback_calls), int(result.outcome["fallback_used"]))
        return model, result

    def assert_regression(self, model, result):
        rows = result.outcome["primary_attempts"]
        features = [model._phi(CONTEXT, row["arm_index"]) for row in rows]
        costs = [row["learning_cost"] for row in rows]
        failures = [float(row["status"] != "success") for row in rows]
        if result.outcome["fallback_learning_cost"] is not None:
            features.append(model._phi(CONTEXT, result.outcome["fallback_arm_index"]))
            costs.append(result.outcome["fallback_learning_cost"])
            failures.append(0.)
        design = np.asarray(features)
        expected_inverse = np.linalg.inv(model.l2_reg*np.eye(model.d_phi) + design.T @ design)
        np.testing.assert_allclose(model.A_inv, expected_inverse, atol=1e-12)
        np.testing.assert_allclose(model.b, design.T @ costs, atol=1e-12)
        if hasattr(model, "failure_b"):
            np.testing.assert_allclose(model.failure_b, design.T @ failures, atol=1e-12)
        self.assertEqual(model.t, len(features))
        self.assertEqual(len(model.history), len(features))
        self.assertTrue(all(np.isfinite(row.loss) for row in model.history))
        self.assertIsNone(model._last_arm)
        self.assertIsNone(model._last_phi)

    def test_successful_fallback_updates_both_models_once(self):
        for rule in ("failure_model_adjacent", "failure_model_suffix", "conservative_adjacent"):
            for joint in (False, True):
                with self.subTest(rule=rule, joint=joint):
                    model, result = self.execute(rule=rule, joint=joint)
                    self.assertEqual(result.outcome["bandit_observation_count"], 4)
                    self.assertEqual(result.outcome["fallback_learning_cost"], 2.)
                    self.assertEqual(model.actions[result.outcome["fallback_arm_index"]], DEFAULT_SETUP_PARAMS)
                    self.assert_regression(model, result)
                    last = result.outcome["primary_attempts"][-1]
                    self.assertAlmostEqual(last["learning_cost"], last["attempt_cost"] + 2.)
                    self.assertGreater(result.timing["update_sec"], 0.)

    def test_failed_fallback_has_no_observation_but_still_costs_time(self):
        model, result = self.execute(fallback_success=False)
        self.assertTrue(result.outcome["failed"])
        self.assertEqual(result.outcome["bandit_observation_count"], 3)
        self.assertIsNone(result.outcome["fallback_learning_cost"])
        self.assertIsNone(result.outcome["fallback_arm_index"])
        last = result.outcome["primary_attempts"][-1]
        self.assertAlmostEqual(last["learning_cost"], last["attempt_cost"] + 2.)
        self.assert_regression(model, result)

    def test_final_model_audit_includes_successful_fallback_observations(self):
        for success, cost in ((True, 2.), (True, 0.), (False, 2.)):
            with self.subTest(success=success, cost=cost):
                model, result = self.execute(fallback_success=success, fallback_cost=cost)
                runtime = SimpleNamespace(branch=SimpleNamespace(policy=policy(model)),
                    args=SimpleNamespace(problem="diffusion"),
                    method=SimpleNamespace(setup_kind="linucb", setup_context="full"))
                record = dict(mkw={}, context=CONTEXT, outcome=result.outcome)
                with patch.object(model_audit, "context_for_setup_method", return_value=CONTEXT):
                    checked = model_audit.audit_final_model(runtime, [record])
                    self.assertTrue(checked["passed"])
                    self.assertEqual(checked["observations"], 3 + int(success))
                    self.assertEqual(checked["failure_observations"], 3)
                    if success:
                        broken = copy.deepcopy(record)
                        broken["outcome"]["fallback_learning_cost"] = None
                        with self.assertRaises(ValueError):
                            model_audit.audit_final_model(runtime, [broken])

    def test_success_stops_before_fallback(self):
        model, result = self.execute(success_on=0)
        self.assertFalse(result.outcome["fallback_used"])
        self.assertEqual(result.outcome["bandit_observation_count"], 1)
        self.assertIsNone(result.outcome["fallback_learning_cost"])
        self.assert_regression(model, result)

    def test_zero_cost_success_still_counts(self):
        model, result = self.execute(fallback_cost=0.)
        self.assertEqual(result.outcome["bandit_observation_count"], 4)
        self.assertEqual(result.outcome["fallback_learning_cost"], 0.)
        self.assert_regression(model, result)

    def test_default_selection_and_fallback_are_two_executions(self):
        model, result = self.execute(default_first=True)
        self.assertEqual(result.outcome["primary_attempts"][0]["arm_index"],
                         result.outcome["fallback_arm_index"])
        self.assert_regression(model, result)

    def test_frozen_fallback_does_not_learn_or_change_observation_counts(self):
        for success in (True, False):
            model, result = self.execute(fallback_success=success, learn=False)
            self.assertEqual(result.outcome["bandit_observation_count"], 0)
            self.assertEqual(result.timing["update_sec"], 0.)
            self.assertIsNone(result.outcome["fallback_learning_cost"])
            self.assertEqual(model.t, 0)
            np.testing.assert_array_equal(model.A_inv, np.eye(model.d_phi))
            np.testing.assert_array_equal(model.b, 0.)
            np.testing.assert_array_equal(model.failure_b, 0.)

    def test_default_fallback_matches_an_ordinary_successful_selection(self):
        for compact in (False, True):
            fallback = learner(compact=compact, default_first=True)
            ordinary = learner(compact=compact, default_first=True)
            ordinary.predict(CONTEXT)
            observation = ordinary.defer_cost(True)
            ordinary.resolve_cost(observation, 2.)
            fallback.update_default_fallback(CONTEXT, DEFAULT_SETUP_PARAMS, 2.)
            for key in ("A_inv", "b", "failure_b"):
                np.testing.assert_allclose(getattr(fallback, key), getattr(ordinary, key), atol=1e-12)
            self.assertEqual(fallback.t, ordinary.t)
            for key, expected in ordinary._cand.export_statistics().items():
                np.testing.assert_array_equal(fallback._cand.export_statistics()[key], expected)

    def test_default_fallback_preserves_candidate_schedule_and_rng(self):
        with TemporaryDirectory() as directory:
            model = learner(directory=Path(directory))
            cursor = model._candidate_schedule.cursor
            rng = copy.deepcopy(model.rng.bit_generator.state)
            with patch.object(model, "_selection_score_subset", side_effect=AssertionError("Scored Default fallback")):
                observation = model.update_default_fallback(CONTEXT, DEFAULT_SETUP_PARAMS, 2.)
            self.assertEqual(observation.arm_index, model.actions.default_arm_index)
            self.assertEqual(model._candidate_schedule.cursor, cursor)
            self.assertEqual(model.rng.bit_generator.state, rng)
            self.assertEqual(model.candidate_stats_history[-1]["strategy"], "default_fallback")
            model.predict(CONTEXT)
            model.cancel_pending()
            self.assertEqual(len(model.history), 1)
            self.assertEqual(len(model.candidate_stats_history), 1)

    def test_default_fallback_rejects_invalid_or_pending_feedback(self):
        model = learner()
        for cost in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                model.update_default_fallback(CONTEXT, DEFAULT_SETUP_PARAMS, cost)
        with self.assertRaises(ValueError):
            model.update_default_fallback(CONTEXT, dict(DEFAULT_SETUP_PARAMS, strong_threshold=.6), 1.)
        self.assertEqual(model.t, 0)
        self.assertEqual(model.history, [])
        model.predict(CONTEXT)
        with self.assertRaises(RuntimeError):
            model.update_default_fallback(CONTEXT, DEFAULT_SETUP_PARAMS, 1.)

    def test_fallback_observation_survives_checkpoint_roundtrip(self):
        model, _ = self.execute()
        with TemporaryDirectory() as directory:
            path = Path(directory)/"setup.npz"
            model.save_mutable_state(path)
            restored = learner()
            restored.load_mutable_state(path)
            for key in ("A_inv", "b", "failure_b"):
                np.testing.assert_array_equal(getattr(restored, key), getattr(model, key))
            self.assertEqual(restored.t, 4)
            for key, expected in model._cand.export_statistics().items():
                np.testing.assert_array_equal(restored._cand.export_statistics()[key], expected)

    def test_immediate_feedback_path_observes_only_successful_fallback(self):
        for success in (True, False):
            model = learner(rule=None)
            result = run_same_context_setup_reselection(policy=policy(model),
                parameter_space={}, context=CONTEXT, solver_fn=lambda params: outcome(False, 1., "setup"),
                fallback_solver_fn=lambda params: outcome(success, 2.),
                default_params=DEFAULT_SETUP_PARAMS, prev_update_est=.125)
            self.assertEqual(result.outcome["bandit_observation_count"], 3 + int(success))
            self.assertEqual(result.outcome["fallback_learning_cost"], 2. if success else None)
            self.assert_regression(model, result)
            reported = report_online_outcome(result.outcome, bandit_timing=result.timing)
            validate_recovery_stream([dict(problem_index=1, outcome=reported)], expect_bandit_updates=True)

    def test_audit_rejects_missing_duplicate_or_failed_fallback_observations(self):
        for success in (True, False):
            _, result = self.execute(fallback_success=success)
            reported = report_online_outcome(result.outcome, bandit_timing=result.timing)
            broken = copy.deepcopy(reported)
            broken["bandit_observation_count"] += 1
            with self.assertRaises(ValueError):
                audit([dict(problem_index=1, outcome=broken)], VARIANT, feedback_mode="adjacent_delayed")
            broken = copy.deepcopy(reported)
            broken["fallback_learning_cost"] = None if success else 2.
            with self.assertRaises(ValueError):
                audit([dict(problem_index=1, outcome=broken)], VARIANT, feedback_mode="adjacent_delayed")


if __name__ == "__main__":
    unittest.main()
