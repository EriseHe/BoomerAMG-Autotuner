"""Frozen deployment must retain retries without entering a learning method."""
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.paper_final.recovery.retry_comparison import DeferredCostLinUCB, run_deferred_case
from experiments.paper_final.recovery.metrics import VARIANT
from experiments.paper_final.recovery.execution import EpisodeRecorder


class FrozenSelector(DeferredCostLinUCB):
    def __init__(self):
        self.t = 17
        self._last_arm = None
        self.choices = []

    def choose(self, arm):
        self._last_arm = arm
        self.choices.append(arm)
        return dict(arm=arm), {}

    def select_recovery(self, context, *, excluded_arms):
        assert self._last_arm is None
        return self.choose(len(excluded_arms))

    def cancel_pending(self):
        self._last_arm = None

    def defer_cost(self, succeeded):
        raise AssertionError('Frozen evaluation attempted setup learning')

    def resolve_cost(self, observation, loss):
        raise AssertionError('Frozen evaluation attempted target feedback')


class FrozenFeedbackTests(unittest.TestCase):
    def run_case(self, success_on=None, fallback_success=True):
        model = FrozenSelector()
        policy = SimpleNamespace(model=model, select=lambda **kwargs:model.choose(0))
        events = []

        def outcome(success):
            return dict(runtime=.1, setup_runtime=.02, solve_runtime=.08, infer_runtime=0.,
                failed=not success, failure_stage='' if success else 'solve',
                failure_reason='' if success else 'synthetic_failure', iterations=2,
                residual_norm=1e-7 if success else 1., cycle_actions=[],cycle_times=[],cycle_residuals=[])

        def fallback():
            events.append('fallback')
            return outcome(fallback_success)

        result = run_deferred_case(policy=policy, parameter_space={}, context=np.ones(1),
            solver=lambda params:outcome(params['arm']==success_on), fallback_solver=fallback,
            variant=VARIANT, previous_update=0., before_attempt=lambda index:events.append(index),
            feedback_mode='adjacent_delayed', learn=False)
        self.assertEqual(model.t,17)
        self.assertIsNone(model._last_arm)
        self.assertFalse(result.outcome['bandit_update_committed'])
        self.assertEqual(result.outcome['bandit_observation_count'],0)
        self.assertEqual(result.timing['update_sec'],0.)
        self.assertTrue(all(a['learning_cost'] is None for a in result.outcome['primary_attempts']))
        return model, events, result

    def test_success_does_not_update(self):
        model, events, result = self.run_case(success_on=0)
        self.assertEqual(model.choices,[0])
        self.assertEqual(events,[0])
        self.assertFalse(result.outcome['failed'])

    def test_frozen_reselection_excludes_failures_and_stops_on_success(self):
        model, events, result = self.run_case(success_on=2)
        self.assertEqual(model.choices,[0,1,2])
        self.assertEqual(events,[0,1,2])
        self.assertFalse(result.outcome['fallback_used'])

    def test_fallback_and_unrecovered_failure_preserve_model(self):
        for recovered in (True,False):
            with self.subTest(recovered=recovered):
                model, events, result = self.run_case(fallback_success=recovered)
                self.assertEqual(model.choices,[0,1,2])
                self.assertEqual(events,[0,1,2,'fallback'])
                self.assertEqual(result.outcome['failed'],not recovered)

    def test_frozen_recorder_needs_no_learning_snapshot(self):
        controller = SimpleNamespace(snapshot_learning_state=lambda: self.fail('Unneeded learning snapshot'))
        recorder = EpisodeRecorder(controller,learn=False)
        self.assertIsNone(recorder.snapshot)
        self.assertEqual(recorder.finalize([object(),object()],None,include_recovery=False),([0.,0.],0))


if __name__ == '__main__':
    unittest.main()
