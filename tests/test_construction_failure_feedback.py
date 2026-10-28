"""Diagnosis-branch checks for construction-aware adjacent setup feedback.

Run from this checkout with:
    python -m unittest discover -s tests -p test_construction_failure_feedback.py
"""
from __future__ import annotations

import ast
import copy
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np

from hypre.bindings import AttemptOutcome, AttemptStatus
from experiments.paper_final.online.case_loop import report_online_outcome
from joint import recovery as feedback
from setup.learners.linucb.feedback import FailureAwareRetryLinUCB
from experiments.paper_final.recovery.audit import audit
from setup.learners.common.action_features import ParameterSpec, ParameterSpaceSpec
from setup.learners.linucb.learner import SharedLinUCB


ROOT = Path(__file__).resolve().parents[1]
BASELINE = subprocess.check_output([
    'git', 'show',
    '67a7890c7a3f0b231679d3e2105a8d15c72a2d77:experiments/paper_final/recovery/setup_feedback.py',
], cwd=ROOT, text=True)
CURRENT = Path(feedback.__file__).read_text()
SUCCESS = AttemptStatus.SUCCESS
CONSTRUCTION = AttemptStatus.SETUP_FAILURE
NONCONVERGENCE = AttemptStatus.NONCONVERGENCE
SOLVE_FAILURE = AttemptStatus.SOLVE_FAILURE
NONFINITE = AttemptStatus.NONFINITE_RESULT


class Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def advance(self, value):
        self.now += value


def load_runner(source, clock):
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'run_deferred_case')
    scope = dict(vars(feedback), time=clock)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<diagnosis-runner>', 'exec'), scope)
    return scope['run_deferred_case']


class Harness:
    def __init__(self, source=CURRENT, *, learn=True, mode='adjacent_delayed', budget=3,
                 real_clock=False):
        self.clock = Clock()
        if real_clock:
            import time
            self.clock.perf_counter = time.perf_counter
        self.run = load_runner(source, self.clock)
        self.context = np.array([0.3])
        self.learn = learn
        self.mode = mode
        self.variant = SimpleNamespace(attempts=budget, default_fallback=True,
                                       rl_recovery_cost=False, unified=True)
        spec = ParameterSpaceSpec((ParameterSpec('strong_threshold', 'continuous',
                                                (.25, .5, .75, 1.0), .25),))
        runtime_model = SharedLinUCB([{'strong_threshold': x} for x in (.25, .5, .75, 1.0)],
                                    1, parameter_spec=spec, context_interaction_indices=(), seed=17)
        self.model = FailureAwareRetryLinUCB.from_runtime_model(runtime_model)
        self.trace = []
        self.wrap('predict', 1/8)
        self.wrap('select_recovery', 1/4)
        self.wrap('defer_cost', 1/32)
        self.wrap('resolve_cost', 1/16)
        self.wrap('update_default_fallback', 1/2)
        self.policy = SimpleNamespace(model=self.model, select=lambda **kw:
                                      (self.model.predict(kw['context']), {}))
        self.previous = 0.0

    def wrap(self, name, duration):
        original = getattr(self.model, name)

        def wrapped(*args, **kwargs):
            self.trace.append(name)
            if name == 'select_recovery' and self.learn:
                # Every preceding failure must reach the binary model before retry selection.
                expected = sum((self.model._phi(self.context, step.arm_index)
                                for step in self.model.history[self.history_start:]),
                               np.zeros_like(self.model.failure_b))
                np.testing.assert_allclose(self.model.failure_b-self.failure_start, expected)
            result = original(*args, **kwargs)
            self.clock.advance(duration)
            return result

        setattr(self.model, name, wrapped)

    def execute(self, statuses, fallback_status=None, *, validate=True):
        self.history_start = len(self.model.history)
        self.failure_start = self.model.failure_b.copy()
        self.local_index = 0

        def before_attempt(index):
            self.local_index = index
            self.clock.advance(1/64)

        def result(status, index, fallback=False):
            native = float(2**index)
            setup = native if status == CONSTRUCTION else native/4
            solve = 0.0 if status == CONSTRUCTION else native-setup
            return AttemptOutcome(status, setup, solve,
                                  0.0 if fallback or status == CONSTRUCTION else 1/32,
                                  residual_norm=1e-10 if status == SUCCESS else 1.0,
                                  cycles=0 if status == CONSTRUCTION else 2)

        def fallback():
            assert fallback_status is not None, 'Unexpected fallback'
            return result(fallback_status, len(statuses), True)

        outcome = self.run(policy=self.policy, parameter_space=None, context=self.context,
                           solver=lambda choice: result(statuses[self.local_index], self.local_index),
                           fallback_solver=fallback, variant=self.variant,
                           previous_update=self.previous, before_attempt=before_attempt,
                           feedback_mode=self.mode, learn=self.learn)
        reported = report_online_outcome(outcome.outcome, bandit_timing=outcome.timing)
        if validate:
            audit([dict(problem_index=1, outcome=reported)], self.variant,
                  feedback_mode=self.mode, learn=self.learn)
        for attempt in outcome.outcome['primary_attempts']:
            assert attempt['attempt_cost'] == sum(attempt[key] for key in
                ('setup_runtime', 'solve_runtime', 'controller_runtime', 'selection_runtime')) + self.previous
        self.previous = outcome.update_runtime_sec
        return outcome, reported

    def state(self):
        return (self.model.A_inv, self.model.b, self.model.failure_b, self.model.t,
                json.dumps(self.model.rng.bit_generator.state, sort_keys=True),
                [(step.arm_index, step.loss) for step in self.model.history])


class ConstructionFeedbackTests(unittest.TestCase):
    def check_targets(self, result, terms):
        rows = result.outcome['primary_attempts']
        costs = [row['attempt_cost'] for row in rows]
        if result.outcome['fallback_used']:
            costs.append(sum(result.outcome[f'fallback_{part}_runtime']
                             for part in ('setup', 'solve', 'controller')))
        self.assertEqual([row['learning_cost'] for row in rows],
                         [sum(costs[index] for index in selected) for selected in terms])

    def check_timing(self, result):
        rows = result.outcome['primary_attempts']
        self.assertEqual(result.timing['select_sec'], sum(row['selection_runtime'] for row in rows))
        self.assertEqual(result.timing['loss_eval_sec'], sum(row['loss_eval_runtime'] for row in rows))
        self.assertEqual(result.timing['update_sec'], sum(row['update_runtime'] for row in rows)
                         + result.outcome['fallback_update_runtime'])
        self.assertEqual(result.update_runtime_sec, result.timing['update_sec'])
        self.assertEqual(result.timing['overhead_sec'], sum(result.timing[key] for key in
                         ('select_sec', 'loss_eval_sec', 'update_sec')))
        self.assertTrue(all(value >= 0 for value in result.timing.values()))

    def check_model(self, harness, result):
        rows = result.outcome['primary_attempts']
        features = [harness.model._phi(harness.context, row['arm_index']) for row in rows]
        targets = [row['learning_cost'] for row in rows]
        failed = [float(row['status'] != 'success') for row in rows]
        if result.outcome['fallback_learning_cost'] is not None:
            features.append(harness.model._phi(harness.context, result.outcome['fallback_arm_index']))
            targets.append(result.outcome['fallback_learning_cost'])
            failed.append(0.0)
        phi = np.asarray(features)
        np.testing.assert_allclose(harness.model.b, phi.T @ targets, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(harness.model.failure_b, phi.T @ failed, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(harness.model.A_inv,
                                  np.linalg.inv(harness.model.l2_reg*np.eye(harness.model.d_phi)+phi.T @ phi),
                                  rtol=1e-12, atol=1e-12)
        self.assertEqual(harness.model.t, len(features))

    def test_explicit_construction_sequences_and_batch_model(self):
        cases = [
            ((SUCCESS,), None, [(0,)]),
            ((CONSTRUCTION, SUCCESS), None, [(0, 1), (1,)]),
            ((CONSTRUCTION, CONSTRUCTION, SUCCESS), None, [(0, 1, 2), (1, 2), (2,)]),
            ((CONSTRUCTION, CONSTRUCTION, NONCONVERGENCE), SUCCESS, [(0, 1, 2), (1, 2), (2, 3)]),
            ((CONSTRUCTION, CONSTRUCTION, NONCONVERGENCE), NONCONVERGENCE, [(0, 1, 2), (1, 2), (2, 3)]),
            ((CONSTRUCTION, NONCONVERGENCE, NONCONVERGENCE), SUCCESS, [(0, 1), (1, 2), (2, 3)]),
            ((NONCONVERGENCE, CONSTRUCTION, CONSTRUCTION), SUCCESS, [(0, 1, 2, 3), (1, 2, 3), (2, 3)]),
            ((NONCONVERGENCE, CONSTRUCTION, CONSTRUCTION), CONSTRUCTION, [(0, 1, 2, 3), (1, 2, 3), (2, 3)]),
            ((NONCONVERGENCE, CONSTRUCTION, CONSTRUCTION), NONCONVERGENCE, [(0, 1, 2, 3), (1, 2, 3), (2, 3)]),
            ((NONCONVERGENCE, CONSTRUCTION, SUCCESS), None, [(0, 1, 2), (1, 2), (2,)]),
            ((NONCONVERGENCE, CONSTRUCTION, NONCONVERGENCE), SUCCESS, [(0, 1, 2), (1, 2), (2, 3)]),
            ((SOLVE_FAILURE, CONSTRUCTION, NONFINITE), SUCCESS, [(0, 1, 2), (1, 2), (2, 3)]),
            ((NONFINITE, CONSTRUCTION, NONCONVERGENCE), SUCCESS, [(0, 1, 2), (1, 2), (2, 3)]),
            ((SOLVE_FAILURE, CONSTRUCTION, CONSTRUCTION), CONSTRUCTION, [(0, 1, 2, 3), (1, 2, 3), (2, 3)]),
            ((CONSTRUCTION, CONSTRUCTION, CONSTRUCTION), CONSTRUCTION, [(0, 1, 2, 3), (1, 2, 3), (2, 3)]),
            ((CONSTRUCTION, CONSTRUCTION, CONSTRUCTION), NONCONVERGENCE, [(0, 1, 2, 3), (1, 2, 3), (2, 3)]),
            ((CONSTRUCTION, CONSTRUCTION, CONSTRUCTION), SUCCESS, [(0, 1, 2, 3), (1, 2, 3), (2, 3)]),
            ((CONSTRUCTION, CONSTRUCTION, SOLVE_FAILURE), SUCCESS, [(0, 1, 2), (1, 2), (2, 3)]),
            ((CONSTRUCTION, CONSTRUCTION, NONFINITE), SUCCESS, [(0, 1, 2), (1, 2), (2, 3)]),
        ]
        for statuses, fallback, terms in cases:
            with self.subTest(statuses=statuses, fallback=fallback):
                harness = Harness()
                result, _ = harness.execute(statuses, fallback)
                self.check_targets(result, terms)
                self.check_timing(result)
                self.check_model(harness, result)

    def test_all_failure_sequences_stop_at_next_built_hierarchy(self):
        failures = (CONSTRUCTION, NONCONVERGENCE, SOLVE_FAILURE, NONFINITE)
        cases = []
        for length in (1, 2, 3):
            cases.extend((prefix+(SUCCESS,), None)
                         for prefix in itertools.product(failures, repeat=length-1))
        cases.extend((statuses, fallback)
                     for statuses in itertools.product(failures, repeat=3)
                     for fallback in (SUCCESS, *failures))
        self.assertEqual(len(cases), 341)
        for statuses, fallback in cases:
            with self.subTest(statuses=statuses, fallback=fallback):
                sequence = statuses+(() if fallback is None else (fallback,))
                terms = []
                for index, status in enumerate(statuses):
                    selected = [index]
                    if status != SUCCESS:
                        for following in range(index+1, len(sequence)):
                            selected.append(following)
                            if sequence[following] != CONSTRUCTION:
                                break
                    terms.append(selected)
                harness = Harness()
                result, _ = harness.execute(statuses, fallback)
                self.check_targets(result, terms)
                self.check_timing(result)
                self.check_model(harness, result)

    def test_no_construction_failures_match_main_baseline(self):
        cases = []
        failures = (NONCONVERGENCE, SOLVE_FAILURE, NONFINITE)
        for length in (1, 2, 3):
            cases.extend((prefix+(SUCCESS,), None)
                         for prefix in itertools.product(failures, repeat=length-1))
        cases.extend((statuses, fallback)
                     for statuses in itertools.product(failures, repeat=3)
                     for fallback in (SUCCESS, NONCONVERGENCE, SOLVE_FAILURE, NONFINITE))
        for learn in (True, False):
            for statuses, fallback in cases:
                with self.subTest(learn=learn, statuses=statuses, fallback=fallback):
                    old = Harness(BASELINE, learn=learn)
                    new = Harness(CURRENT, learn=learn)
                    before, _ = old.execute(statuses, fallback)
                    after, _ = new.execute(statuses, fallback)
                    self.assertEqual(json.dumps(before.outcome, sort_keys=True), json.dumps(after.outcome, sort_keys=True))
                    self.assertEqual(before.timing, after.timing)
                    self.assertEqual(old.trace, new.trace)
                    for left, right in zip(old.state(), new.state()):
                        if isinstance(left, np.ndarray):
                            np.testing.assert_array_equal(left, right)
                        else:
                            self.assertEqual(left, right)

    def test_frozen_construction_failures_do_not_update_models(self):
        for fallback in (SUCCESS, NONCONVERGENCE, CONSTRUCTION):
            harness = Harness(learn=False)
            state = [value.copy() for value in harness.state()[:3]]
            result, _ = harness.execute((CONSTRUCTION, CONSTRUCTION, CONSTRUCTION), fallback)
            self.assertTrue(all(row['learning_cost'] is None for row in result.outcome['primary_attempts']))
            self.assertEqual(result.timing['update_sec'], 0)
            self.assertEqual(harness.model.t, 0)
            for before, after in zip(state, harness.state()[:3]):
                np.testing.assert_array_equal(before, after)
            self.check_timing(result)

    def test_suffix_mode_unchanged(self):
        old, new = Harness(BASELINE, mode='suffix_delayed'), Harness(CURRENT, mode='suffix_delayed')
        before, _ = old.execute((CONSTRUCTION, NONCONVERGENCE, CONSTRUCTION), SUCCESS)
        after, _ = new.execute((CONSTRUCTION, NONCONVERGENCE, CONSTRUCTION), SUCCESS)
        self.assertEqual(json.dumps(before.outcome, sort_keys=True), json.dumps(after.outcome, sort_keys=True))
        self.check_targets(after, [(0, 1, 2, 3), (1, 2, 3), (2, 3)])

    def test_single_attempt_budget_and_carried_update(self):
        for fallback in (SUCCESS, NONCONVERGENCE, CONSTRUCTION):
            harness = Harness(budget=1)
            result, _ = harness.execute((CONSTRUCTION,), fallback)
            self.check_targets(result, [(0, 1)])
            carried = result.update_runtime_sec
            followup, _ = harness.execute((SUCCESS,))
            self.assertGreater(carried, 0)
            self.check_timing(followup)

    def test_runtime_audit_rejects_old_short_construction_target(self):
        for failure in (CONSTRUCTION, NONCONVERGENCE, SOLVE_FAILURE, NONFINITE):
            with self.subTest(failure=failure):
                harness = Harness()
                _, reported = harness.execute((failure, CONSTRUCTION, NONCONVERGENCE), SUCCESS)
                bad = copy.deepcopy(reported)
                rows = bad['primary_attempts']
                rows[0]['learning_cost'] = rows[0]['attempt_cost']+rows[1]['attempt_cost']
                with self.assertRaisesRegex(ValueError, 'Setup target does not match'):
                    audit([dict(problem_index=1, outcome=bad)], harness.variant,
                          feedback_mode='adjacent_delayed')

    def test_real_clock_timing_totals_close(self):
        for statuses, fallback in (((CONSTRUCTION, CONSTRUCTION, NONCONVERGENCE), SUCCESS),
                                   ((CONSTRUCTION, CONSTRUCTION, CONSTRUCTION), CONSTRUCTION)):
            harness = Harness(real_clock=True)
            result, _ = harness.execute(statuses, fallback)
            self.check_timing(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
