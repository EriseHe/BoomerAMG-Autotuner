"""Protect stream continuity and freeze both live learners without a reload."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final import run_online as runner
from experiments.paper_final.online.configuration import (
    FrozenEvaluationSpec, evaluation_index_offset, parse_joint_experiment_config,
)
from solve.controllers.common.state_encoder import STATE_ENCODING_VERSION


ROOT = Path(__file__).resolve().parents[2]


def configuration():
    raw = runner.read(ROOT/'data/module1/configurations/diffusion_60/seed_4.json')
    raw['solve']['state_encoding'] = STATE_ENCODING_VERSION
    raw['solve']['cost_target'] = 'native_plus_controller'
    raw['frozen_stream'] = runner.read(ROOT/'data/module1/frozen/configurations/diffusion_60/seed_4.json')['stream']
    return raw


class ContinuousModuleOneTests(unittest.TestCase):
    def test_stream_preserves_both_pinned_blocks_without_reshuffling(self):
        raw = configuration()
        full = runner.decode(raw, ROOT/'results/unused-test-path')
        old = copy.deepcopy(raw)
        old.pop('frozen_stream')
        training = runner.decode(old, ROOT/'results/unused-test-path')
        old['stream'] = raw['frozen_stream']
        tail = runner.decode(old, ROOT/'results/unused-test-path')
        self.assertEqual(full.freeze_after, 5000)
        self.assertEqual(full.args.online_cases, 6000)
        self.assertTrue(full.args.continuous_stream)
        self.assertEqual(runner.digest(runner.input_rows(full, 5000)),
                         runner.digest(runner.input_rows(training, 5000)))
        self.assertEqual([runner.digest(mkw) for mkw, _ in full.stream[5000:]],
                         [runner.digest(mkw) for mkw, _ in tail.stream])
        self.assertEqual([row['problem_index'] for row in runner.input_rows(full, 6000)], list(range(1,6001)))

    def test_continuous_and_standalone_frozen_indices_differ(self):
        args = runner.decode(configuration(), ROOT/'results/unused-test-path').args
        frozen = replace(args, evaluation=FrozenEvaluationSpec(Path('.'), Path('config'), 5000, False))
        self.assertEqual(evaluation_index_offset(frozen), 0)
        self.assertEqual(evaluation_index_offset(replace(frozen, continuous_stream=False)), 5000)

    def test_invalid_phase_lengths_are_rejected(self):
        raw = configuration()
        raw['frozen_stream'].update(cases=2000, online_cases=2000)
        with self.assertRaisesRegex(ValueError, '5,000 learning inputs'):
            parse_joint_experiment_config(raw)

    def test_freeze_preserves_live_objects_and_guards_every_update(self):
        args = runner.decode(configuration(), ROOT/'results/unused-test-path').args
        with TemporaryDirectory() as directory:
            stage = Path(directory)
            model = SimpleNamespace(A_inv=np.eye(2), b=np.ones(2), failure_b=np.ones(2),
                _last_arm=None, _last_phi=None, history=[object()], rng=np.random.default_rng(42))
            model.save_mutable_state = lambda path: np.savez(path, A_inv=model.A_inv, b=model.b,
                                                           failure_b=model.failure_b)
            controller = SimpleNamespace(theta=np.ones(2), a_matrix=np.eye(2), a_inverse=np.eye(2),
                b=np.ones(2), episode_moment_covariance=np.eye(2), rng=np.random.default_rng(23))
            def save_controller(path):
                np.savez(path, theta=controller.theta, a_matrix=controller.a_matrix,
                         a_inverse=controller.a_inverse, b=controller.b,
                         episode_moment_covariance=controller.episode_moment_covariance)
            bundle = SimpleNamespace(controller=controller, save=save_controller)
            runtime = SimpleNamespace(args=args, bundle=bundle, branch=SimpleNamespace(policy=SimpleNamespace(model=model)))
            original = (id(runtime), id(model), id(controller), id(model.history), id(model.rng), id(controller.rng))
            runner.freeze_live_runtimes(stage, {'bandit_lstdq': runtime}, 5000)
            self.assertEqual(original, (id(runtime), id(model), id(controller), id(model.history), id(model.rng), id(controller.rng)))
            self.assertFalse(runtime.args.evaluation.explore)
            for field in ('update', 'defer_cost', 'resolve_cost'):
                with self.assertRaisesRegex(RuntimeError, 'update was attempted'):
                    getattr(model, field)()
            with self.assertRaisesRegex(RuntimeError, 'update was attempted'):
                controller.update()
            for array in (model.A_inv, model.b, model.failure_b, controller.theta, controller.a_matrix,
                          controller.a_inverse, controller.b, controller.episode_moment_covariance):
                self.assertFalse(array.flags.writeable)
            runner.save_checkpoints(stage, {'bandit_lstdq': runtime}, 'final')
            self.assertTrue(runner.frozen_state_audit(stage, {'bandit_lstdq': runtime}, continuous=True)['passed'])
            controller.theta.setflags(write=True)
            controller.theta[0] += 1
            runner.save_checkpoints(stage, {'bandit_lstdq': runtime}, 'final')
            with self.assertRaisesRegex(ValueError, 'Frozen learner state changed'):
                runner.frozen_state_audit(stage, {'bandit_lstdq': runtime}, continuous=True)

    def test_native_worker_crosses_boundary_in_one_process_with_overhead(self):
        raw = configuration()
        raw['problem']['grid'] = [12, 12, 12]
        for field in ('stream', 'frozen_stream'):
            separate = copy.deepcopy(raw)
            separate['stream'] = separate[field]
            separate.pop('frozen_stream')
            separate['stream']['expected_sha256'] = ''
            spec = parse_joint_experiment_config(separate)
            args = runner.runtime_config_from_spec(spec)
            runner.configure_paired_environment(args)
            _, manifest = runner.build_paired_instance_stream(args)
            raw[field]['expected_sha256'] = manifest['sha256']
        original_decode = runner.decode
        def diagnostic_decode(config, directory):
            runtime = original_decode(config, directory)
            runtime.stream = runtime.stream[:2] + runtime.stream[5000:5003]
            runtime.args = replace(runtime.args, online_cases=5)
            runtime.freeze_after = 2
            return runtime
        with TemporaryDirectory(dir=ROOT/'results', prefix='continuous_native_validation_') as directory:
            base = Path(directory)
            runner.dump(base/'config.json', raw)
            runner.dump(base/'suite.json', dict(name='Continuous native validation only',
                seeds=[dict(label='4', config=str(base/'config.json'))]))
            output = base/'run'
            with patch.object(runner, 'decode', diagnostic_decode), \
                 patch.object(runner, 'initialize', wraps=runner.initialize) as initialize:
                runner.prepare(base/'suite.json', output, cases=5)
                runner.dump(output/'release.json', {'validation_only': True})
                runner.worker(output, '4')
                self.assertEqual(initialize.call_count, 1)
            stage = output/'seed_4'
            self.assertEqual(runner.read(stage/'status.json')['status'], 'complete')
            self.assertTrue(runner.read(stage/'frozen_state_audit.json')['passed'])
            pids = {event['pid'] for event in runner.read_records(stage/'execution.jsonl')}
            self.assertEqual(len(pids), 1)
            for name in runner.factorial.METHODS:
                rows = list(runner.read_records(stage/f'trajectories/{name}.jsonl'))
                self.assertEqual([r['problem_index'] for r in rows], list(range(1,6)))
                self.assertTrue(all(r['frozen_evaluation'] for r in rows[2:]))
                for row in rows[2:]:
                    out = row['outcome']
                    self.assertFalse(out['bandit_update_committed'])
                    self.assertFalse(out['controller_update_committed'])
                    if name != runner.factorial.METHODS[0]:
                        self.assertGreater(out['end_to_end_runtime'], out['runtime'])
                    for attempt in out['primary_attempts']:
                        self.assertFalse(any(attempt.get('cycle_explored', [])))
