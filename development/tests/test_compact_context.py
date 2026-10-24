"""Check normalized solve context, time units, checkpoints and native execution."""
import copy
from dataclasses import replace
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from development.compact_context import CompactSolveStateEncoder, log_cycle_time
from development.run_compact_context import build_solve_runtime, check_protocol, configured_runner, prepare, runner
from development.setup_normalization import make_range_normalized_setup_encoder
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace


ROOT = Path(__file__).resolve().parents[2]


def configuration():
    return runner.read(ROOT/'configs/module1/advection_60/seed_1.json')


def encoder(cycle_time_mode="log"):
    raw = configuration()
    space = SetupConfigurationSpace.from_mapping('expanded', raw['setup']['configuration_spaces']['expanded_no_endpoints'])
    return CompactSolveStateEncoder(cycle_time_mode=cycle_time_mode, tol=1e-6,
        max_cycles=50, c_max=1000, mode='setup_full',
        setup_obs_encoder=make_range_normalized_setup_encoder(configuration_space=space, parameter_resolution=20),
        problem_context_mode='canonical_no_c_mean', weight_bounds=(1., 3.))


def observation(**changes):
    values = dict(mkw=dict(nx=60, ny=60, nz=60, k=10., c=20., a0=30., a1=40., a2=50., a3=60.),
        setup_params=dict(DEFAULT_SETUP_PARAMS), initial_residual=1., residual=.01,
        previous_residual=.1, cycle=2, last_weight=1.5, last_cycle_time=.04, first_cycle_time=.02)
    values.update(changes)
    return values


class CompactContextTests(unittest.TestCase):
    def test_log_time_is_absolute_unclipped_and_independent_of_first_cycle(self):
        value = encoder()
        for seconds in (0., 1e-9, .001, .01, .1, 1., 1000.):
            a = value.encode(**observation(last_cycle_time=seconds, first_cycle_time=.001))
            b = value.encode(**observation(last_cycle_time=seconds, first_cycle_time=10.))
            self.assertAlmostEqual(a[5], math.log1p(seconds/.001))
            np.testing.assert_array_equal(a, b)
        self.assertEqual(value.encode(**observation(cycle=0, last_cycle_time=0., first_cycle_time=None))[5], 0.)
        self.assertGreater(log_cycle_time(1.), 1.)
        self.assertAlmostEqual(log_cycle_time(.001), math.log(2.))
        for seconds in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                log_cycle_time(seconds)

    def test_time_modes_differ_only_in_timing_and_reject_mixed_checkpoints(self):
        log, ratio = encoder(), encoder("first_cycle_ratio")
        for cycle in (0, 1, 5, 49):
            for seconds in (.001, .04, 1.):
                a = log.encode(**observation(cycle=cycle, last_cycle_time=seconds))
                b = ratio.encode(**observation(cycle=cycle, last_cycle_time=seconds))
                np.testing.assert_array_equal(np.delete(a, 5), np.delete(b, 5))
                self.assertEqual(a.shape, b.shape)
                self.assertAlmostEqual(b[5], 0. if cycle == 0 else seconds/.02)
        self.assertGreater(ratio.encode(**observation(last_cycle_time=10.))[5], 1.)
        spec = runner.parse_joint_experiment_config(configuration())
        args = runner.runtime_config_from_spec(spec)
        bundles = [build_solve_runtime(args, specs=spec.methods, cycle_time_mode=mode).controller_bundles
                   for mode in ("log", "first_cycle_ratio")]
        for method in (runner.factorial.METHODS[1], runner.factorial.METHODS[3]):
            with TemporaryDirectory() as directory:
                path = Path(directory)/"controller.npz"
                bundles[0][method].save(path)
                with self.assertRaises(ValueError):
                    bundles[1][method].load(path)

    def test_all_other_basic_features_match_and_remain_unclipped(self):
        value = encoder()
        old = value.base
        for cycle, residual, previous in ((0, 1., 1.), (3, 100., .1), (51, 1e-8, .01)):
            kwargs = observation(cycle=cycle, residual=residual, previous_residual=previous)
            expected = old.encode(**{k:v for k,v in kwargs.items() if k != "first_cycle_time"})
            actual = value.encode(**kwargs)
            self.assertEqual(actual.size, 32)
            np.testing.assert_array_equal(np.delete(actual, 5), np.delete(expected, 5))
            np.testing.assert_array_equal(actual[13:], old.setup_obs_encoder.encode_setup(DEFAULT_SETUP_PARAMS))
        self.assertGreater(value.encode(**observation(residual=100.))[2], 1.)
        self.assertLess(value.encode(**observation(residual=1e-8))[2], 0.)
        np.testing.assert_array_equal(value.constant_value_parameters(4.), np.r_[4., np.zeros(31)])

    def test_checkpoint_records_features_and_rejects_changed_timing(self):
        spec = runner.parse_joint_experiment_config(configuration())
        args = runner.runtime_config_from_spec(spec)
        built = build_solve_runtime(args, specs=spec.methods).controller_bundles
        fixed, joint = (built[n] for n in (runner.factorial.METHODS[1], runner.factorial.METHODS[3]))
        self.assertEqual((fixed.encoder.feature_dim, joint.encoder.feature_dim), (32, 32))
        self.assertEqual((fixed.controller.joint_dim, joint.controller.joint_dim), (288, 288))
        for bundle in (fixed, joint):
            metadata = bundle.protocol_metadata()['state_encoder']
            self.assertEqual(metadata['cycle_time_reference_sec'], .001)
            self.assertEqual(metadata['setup_descriptor_dim'], 19)
            self.assertFalse(metadata['cycle_time_clipping'])
            with TemporaryDirectory() as directory:
                path = Path(directory)/'controller.npz'
                bundle.save(path)
                bundle.load(path)
                saved = runner.read(path.with_suffix('.npz.encoder.json'))
                saved['cycle_time_normalization'] = 'first_native_cycle_per_attempt'
                runner.dump(path.with_suffix('.npz.encoder.json'), saved)
                with self.assertRaisesRegex(ValueError, 'state encoder does not match'):
                    bundle.load(path)

    def test_native_four_branch_worker(self):
        for mode in ("log", "first_cycle_ratio"):
            with self.subTest(mode=mode):
                self._check_native_worker(mode)

    def _check_native_worker(self, mode):
        raw = copy.deepcopy(configuration())
        raw.pop('frozen_stream')
        raw['problem']['grid'] = [12, 12, 12]
        raw['stream']['expected_sha256'] = ''
        spec = runner.parse_joint_experiment_config(raw)
        args = runner.runtime_config_from_spec(spec)
        runner.configure_paired_environment(args)
        _, manifest = runner.build_paired_instance_stream(args)
        raw['stream']['expected_sha256'] = manifest['sha256']
        original_decode = runner.decode
        def short_decode(config, directory):
            runtime = original_decode(config, directory)
            runtime.stream = runtime.stream[:5]
            runtime.args = replace(runtime.args, online_cases=5)
            return runtime
        with TemporaryDirectory(dir=ROOT/'results', prefix='compact_context_validation_') as directory:
            base = Path(directory)
            runner.dump(base/'config.json', raw)
            runner.dump(base/'suite.json', dict(name='Compact context native validation',
                seeds=[dict(label='1', config=str(base/'config.json'))]))
            output = base/'run'
            with patch.object(runner, 'decode', short_decode):
                prepare(base/'suite.json', output, cases=5, cycle_time_mode=mode)
                self.assertTrue(check_protocol(output))
                runner.dump(output/'release.json', {'validation_only': True})
                with configured_runner(cycle_time_mode=mode):
                    runner.worker(output, '1')
            stage = output/'seed_1'
            self.assertTrue(runner.read(stage/'result.json')['passed'])
            self.assertEqual(runner.read(stage/'status.json')['processed'], 5)
            self.assertFalse((stage/'freeze_boundary.json').exists())
            audit = runner.read(stage/'compact_context_audit.json')
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['controllers']['bandit_lstdq']['state_features'], 32)
            self.assertEqual(audit['controllers']['default_setup_lstdq']['state_features'], 32)
            self.assertEqual(runner.read(output/'protocol.json')['measured_frozen_inputs'], [])
            for method in runner.factorial.METHODS:
                rows = list(runner.read_records(stage/f'trajectories/{method}.jsonl'))
                self.assertEqual([r['problem_index'] for r in rows], list(range(1, 6)))
                if method != runner.factorial.METHODS[0]:
                    self.assertTrue(all(r['outcome']['end_to_end_runtime'] > r['outcome']['runtime'] for r in rows))


if __name__ == '__main__':
    unittest.main()
