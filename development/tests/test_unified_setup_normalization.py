"""Verify shared setup scaling against a selected algorithm checkout."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('AUTOTUNER_TEST_ROOT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
from experiments.runtime import configure_single_thread
configure_single_thread()
import numpy as np
from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.learners.common import GenericActionFeatureEncoder, FactorizedActionFeatureCache
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace, build_setup_parameter_spec, build_setup_param_space


def configuration():
    return json.loads((ROOT / 'configs/module1/diffusion_60/seed_3.json').read_text())


def space_from(raw):
    name, values = next(iter(raw['setup']['configuration_spaces'].items()))
    return SetupConfigurationSpace.from_mapping(name, values)


def old_module(name, path):
    module = types.ModuleType(name)
    sys.modules[name] = module
    source = subprocess.check_output(['git', 'show', f'HEAD:{path}'], cwd=ROOT, text=True)
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


class UnifiedSetupNormalizationTests(unittest.TestCase):
    def test_all_configured_spaces_share_scaling_and_feature_blocks(self):
        signatures = set()
        for path in sorted((ROOT / 'configs').glob('module*/**/seed_*.json')):
            raw = json.loads(path.read_text())
            if 'configuration_spaces' not in raw.get('setup', {}):
                continue
            signature = json.dumps(raw['setup'], sort_keys=True)
            if signature in signatures:
                continue
            signatures.add(signature)
            space = space_from(raw)
            resolution = raw['setup']['parameter_resolution']
            catalog = build_setup_param_space(tune_dim=7, configuration_space=space,
                                             parameter_resolution=resolution, materialize=False)
            bandit = GenericActionFeatureEncoder(catalog.parameter_spec)
            solve = make_setup_feature_encoder(configuration_space=space, parameter_resolution=resolution)
            self.assertEqual(bandit.parameter_spec, solve.parameter_spec)
            self.assertEqual((bandit.feature_dim, solve.descriptor_dim), (104, 19))
            for arm in [catalog.default_arm_index, *np.random.default_rng(7).integers(len(catalog.actions), size=100)]:
                action = catalog.actions[int(arm)]
                features, descriptor = bandit.encode_action(action), solve.encode_setup(action)
                np.testing.assert_array_equal(features[:5], descriptor[:5])
                np.testing.assert_array_equal(features[20:34], descriptor[5:])
                self.assertTrue(np.all(np.abs(features) <= 1 + 1e-12))
            for index, parameter in enumerate(bandit.numeric_params):
                allowed = (*parameter.values, parameter.default)
                low, high = min(allowed), max(allowed)
                for value in allowed:
                    action = dict(DEFAULT_SETUP_PARAMS, **{parameter.name: value})
                    expected = 2 * (value - low) / (high - low) - 1 if high > low else 0
                    self.assertAlmostEqual(bandit.encode_setup(action)[index], expected)
        self.assertTrue(signatures)

    def test_solve_descriptors_are_bitwise_unchanged(self):
        old_space = old_module('_normalization_before_space', 'setup/space.py')
        old_factory = old_module('_normalization_before_factory', 'experiments/paper_final/online/controllers.py')
        old_factory.build_setup_parameter_spec = old_space.build_setup_parameter_spec
        space = space_from(configuration())
        before = old_factory.make_setup_feature_encoder(configuration_space=space, parameter_resolution=20)
        after = make_setup_feature_encoder(configuration_space=space, parameter_resolution=20)
        self.assertEqual(before.parameter_spec, after.parameter_spec)
        catalog = build_setup_param_space(tune_dim=7, configuration_space=space, materialize=False)
        for arm in [catalog.default_arm_index, *np.random.default_rng(8).integers(len(catalog.actions), size=100)]:
            action = catalog.actions[int(arm)]
            np.testing.assert_array_equal(before.encode_setup(action), after.encode_setup(action))

    def test_factorized_cache_matches_direct_features(self):
        for space in (space_from(configuration()), SetupConfigurationSpace(
            name='small', coarsen_types=(6, 10), interp_types=(6, 8), agg_interp_types=(4, 6),
            numeric_grids={'strong_threshold': (.1, .95), 'max_row_sum': (.2, .95),
                           'trunc_factor': (0.,), 'P_max_elmts': (4,), 'agg_num_levels': (0, 1)})):
            catalog = build_setup_param_space(tune_dim=7, configuration_space=space, materialize=False)
            encoder = GenericActionFeatureEncoder(catalog.parameter_spec)
            cache = FactorizedActionFeatureCache(catalog.actions, encoder)
            ids = np.asarray([catalog.default_arm_index, 0, len(catalog.actions)-1,
                              *np.random.default_rng(9).integers(len(catalog.actions), size=256)])
            expected = encoder.encode_actions([catalog.actions[int(i)] for i in ids])
            np.testing.assert_array_equal(cache.features(ids), expected)

    def test_default_outside_grid_and_constant_ranges(self):
        spec, _ = build_setup_parameter_spec(tune_dim=5, numeric_grids={
            'strong_threshold': (.5, .75), 'max_row_sum': (.9,), 'trunc_factor': (0.,),
            'P_max_elmts': (4,), 'agg_num_levels': (0,)})
        encoder = GenericActionFeatureEncoder(spec)
        np.testing.assert_array_equal(encoder.encode_setup(DEFAULT_SETUP_PARAMS), [-1, 0, 0, 0, 0])
        self.assertEqual(encoder.encode_setup(dict(DEFAULT_SETUP_PARAMS, strong_threshold=.75))[0], 1.)
        self.assertTrue(np.isfinite(encoder.encode_action(DEFAULT_SETUP_PARAMS)).all())

    def test_conditional_numeric_and_categorical_masks(self):
        spec, _ = build_setup_parameter_spec(tune_dim=7, tune7_variant='agg_conditional')
        encoder = GenericActionFeatureEncoder(spec)
        inactive = dict(DEFAULT_SETUP_PARAMS, agg_tr=.1, agg_Pmx=4)
        np.testing.assert_array_equal(encoder.encode_setup(inactive)[-2:], [0, 0])
        active = dict(inactive, agg_num_levels=1)
        np.testing.assert_array_equal(encoder.encode_setup(active)[-2:], [1, 1])
        encoder = make_setup_feature_encoder(configuration_space=space_from(configuration()))
        descriptor = encoder.encode_setup(dict(DEFAULT_SETUP_PARAMS, agg_interp_type=7))
        np.testing.assert_array_equal(descriptor[-2:], [0, 0])
        descriptor = encoder.encode_setup(dict(DEFAULT_SETUP_PARAMS, agg_num_levels=1, agg_interp_type=7))
        np.testing.assert_array_equal(descriptor[-2:], [0, 1])

    def test_native_four_branch_run_and_freeze(self):
        from experiments.paper_final import run_online as runner
        self.assertEqual(runner.ROOT.resolve(), ROOT)
        raw = configuration()
        raw['problem']['grid'] = [12, 12, 12]
        for field in ('stream', 'frozen_stream'):
            separate = copy.deepcopy(raw)
            separate['stream'] = separate[field]
            separate.pop('frozen_stream')
            separate['stream']['expected_sha256'] = ''
            spec = runner.parse_joint_experiment_config(separate)
            args = runner.runtime_config_from_spec(spec)
            runner.configure_paired_environment(args)
            _, manifest = runner.build_paired_instance_stream(args)
            raw[field]['expected_sha256'] = manifest['sha256']
        decode, initialize = runner.decode, runner.initialize
        inspected = []
        def short_decode(config, directory):
            runtime = decode(config, directory)
            runtime.stream = runtime.stream[:3] + runtime.stream[5000:5002]
            runtime.args = replace(runtime.args, online_cases=5)
            runtime.freeze_after = 3
            return runtime
        def checked_initialize(output, label):
            runtimes = initialize(output, label)
            fixed_rl = runtimes[runner.factorial.METHODS[1]].bundle.encoder.setup_obs_encoder
            for name in runner.factorial.METHODS[2:]:
                encoder = runtimes[name].branch.policy.model._encoder
                self.assertEqual(encoder.parameter_spec, fixed_rl.parameter_spec)
            joint_rl = runtimes[runner.factorial.METHODS[3]].bundle.encoder.setup_obs_encoder
            self.assertEqual(joint_rl.parameter_spec, fixed_rl.parameter_spec)
            inspected.append(True)
            return runtimes
        with TemporaryDirectory(prefix='unified_setup_normalization_') as directory:
            base = Path(directory)
            runner.dump(base/'config.json', raw)
            runner.dump(base/'suite.json', {'name': 'Shared normalization smoke',
                'seeds': [{'label': '3', 'config': str(base/'config.json')}]})
            output = base/'run'
            with patch.object(runner, 'decode', short_decode), patch.object(runner, 'initialize', checked_initialize):
                runner.prepare(base/'suite.json', output, cases=5)
                runner.dump(output/'release.json', {'validation_only': True})
                runner.worker(output, '3')
            self.assertEqual(inspected, [True])
            stage = output/'seed_3'
            self.assertEqual(runner.read(stage/'status.json')['status'], 'complete')
            self.assertTrue(runner.read(stage/'frozen_state_audit.json')['passed'])
            for method in runner.factorial.METHODS:
                rows = list(runner.read_records(stage/f'trajectories/{method}.jsonl'))
                self.assertEqual([r['problem_index'] for r in rows], list(range(1, 6)))
                for row in rows[3:]:
                    self.assertFalse(row['outcome']['bandit_update_committed'])
                    self.assertFalse(row['outcome']['controller_update_committed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
