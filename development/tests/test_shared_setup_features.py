"""Verify exact action-feature sharing and the four-branch native diagnostic."""
import copy
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from development.run_shared_setup_features import configured_runner, prepare, runner
from development.shared_setup_features import (
    make_shared_setup_action_encoder, make_range_normalized_setup_action_encoder,
)
from development.setup_normalization import make_range_normalized_setup_encoder
from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.learners.common.aot_candidates import CompactActionCatalog, FactorizedActionFeatureCache
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace
from solve.controllers.common import OnlineControllerFactoryRequest, SharedActionSpec, SolveStateSpec
from solve.controllers.recursive_lstdq import RecursiveLstdqSpec
from solve.controllers.recursive_lstdq.factory import build_recursive_lstdq_controller


ROOT = Path(__file__).resolve().parents[2]


def configuration():
    return runner.read(ROOT/'configs/module1/advection_60/seed_1.json')


def encoders():
    raw = configuration()
    space = SetupConfigurationSpace.from_mapping('expanded', raw['setup']['configuration_spaces']['expanded_no_endpoints'])
    kwargs = dict(configuration_space=space, parameter_resolution=20)
    return make_setup_feature_encoder(**kwargs), make_shared_setup_action_encoder(**kwargs)


def normalized_encoders():
    raw = configuration()
    space = SetupConfigurationSpace.from_mapping('expanded', raw['setup']['configuration_spaces']['expanded_no_endpoints'])
    kwargs = dict(configuration_space=space, parameter_resolution=20)
    return make_range_normalized_setup_encoder(**kwargs), make_range_normalized_setup_action_encoder(**kwargs)


def bundle(encoder):
    return build_recursive_lstdq_controller(OnlineControllerFactoryRequest(
        state=SolveStateSpec(tol=1e-6, max_cycles=50, c_max=1000,
                             problem_context_mode='canonical_no_c_mean'),
        actions=SharedActionSpec(weights=(1., 2., 3.), rbf_centers=(1., 2., 3.)),
        algorithm=RecursiveLstdqSpec(), trace_lambda=.8, setup_obs_encoder=encoder, seed=7))


class SharedSetupFeatureTests(unittest.TestCase):
    def test_normalized_expansion_is_bounded_at_every_numerical_corner_and_category(self):
        old, normalized = normalized_encoders()
        bandit, _ = encoders()
        numeric = [p for p in bandit.parameter_spec.parameters if p.kind != 'categorical']
        categorical = [p for p in bandit.parameter_spec.parameters if p.kind == 'categorical']
        choices = [(min(p.values), max(p.values)) for p in numeric]+[p.values for p in categorical]
        names = [p.name for p in numeric+categorical]
        for values in product(*choices):
            params = dict(DEFAULT_SETUP_PARAMS, **dict(zip(names, values)))
            expanded = normalized.encode_setup(params)
            basic = old.encode_setup(params)
            self.assertEqual(expanded.shape, (104,))
            self.assertTrue(np.all(np.abs(expanded) <= 1+1e-12))
            np.testing.assert_array_equal(expanded[:5], basic[:5])
            np.testing.assert_array_equal(expanded[20:34], basic[5:])
            np.testing.assert_array_equal(expanded[20:34], bandit.encode_action(params)[20:34])
            self.assertTrue(np.all(expanded[5:10] >= 0))

    def test_normalization_retains_every_grid_value_and_prior_19_feature_scaling(self):
        basic, normalized = normalized_encoders()
        bandit, _ = encoders()
        for index, parameter in enumerate(p for p in bandit.parameter_spec.parameters if p.kind != 'categorical'):
            coordinates = []
            for value in sorted(parameter.values):
                params = dict(DEFAULT_SETUP_PARAMS, **{parameter.name: value})
                vector = normalized.encode_setup(params)
                np.testing.assert_array_equal(vector, basic.encode_action(params))
                coordinates.append(vector[index])
            self.assertAlmostEqual(coordinates[0], -1)
            self.assertAlmostEqual(coordinates[-1], 1)
            self.assertTrue(np.all(np.diff(coordinates) > 0))
        params = dict(DEFAULT_SETUP_PARAMS, max_row_sum=1e-6, interp_type=17)
        values = normalized.encode_setup(params)
        self.assertAlmostEqual(values[1], -1)
        self.assertAlmostEqual(values[6], 1)
        self.assertEqual(values[normalized.descriptor_names.index('interp_type=17')], 1)
        self.assertAlmostEqual(bandit.encode_action(params)[1], -8.99999)

    def test_normalized_checkpoint_rejects_same_dimension_unscaled_state(self):
        _, raw = encoders()
        _, normalized = normalized_encoders()
        before, after = bundle(raw), bundle(normalized)
        params = dict(DEFAULT_SETUP_PARAMS, interp_type=17, max_row_sum=.05)
        kwargs = dict(mkw=dict(nx=60, ny=60, nz=60, k=10., c=20., a0=30., a1=40., a2=50., a3=60.),
            setup_params=params, initial_residual=1., residual=.01, previous_residual=.1,
            cycle=2, last_weight=1.5, last_cycle_time=.04, first_cycle_time=.02)
        a, b = before.encoder.encode(**kwargs), after.encoder.encode(**kwargs)
        self.assertEqual((a.size, b.size), (117, 117))
        np.testing.assert_array_equal(a[:13], b[:13])
        with TemporaryDirectory() as directory:
            path = Path(directory)/'controller.npz'
            before.save(path)
            with self.assertRaisesRegex(ValueError, 'state encoder does not match'):
                after.load(path)
            after.save(path)
            restored = bundle(normalized)
            restored.load(path)
            np.testing.assert_array_equal(restored.controller.theta, after.controller.theta)

    def test_matches_bandit_cache_for_default_and_varied_configurations(self):
        bandit, shared = encoders()
        self.assertEqual(shared.parameter_spec, bandit.parameter_spec)
        self.assertEqual(shared.descriptor_dim, 104)
        self.assertEqual(len(set(shared.descriptor_names)), 104)
        catalog = CompactActionCatalog(bandit.parameter_spec)
        cache = FactorizedActionFeatureCache(catalog, bandit)
        arms = [catalog.default_arm_index, *np.random.default_rng(4).integers(len(catalog), size=100)]
        for arm in arms:
            np.testing.assert_array_equal(shared.encode_setup(catalog[int(arm)]), cache.feature(int(arm)))
        np.testing.assert_array_equal(shared.encode_setup(DEFAULT_SETUP_PARAMS), np.zeros(104))

    def test_keeps_scaling_categories_and_conditional_interactions(self):
        bandit, shared = encoders()
        params = dict(DEFAULT_SETUP_PARAMS, max_row_sum=.000001, interp_type=17,
                      agg_num_levels=0, agg_interp_type=7)
        values = shared.encode_setup(params)
        self.assertAlmostEqual(values[shared.descriptor_names.index('max_row_sum')], -8.99999)
        self.assertAlmostEqual(values[shared.descriptor_names.index('max_row_sum^2')], (-8.99999)**2)
        self.assertEqual(values[shared.descriptor_names.index('interp_type=17')], 1.)
        self.assertAlmostEqual(values[shared.descriptor_names.index('interp_type=17*max_row_sum')], -8.99999)
        for levels in (0, 1, 2):
            params['agg_num_levels'] = levels
            value = shared.encode_setup(params)
            np.testing.assert_array_equal(value, bandit.encode_action(params))
            self.assertEqual(value[shared.descriptor_names.index('agg_interp_type=7')], float(levels > 0))
        inactive = shared.encode_setup(dict(params, agg_num_levels=0))
        self.assertTrue(all(inactive[i] == 0. for i, name in enumerate(shared.descriptor_names)
                            if name.startswith('agg_interp_type=')))

    def test_other_state_features_and_checkpoint_compatibility(self):
        bandit, shared = encoders()
        old, new = bundle(bandit), bundle(shared)
        params = dict(DEFAULT_SETUP_PARAMS, interp_type=17, max_row_sum=.05)
        kwargs = dict(mkw=dict(nx=60, ny=60, nz=60, k=10., c=20., a0=30., a1=40., a2=50., a3=60.),
            setup_params=params, initial_residual=1., residual=.01, previous_residual=.1,
            cycle=2, last_weight=1.5, last_cycle_time=.04, first_cycle_time=.02)
        before, after = old.encoder.encode(**kwargs), new.encoder.encode(**kwargs)
        self.assertEqual((before.size, after.size), (32, 117))
        np.testing.assert_array_equal(before[:13], after[:13])
        np.testing.assert_array_equal(after[13:], bandit.encode_action(params))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'controller.npz'
            old.save(path)
            with self.assertRaisesRegex(ValueError, 'state encoder does not match'):
                new.load(path)
            new.save(path)
            restored = bundle(shared)
            restored.load(path)
            np.testing.assert_array_equal(restored.controller.theta, new.controller.theta)

    def test_native_four_branch_worker_uses_full_features_and_stops_at_limit(self):
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
            runtime.stream = runtime.stream[:3]
            runtime.args = replace(runtime.args, online_cases=3)
            return runtime
        with TemporaryDirectory(dir=ROOT/'results', prefix='shared_setup_feature_validation_') as directory:
            base = Path(directory)
            runner.dump(base/'config.json', raw)
            runner.dump(base/'suite.json', dict(name='Shared action features native validation',
                seeds=[dict(label='1', config=str(base/'config.json'))]))
            output = base/'run'
            with patch.object(runner, 'decode', short_decode):
                prepare(base/'suite.json', output, cases=3)
                runner.dump(output/'release.json', {'validation_only': True})
                with configured_runner():
                    runner.worker(output, '1')
            stage = output/'seed_1'
            self.assertEqual(runner.read(stage/'status.json')['status'], 'complete')
            self.assertTrue(runner.read(stage/'result.json')['passed'])
            audit = runner.read(stage/'shared_setup_feature_audit.json')
            self.assertTrue(audit['passed'])
            for controller in audit['controllers'].values():
                self.assertEqual(controller['setup_features'], 104)
                self.assertEqual(controller['state_features'], 117)
                self.assertEqual(controller['state_action_features'], 1053)
                self.assertTrue(controller['normalized_range_verified'])
                self.assertFalse(controller['exact_bandit_features'])
            self.assertTrue(audit['bandit_scaling_unchanged'])
            self.assertTrue(audit['rl_only_rescaling'])
            protocol = runner.read(output/'protocol.json')
            self.assertEqual(protocol['measured_learning_inputs'], [1, 3])
            self.assertEqual(protocol['measured_frozen_inputs'], [])
            self.assertEqual(protocol['shared_setup_features']['normalization'], 'range')
            for method in runner.factorial.METHODS:
                rows = list(runner.read_records(stage/f'trajectories/{method}.jsonl'))
                self.assertEqual([r['problem_index'] for r in rows], [1, 2, 3])
                if method != runner.factorial.METHODS[0]:
                    self.assertTrue(all(r['outcome']['end_to_end_runtime'] > r['outcome']['runtime'] for r in rows))
            self.assertFalse((stage/'freeze_boundary.json').exists())
