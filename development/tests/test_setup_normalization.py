"""Check numerical range scaling without changing categories or bandit features."""
import copy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from development.run_setup_normalization import configured_runner, normalization_source_hashes, runner
from development.setup_normalization import make_range_normalized_setup_encoder
from development.setup_normalization import make_bandit_scaled_setup_encoder as make_setup_feature_encoder
from setup.learners.common.aot_candidates import CompactActionCatalog
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
    return make_setup_feature_encoder(**kwargs), make_range_normalized_setup_encoder(**kwargs)


def bundle(encoder):
    return build_recursive_lstdq_controller(OnlineControllerFactoryRequest(
        state=SolveStateSpec(tol=1e-6, max_cycles=50, c_max=1000,
                             problem_context_mode='canonical_no_c_mean'),
        actions=SharedActionSpec(weights=(1., 2., 3.), rbf_centers=(1., 2., 3.)),
        algorithm=RecursiveLstdqSpec(), trace_lambda=.8, setup_obs_encoder=encoder, seed=7))


class SetupNormalizationTests(unittest.TestCase):
    def test_allowed_values_use_physical_distances_and_fixed_bounds(self):
        shared, normalized = encoders()
        self.assertEqual(shared.descriptor_names, normalized.descriptor_names)
        self.assertEqual(normalized.descriptor_dim, 19)
        for index, parameter in enumerate(shared.numeric_params):
            low, high = min(parameter.values), max(parameter.values)
            for value in parameter.values:
                params = dict(DEFAULT_SETUP_PARAMS, **{parameter.name: value})
                self.assertAlmostEqual(normalized.encode_setup(params)[index],
                                       2 * (value - low) / (high - low) - 1)
        # The nonuniform integer grid preserves distances, rather than its rank.
        self.assertAlmostEqual(normalized.encode_setup(dict(DEFAULT_SETUP_PARAMS, P_max_elmts=8))[3], -1/7)
        self.assertEqual(normalized.encode_setup(dict(DEFAULT_SETUP_PARAMS, agg_num_levels=1))[4], 0.)
        params = dict(DEFAULT_SETUP_PARAMS, strong_threshold=.95, max_row_sum=.000001,
                      trunc_factor=.75, P_max_elmts=16, agg_num_levels=1,
                      coarsen_type=8, interp_type=3, agg_interp_type=7)
        np.testing.assert_allclose(normalized.encode_setup(params)[:5], [1, -1, 2*.75/.95-1, 1, 0])
        self.assertLess(shared.encode_setup(params)[1], -8.)

    def test_categories_masks_and_bandit_remain_identical(self):
        shared, normalized = encoders()
        original_spec = shared.parameter_spec
        catalog = CompactActionCatalog(shared.parameter_spec)
        for arm in np.random.default_rng(32).integers(len(catalog), size=100):
            params = catalog[int(arm)]
            np.testing.assert_array_equal(shared.encode_setup(params)[5:], normalized.encode_setup(params)[5:])
        for levels in (0, 1, 2):
            params = dict(DEFAULT_SETUP_PARAMS, interp_type=17, agg_num_levels=levels, agg_interp_type=7)
            np.testing.assert_array_equal(shared.encode_setup(params)[5:], normalized.encode_setup(params)[5:])
        self.assertIs(shared.parameter_spec, original_spec)
        untouched, _ = encoders()
        self.assertEqual(untouched.parameter_spec, original_spec)
        np.testing.assert_array_equal(shared.encode_action(params), untouched.encode_action(params))

    def test_checkpoint_rejects_a_different_normalization(self):
        shared, normalized = encoders()
        old, new = bundle(shared), bundle(normalized)
        self.assertEqual(old.encoder.feature_dim, new.encoder.feature_dim)
        self.assertEqual(new.encoder.feature_dim, 32)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'controller.npz'
            old.save(path)
            with self.assertRaisesRegex(ValueError, 'state encoder does not match'):
                new.load(path)
            new.save(path)
            bundle(normalized).load(path)

    def test_native_worker_uses_new_encoding_through_freeze(self):
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
        original_decode, original_initialize = runner.decode, runner.initialize
        def short_decode(config, directory):
            runtime = original_decode(config, directory)
            runtime.stream = runtime.stream[:2] + runtime.stream[5000:5003]
            runtime.args = replace(runtime.args, online_cases=5)
            runtime.freeze_after = 2
            return runtime
        initialized = []
        shared, normalized = encoders()
        def checked_initialize(output, label):
            runtimes = original_initialize(output, label)
            for name in (runner.factorial.METHODS[1], runner.factorial.METHODS[3]):
                encoder = runtimes[name].bundle.encoder.setup_obs_encoder
                self.assertEqual(encoder.parameter_spec, normalized.parameter_spec)
                self.assertEqual(runtimes[name].bundle.encoder.feature_dim, 32)
            for name in runner.factorial.METHODS[2:]:
                self.assertEqual(runtimes[name].branch.policy.model._encoder.parameter_spec, shared.parameter_spec)
            initialized.append(runtimes)
            return runtimes
        with TemporaryDirectory(dir=ROOT/'results', prefix='setup_normalization_validation_') as directory:
            base = Path(directory)
            runner.dump(base/'config.json', raw)
            runner.dump(base/'suite.json', dict(name='Setup normalization native validation',
                seeds=[dict(label='1', config=str(base/'config.json'))]))
            output = base/'run'
            with configured_runner(), patch.object(runner, 'decode', short_decode), \
                 patch.object(runner, 'initialize', checked_initialize):
                runner.prepare(base/'suite.json', output, cases=5)
                runner.dump(output/'release.json', {'validation_only': True})
                runner.worker(output, '1')
                self.assertEqual(runner.read(output/'source_manifest.json'), normalization_source_hashes())
            self.assertEqual(len(initialized), 1)
            stage = output/'seed_1'
            self.assertEqual(runner.read(stage/'status.json')['status'], 'complete')
            self.assertTrue(runner.read(stage/'frozen_state_audit.json')['passed'])
            self.assertEqual(len({e['pid'] for e in runner.read_records(stage/'execution.jsonl')}), 1)
            for name in runner.factorial.METHODS:
                rows = list(runner.read_records(stage/f'trajectories/{name}.jsonl'))
                self.assertEqual([r['problem_index'] for r in rows], list(range(1, 6)))
                for row in rows[2:]:
                    self.assertTrue(row['frozen_evaluation'])
                    self.assertFalse(row['outcome']['bandit_update_committed'])
                    self.assertFalse(row['outcome']['controller_update_committed'])
                    if name != runner.factorial.METHODS[0]:
                        self.assertGreater(row['outcome']['end_to_end_runtime'], row['outcome']['runtime'])
