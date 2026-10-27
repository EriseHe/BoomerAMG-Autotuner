"""Behavioral checks for shared setup features and native-only RL cost accounting."""
from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final.online.controllers import make_setup_feature_encoder
from experiments.paper_final.recovery.execution import EpisodeRecorder
from hypre.bindings import AMGNativeError, AttemptOutcome, SolveStatus
from hypre.bindings.boomeramg import NativeCode
from setup.learners.common.aot_candidates import CompactActionCatalog, FactorizedActionFeatureCache
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace
from solve.controllers.common import OnlineControllerFactoryRequest, SharedActionSpec, SolveStateSpec
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.controllers.recursive_lstdq import RecursiveLstdqController, RecursiveLstdqSpec
from solve.controllers.recursive_lstdq.factory import build_recursive_lstdq_controller
from solve.core.episode import run_td_episode


ROOT = Path(__file__).resolve().parents[2]


def setup_encoder():
    raw = json.loads((ROOT / 'data/module1/online/configurations/diffusion_40/seed_1.json').read_text())
    space = SetupConfigurationSpace.from_mapping(
        'expanded', raw['setup']['configuration_spaces']['expanded_no_endpoints'])
    return make_setup_feature_encoder(configuration_space=space, parameter_resolution=20)


def bundle():
    return build_recursive_lstdq_controller(OnlineControllerFactoryRequest(
        state=SolveStateSpec(tol=1e-6, max_cycles=50, c_max=1000,
                             problem_context_mode='diffusion3d'),
        actions=SharedActionSpec(weights=(1., 2., 3.), rbf_centers=(1., 2., 3.)),
        algorithm=RecursiveLstdqSpec(), trace_lambda=.8,
        setup_obs_encoder=setup_encoder(), seed=7))


class SharedSetupTests(unittest.TestCase):
    def test_trained_models_load_with_the_permanent_name(self):
        original = bundle()
        self.assertEqual(original.encoder.encoding, 'shared_context')
        with TemporaryDirectory() as directory:
            path = Path(directory)/'controller.npz'
            original.save(path)
            sidecar = path.with_suffix('.npz.encoder.json')
            metadata = json.loads(sidecar.read_text())
            self.assertEqual(metadata.pop('name'), 'shared_context')
            metadata['encoding_version'] = 'shared_setup_one_hot_tolerance_residuals_v3'
            sidecar.write_text(json.dumps(metadata))
            restored = bundle()
            restored.load(path)
            np.testing.assert_array_equal(restored.controller.theta, original.controller.theta)
            self.assertEqual(restored.encoder.encoding, 'shared_context')

    def test_descriptor_matches_its_factorized_feature_representation(self):
        encoder = setup_encoder()
        catalog = CompactActionCatalog(encoder.parameter_spec)
        cache = FactorizedActionFeatureCache(catalog, encoder)
        arms = np.r_[catalog.default_arm_index,
                     np.random.default_rng(2).integers(len(catalog), size=100)]
        for arm in arms:
            params = catalog[int(arm)]
            bandit = cache.feature(int(arm))
            np.testing.assert_array_equal(encoder.encode_action(params), bandit)
            np.testing.assert_array_equal(encoder.encode_setup(params),
                np.r_[bandit[:encoder.numeric_dim],
                      bandit[encoder.numeric_feature_dim:
                             encoder.numeric_feature_dim + encoder.categorical_dim]])
        self.assertEqual((encoder.descriptor_dim, encoder.feature_dim), (19, 104))

    def test_categories_are_indicators_and_inactive_values_are_masked(self):
        encoder = setup_encoder()
        indexes = [i for i, name in enumerate(encoder.descriptor_names)
                   if name.startswith('interp_type=')]
        encoded = encoder.encode_setup(dict(DEFAULT_SETUP_PARAMS, interp_type=17))
        np.testing.assert_array_equal(encoded[indexes], [0, 0, 0, 0, 0, 1, 0])
        np.testing.assert_array_equal(encoder.encode_setup(DEFAULT_SETUP_PARAMS)[indexes], 0)
        np.testing.assert_array_equal(
            encoder.encode_setup(dict(DEFAULT_SETUP_PARAMS, agg_num_levels=0, agg_interp_type=7)),
            encoder.encode_setup(dict(DEFAULT_SETUP_PARAMS, agg_num_levels=0, agg_interp_type=4)))
        active = encoder.encode_setup(dict(DEFAULT_SETUP_PARAMS, agg_num_levels=1, agg_interp_type=7))
        self.assertEqual(active[encoder.descriptor_names.index('agg_interp_type=7')], 1)
        with self.assertRaises(ValueError):
            encoder.encode_setup(dict(DEFAULT_SETUP_PARAMS, interp_type=14))

    def test_solve_uses_unclipped_shared_descriptors(self):
        shared = setup_encoder()
        encoder = SolveStateEncoder(tol=1e-6, max_cycles=50, c_max=1000,
            mode='setup_full', setup_obs_encoder=shared, problem_context_mode='diffusion3d',
            weight_bounds=(1., 3.))
        params = dict(DEFAULT_SETUP_PARAMS, strong_threshold=.95, max_row_sum=.05,
                      trunc_factor=.95, P_max_elmts=16, interp_type=17)
        features = encoder.encode(mkw=dict(nx=40, ny=40, nz=40, k=1., c=10., a0=100.),
            setup_params=params, initial_residual=1., residual=1., previous_residual=1.,
            cycle=0, last_weight=1., last_cycle_time=0.)
        np.testing.assert_array_equal(features[-19:], shared.encode_setup(params))
        self.assertLessEqual(np.max(np.abs(features[-19:])), 1.)
        self.assertEqual(encoder.feature_dim, 29)
        for obsolete in ('space_aware_v2', 'shared_setup_one_hot_tolerance_residuals',
                         'shared_setup_one_hot_tolerance_residuals_v2'):
            with self.subTest(encoding=obsolete), self.assertRaises(ValueError):
                SolveStateSpec(tol=1e-6, max_cycles=50, c_max=1000,
                               encoding=obsolete)

    def test_cycle_count_only_changes_its_normalized_scalar(self):
        encoder = bundle().encoder
        kwargs = dict(mkw=dict(nx=40, ny=40, nz=40, k=1., c=10., a0=100.),
            setup_params=DEFAULT_SETUP_PARAMS, initial_residual=1., residual=.1,
            previous_residual=1., last_weight=1., last_cycle_time=.02)
        reference = encoder.encode(**kwargs, cycle=1)
        for cycle in range(1, encoder.max_cycles + 1):
            features = encoder.encode(**kwargs, cycle=cycle)
            self.assertAlmostEqual(features[1], cycle / encoder.max_cycles)
            np.testing.assert_array_equal(np.delete(features, 1), np.delete(reference, 1))

    def test_checkpoint_and_rollback_preserve_causal_estimator(self):
        original = bundle()
        controller = original.controller
        controller.record_update_runtime(.01)
        snapshot = controller.snapshot_learning_state()
        controller.record_update_runtime(.03)
        self.assertEqual(controller.estimated_update_runtime_sec, .03)
        controller.restore_learning_state(snapshot)
        self.assertEqual(controller.estimated_update_runtime_sec, .01)
        self.assertEqual(controller.update_runtime_count, 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'controller.npz'
            original.save(path)
            restored = bundle()
            restored.load(path)
            self.assertEqual(restored.controller.estimated_update_runtime_sec, .01)
            self.assertEqual(restored.controller.update_runtime_count, 1)
            sidecar = path.with_suffix('.npz.encoder.json')
            metadata = json.loads(sidecar.read_text())
            outdated = dict(metadata, encoding_version='shared_setup_one_hot_tolerance_residuals_v2')
            sidecar.write_text(json.dumps(outdated))
            with self.assertRaisesRegex(ValueError, 'state encoder does not match'):
                restored.load(path)
            self.assertEqual(restored.controller.update_runtime_count, 1)
            metadata['setup_descriptor_names'] = list(reversed(metadata['setup_descriptor_names']))
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                restored.load(path)
            self.assertEqual(restored.controller.update_runtime_count, 1)
        for invalid in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                controller.record_update_runtime(invalid)

    def test_cycle_time_is_absolute_logarithmic_without_clipping(self):
        encoder = bundle().encoder
        kwargs = dict(mkw=dict(nx=40, ny=40, nz=40, k=1., c=10., a0=100.),
            setup_params=DEFAULT_SETUP_PARAMS, initial_residual=1., residual=.1,
            previous_residual=1., cycle=2, last_weight=1.)
        reference = encoder.encode(**kwargs, last_cycle_time=0.)
        for duration in (1e-9, .001, .02, .06, 1., 1000.):
            features = encoder.encode(**kwargs, last_cycle_time=duration)
            self.assertAlmostEqual(features[5], math.log1p(duration/.001))
            np.testing.assert_array_equal(np.delete(features, 5), np.delete(reference, 5))
        self.assertEqual(encoder.encode(**dict(kwargs, cycle=0), last_cycle_time=0.)[5], 0.)
        self.assertGreater(encoder.encode(**kwargs, last_cycle_time=1.)[5], 1.)

    def test_cycle_time_rejects_invalid_observations(self):
        encoder = bundle().encoder
        kwargs = dict(mkw=dict(nx=40, ny=40, nz=40, k=1., c=10., a0=100.),
            setup_params=DEFAULT_SETUP_PARAMS, initial_residual=1., residual=.1,
            previous_residual=1., cycle=1, last_weight=1.)
        for invalid in (-1., float('inf'), float('nan')):
            with self.subTest(duration=invalid), self.assertRaises(ValueError):
                encoder.encode(**kwargs, last_cycle_time=invalid)

    def test_residual_features_share_tolerance_scale_and_telescope(self):
        encoder = bundle().encoder
        kwargs = dict(mkw=dict(nx=40, ny=40, nz=40, k=1., c=10., a0=100.),
            setup_params=DEFAULT_SETUP_PARAMS, initial_residual=1.,
            last_weight=1., last_cycle_time=.002)
        residuals = (1., .1, 1e-3, 1e6, 1e-6, 1e-9)
        expected_gaps = (1., 5/6, .5, 2., 0., -.5)
        last_gap = 1.
        signed_changes = []
        for cycle, (residual, gap) in enumerate(zip(residuals, expected_gaps)):
            features = encoder.encode(**kwargs, cycle=cycle, residual=residual,
                previous_residual=residuals[max(0, cycle-1)])
            self.assertAlmostEqual(features[2], gap)
            self.assertAlmostEqual(features[3], gap * gap)
            self.assertAlmostEqual(features[4], gap - last_gap)
            signed_changes.append(features[4])
            last_gap = gap
        self.assertAlmostEqual(sum(signed_changes), expected_gaps[-1] - 1.)

    def test_residual_normalization_is_invariant_to_consistent_units(self):
        original = bundle().encoder
        scaled = SolveStateEncoder(tol=1e-3, max_cycles=50, c_max=1000,
            mode='setup_full', setup_obs_encoder=setup_encoder(),
            problem_context_mode='diffusion3d', weight_bounds=(1., 3.))
        kwargs = dict(mkw=dict(nx=40, ny=40, nz=40, k=1., c=10., a0=100.),
            setup_params=DEFAULT_SETUP_PARAMS, cycle=2, last_weight=1., last_cycle_time=.002,
            )
        first = original.encode(**kwargs, initial_residual=1., residual=1e-3, previous_residual=.1)
        second = scaled.encode(**kwargs, initial_residual=1000., residual=1., previous_residual=100.)
        np.testing.assert_allclose(first, second, rtol=1e-14, atol=1e-14)

    def test_residual_normalization_handles_large_finite_ratios(self):
        encoder = bundle().encoder
        kwargs = dict(mkw=dict(nx=40, ny=40, nz=40, k=1., c=10., a0=100.),
            setup_params=DEFAULT_SETUP_PARAMS, initial_residual=1., cycle=1,
            last_weight=1., last_cycle_time=.002)
        features = encoder.encode(**kwargs, residual=1e200, previous_residual=1e-200)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertAlmostEqual(features[4], 400/6)
        for residual in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                encoder.encode(**kwargs, residual=residual, previous_residual=1.)
        with self.assertRaises(ValueError):
            encoder.encode(**dict(kwargs, initial_residual=1e-6), residual=1e-6, previous_residual=1e-6)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, duration):
        self.now += duration


class TimedController(RecursiveLstdqController):
    def __init__(self, clock):
        super().__init__(feature_dim=2, seed=2, spec=RecursiveLstdqSpec(),
            config=SharedActionSpec(weights=(1., 2.), rbf_centers=(1., 2.)).to_td_config(trace_lambda=.8))
        self.clock = clock
        self.executed_costs = []
        self.selections = 0

    def snapshot_learning_state(self):
        self.clock.advance(.017)
        return super().snapshot_learning_state()

    def start_episode(self, **kwargs):
        self.clock.advance(.013)
        return super().start_episode(**kwargs)

    def select_action(self, *args, **kwargs):
        self.clock.advance(.003 + .001 * self.selections)
        self.selections += 1
        return super().select_action(*args, **kwargs)

    def update(self, **kwargs):
        self.clock.advance(.005 + .002 * len(self.executed_costs))
        self.executed_costs.append(kwargs['cost'])
        return super().update(**kwargs)

    def finish_episode(self, **kwargs):
        self.clock.advance(.011)
        return super().finish_episode(**kwargs)


class TimedEncoder:
    def __init__(self, clock):
        self.clock = clock

    def encode(self, **kwargs):
        self.clock.advance(.001 * (kwargs['cycle'] + 1))
        return np.array([1., kwargs['cycle'] * .1])


class Environment:
    def __init__(self, *, fail=False, error=False, cycle_times=(.02, .03, .04)):
        self.cycles = 0
        self.cycle_times = cycle_times
        self.fail, self.error = fail, error
        self.last_step = SimpleNamespace(status=SolveStatus.CONTINUE)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def prepare_rl(self, **kwargs):
        return SimpleNamespace(setup_runtime_sec=.03, initial_residual_norm=1., initial_relax_weight=1.)

    def step_rl(self, **kwargs):
        self.cycles += 1
        if self.error:
            raise AMGNativeError(operation='step', code=NativeCode.SOLVE_ERROR,
                                 solve_runtime_sec=.02)
        if self.cycles == 3:
            self.last_step.status = SolveStatus.MAX_CYCLES if self.fail else SolveStatus.CONVERGED
        return (.1 if self.fail or self.cycles < 3 else 1e-7), self.cycle_times[self.cycles - 1]


class CostAccountingTests(unittest.TestCase):
    def test_log_time_context_and_reward_use_native_measurements(self):
        used = bundle()
        original_encode = used.encoder.encode
        observations = []
        def record_state(**kwargs):
            features = original_encode(**kwargs)
            observations.append((kwargs['cycle'], kwargs['last_cycle_time'], features[5]))
            return features
        # Reuse both learners across attempts, with different native time scales.
        for durations in ((.02, .03, .04), (.10, .04, .02)):
            expected = [0., math.log1p(durations[0]/.001), math.log1p(durations[1]/.001)]
            observations.clear()
            used.controller.record_update_runtime(.75)
            with patch.object(used.encoder, 'encode', side_effect=record_state), \
                 patch('solve.core.episode.create_env', return_value=Environment(cycle_times=durations)):
                result = run_td_episode(
                    mkw=dict(nx=40, ny=40, nz=40, k=1., c=10., a0=100.),
                    params=DEFAULT_SETUP_PARAMS, controller=used.controller, encoder=used.encoder,
                    solve_tol=1e-6, solve_max_cycles=3, learn=True, explore=False,
                    failure_penalty_sec=0.)
            np.testing.assert_allclose([v[2] for v in observations], expected)
            self.assertEqual([v[1] for v in observations], [0., durations[0], durations[1]])
            self.assertEqual(result['cycle_times'], list(durations))
            self.assertEqual(result['cycle_assigned_costs'], list(durations))
            self.assertGreater(result['infer_runtime'], 0.)

    def run_episode(self, *, learn=True, fail=False, error=False, recorder=False):
        clock = Clock()
        controller = TimedController(clock)
        if not learn:
            controller.record_update_runtime(.2)
        used = EpisodeRecorder(controller) if recorder else controller
        with patch('solve.core.episode.time.perf_counter', clock), \
             patch('solve.core.episode.create_env', return_value=Environment(fail=fail, error=error)):
            result = run_td_episode(mkw={}, params=DEFAULT_SETUP_PARAMS, controller=used,
                encoder=TimedEncoder(clock), solve_tol=1e-6, solve_max_cycles=3,
                learn=learn, explore=False, record_action_metadata=True, failure_penalty_sec=0.)
        return controller, used, result

    def test_native_targets_exclude_overhead_while_evaluation_measures_it(self):
        controller, _, result = self.run_episode()
        np.testing.assert_allclose(controller.executed_costs, [.02, .03, .04])
        np.testing.assert_allclose(result['cycle_update_time_estimates'], [0., 0., 0.])
        np.testing.assert_allclose(result['cycle_update_times'], [.005, .007, .009])
        np.testing.assert_allclose(result['cycle_assigned_costs'], controller.executed_costs)
        self.assertEqual(controller.selections, 3)
        self.assertAlmostEqual(result['feature_runtime'], .006)
        self.assertAlmostEqual(result['decision_runtime'], .012)
        self.assertAlmostEqual(result['update_runtime'], .021 + .011)
        self.assertAlmostEqual(result['lifecycle_runtime'], .017 + .013)
        self.assertAlmostEqual(result['infer_runtime'], .080)
        self.assertAlmostEqual(result['native_solve_runtime'], .09)
        self.assertAlmostEqual(controller.estimated_update_runtime_sec, .009)

    def test_frozen_solve_does_not_charge_or_change_update_estimate(self):
        controller, _, result = self.run_episode(learn=False)
        self.assertEqual(controller.executed_costs, [])
        self.assertEqual(result['cycle_update_time_estimates'], [0., 0., 0.])
        self.assertEqual(result['cycle_update_times'], [0., 0., 0.])
        self.assertEqual(controller.estimated_update_runtime_sec, .2)
        self.assertEqual(controller.update_runtime_count, 1)
        self.assertFalse(result['controller_update_committed'])
        self.assertAlmostEqual(result['feature_runtime'], .006)
        self.assertAlmostEqual(result['decision_runtime'], .012)
        self.assertGreater(result['infer_runtime'], 0.)

    def test_terminal_failure_retains_cost_then_whole_input_failure_rolls_back(self):
        controller, recorder, result = self.run_episode(fail=True, recorder=True)
        self.assertEqual(controller.update_runtime_count, 3)
        attempt = AttemptOutcome.from_mapping(result)
        recorder.finalize([attempt], None, include_recovery=False)
        self.assertEqual(controller.update_runtime_count, 0)
        self.assertEqual(controller.estimated_update_runtime_sec, 0.)
        self.assertEqual(controller.steps, 0)
        np.testing.assert_array_equal(controller.b, 0.)

    def test_successful_default_fallback_retains_only_learned_updates(self):
        controller, recorder, result = self.run_episode(fail=True, recorder=True)
        fallback = AttemptOutcome.from_mapping(dict(runtime=10., setup_runtime=2.,
            solve_runtime=8., failed=False, iterations=5, residual_norm=1e-7))
        recorder.finalize([AttemptOutcome.from_mapping(result)], fallback, include_recovery=False)
        np.testing.assert_allclose(controller.executed_costs, [.02, .03, .04])
        self.assertEqual(controller.update_runtime_count, 3)
        self.assertAlmostEqual(controller.estimated_update_runtime_sec, .009)

    def test_native_exception_retains_assigned_cost_diagnostics(self):
        controller, _, result = self.run_episode(error=True)
        np.testing.assert_allclose(controller.executed_costs, [.02])
        self.assertEqual(len(result['cycle_selected_q_values']), 1)
        self.assertTrue(result['failed'])
        self.assertEqual(controller.update_runtime_count, 1)


if __name__ == '__main__':
    unittest.main()
