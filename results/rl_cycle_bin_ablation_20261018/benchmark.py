"""Measure cycle-bin removal on fixed recorded trajectories, not new policies."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
sys.path.insert(0, str(REPO))
from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.space import SetupConfigurationSpace
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig
from solve.controllers.recursive_lstdq import RecursiveLstdqController, RecursiveLstdqSpec

SOURCE = REPO / 'results/rl_cycle_time_feature_ablation_20261018/native_1000_3seeds'
METHODS = ('default_setup_lstdq', 'bandit_lstdq')
REPETITIONS = 3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest():
    paths = sorted((REPO / 'solve').rglob('*.py'))
    return {str(p.relative_to(REPO)): digest(p) for p in paths}


original_manifest = source_manifest()
source_text = (REPO / 'solve/controllers/common/state_encoder.py').read_text()
replacements = {
    'shared_setup_one_hot_tolerance_residuals': 'shared_setup_one_hot_tolerance_residuals_without_cycle_bins',
    'self._setup_offset = self._cycle_offset + 2 * self.cycle_bins': 'self._setup_offset = self._cycle_offset',
    '        cycle_bin = int(np.searchsorted(self._CYCLE_BOUNDS, int(cycle), side="right"))\n': '',
    '        features[self._cycle_offset + cycle_bin] = 1.0\n': '',
    '        features[self._cycle_offset + self.cycle_bins + cycle_bin] = gap_fraction\n': '',
}
for old, new in replacements.items():
    assert source_text.count(old) == 1, old
    source_text = source_text.replace(old, new)
ablation_path = OUT / 'ablation_state_encoder.py'
ablation_path.write_text(source_text)
module_spec = importlib.util.spec_from_file_location('isolated_bin_ablation', ablation_path)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
ENCODERS = {'with_bins': SolveStateEncoder, 'without_bins': module.SolveStateEncoder}


def make_encoder(cls, experiment):
    space = SetupConfigurationSpace.from_mapping(
        'expanded', experiment['setup']['configuration_spaces']['expanded_no_endpoints'])
    return cls(
        tol=experiment['solve']['tolerance'],
        max_cycles=experiment['solve']['max_cycles'],
        c_max=experiment['problem']['c_max'], mode='setup_full',
        setup_obs_encoder=make_setup_feature_encoder(
            configuration_space=space,
            parameter_resolution=experiment['setup']['parameter_resolution']),
        problem_context_mode='diffusion3d', weight_bounds=(1., 3.))


def prepare(run, method, experiment):
    checkpoint = run / f'checkpoints/{method}_final.npz'
    with np.load(checkpoint, allow_pickle=False) as saved:
        cfg = ExpectedSarsaLambdaConfig(**json.loads(str(saved['config'].item())))
        spec = RecursiveLstdqSpec(**json.loads(str(saved['spec'].item())))
    weights = np.asarray(cfg.weights)
    original = make_encoder(SolveStateEncoder, experiment)
    reduced = make_encoder(module.SolveStateEncoder, experiment)
    keep = np.r_[0:original._cycle_offset, original._setup_offset:original.feature_dim]
    assert (original.feature_dim, reduced.feature_dim) == (43, 29)
    cases, state_count, recorded_total, recorded_overhead = [], 0, 0., 0.
    path = run / f'trajectories/{method}.jsonl'
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            outcome = row['outcome']
            assert not outcome['unrecovered_failure']
            assert not outcome['fallback_used']
            recorded_total += outcome['end_to_end_runtime']
            recorded_overhead += outcome['infer_runtime']
            episodes = []
            for attempt in outcome['primary_attempts']:
                if not attempt['rl_episode_recorded']:
                    continue
                actions = attempt['cycle_actions']
                initial_weight = float(outcome['initial_environment_weight'])
                residuals = [1., *attempt['cycle_residuals']]
                times = [0., *attempt['cycle_times']]
                previous_weights = [initial_weight, *actions]
                indices = [int(np.argmin(np.abs(weights - a))) for a in actions]
                observations = []
                for k in range(len(actions)):
                    kwargs = dict(
                        mkw=row['mkw'], setup_params=attempt['params'],
                        problem_context=row['context'], initial_residual=1.,
                        residual=residuals[k], previous_residual=residuals[max(0, k-1)],
                        cycle=k, last_weight=previous_weights[k], last_cycle_time=times[k])
                    full = original.encode(**kwargs)
                    small = reduced.encode(**kwargs)
                    np.testing.assert_array_equal(full[keep], small)
                    assert np.isfinite(small).all()
                    observations.append(kwargs)
                    state_count += 1
                assert len(indices) == len(attempt['cycle_assigned_costs']) == len(attempt['cycle_update_times'])
                episodes.append((observations, indices, attempt['cycle_assigned_costs'],
                                 attempt['cycle_update_times'], initial_weight))
            cases.append(episodes)
    assert len(cases) == 1000
    return cfg, spec, cases, {
        'inputs': len(cases), 'cycles': state_count,
        'recorded_total_seconds': recorded_total,
        'recorded_rl_overhead_seconds': recorded_overhead,
        'trajectory_sha256': digest(path),
        'encoder_projection_exact': True,
    }


def timed(variant, cfg, spec, cases, experiment, seed, checkpoint=None):
    encoder = make_encoder(ENCODERS[variant], experiment)
    controller = RecursiveLstdqController(
        feature_dim=encoder.feature_dim, config=cfg, spec=spec, seed=seed)
    start = time.perf_counter()
    for episodes in cases:
        case_snapshot = controller.snapshot_learning_state()
        for observations, indices, costs, durations, initial_weight in episodes:
            attempt_snapshot = controller.snapshot_learning_state()
            controller.start_episode(initial_environment_weight=initial_weight)
            features = encoder.encode(**observations[0])
            controller.select_action(features, explore=True, cycle=0, include_metadata=True)
            for k, (index, cost, duration) in enumerate(zip(indices, costs, durations)):
                terminal = k + 1 == len(indices)
                if terminal:
                    next_features, next_index = features, None
                else:
                    next_features = encoder.encode(**observations[k+1])
                    controller.select_action(next_features, explore=True, cycle=k+1, include_metadata=True)
                    next_index = indices[k+1]
                controller.update(
                    features=features, action_index=index, cost=cost,
                    next_features=next_features, terminal=terminal,
                    next_cycle=k+1, next_action_index=next_index)
                controller.record_update_runtime(duration)
                features = next_features
            controller.finish_episode(learned=True)
    elapsed = time.perf_counter() - start
    assert np.isfinite(controller.theta).all()
    checks = {}
    if checkpoint is not None:
        with np.load(checkpoint, allow_pickle=False) as saved:
            for name in ('a_matrix', 'a_inverse', 'b', 'theta', 'episode_moment_covariance'):
                np.testing.assert_allclose(getattr(controller, name), saved[name], atol=1e-8, rtol=1e-7)
            checks['baseline_checkpoint_reproduced'] = True
    return dict(seconds=elapsed, samples=int(controller.sample_count),
                inverse_rebuilds=int(controller.inverse_rebuild_count),
                feature_dim=encoder.feature_dim, joint_dim=controller.joint_dim, **checks)


result = {
    'scope': 'Fixed-trajectory controller timing; this is not a policy-performance experiment.',
    'included': ['state encoding', 'action scoring and exploration', 'LSTDQ updates',
                 'per-problem and per-attempt snapshots', 'episode start/finish'],
    'excluded': ['native solver', 'bandit computation', 'trajectory logging',
                 'per-operation timer instrumentation', 'controller construction'],
    'fixed_across_variants': ['observations', 'executed actions', 'assigned costs',
                              'recorded update durations', 'setup one-hot descriptors',
                              'residual normalization', 'cycle-time feature'],
    'removed_features': ['7 cycle-bin indicators', '7 residual-gap by cycle-bin interactions'],
    'state_dimensions': [43, 29], 'joint_dimensions': [387, 261],
    'dense_matrix_entries': [387**2, 261**2],
    'dense_matrix_entry_reduction_pct': 100*(1-(29/43)**2),
    'seeds': [1, 2, 3], 'inputs_per_seed': 1000, 'repetitions_per_seed': REPETITIONS,
    'threads': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
    'source_manifest': original_manifest, 'runs': [],
}
for seed in result['seeds']:
    run = SOURCE / f'seed_{seed}_with_time/seed_{seed}'
    experiment = json.loads((run / 'experiment_config.json').read_text())
    for method in METHODS:
        cfg, spec, cases, entry = prepare(run, method, experiment)
        entry.update(seed=seed, method=method, measurements={'with_bins': [], 'without_bins': []})
        for variant in ENCODERS:
            timed(variant, cfg, spec, cases[:20], experiment, seed)
        for repeat in range(REPETITIONS):
            order = ['with_bins', 'without_bins']
            if (seed + repeat) % 2 == 0:
                order.reverse()
            for variant in order:
                checkpoint = run / f'checkpoints/{method}_final.npz' if variant == 'with_bins' and repeat == 0 else None
                measurement = timed(variant, cfg, spec, cases, experiment, seed, checkpoint)
                assert measurement['samples'] == entry['cycles']
                entry['measurements'][variant].append(measurement)
        entry['median_seconds'] = {variant: statistics.median(m['seconds'] for m in measurements)
                                   for variant, measurements in entry['measurements'].items()}
        entry['reduction_pct'] = 100*(1-entry['median_seconds']['without_bins']/entry['median_seconds']['with_bins'])
        result['runs'].append(entry)
        (OUT / 'controller_replay.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:entry[k] for k in ('seed', 'method', 'cycles', 'median_seconds', 'reduction_pct')}), flush=True)

result['aggregate'] = {}
for method in METHODS:
    entries = [entry for entry in result['runs'] if entry['method'] == method]
    medians = {variant: statistics.mean(e['median_seconds'][variant] for e in entries) for variant in ENCODERS}
    saved_seconds = medians['with_bins'] - medians['without_bins']
    recorded_total = statistics.mean(e['recorded_total_seconds'] for e in entries)
    result['aggregate'][method] = dict(
        mean_of_seed_median_seconds=medians,
        controller_replay_reduction_pct=100*saved_seconds/medians['with_bins'],
        saved_seconds_per_1000_inputs=saved_seconds,
        recorded_total_seconds_per_1000_inputs=recorded_total,
        fixed_trajectory_total_reduction_estimate_pct=100*saved_seconds/recorded_total,
        seed_reduction_pct=[e['reduction_pct'] for e in entries])
assert source_manifest() == original_manifest
result['production_source_unchanged'] = True
(OUT / 'controller_replay.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result['aggregate'], indent=2), flush=True)
