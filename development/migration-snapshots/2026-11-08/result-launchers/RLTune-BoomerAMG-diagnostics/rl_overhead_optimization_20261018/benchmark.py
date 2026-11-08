"""Replay fixed recorded inputs to separate implementation cost from policy changes."""
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.space import SetupConfigurationSpace
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig
from solve.controllers.common.linear_lcb import _greedy_cost_index
from solve.controllers.recursive_lstdq import RecursiveLstdqController, RecursiveLstdqSpec

package = types.ModuleType('reference_lstdq')
package.__path__ = [str(ROOT / 'baseline')]
sys.modules['reference_lstdq'] = package
sys.modules['reference_lstdq.config'] = importlib.import_module('solve.controllers.recursive_lstdq.config')
Reference = importlib.import_module('reference_lstdq.controller').RecursiveLstdqController
RUN = REPO / 'results/shared_setup_tolerance_diffusion40_5000_3seeds_20261018'
config = json.loads((RUN / 'seed_1/experiment_config.json').read_text())
space = SetupConfigurationSpace.from_mapping(
    'expanded', config['setup']['configuration_spaces']['expanded_no_endpoints'])
encoder = SolveStateEncoder(
    tol=1e-6, max_cycles=50, c_max=1000., mode='setup_full',
    setup_obs_encoder=make_setup_feature_encoder(configuration_space=space, parameter_resolution=20),
    problem_context_mode='diffusion3d', weight_bounds=(1., 3.))


def prepare(method):
    checkpoint = RUN / f'seed_1/checkpoints/{method}_final.npz'
    with np.load(checkpoint, allow_pickle=False) as saved:
        cfg = ExpectedSarsaLambdaConfig(**json.loads(str(saved['config'].item())))
        spec = RecursiveLstdqSpec(**json.loads(str(saved['spec'].item())))
    weights = np.asarray(cfg.weights)
    cases = []
    with (RUN / f'seed_1/trajectories/{method}.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            episodes = []
            for attempt in row['outcome']['primary_attempts']:
                if not attempt['rl_episode_recorded']:
                    continue
                actions = attempt['cycle_actions']
                residuals = [1., *attempt['cycle_residuals']]
                times = [0., *attempt['cycle_times']]
                previous_weights = [1., *actions]
                states = [encoder.encode(
                    mkw=row['mkw'], setup_params=attempt['params'],
                    problem_context=row['context'], initial_residual=1.,
                    residual=residuals[k], previous_residual=residuals[max(0, k-1)],
                    cycle=k, last_weight=previous_weights[k], last_cycle_time=times[k],
                ) for k in range(len(actions))]
                indices = [int(np.argmin(np.abs(weights - action))) for action in actions]
                transitions = []
                for k in range(len(actions)):
                    terminal = k == len(actions) - 1
                    transitions.append(dict(
                        features=states[k], action_index=indices[k],
                        cost=attempt['cycle_assigned_costs'][k],
                        next_features=states[k] if terminal else states[k+1],
                        terminal=terminal, next_cycle=k+1,
                        next_action_index=None if terminal else indices[k+1],
                    ))
                episodes.append((transitions, attempt['cycle_update_times']))
            cases.append(episodes)
    return cfg, spec, cases


def model(cls, cfg, spec):
    return cls(feature_dim=43, config=cfg, spec=spec, seed=1)


def timed(cls, cfg, spec, cases):
    controller = model(cls, cfg, spec)
    start = time.perf_counter()
    for episodes in cases:
        case_snapshot = controller.snapshot_learning_state()
        for transitions, durations in episodes:
            attempt_snapshot = controller.snapshot_learning_state()
            controller.start_episode(initial_environment_weight=1.)
            for transition, duration in zip(transitions, durations):
                controller.select_action(transition['features'], explore=True, include_metadata=True)
                controller.update(**transition)
                controller.record_update_runtime(duration)
            controller.finish_episode(learned=True)
    return time.perf_counter() - start


def validate(cfg, spec, cases, checkpoint):
    reference = model(Reference, cfg, spec)
    optimized = model(RecursiveLstdqController, cfg, spec)
    count = checks = 0
    errors = np.zeros(3)
    for episodes in cases:
        for transitions, durations in episodes:
            for c in (reference, optimized):
                c.start_episode(initial_environment_weight=1.)
            for transition, duration in zip(transitions, durations):
                if count % 97 == 0:
                    old = reference._values(transition['features'], cycle=0)
                    new = optimized._values(transition['features'], cycle=0)
                    for i, (a, b) in enumerate(zip(new, old)):
                        np.testing.assert_allclose(a, b, atol=1e-10, rtol=2e-8)
                        errors[i] = max(errors[i], float(np.max(np.abs(a-b))))
                    choices = [_greedy_cost_index(v[2], allowed_indices=c._all_action_indices,
                               anchor_index=c.anchor_index, secondary_values=v[0])
                               for c, v in ((reference, old), (optimized, new))]
                    assert choices[0] == choices[1]
                    checks += 1
                for c in (reference, optimized):
                    c.update(**transition)
                    c.record_update_runtime(duration)
                count += 1
            for c in (reference, optimized):
                c.finish_episode(learned=True)
    for name in ('a_matrix', 'a_inverse', 'b', 'theta', 'episode_moment_covariance'):
        np.testing.assert_allclose(getattr(optimized, name), getattr(reference, name), atol=1e-9, rtol=2e-9)
    with np.load(checkpoint, allow_pickle=False) as saved:
        for name in ('a_matrix', 'a_inverse', 'b', 'theta', 'episode_moment_covariance'):
            np.testing.assert_allclose(getattr(reference, name), saved[name], atol=1e-9, rtol=2e-9)
    return {'transitions': count, 'score_checks': checks, 'greedy_actions_match': True,
            'max_absolute_errors_mean_uncertainty_score': errors.tolist(),
            'original_checkpoint_reproduced': True}


result = {'state_features': 43, 'joint_features': 387, 'methods': {}}
for method in ('default_setup_lstdq', 'bandit_lstdq'):
    cfg, spec, cases = prepare(method)
    print(method, 'prepared', sum(len(t) for case in cases for t, _ in case), 'transitions', flush=True)
    timings = {'reference': [], 'optimized': []}
    for repetition in range(5):
        order = [('reference', Reference), ('optimized', RecursiveLstdqController)]
        if repetition % 2:
            order.reverse()
        for label, cls in order:
            timings[label].append(timed(cls, cfg, spec, cases[:200]))
    medians = {k: statistics.median(v) for k, v in timings.items()}
    entry = {'benchmark_inputs': 200, 'repetitions': 5, 'timings_sec': timings,
             'median_sec': medians, 'controller_time_reduction_pct': 100*(1-medians['optimized']/medians['reference'])}
    print(method, 'benchmark', json.dumps(entry), flush=True)
    entry['equivalence'] = validate(cfg, spec, cases, RUN / f'seed_1/checkpoints/{method}_final.npz')
    result['methods'][method] = entry
    (ROOT / 'benchmark_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(method, 'equivalence', json.dumps(entry['equivalence']), flush=True)
print('All fixed-data benchmark and equivalence checks passed', flush=True)
