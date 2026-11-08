"""Small native ablation; alter only isolated copies of the state encoder."""
from pathlib import Path
import importlib.util
import itertools
import json
import statistics
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from experiments.runtime import single_thread_environment
from experiments.paper_final.common.artifacts import dump, read, file_hash, now
from experiments.paper_final.run_online import prepare, source_hashes
from experiments.paper_final.online.factorial import METHODS, audit_execution
from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.space import SetupConfigurationSpace
from solve.controllers.common.state_encoder import SolveStateEncoder

SOURCE_RUN = REPO / 'results/shared_setup_tolerance_diffusion40_5000_3seeds_20261018'
ENCODER_PATH = 'solve/controllers/common/state_encoder.py'
BASE_VERSION = 'shared_setup_one_hot_tolerance_residuals'
ABLATION_VERSION = BASE_VERSION + '_without_cycle_time'
CASES = 1000
PREVIOUS = HERE.parent / 'native_100_3seeds'
ORDER = [(1, 'with_time'), (1, 'without_time'),
         (2, 'without_time'), (2, 'with_time'),
         (3, 'with_time'), (3, 'without_time')]


def ablated_source(source):
    replacements = [
        (f'STATE_ENCODING_VERSION = "{BASE_VERSION}"',
         f'STATE_ENCODING_VERSION = "{ABLATION_VERSION}"'),
        ('self._problem_offset = 7', 'self._problem_offset = 6'),
        ('        # Preserve the existing 10 ms reference scale without merging longer\n'
         '        # cycles into a single saturated observation.\n'
         '        cycle_time = float(last_cycle_time) / (5.0 * max(self.time_scale_sec, eps))\n', ''),
        ('        features[:7] = (', '        features[:6] = ('),
        ('            cycle_time,\n', ''),
    ]
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source


def validate_encoder(source):
    module_path = HERE / 'ablation_state_encoder.py'
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location('ablation_state_encoder', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = read(SOURCE_RUN / 'seed_1/experiment_config.json')
    space = SetupConfigurationSpace.from_mapping(
        'expanded', raw['setup']['configuration_spaces']['expanded_no_endpoints'])
    kwargs = dict(tol=1e-6, max_cycles=50, c_max=1000., mode='setup_full',
        setup_obs_encoder=make_setup_feature_encoder(configuration_space=space, parameter_resolution=20),
        problem_context_mode='diffusion3d', weight_bounds=(1., 3.))
    original, reduced = SolveStateEncoder(**kwargs), module.SolveStateEncoder(**kwargs)
    checked = 0
    for seed in (1, 2, 3):
        for method in ('default_setup_lstdq', 'bandit_lstdq'):
            with (SOURCE_RUN / f'seed_{seed}/trajectories/{method}.jsonl').open() as handle:
                for line in itertools.islice(handle, 10):
                    row = json.loads(line)
                    for attempt in row['outcome']['primary_attempts']:
                        if not attempt.get('rl_episode_recorded'):
                            continue
                        residuals = [1., *attempt['cycle_residuals']]
                        times = [0., *attempt['cycle_times']]
                        weights = [1., *attempt['cycle_actions']]
                        for k in range(len(attempt['cycle_actions'])):
                            obs = dict(mkw=row['mkw'], setup_params=attempt['params'],
                                problem_context=row['context'], initial_residual=1.,
                                residual=residuals[k], previous_residual=residuals[max(0, k-1)],
                                cycle=k, last_weight=weights[k], last_cycle_time=times[k])
                            before, after = original.encode(**obs), reduced.encode(**obs)
                            assert before.shape == (43,) and after.shape == (42,)
                            np.testing.assert_array_equal(np.delete(before, 5), after)
                            np.testing.assert_array_equal(after, reduced.encode(**dict(obs, last_cycle_time=123.)))
                            checked += 1
    result = dict(passed=True, states_checked=checked, original_dim=43, ablated_dim=42,
        all_other_features_exactly_equal=True, time_feature_absent=True,
        new_encoding_version=ABLATION_VERSION)
    dump(HERE / 'encoder_validation.json', result)
    print('ENCODER VALIDATION ' + json.dumps(result), flush=True)


def main():
    source_before = source_hashes()
    dump(HERE / 'original_source_manifest.json', source_before)
    new_source = ablated_source((REPO / ENCODER_PATH).read_text())
    assert source_before == read(PREVIOUS / 'original_source_manifest.json')
    assert new_source == (PREVIOUS / 'ablation_state_encoder.py').read_text()
    shutil.copy2(PREVIOUS / 'encoder_validation.json', HERE / 'encoder_validation.json')
    (HERE / 'ablation_state_encoder.py').write_text(new_source)
    dump(HERE / 'diagnostic_protocol.json', dict(prepared_at=now(),
        inputs_per_branch=CASES, seeds=[1, 2, 3], methods=METHODS, run_order=ORDER,
        learning='Cold start; online learning active from input 1 for all learned branches.',
        only_source_change=ENCODER_PATH, original_state_dim=43, ablated_state_dim=42,
        original_joint_dim=387, ablated_joint_dim=378,
        native_cost_and_controller_overhead_in_reward=True,
        timing='Measured native setup and solve plus setup and controller overhead.',
        pairing='Same stream prefix, candidate schedule, method order, seeds and hyperparameters within each pair.',
        execution='Six serial, single-thread workers; four branches per worker.',
        limits='Short cold-start diagnostic, not a long-run or policy-equivalence claim.'))
    results = []
    paired = {}
    for seed, version in ORDER:
        label = str(seed)
        output = HERE / f'seed_{seed}_{version}'
        suite_path = HERE / f'suite_seed_{seed}.json'
        source_config = SOURCE_RUN / f'seed_{seed}/experiment_config.json'
        if not suite_path.exists():
            dump(suite_path, dict(name=f'Cycle-time feature diagnostic seed {seed}',
                seeds=[dict(label=label, config=str(source_config))]))
        print(f'PREPARING seed {seed} {version}', flush=True)
        protocol = prepare(suite_path, output, cases=CASES)
        manifest = read(output / 'source_manifest.json')
        assert manifest == source_before
        if version == 'without_time':
            (output / 'source' / ENCODER_PATH).write_text(new_source)
            manifest[ENCODER_PATH] = file_hash(output / 'source' / ENCODER_PATH)
            assert {key for key in manifest if manifest[key] != source_before[key]} == {ENCODER_PATH}
            dump(output / 'source_manifest.json', manifest)
            config = read(output / f'seed_{label}/experiment_config.json')
            assert config['solve']['state_encoding'] == BASE_VERSION
            config['solve']['state_encoding'] = ABLATION_VERSION
            dump(output / f'seed_{label}/experiment_config.json', config)
            protocol['stages'][label]['config_sha256'] = file_hash(output / f'seed_{label}/experiment_config.json')
            protocol['solve_state_encoding'] = ABLATION_VERSION
            protocol['ablation'] = dict(name='Remove previous native cycle time from state only',
                source_configuration_transform={'solve.state_encoding': ABLATION_VERSION})
            dump(output / 'protocol.json', protocol)
        input_hash = protocol['stages'][label]['inputs_sha256']
        paired.setdefault(seed, {})[version] = {'inputs_sha256': input_hash}
        if len(paired[seed]) == 2:
            assert paired[seed]['with_time']['inputs_sha256'] == paired[seed]['without_time']['inputs_sha256']
        cached = PREVIOUS / f'seed_{seed}_with_time/seed_{seed}/aot_candidate_schedules'
        destination = output / f'seed_{label}/aot_candidate_schedules'
        shutil.copytree(cached, destination)
        cache_hashes = {str(p.relative_to(cached)): file_hash(p) for p in cached.rglob('*') if p.is_file()}
        assert cache_hashes == {str(p.relative_to(destination)): file_hash(p) for p in destination.rglob('*') if p.is_file()}
        dump(output / 'candidate_schedule_reuse.json', dict(source=str(cached), hashes=cache_hashes))
        dump(output / 'release.json', dict(at=now(), serial_single_worker=True))
        print(f'RUNNING seed {seed} {version}: four branches x {CASES} inputs', flush=True)
        with (output / f'seed_{label}/worker.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'experiments.paper_final.run_online', 'worker',
                '--output', str(output), '--label', label], cwd=output / 'source',
                env=single_thread_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
        audit_execution(output)
        stage = output / f'seed_{label}'
        result = read(stage / 'result.json')
        assert result['passed'] and result['complete'] and result['cases'] == CASES
        initial = read(stage / 'initial_audit.json')
        assert set(initial['candidates'].values()) == {cache_hashes['expanded_no_endpoints/structured512/candidate_ids.npy']}
        paired[seed][version]['candidate_hashes'] = initial['candidates']
        if all('candidate_hashes' in record for record in paired[seed].values()) and len(paired[seed]) == 2:
            assert paired[seed]['with_time']['candidate_hashes'] == paired[seed]['without_time']['candidate_hashes']
        for method in METHODS:
            metrics = dict(result['methods'][method])
            rows = [json.loads(line) for line in (stage / f'trajectories/{method}.jsonl').read_text().splitlines()]
            outcomes = [row['outcome'] for row in rows]
            attempts = [attempt for outcome in outcomes for attempt in outcome['primary_attempts']]
            metrics.update(
                rl_overhead_sec=sum(o['infer_runtime'] for o in outcomes),
                setup_overhead_sec=sum(o['bandit_overhead_runtime'] for o in outcomes),
                native_setup_sec=sum(o['setup_runtime'] for o in outcomes),
                native_solve_sec=sum(o['native_solve_runtime'] for o in outcomes),
                retries=sum(o['learned_reselection_count'] for o in outcomes),
                primary_attempts=len(attempts),
                failed_primary_attempts=sum(a['status'] != 'success' for a in attempts),
                controller_feature_sec=sum(o.get('feature_runtime', 0.) for o in outcomes),
                controller_decision_sec=sum(o.get('decision_runtime', 0.) for o in outcomes),
                controller_update_sec=sum(o.get('update_runtime', 0.) for o in outcomes),
            )
            assert abs(metrics['total_sec'] - metrics['native_sec'] - metrics['rl_overhead_sec'] - metrics['setup_overhead_sec']) < 1e-8
            metrics['rl_overhead_us_per_action'] = 1e6 * metrics['rl_overhead_sec'] / metrics['rl_actions'] if metrics['rl_actions'] else None
            if method in (METHODS[1], METHODS[3]):
                checkpoint = stage / f'checkpoints/{method}_final.npz.encoder.json'
                metadata = read(checkpoint)
                expected_dim = 43 if version == 'with_time' else 42
                assert metadata['feature_dim'] == expected_dim
                assert metadata['encoding_version'] == (BASE_VERSION if version == 'with_time' else ABLATION_VERSION)
                assert result['audits'][method]['controller']['joint_feature_dim'] == 9 * expected_dim
            results.append(dict(seed=seed, version=version, method=method, **metrics))
        dump(output / 'status.json', dict(status='complete', passed=True, at=now()))
        dump(HERE / 'raw_results.json', results)
        print('COMPLETED ' + json.dumps(dict(seed=seed, version=version, methods=result['methods'])), flush=True)
    assert source_hashes() == source_before, 'Production execution source changed during diagnostic'
    summary = []
    fields = ('total_sec', 'native_sec', 'native_setup_sec', 'native_solve_sec',
        'rl_overhead_sec', 'setup_overhead_sec', 'cycles', 'retries', 'first_failures',
        'primary_attempts', 'failed_primary_attempts', 'controller_feature_sec',
        'controller_decision_sec', 'controller_update_sec')
    for method in METHODS:
        entries = {version: [r for r in results if r['method'] == method and r['version'] == version]
            for version in ('with_time', 'without_time')}
        means = {version: {field: statistics.mean(r[field] for r in rows) for field in fields}
            for version, rows in entries.items()}
        pairs = []
        for seed in (1, 2, 3):
            a = next(r for r in entries['with_time'] if r['seed'] == seed)
            b = next(r for r in entries['without_time'] if r['seed'] == seed)
            da = next(r for r in results if r['method'] == METHODS[0] and r['seed'] == seed and r['version'] == 'with_time')
            db = next(r for r in results if r['method'] == METHODS[0] and r['seed'] == seed and r['version'] == 'without_time')
            pairs.append(dict(seed=seed, total_reduction_pct=100*(1-b['total_sec']/a['total_sec']),
                default_normalized_reduction_pct=100*(1-(b['total_sec']/db['total_sec'])/(a['total_sec']/da['total_sec'])),
                cycle_change=b['cycles']-a['cycles'], retries_change=b['retries']-a['retries']))
        summary.append(dict(method=method, mean_per_1000_inputs=means, paired=pairs,
            total_reduction_pct=100*(1-means['without_time']['total_sec']/means['with_time']['total_sec']),
            failures={version: sum(r['failures'] for r in rows) for version, rows in entries.items()},
            fallbacks={version: sum(r['fallbacks'] for r in rows) for version, rows in entries.items()},
            pooled_rl_overhead_us_per_action={version: 1e6*sum(r['rl_overhead_sec'] for r in rows)/sum(r['rl_actions'] for r in rows) if sum(r['rl_actions'] for r in rows) else None for version, rows in entries.items()}))
    final = dict(passed=True, inputs_per_branch=CASES, seeds=[1, 2, 3], results=summary,
        pairing_audit=paired, production_source_unchanged=True,
        interpretation='Short online-learning diagnostic; runtime feedback can change policies. Same seeds do not eliminate runtime-driven variation. No long-run inference.')
    dump(HERE / 'summary.json', final)
    print('FINAL SUMMARY ' + json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
