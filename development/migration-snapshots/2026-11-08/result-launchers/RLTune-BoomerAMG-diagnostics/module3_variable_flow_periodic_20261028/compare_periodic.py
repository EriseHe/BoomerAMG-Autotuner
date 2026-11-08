"""Local three-policy test of the six saved Module 3 variable-flow models.

Reuse the established frozen-policy preparation, execution, recovery, and audit
functions. No fixed-weight search or learner updates are performed.
"""
from pathlib import Path
import argparse
import itertools
import json
import random
import shutil
import signal

from experiments.runtime import configure_single_thread
configure_single_thread()
from experiments.runtime import prevent_sleep, stop_sleep_prevention
from experiments.paper_final.common import artifacts as a, frozen_policy as policy
from experiments.paper_final.common.frozen_inputs import execution_source_paths
from experiments.paper_final.common.workers import run_frozen_worker, run_frozen_phase
from experiments.paper_final.matched_policy.configuration import load_config, policy_library
from experiments.paper_final.matched_policy.planning import make_inputs, freeze_hierarchies
from experiments.paper_final.matched_policy.workflow import (
    FrozenRuntime, verify, preflight, phase_rows,
)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'run'
RUNNER = 'results.module3_variable_flow_periodic_20261028.compare_periodic'
METHODS = ('reference', 'periodic', 'rl')
POLICY_NAMES = {'reference': 'fixed_1', 'periodic': 'periodic', 'rl': 'rl_frozen'}


def prepare():
    if OUTPUT.exists():
        verify(OUTPUT)
        return
    source = a.ROOT / 'results/modules_2_3_20261027/study/module3'
    config = {
        'schema_version': 1, 'name': 'module3_variable_flow_frozen_three_policy',
        'cases': 100, 'evaluation_repetitions': 3, 'weights': [1],
        'periodic_pattern': [2.85, 1.10], 'workers': 3,
        'seeds': {'inputs': 20261028, 'candidates': 20261029,
                  'order': 20261030, 'repeats': 20261031},
        'training_runs': [
            {'seed': seed, 'method': 'bandit_lstdq',
             'run_dir': str(source / f'variable_flow_60_batch{1 if seed <= 3 else 2}' / f'seed_{seed}')}
            for seed in range(1, 7)
        ],
    }
    config_path = HERE / 'configuration.json'
    a.dump(config_path, config)
    cfg = load_config(config_path)
    inputs = make_inputs(cfg)
    OUTPUT.mkdir()
    for folder in ('analysis', 'progress', 'raw/test', 'provenance/source'):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    a.dump(OUTPUT / 'configuration.json', cfg)
    base, audits, sources = freeze_hierarchies(OUTPUT, cfg, inputs)
    policies = {name: policy_library(cfg)[name] for name in POLICY_NAMES.values()}
    protocol = {
        'module': 3, 'format': 'module3-variable-flow-frozen-three-policy',
        'name': cfg['name'], 'methods': list(METHODS),
        'training_seeds': list(range(1, 7)), 'problem': cfg['problem'],
        'test_cases': len(inputs), 'repetitions_per_case': 3,
        'policies': policies, 'sources': sources, 'max_workers': 3,
        'tolerance': cfg['solver']['tolerance'],
        'max_cycles': cfg['solver']['max_cycles'], 'profile': cfg['solver']['profile'],
        'seeds': cfg['seeds'], 'created_at': a.now(),
        'selection': 'No fixed-weight search. Three prescribed policy roles.',
        'recovery': 'One primary matched hierarchy, then one fresh Default fallback on numerical failure; no setup reselection or learning.',
        'evaluation': '100 new paired inputs, same inputs for all seeds; three independent repetitions; balanced random policy order; all failures and recovery costs retained.',
        'cost': 'Inclusive continuation excludes common primary setup, includes recovery setup and solve and controller computation.',
    }
    rng = random.Random(cfg['seeds']['order'])
    jobs = []
    for seed in range(1, 7):
        orders = list(itertools.permutations(POLICY_NAMES.values())) * 50
        rng.shuffle(orders)
        seed_jobs = [row for row in base if row['seed'] == seed]
        n = 0
        for row in seed_jobs:
            for rep in range(3):
                jobs.append({**row, 'repeat': rep,
                             'method_policies': POLICY_NAMES,
                             'policy_order': list(orders[n])})
                n += 1
        assert n == len(orders)
    rng.shuffle(jobs)
    for name, value in [('protocol.json', protocol), ('inputs.json', {'test': inputs}),
                        ('jobs_base.json', base), ('jobs_test.json', jobs),
                        ('setup_freeze_audit.json', audits)]:
        a.dump(OUTPUT / name, value)
    hashes = {name: a.file_hash(a.ROOT / name) for name in sorted(execution_source_paths())}
    for name in hashes:
        dest = OUTPUT / 'provenance/source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.ROOT / name, dest)
    shutil.copy2(__file__, OUTPUT / 'diagnostic_runner.py')
    a.dump(OUTPUT / 'source_manifest.json', hashes)
    env = policy.environment_check(OUTPUT)
    files = ['configuration.json', 'protocol.json', 'inputs.json', 'jobs_base.json',
             'jobs_test.json', 'setup_freeze_audit.json', 'source_manifest.json',
             'diagnostic_runner.py']
    files += [str(p.relative_to(OUTPUT)) for p in (OUTPUT / 'checkpoints').rglob('*') if p.is_file()]
    a.dump(OUTPUT / 'prepared.json', {'at': a.now(),
           'files_sha256': {name: a.file_hash(OUTPUT / name) for name in files},
           'execution_environment': policy.environment_identity(env)})
    a.dump(OUTPUT / 'status.json', {'status': 'prepared', 'trials': 5400})
    a.emit(OUTPUT, 'Variable-flow three-policy test prepared', seeds=6, cases=100, trials=5400)


def analyze():
    import math
    import statistics
    from collections import Counter
    rows = phase_rows(OUTPUT, 'test')
    protocol = a.read(OUTPUT / 'protocol.json')
    fields = ('native_continuation_sec', 'inclusive_continuation_sec',
              'native_total_sec', 'inclusive_total_sec')
    per_case, per_seed = [], []
    for seed in protocol['training_seeds']:
        for method, name in POLICY_NAMES.items():
            selected = [r for r in rows if r['seed'] == seed and r['policy'] == name]
            assert len(selected) == 300
            for case in range(100):
                records = [r for r in selected if r['case_id'] == case]
                assert len(records) == 3
                per_case.append({
                    'seed': seed, 'method': method, 'case_id': case,
                    'success': all(r['success'] for r in records),
                    'fallback': any(r['outcome'].get('fallback_used') for r in records),
                    **{f: statistics.mean(r[f] for r in records) for f in fields},
                })
            cases = [r for r in per_case if r['seed'] == seed and r['method'] == method]
            per_seed.append({
                'seed': seed, 'method': method, 'cases': 100,
                'completed': sum(r['success'] for r in cases),
                'fallback_cases': sum(r['fallback'] for r in cases),
                **{f: math.fsum(r[f] for r in cases) for f in fields},
                'primary_failure_reasons': dict(Counter(
                    str(r['outcome'].get('primary_failure_reason')) for r in selected
                    if r['outcome'].get('fallback_used'))),
            })
    a.write_csv(OUTPUT / 'analysis/per_case.csv', per_case)
    a.write_csv(OUTPUT / 'analysis/per_seed.csv', per_seed)
    audit = {'exact_coverage': True, 'evaluation_trials': len(rows),
             'evaluation_failures': sum(not r['success'] for r in rows),
             'recovered_evaluations': sum(bool(r['outcome'].get('fallback_used')) for r in rows)}
    a.dump(OUTPUT / 'analysis/summary.json', {'protocol': protocol, 'per_seed': per_seed, 'audit': audit})
    a.dump(OUTPUT / 'complete.json', {'at': a.now(), 'audit': audit})
    a.dump(OUTPUT / 'status.json', {'status': 'complete', 'audit': audit})
    print(json.dumps(per_seed, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'worker', 'analyze'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--phase', default='test')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.command == 'worker':
        run_frozen_worker(args.output, args.phase, args.worker, 3, run_number=3,
                          verify=verify, policy=FrozenRuntime(args.output))
    elif args.command == 'analyze':
        analyze()
    elif args.command == 'prepare':
        prepare()
    else:
        prepare()
        verify(OUTPUT)
        awake = prevent_sleep()
        try:
            if not (OUTPUT / 'preflight.json').exists():
                preflight(OUTPUT)
            if not (OUTPUT / 'raw/test/complete.json').exists():
                run_frozen_phase(OUTPUT, 'test', runner_module=RUNNER, run_number=3, workers=3)
            analyze()
        finally:
            stop_sleep_prevention(awake)


if __name__ == '__main__':
    main()
