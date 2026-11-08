"""Summarize paired early and late windows after all native runs complete."""
from pathlib import Path
import json
import statistics

HERE = Path(__file__).resolve().parent
METHODS = ('default_setup_default_solve', 'default_setup_lstdq', 'bandit_default', 'bandit_lstdq')
WINDOWS = {'all_1000': (0, 1000), 'first_100': (0, 100),
           'inputs_101_to_500': (100, 500), 'last_500': (500, 1000)}
FIELDS = ('total_sec', 'native_sec', 'setup_sec', 'solve_sec', 'rl_overhead_sec',
          'setup_overhead_sec', 'cycles', 'rl_actions', 'retries', 'final_failures',
          'first_attempt_failures')


def main():
    summary = json.loads((HERE / 'summary.json').read_text())
    assert summary['passed'] and summary['production_source_unchanged']
    records = []
    for seed in (1, 2, 3):
        for version in ('with_time', 'without_time'):
            stage = HERE / f'seed_{seed}_{version}/seed_{seed}'
            for method in METHODS:
                windows = {name: dict.fromkeys(FIELDS, 0.) for name in WINDOWS}
                count = 0
                with (stage / f'trajectories/{method}.jsonl').open() as handle:
                    for index, line in enumerate(handle):
                        row = json.loads(line)
                        outcome = row['outcome']
                        attempts = outcome['primary_attempts']
                        values = dict(total_sec=outcome['end_to_end_runtime'],
                            native_sec=outcome['runtime'], setup_sec=outcome['setup_runtime'],
                            solve_sec=outcome['native_solve_runtime'],
                            rl_overhead_sec=outcome['infer_runtime'],
                            setup_overhead_sec=outcome['bandit_overhead_runtime'],
                            cycles=sum(a['cycles'] for a in attempts)+outcome['fallback_cycles'],
                            rl_actions=sum(len(a.get('cycle_actions', [])) for a in attempts),
                            retries=outcome['learned_reselection_count'],
                            final_failures=int(outcome['failed']),
                            first_attempt_failures=int(outcome['first_primary_status'] != 'success'))
                        for name, (start, stop) in WINDOWS.items():
                            if start <= index < stop:
                                for field in FIELDS:
                                    windows[name][field] += values[field]
                        count += 1
                assert count == 1000
                for name, values in windows.items():
                    records.append(dict(seed=seed, version=version, method=method,
                        window=name, inputs=WINDOWS[name][1]-WINDOWS[name][0], **values))
    result = dict(passed=True, records=records, comparisons=[])
    for window in WINDOWS:
        for method in METHODS:
            entries = {version: [r for r in records if r['window'] == window and
                r['method'] == method and r['version'] == version]
                for version in ('with_time', 'without_time')}
            means = {version: {field: statistics.mean(r[field] for r in rows)
                for field in FIELDS} for version, rows in entries.items()}
            paired = []
            for seed in (1, 2, 3):
                a, b = (next(r for r in entries[v] if r['seed'] == seed)
                    for v in ('with_time', 'without_time'))
                da, db = (next(r for r in records if r['window'] == window and
                    r['method'] == METHODS[0] and r['seed'] == seed and r['version'] == v)
                    for v in ('with_time', 'without_time'))
                paired.append(dict(seed=seed, time_change_pct=100*(b['total_sec']/a['total_sec']-1),
                    default_normalized_time_change_pct=100*((b['total_sec']/db['total_sec'])/(a['total_sec']/da['total_sec'])-1),
                    cycle_change_pct=100*(b['cycles']/a['cycles']-1)))
            a, b = means['with_time'], means['without_time']
            entry = dict(window=window, method=method, mean_per_seed=means, paired=paired,
                time_change_pct=100*(b['total_sec']/a['total_sec']-1),
                cycle_change_pct=100*(b['cycles']/a['cycles']-1),
                rl_overhead_per_action_change_pct=100*((b['rl_overhead_sec']/b['rl_actions'])/(a['rl_overhead_sec']/a['rl_actions'])-1) if a['rl_actions'] else None)
            result['comparisons'].append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k != 'mean_per_seed'}))
    (HERE / 'window_analysis.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
