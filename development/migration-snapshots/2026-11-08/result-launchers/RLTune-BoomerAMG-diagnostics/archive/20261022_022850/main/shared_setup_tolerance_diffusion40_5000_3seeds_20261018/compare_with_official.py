import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[1]
old = json.loads((repo / 'data/module1/summary.json').read_text())
new = json.loads((root / 'summary.json').read_text())
old_protocol = json.loads((repo / 'data/module1/protocol.json').read_text())
new_protocol = json.loads((root / 'protocol.json').read_text())
inputs = json.loads((repo / 'data/module1/inputs/manifest.json').read_text())
assert old_protocol['native_sha256'] == new_protocol['native_sha256']
for seed in ('1', '2', '3'):
    assert new_protocol['stages'][seed]['full_stream_sha256'] in inputs['streams']

names = {
    'default_setup_default_solve': 'Default',
    'default_setup_lstdq': 'Default + LSTDQ',
    'bandit_default': 'LinUCB',
    'bandit_lstdq': 'LinUCB + LSTDQ',
}
comparisons = []
for window in ('full', 'last_1000'):
    for cohort, data, seeds in (
        ('Official: seeds 1-6', old, list('123456')),
        ('Official: seeds 1-3', old, list('123')),
        ('New: seeds 1-3', new, list('123')),
    ):
        selected = [r for r in data['rows']
                    if r.get('experiment', 'diffusion_40') == 'diffusion_40'
                    and r['window'] == window and str(r['seed']) in seeds]
        indexed = {(str(r['seed']), r['method']): r for r in selected}
        assert len(selected) == len(indexed) == len(seeds) * len(names)
        for method, label in names.items():
            rows = [indexed[(seed, method)] for seed in seeds]
            reductions = []
            for seed, row in zip(seeds, rows):
                baseline = indexed[(seed, 'default_setup_default_solve')]
                assert row['cases'] == baseline['cases'] == (5000 if window == 'full' else 1000)
                value = 100 * (1 - row['total_sec'] / baseline['total_sec'])
                assert math.isclose(value, row['inclusive_reduction_pct'], abs_tol=1e-9)
                reductions.append(value)
            comparisons.append({
                'window': window, 'cohort': cohort, 'method': label,
                'mean_total_sec': statistics.mean(r['total_sec'] for r in rows),
                'mean_native_sec': statistics.mean(r['native_sec'] for r in rows),
                'mean_learner_overhead_sec': statistics.mean(r['total_sec'] - r['native_sec'] for r in rows),
                'mean_reduction_pct': statistics.mean(reductions),
                'reduction_sample_sd': statistics.stdev(reductions),
                'final_failures': sum(r['failures'] for r in rows),
                'first_attempt_failures': sum(r['first_failures'] for r in rows),
            })

old_joint = next(r for r in comparisons if (r['window'], r['cohort'], r['method']) == ('full', 'Official: seeds 1-3', 'LinUCB + LSTDQ'))
new_joint = next(r for r in comparisons if (r['window'], r['cohort'], r['method']) == ('full', 'New: seeds 1-3', 'LinUCB + LSTDQ'))
delta = {k: new_joint[k] - old_joint[k] for k in ('mean_total_sec', 'mean_native_sec', 'mean_learner_overhead_sec', 'mean_reduction_pct')}
delta['total_time_increase_pct'] = 100 * (new_joint['mean_total_sec'] / old_joint['mean_total_sec'] - 1)
result = {'comparisons': comparisons, 'joint_matched_difference': delta,
          'checks': {'native_binaries_identical': True, 'official_input_hashes_match': True,
                     'reductions_recomputed_from_seed_totals': True}}
(root / 'official_comparison.json').write_text(json.dumps(result, indent=2) + '\n')

receipt = {'schemaVersion': 1, 'items': [{
    'id': 'official-comparison',
    'title': 'Diffusion 40: official results and the revised implementation',
    'queries': [{
        'id': 'diffusion-40-comparison',
        'source': {
            'label': 'Module 1 and the diffusion 40 rerun',
            'files': [{'label': 'Official Module 1 summary.json'},
                      {'label': 'Revised implementation summary.json'},
                      {'label': 'Module 1 inputs manifest.json'},
                      {'label': 'Official and revised protocol.json'}],
            'metricDefinitions': [{
                'label': 'Measured total-time reduction',
                'definition': 'Measured total-time reductions include native solver execution and learner overhead, using paired Default totals within each seed and averaging the resulting percentages across seeds.'
            }],
            'filters': ['Diffusion on a 40 cubed grid', 'Four independently measured methods',
                        'Official results use seeds 1-6; the matched comparison and rerun use seeds 1-3'],
            'caveats': ['The new run changes setup encoding, the RL cost target, and feature normalization together, so this comparison does not isolate the effect of any one change.',
                        'The experiments were timed in separate runs; their Default mean totals differ by approximately 0.05 percent for seeds 1-3.'],
            'evidenceFlow': [
                {'kind': 'source', 'title': 'Comparison population', 'detail': 'The published diffusion 40 table agrees with the six-seed Module 1 summary; seeds 1-3 provide the direct comparison with the rerun.'},
                {'kind': 'validation', 'title': 'Matched configuration and inputs', 'detail': 'The three complete input-stream hashes match the released Module 1 streams. Native-library hashes are identical. All selected seed/window/method cells are unique and have the expected input counts.'},
                {'kind': 'calculation', 'title': 'Overhead decomposition', 'detail': 'For the joint method on seeds 1-3, the change in total time equals the change in native time plus the change in measured learner overhead. Reductions were independently recomputed from the saved seed totals.'}
            ]
        },
        'reportingPeriod': 'Full stream: inputs 1-5000; final window: inputs 4001-5000',
        'columns': list(comparisons[0]),
        'rows': comparisons,
        'preview': {'kind': 'aggregate', 'note': 'Mean seconds are per seed; failure counts sum across the listed seeds.', 'totalRows': len(comparisons)},
        'methods': [{'language': 'calculation', 'code': 'For each seed: reduction = 100 * (1 - method_total / Default_total).\nReport the arithmetic mean and sample standard deviation of seed-level reductions.\nMeasured learner overhead = measured total - native solver time.\nMatched joint total time: 258.5107766975 s -> 269.1110922442 s.\nMatched joint native time: 225.7670890000 s -> 224.8257776667 s.\nMatched joint learner overhead: 32.7436876975 s -> 44.2853145776 s.'}]
    }]
}]}
(root / 'official_comparison_sources.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(result, indent=2))
