"""Audit and show the six-seed variable-flow comparison after completion."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'
OUT = ROOT / 'readout'
METHODS = {'reference': 'fixed_1', 'periodic': 'periodic', 'rl': 'rl_frozen'}
FIELDS = ('native_continuation_sec', 'inclusive_continuation_sec')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (RUN / 'complete.json').exists(), 'Timing run is not complete'
    protocol = read(RUN / 'protocol.json')
    for marker, field in [('prepared.json', 'files_sha256'),
                          ('raw/test/complete.json', 'raw_sha256')]:
        for rel, expected in read(RUN / marker)[field].items():
            assert sha(RUN / rel) == expected, (marker, rel)
    inputs = read(RUN / 'inputs.json')['test']
    new_matrices = {json.dumps(row['mkw'], sort_keys=True) for row in inputs}
    assert len(new_matrices) == 100
    for source in protocol['sources'].values():
        for f in source['original_files'].values():
            assert sha(Path(f['path'])) == f['sha256']
        training = Path(source['original_files']['config']['path']).parent
        seen = {json.dumps(json.loads(line)['mkw'], sort_keys=True)
                for line in (training / 'inputs.jsonl').open()}
        assert not (new_matrices & seen), 'Evaluation overlaps training inputs'

    jobs = read(RUN / 'jobs_test.json')
    expected = {(j['seed'], j['case_id'], j['repeat'], p): j
                for j in jobs for p in j['policy_order']}
    assert len(expected) == 5400
    raw = {}
    for path in sorted((RUN / 'raw/test').glob('worker_*.jsonl')):
        for line in path.open():
            row = json.loads(line)
            key = (row['seed'], row['case_id'], row['repeat'], row['policy'])
            assert key not in raw
            raw[key] = row
            j = expected[key]
            assert (row['input_id'], row['hierarchy_id']) == (j['input_id'], j['hierarchy_id'])
            o = row['outcome']
            assert math.isclose(row[FIELDS[0]], o['solve_runtime'] + o.get('fallback_setup_runtime', 0), abs_tol=1e-9)
            assert math.isclose(row[FIELDS[1]], row[FIELDS[0]] + o['infer_runtime'], abs_tol=1e-9)
            if row['success']:
                assert math.isfinite(o['completed_residual_norm'])
                assert o['completed_residual_norm'] <= protocol['tolerance']
            spec = protocol['policies'][row['policy']]
            if spec['kind'] in ('fixed', 'periodic'):
                pattern = spec['pattern']
                assert o['cycle_actions'] == [pattern[k % len(pattern)] for k in range(len(o['cycle_actions']))]
    assert set(raw) == set(expected)
    for seed in range(1, 7):
        counts = Counter(tuple(j['policy_order']) for j in jobs if j['seed'] == seed)
        assert set(counts) == set(itertools.permutations(METHODS.values()))
        assert set(counts.values()) == {50}

    cases = {}
    for seed in range(1, 7):
        for case in range(100):
            for method, policy in METHODS.items():
                rr = [raw[seed, case, rep, policy] for rep in range(3)]
                cases[seed, case, method] = {
                    'success': all(r['success'] for r in rr),
                    'fallback': any(r['outcome'].get('fallback_used') for r in rr),
                    **{field: statistics.mean(r[field] for r in rr) for field in FIELDS},
                }

    rows = []
    for seed in range(1, 7):
        row = {'seed': seed}
        for method in METHODS:
            rr = [cases[seed, c, method] for c in range(100)]
            row[method + '_completed'] = sum(r['success'] for r in rr)
            row[method + '_fallback_cases'] = sum(r['fallback'] for r in rr)
            row[method + '_measured_inclusive_ms_per_problem'] = 10 * math.fsum(r[FIELDS[1]] for r in rr)
        row['rl_vs_reference_inclusive_reduction_pct'] = (
            100 * (1 - row['rl_measured_inclusive_ms_per_problem'] /
                   row['reference_measured_inclusive_ms_per_problem'])
            if row['reference_completed'] == row['rl_completed'] == 100 else None)
        common = [c for c in range(100) if cases[seed, c, 'periodic']['success'] and cases[seed, c, 'rl']['success']]
        row['common_completed_periodic_rl'] = len(common)
        for field, label in zip(FIELDS, ('solver', 'inclusive')):
            periodic = math.fsum(cases[seed, c, 'periodic'][field] for c in common)
            rl = math.fsum(cases[seed, c, 'rl'][field] for c in common)
            row[f'rl_vs_periodic_{label}_reduction_pct'] = 100 * (1 - rl / periodic) if periodic else None
        rows.append(row)
    OUT.mkdir(exist_ok=True)
    with (OUT / 'six_seed_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    a = [r['rl_vs_periodic_inclusive_reduction_pct'] for r in rows
         if r['rl_vs_periodic_inclusive_reduction_pct'] is not None]
    result = {
        'source_run': str(RUN), 'seed_results': rows,
        'matched_successful_subset_mean_rl_reduction_pct': statistics.mean(a),
        'matched_successful_subset_sample_sd_pct': statistics.stdev(a),
        'checks': {'raw_hashes': True, 'all_5400_cells': True, 'paired_hierarchies': True,
                   'timing_accounting': True, 'prescribed_weights': True,
                   'balanced_execution_order': True, 'no_training_input_overlap': True},
        'timing_scope': 'Continuation excludes primary setup, includes recovery setup and solve. Inclusive includes controller computation. Ratios use only problems both compared policies finish in all repetitions.',
        'completion_scope': 'All 100 problems for every seed; completion after the common fallback.',
    }
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif', 'font.size': 8.5, 'axes.labelsize': 8.5,
        'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.3, 3.25), gridspec_kw={'width_ratios': [1, 1.3]})
    styles = [('reference', 'Fixed $w=1$', 'o', -.18, 'white'),
              ('periodic', 'Period-two', 's', 0, 'white'),
              ('rl', 'Frozen LSTDQ', 'D', .18, '#222222')]
    for method, label, marker, offset, face in styles:
        ax.scatter([100 - r[method + '_fallback_cases'] for r in rows],
                   [r['seed'] + offset for r in rows], s=28, marker=marker,
                   edgecolors='#222222', facecolors=face, linewidths=.9, label=label, zorder=3)
    ax.set(xlim=(-2, 104), ylim=(6.6, .4), yticks=range(1, 7),
           yticklabels=[f'Seed {s}' for s in range(1, 7)],
           xlabel='Converged without fallback (out of 100)')
    ax.set_title('(a) Primary-attempt convergence', loc='left')
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis='x', color='#e6e6e6', linewidth=.5)
    for field, marker, offset, label, face in [('solver', 'o', -.1, 'Solver time', 'white'),
                                            ('inclusive', 'D', .1, 'Including controller overhead', '#222222')]:
        xx = [r[f'rl_vs_periodic_{field}_reduction_pct'] for r in rows]
        bx.scatter(xx, [r['seed'] + offset for r in rows], marker=marker, s=28,
                   edgecolors='#222222', facecolors=face, linewidths=.9, label=label, zorder=3)
    bx.axvline(0, color='#777777', linewidth=.7)
    bx.set(ylim=(6.6, .4), yticks=range(1, 7),
           yticklabels=[f"Seed {r['seed']} (n={r['common_completed_periodic_rl']})" for r in rows],
           xlabel='LSTDQ time reduction vs period-two (%)')
    bx.set_title('(b) Problems both policies complete', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncol=3, frameon=False,
              fontsize=7.4, columnspacing=1, handletextpad=.4)
    bx.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2, frameon=False,
              fontsize=7.4, columnspacing=1, handletextpad=.4)
    fig.suptitle(r'Variable-flow $60^3$: six frozen models, 100 new problems each', fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.29, wspace=.55)
    fig.savefig(OUT / 'six_seed_comparison.png', dpi=220, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
