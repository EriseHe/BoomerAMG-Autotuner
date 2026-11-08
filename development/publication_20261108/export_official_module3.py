"""Reproduce the official paper's preselected first test repetition, read-only sources.

Writes only the isolated publication clone's data/module3 subtree and work evidence.
The 3-repeat measurement run and original research repositories are never changed.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np

BASE = Path(__file__).resolve().parents[2]
CLONE = BASE / 'work/boomer-publication'
OUT = CLONE / 'data/module3'
SOURCE = Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG-diagnostics/results/module3_first_attempt_20261030')
PAPER = Path('/Users/erisehe/Documents/GitHub/SISC-Overleaf')
METHODS = ['reference', 'fixed', 'oracle', 'periodic', 'rl']
LABELS = ['Fixed w=1', 'Global best fixed', 'Per-problem best fixed', 'Periodic (2.85,1.10)', 'Frozen LSTDQ']
FIELDS = ['native_solve_sec', 'inclusive_solve_sec', 'native_total_sec', 'inclusive_total_sec']


def hashes(path):
    sha = hashlib.sha256()
    git = hashlib.sha1()
    size = path.stat().st_size
    git.update(b'blob ' + str(size).encode() + b'\0')
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
            git.update(block)
    return {'bytes': size, 'sha256': sha.hexdigest(), 'git_blob_sha1': git.hexdigest()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def copy(path, dest):
    if path.is_symlink():
        raise RuntimeError('Unexpected source symlink: ' + str(path))
    before = path.stat()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError('Source changed during copy: ' + str(path))
    assert hashes(path)['sha256'] == hashes(dest)['sha256']


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert OUT.name == 'module3' and OUT.parent.name == 'data' and CLONE in OUT.parents
    assert (CLONE / '.git').exists(), 'Expected isolated publication clone'
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    archive_plan = []
    families = {}
    seed_gains = {}
    figure7_weights = None
    figure7_cases = None
    for family in ['diffusion', 'variable_flow']:
        run = SOURCE / family / 'run'
        dest = OUT / family
        dest.mkdir()
        source_protocol = json.loads((run / 'protocol.json').read_text())
        assert source_protocol['timing_protocol'] == 'first_attempt_native_solve_v1'
        assert source_protocol['repetitions_per_case'] == source_protocol['scan_repetitions'] == 3
        assert not source_protocol['recovery_enabled'] and source_protocol['failed_attempt_times_included']
        jobs = json.loads((run / 'jobs_test.json').read_text())
        raw = {}
        raw_count = 0
        source_evidence = {}
        for phase in ['scan', 'test']:
            marker = json.loads((run / f'raw/{phase}/complete.json').read_text())
            for path in sorted((run / f'raw/{phase}').iterdir()):
                if not path.is_file() or path.suffix not in ('.jsonl', '.json'):
                    continue
                info = hashes(path)
                relative = path.relative_to(run).as_posix()
                if relative in marker.get('raw_sha256', {}):
                    assert marker['raw_sha256'][relative] == info['sha256']
                archive_plan.append({'source': str(path), 'target': f'data/official/module3/{family}/{relative}', **info})
                source_evidence[f'data/official/module3/{family}/{relative}'] = info
                if phase == 'test' and path.name.startswith('worker_') and path.suffix == '.jsonl':
                    with path.open() as stream:
                        for line in stream:
                            r = json.loads(line)
                            key = (r['seed'], r['source'], r['case_id'], r['repeat'], r['policy'])
                            assert key not in raw, 'Duplicate source measurement'
                            raw[key] = r
                            raw_count += 1
            if phase == 'test':
                assert raw_count == marker['trials']
        base_jobs = {(j['seed'], j['case_id']): j for j in jobs if j['repeat'] == 0}
        seeds = sorted({j['seed'] for j in jobs})
        cases = sorted({j['case_id'] for j in jobs})
        assert seeds == [1, 2, 3, 4, 5, 6] and len(cases) == 100 and len(base_jobs) == 600
        costs = np.empty((6, 5, 100, 4))
        success = np.empty((6, 5, 100), dtype=bool)
        weights = np.full((6, 5, 100, 50), np.nan)
        cycles = np.zeros((6, 5, 100), dtype=np.int64)
        selected_keys = []
        per_case = []
        for i, seed in enumerate(seeds):
            for c, case in enumerate(cases):
                job = base_jobs[(seed, case)]
                for m, method in enumerate(METHODS):
                    policy = job['method_policies'][method]
                    assert policy is not None
                    key = (seed, job['source'], case, 0, policy)
                    r = raw[key]
                    selected_keys.append({'seed': seed, 'source': job['source'], 'case_id': case,
                                          'method': method, 'policy': policy, 'repeat': 0})
                    costs[i, m, c] = [r[k] for k in FIELDS]
                    success[i, m, c] = r['success']
                    outcome = r['outcome']
                    primary = (outcome.get('primary_attempts') or [outcome])[0]
                    ncycle = int(primary.get('cycles', outcome.get('primary_cycles', outcome.get('iterations', 0))))
                    actions = primary.get('cycle_actions', outcome.get('cycle_actions', outcome.get('cycle_action_values', [])))
                    if not actions and ncycle:
                        actions = outcome.get('cycle_action_values', outcome.get('cycle_actions', []))
                    assert len(actions) == ncycle and 0 <= ncycle <= 50
                    np.testing.assert_allclose(costs[i, m, c, 0], outcome['primary_solve_runtime'], atol=1e-8)
                    weights[i, m, c, :ncycle] = actions
                    cycles[i, m, c] = ncycle
                    per_case.append({'seed': seed, 'case_id': case, 'method': method, 'policy': policy,
                                     'test_repeat': 0, 'first_attempt_success': bool(success[i, m, c]),
                                     'primary_cycles': ncycle, **dict(zip(FIELDS, costs[i, m, c].tolist()))})
        assert np.isfinite(costs).all() and (costs >= 0).all()
        totals = costs.sum(axis=2)
        gains = 100 * (1 - totals[:, :, 0] / totals[:, :1, 0])
        inclusive_gains = 100 * (1 - totals[:, :, 1] / totals[:, :1, 1])
        rl_compare = 100 * (1 - totals[:, 4, 0, None] / totals[:, :4, 0])
        seed_gains[family] = gains[:, 1:]
        order = np.stack([np.lexsort((np.array(cases), -cycles[i, 0])) for i in range(6)])
        with np.load(run / 'paper_figures/plot_data.npz', allow_pickle=False) as existing:
            np.testing.assert_equal(weights, existing['first_repetition_primary_weights'])
        if family == 'variable_flow':
            figure7_weights = weights[0, [0, 2, 3, 4]][:, order[0], :]
            figure7_cases = np.array(cases)[order[0]]
            with np.load(SOURCE / 'figures_final/figure7_displayed_data.npz', allow_pickle=False) as shown:
                np.testing.assert_equal(figure7_weights, shown['weights'])
                np.testing.assert_equal(figure7_cases, shown['case_ids'])
            copy(SOURCE / 'figures_final/figure7_displayed_data.npz', dest / 'figure7_displayed_data.npz')
        per_seed = []
        for i, seed in enumerate(seeds):
            for m, method in enumerate(METHODS):
                per_seed.append({'seed': seed, 'method': method, 'label': LABELS[m], 'cases': 100,
                                 'test_repeat': 0, 'failed_cases': int((~success[i, m]).sum()),
                                 **dict(zip(FIELDS, totals[i, m].tolist())),
                                 'native_solve_reduction_pct_vs_reference': float(gains[i, m]),
                                 'inclusive_solve_reduction_pct_vs_reference': float(inclusive_gains[i, m])})
        aggregate = [{'method': method, 'label': LABELS[m], 'seeds': 6, 'cases': 600,
                      'test_repeat': 0, 'failed_cases': int((~success[:, m]).sum()),
                      'failure_rate_pct': float((~success[:, m]).mean() * 100),
                      'native_solve_reduction_pct_mean': float(gains[:, m].mean()),
                      'native_solve_reduction_pct_sample_sd': float(gains[:, m].std(ddof=1))}
                     for m, method in enumerate(METHODS)]
        comparisons = [{'reference': METHODS[m], 'reference_label': LABELS[m],
                        'mean_pct': float(rl_compare[:, m].mean()),
                        'sample_sd_pct': float(rl_compare[:, m].std(ddof=1)),
                        'reference_failure_rate_pct': aggregate[m]['failure_rate_pct'],
                        'displayed_time_reduction': f'{rl_compare[:, m].mean():.1f} ± {rl_compare[:, m].std(ddof=1):.1f}'}
                       for m in range(4)]
        official_protocol = {
            'module': 3, 'format': 'official-module3-first-test-repetition-v1', 'family': family,
            'metric': 'First-attempt native solve time; setup, learner computation, recovery and fallback excluded.',
            'reported_test_repetition_index_zero_based': 0, 'reported_test_repetitions': 1,
            'source_measurement_test_repetitions': 3, 'scan_repetitions_per_problem_weight': 3,
            'selection_basis': 'Use frozen selections.json chosen by the original three-repetition scan. No selection is recomputed from test times.',
            'case_aggregation': 'For each seed/policy, sum native solve times over all 100 paired problems using test repeat 0; compute 100*(1-policy_sum/reference_sum).',
            'seed_aggregation': 'Unweighted arithmetic mean and sample standard deviation (ddof=1) over six seed-level reductions.',
            'failed_attempt_times_included': True, 'failure_population': '600 first attempts per policy/family at test repeat0.',
            'repeat_policy': 'Repeat 0 is the first prescribed test repetition. Repeats 1 and 2 remain in the raw-source archive as additional measurements and are not averaged into official paper numbers.',
            'source_protocol_file': 'source_protocol.json', 'source_summary_file': 'source_summary.json',
            'raw_source_root': f'data/official/module3/{family}', 'methods': METHODS, 'labels': LABELS,
            'seeds': seeds, 'case_ids': cases, 'cost_fields': FIELDS,
            'weights': source_protocol['weights'], 'periodic_pair': [2.85, 1.10],
            'recovery_enabled': False, 'max_cycles': source_protocol['max_cycles'],
            'tolerance': source_protocol['tolerance'], 'native_checkpoints': 'checkpoints/seed_{1..6}/',
        }
        write_json(dest / 'protocol.json', official_protocol)
        summary = {'module': 3, 'family': family, 'official_protocol': 'protocol.json',
                   'per_seed': per_seed, 'aggregate': aggregate, 'frozen_lstdq_comparisons': comparisons,
                   'source_measurement_rows': raw_count, 'reported_method_case_records': 3000,
                   'all_failed_first_attempts_retained': True}
        write_json(dest / 'summary.json', summary)
        write_csv(dest / 'per_case.csv', per_case)
        write_csv(dest / 'per_seed.csv', per_seed)
        np.savez_compressed(dest / 'arrays.npz', seeds=np.array(seeds), methods=np.array(METHODS),
                            case_ids=np.array(cases), cost_fields=np.array(FIELDS), costs_sec=costs,
                            reported_test_repetition=np.array(0), success=success,
                            primary_cycles=cycles, first_repetition_primary_weights=weights,
                            case_order_0_based=order, native_reductions_pct=gains,
                            inclusive_reductions_pct=inclusive_gains)
        for source_name, dest_name in [('protocol.json', 'source_protocol.json'),
                                       ('analysis/summary.json', 'source_summary.json'),
                                       ('configuration.json', 'configuration.json'), ('inputs.json', 'inputs.json'),
                                       ('selections.json', 'selections.json'), ('jobs_test.json', 'jobs_test.json'),
                                       ('jobs_scan.json', 'jobs_scan.json'), ('source_manifest.json', 'source_manifest.json'),
                                       ('environment.json', 'environment.json'), ('setup_freeze_audit.json', 'setup_freeze_audit.json')]:
            copy(run / source_name, dest / dest_name)
        for path in sorted((run / 'checkpoints').rglob('*')):
            if path.is_file():
                copy(path, dest / path.relative_to(run))
        source_hashes = {name: hashes(run / name) for name in ['protocol.json', 'analysis/summary.json',
                         'configuration.json', 'inputs.json', 'selections.json', 'jobs_test.json', 'jobs_scan.json']}
        write_json(dest / 'provenance.json', {'authoritative_measurement_run': str(run),
                   'original_run_files': source_hashes, 'raw_source_files': source_evidence,
                   'selected_test_measurement_keys': selected_keys,
                   'source_check': 'Raw test rows read independently; unique trial keys, expected trial count, immutable source marker SHA256s and repeat0 mapping checked.',
                   'no_source_measurements_modified': True})
        with np.load(dest / 'arrays.npz', allow_pickle=False) as z:
            np.testing.assert_equal(z['costs_sec'], costs)
            np.testing.assert_equal(z['success'], success)
        families[family] = {'summary': summary, 'protocol': official_protocol,
                            'paper_figure6_annotations': [f'{gains[:, m].mean():.1f} ± {gains[:, m].std(ddof=1):.1f}' for m in range(1, 5)]}

    # Compare actual rendered SVG marker positions/accessible labels, not stale SVG comments.
    svg = ET.parse(PAPER / 'figures/matched_policy_savings.svg').getroot()
    ns = '{http://www.w3.org/2000/svg}'
    by_id = {e.get('id'): e for e in svg.iter() if e.get('id')}
    ticks = []
    for e in svg.iter():
        if e.get('id', '').startswith('xtick_'):
            labels = [a.get('aria-label') for a in e.iter() if a.get('aria-label')]
            locations = [float(a.get('x')) for a in e.iter(ns + 'use') if a.get('x')]
            if len(labels) == 1 and locations:
                try:
                    ticks.append((float(labels[0].replace('−', '-')), locations[0]))
                except ValueError:
                    pass
    slope, intercept = np.polyfit([a for a, b in ticks], [b for a, b in ticks], 1)
    labels = [e.get('aria-label') for e in svg.iter()
              if re.fullmatch(r'-?\d+\.\d ± \d+\.\d', e.get('aria-label', ''))]
    expected_labels = sum([families[f]['paper_figure6_annotations'] for f in ['diffusion', 'variable_flow']], [])
    assert labels == expected_labels
    max_error = 0.0
    for family_index, family in enumerate(['diffusion', 'variable_flow']):
        for method_index in range(4):
            n = family_index * 4 + method_index + 1
            marker_x = np.array([float(e.get('x')) for e in by_id[f'PathCollection_{n}'].iter(ns + 'use') if e.get('x')])
            observed = (marker_x - intercept) / slope
            expected = seed_gains[family][:, method_index]
            np.testing.assert_allclose(observed, expected, atol=3e-7, rtol=0)
            max_error = max(max_error, float(np.max(np.abs(observed - expected))))
    expected_table = {'diffusion': ['39.6 ± 1.5', '8.5 ± 2.1', '5.9 ± 2.0', '-2.8 ± 3.8'],
                      'variable_flow': ['23.6 ± 5.8', '0.4 ± 6.1', '-10.6 ± 7.0', '25.8 ± 6.0']}
    for family in families:
        assert [r['displayed_time_reduction'] for r in families[family]['summary']['frozen_lstdq_comparisons']] == expected_table[family]
    expected_failures = {'diffusion': [3, 1, 0, 2, 1], 'variable_flow': [31, 6, 1, 258, 2]}
    for family in families:
        assert [r['failed_cases'] for r in families[family]['summary']['aggregate']] == expected_failures[family]
    root_protocol = {'module': 3, 'format': 'official-module3-first-test-repetition-v1',
                     'families': ['diffusion', 'variable_flow'], 'official_test_repeat_zero_based': 0,
                     'cases_per_seed_per_policy': 100, 'seeds_per_family': 6, 'policies': METHODS,
                     'metric': 'First-attempt native solve time, retaining every failed attempt.',
                     'source_test_repetitions': 3, 'official_reported_test_repetitions': 1,
                     'selection_scan_repetitions': 3, 'archived_additional_test_repetitions': [1, 2],
                     'basis': 'Exact current official paper Figure6 marker coordinates/annotations, Figure7 trajectories, and table/text numeric values.',
                     'legacy_continuation_and_three_repeat_mean_not_the_official_paper_basis': True,
                     'family_protocols': {f: f'{f}/protocol.json' for f in families}}
    validation = {'passed': True, 'raw_test_rows_independently_read': {f: families[f]['summary']['source_measurement_rows'] for f in families},
                  'figure6_all_48_seed_points_match_repeat0': True, 'figure6_max_marker_error_percentage_points': max_error,
                  'figure6_accessible_annotations_match': labels, 'paper_table_time_reductions_match': expected_table,
                  'paper_first_attempt_failure_counts_match': expected_failures,
                  'figure7_displayed_weights_and_case_order_exact_match': True,
                  'paper_and_source_files_unchanged': True,
                  'original_source_repetition_count': 3, 'official_repetition_zero_based': 0,
                  'official_paper_files_sha256': {name: hashes(PAPER / name)['sha256'] for name in
                      ['sections/06_experiments.tex', 'tables/matched_relaxation_rows.tex',
                       'figures/matched_policy_savings.svg', 'figures/matched_policy_savings.png',
                       'figures/matched_policy_weights.svg', 'figures/matched_policy_weights.png']}}
    write_json(OUT / 'protocol.json', root_protocol)
    write_json(OUT / 'summary.json', {'module': 3, 'official_protocol': 'protocol.json',
               'families': {f: families[f]['summary'] for f in families}})
    write_json(OUT / 'validation.json', validation)
    write_json(OUT / 'raw_archive_manifest.json', {'files': {x['target']: {k: x[k] for k in ['bytes', 'sha256']} for x in archive_plan}})
    generated = [{'path': p.relative_to(CLONE).as_posix(), **hashes(p)}
                 for p in sorted(OUT.rglob('*')) if p.is_file()]
    plan = {'complete': True, 'source_originals_unchanged': True, 'publication_subtree': str(OUT),
            'exact_official_basis_resolved': True, 'basis': root_protocol,
            'validation': validation, 'families': {f: families[f]['paper_figure6_annotations'] for f in families},
            'generated_file_count': len(generated), 'generated_bytes': sum(x['bytes'] for x in generated),
            'generated_files': generated, 'raw_archive_entries': archive_plan,
            'exporter': str(Path(__file__).resolve()), 'not_committed_or_pushed': True}
    write_json(BASE / 'work/module3-official-publication-plan.json', plan)
    write_json(BASE / 'work/paper-data-audit/module3-raw-archive-plan.json', {'entries': archive_plan})
    print(json.dumps({k: v for k, v in plan.items() if k not in ('generated_files', 'raw_archive_entries', 'basis')}, indent=2))


if __name__ == '__main__':
    main()
