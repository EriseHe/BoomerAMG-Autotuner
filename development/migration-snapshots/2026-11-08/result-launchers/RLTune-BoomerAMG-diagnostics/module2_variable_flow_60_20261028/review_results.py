"""Recheck official evidence and compare policies on a shared successful set."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'source'))
from experiments.runtime import configure_single_thread
configure_single_thread()
from experiments.paper_final.matched_policy.workflow import verify, phase_rows

run = BASE / 'run'
verify(run)
for phase in ('scan', 'test'):
    marker = json.loads((run / f'raw/{phase}/complete.json').read_text())
    for name, digest in marker['raw_sha256'].items():
        h = hashlib.sha256()
        with (run / name).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        assert h.hexdigest() == digest

raw = phase_rows(run, 'test')  # official coverage, schedule, residual, frozen-model audits
lookup = collections.defaultdict(list)
hierarchies = collections.defaultdict(set)
for row in raw:
    lookup[row['seed'], row['case_id'], row['policy']].append(row)
    hierarchies[row['seed'], row['case_id']].add((row['input_id'], row['hierarchy_id']))
    o = row['outcome']
    assert abs(row['native_continuation_sec'] - (o['solve_runtime'] + o.get('fallback_setup_runtime', 0))) < 1e-8
    assert abs(row['inclusive_continuation_sec'] - row['native_continuation_sec'] - o['infer_runtime']) < 1e-8
assert all(len(v) == 1 for v in hierarchies.values())
cases = list(csv.DictReader((run / 'analysis/per_case.csv').open()))
methods = ['reference', 'fixed', 'oracle', 'periodic', 'rl']
fields = ['native_continuation_sec', 'inclusive_continuation_sec']
by_case = {}
for c in cases:
    seed, case = int(c['seed']), int(c['case_id'])
    rows = lookup[seed, case, c['policy']]
    assert {r['repeat'] for r in rows} == {0, 1, 2} and len(rows) == 3
    assert len({(r['success'], r['outcome']['primary_status']) for r in rows}) == 1
    c['primary_failed'] = any(r['outcome']['primary_status'] != 'success' for r in rows)
    c['success'] = c['success'] == 'True'
    for f in fields:
        c[f] = float(c[f])
        assert abs(c[f] - sum(r[f] for r in rows)/3) < 1e-9
    by_case[seed, case, c['method']] = c

selections = json.loads((run / 'selections.json').read_text())
result = {'audit': {'official_verification': True, 'raw_hashes_verified': True,
                    'matched_input_and_hierarchy': True, 'stable_outcomes_across_repetitions': True,
                    'timing_accounting_verified': True}, 'seeds': []}
for seed in range(1, 7):
    common = [i for i in range(100) if all(by_case[seed, i, m]['success'] for m in methods)]
    per = [float(v.split('_')[1]) for v in selections[str(seed)]['per_instance'].values()]
    entry = {'seed': seed, 'global_fixed_weight': float(selections[str(seed)]['global'].split('_')[1]),
             'per_instance_weight_range': [min(per), max(per)], 'methods': {},
             'common_success_count': len(common), 'common_success_case_ids_0_based': common}
    for m in methods:
        group = [by_case[seed, i, m] for i in range(100)]
        totals = {f: sum(c[f] for c in group) for f in fields}
        entry['methods'][m] = {'first_attempt_failures': sum(c['primary_failed'] for c in group),
                              'final_failures': sum(not c['success'] for c in group),
                              'full_set_cost_sec': totals,
                              'common_success_reduction_pct': {
                                  f: 100*(1-sum(by_case[seed,i,m][f] for i in common)/
                                          sum(by_case[seed,i,'reference'][f] for i in common))
                                  for f in fields}}
    result['seeds'].append(entry)
out = BASE / 'readout'
out.mkdir(exist_ok=True)
(out / 'verified_results.json').write_text(json.dumps(result, indent=2)+'\n')
for s in result['seeds']:
    print('SEED',s['seed'],'global',s['global_fixed_weight'],'oracle range',s['per_instance_weight_range'],'common n',s['common_success_count'])
    for m,d in s['methods'].items():
        print(m, 'fail', d['first_attempt_failures'],d['final_failures'],
              'native/inclusive sec',[round(v,3) for v in d['full_set_cost_sec'].values()],
              'common reductions',[round(v,2) for v in d['common_success_reduction_pct'].values()])
print('AUDIT',result['audit'])
