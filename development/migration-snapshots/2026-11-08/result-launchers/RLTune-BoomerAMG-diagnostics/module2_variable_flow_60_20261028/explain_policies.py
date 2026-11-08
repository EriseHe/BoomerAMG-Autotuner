"""Descriptive diagnostics of recorded matched-policy trials; no new solves."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import statistics as st
import math

BASE = Path(__file__).resolve().parent
RUNS = {
    'diffusion': BASE.parent / 'modules_2_3_20261027/study/module2/run',
    'variable_flow': BASE / 'run',
}

def describe(values):
    return {'n': len(values), 'mean': st.mean(values), 'median': st.median(values),
            'min': min(values), 'max': max(values)} if values else None

def trajectory(row):
    o = row['outcome']
    p = (o.get('primary_attempts') or [o])[0]
    cycles = p.get('cycles', o.get('primary_cycles', o.get('iterations', 0)))
    acts = p.get('cycle_actions') or o.get('cycle_action_values') or o.get('cycle_actions', [])
    residuals = p.get('cycle_residuals') or o.get('cycle_residuals', [])
    return cycles, acts, residuals

report = {}
for family, run in RUNS.items():
    selections = json.loads((run / 'selections.json').read_text())
    jobs = {(j['seed'], j['case_id']): j for j in json.loads((run / 'jobs_base.json').read_text())}
    data = {}
    for f in sorted((run / 'raw/test').glob('worker_*.jsonl')):
        for line in f.open():
            r = json.loads(line)
            if r['repeat'] == 0:
                data[r['seed'], r['case_id'], r['policy']] = r
    seeds = []
    allweights = Counter()
    pairwise = defaultdict(list)
    bins = {i: {'n': 0, 'periodic_first_failures': 0, 'rl_first_failures': 0,
                'rl_minus_oracle_cycles': [], 'periodic_cycles': [], 'rl_cycles': [], 'oracle_cycles': []}
            for i in range(3)}
    examples = []
    for seed in range(1, 7):
        metrics = defaultdict(list)
        weights = Counter()
        for case in range(100):
            sel = selections[str(seed)]
            names = {'reference': 'fixed_1', 'fixed': sel['global'],
                     'oracle': sel['per_instance'][str(case)], 'periodic': 'periodic', 'rl': 'rl_frozen'}
            rows = {m: data[seed, case, name] for m, name in names.items()}
            ts = {m: trajectory(row) for m, row in rows.items()}
            weight = float(names['oracle'].split('_')[1]); weights[weight] += 1
            for m, row in rows.items():
                metrics[m + '_cycles'].append(ts[m][0])
                metrics[m + '_first_failure'].append(row['outcome']['primary_status'] != 'success')
                metrics[m + '_final_failure'].append(not row['success'])
            rl, oracle = rows['rl'], rows['oracle']
            k, actions, residuals = ts['rl']
            assert len(actions) == k, (family, seed, case, len(actions), k)
            metrics['rl_initial_weight'].append(actions[0])
            metrics['rl_w1_first5_fraction'].append(sum(w == 1 for w in actions[:5]) / min(5,k))
            metrics['rl_zero_score_fraction'].append(sum(v == 0 for v in rl['outcome']['cycle_selection_scores']) / k)
            metrics['rl_forced_default_fraction'].append(sum(rl['outcome']['cycle_forced_default_actions']) / k)
            if all(rows[m]['outcome']['primary_status'] == 'success' for m in ['rl','oracle']):
                metrics['rl_minus_oracle_cycles_successful_primary'].append(k-ts['oracle'][0])
                metrics['rl_native_per_cycle_successful_primary'].append(rl['native_continuation_sec']/k)
                metrics['oracle_native_per_cycle_successful_primary'].append(oracle['native_continuation_sec']/ts['oracle'][0])
            metrics['rl_infer_fraction_of_inclusive'].append((rl['inclusive_continuation_sec']-rl['native_continuation_sec'])/rl['inclusive_continuation_sec'])
            if len(residuals) == k:
                previous = [1.] + residuals[:-1]
                for w, before, after in zip(actions,previous,residuals):
                    pairwise['rl_large_weight_growth' if w >= 2 else 'rl_small_weight_growth'].append(after > before)
            if family == 'variable_flow':
                eps=jobs[seed,case]['mkw']['k']; b=min(2,max(0,int(math.log10(eps)+4)))
                bucket=bins[b];bucket['n']+=1
                bucket['periodic_first_failures']+=rows['periodic']['outcome']['primary_status']!='success'
                bucket['rl_first_failures']+=rl['outcome']['primary_status']!='success'
                for m in ['periodic','rl','oracle']:bucket[m+'_cycles'].append(ts[m][0])
                if all(rows[m]['outcome']['primary_status'] == 'success' for m in ['rl','oracle']):bucket['rl_minus_oracle_cycles'].append(k-ts['oracle'][0])
                if seed==1 and case in [0,30,73]:
                    examples.append({'seed':seed,'case_id_0_based':case,'epsilon':eps,'oracle_weight':weight,
                                     'cycles':{m:t[0] for m,t in ts.items()},'rl_actions':actions,
                                     'rl_residuals':residuals})
        allweights.update(weights)
        seeds.append({'seed':seed,'oracle_weights':dict(sorted(weights.items())),
                      'metrics':{m:describe(v) for m,v in metrics.items()}})
    pooled = {}
    for m in seeds[0]['metrics']:
        ds=[s['metrics'][m] for s in seeds if s['metrics'][m]]
        pooled[m]=sum(d['mean']*d['n'] for d in ds)/sum(d['n'] for d in ds)
    report[family]={'seeds':seeds,'pooled_means':pooled,'all_oracle_weights':dict(sorted(allweights.items())),
                    'growth_by_weight':{k:describe(v) for k,v in pairwise.items()},
                    'epsilon_bins':{i:{k:describe(v) if isinstance(v,list) else v for k,v in d.items()} for i,d in bins.items()},
                    'examples':examples,'trajectory_repetition_0_based':0}
    print(family,'POOLED',json.dumps(pooled,indent=2))
    print('ORACLE_WEIGHTS',dict(sorted(allweights.items())))
    print('GROWTH',report[family]['growth_by_weight'])
    if family=='variable_flow':print('EPSILON_BINS',json.dumps(report[family]['epsilon_bins'],indent=2))
(BASE/'readout/policy_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
