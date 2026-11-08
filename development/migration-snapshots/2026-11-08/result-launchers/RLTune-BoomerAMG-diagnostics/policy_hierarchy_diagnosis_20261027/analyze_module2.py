"""Read-only diagnosis of completed matched-policy runs; no solver execution."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG')
sys.path.insert(0, str(REPO))
from experiments.paper_final.plot_policy import read_study, paired_gains
from experiments.paper_final.matched_policy.selection import trial_key

OUT = Path(__file__).resolve().parent
ROOTS = {
    'advection_60': Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG-diagnostics/results/module2_advection_60_20261027_133028/run'),
    'diffusion_60': Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG-diagnostics/results/modules_2_3_20261027/study/module2/run'),
}
METHODS = ('reference', 'fixed', 'oracle', 'periodic', 'rl')

def counts(values):
    return dict(Counter(str(v) for v in values).most_common())

def mean(values):
    return float(np.mean(values)) if len(values) else None

def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a,.9)), maximum=float(a.max()))

def analyze(root):
    summary, seeds, case_ids, costs, success, weights, sources = read_study(root)
    protocol = summary['protocol']
    base = json.loads((root/'jobs_base.json').read_text())
    jobs = json.loads((root/'jobs_test.json').read_text())
    selections = json.loads((root/'selections.json').read_text())
    raw = {}
    for path in sorted((root/'raw/test').glob('worker_*.jsonl')):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                key = trial_key(row)
                assert key not in raw
                raw[key] = row
    cases = {s: {r['case_id']: r for r in base if r['seed']==s} for s in seeds}
    for s in seeds:
        assert set(cases[s])==set(case_ids)
        for c in case_ids:
            assert cases[s][c]['input_id']==cases[seeds[0]][c]['input_id']
    result = {'sources':sources, 'problem':protocol['problem'], 'seeds':{}, 'paired_inputs_verified':True}
    gains = paired_gains(costs,success)
    paired = []
    for si,s in enumerate(seeds):
        sjobs = [j for j in jobs if j['seed']==s]
        params = [cases[s][c]['params'] for c in case_ids]
        seed = {
            'setup_counts':{k:counts([p[k] for p in params]) for k in params[0]},
            'unique_setup_configurations':len({json.dumps(p,sort_keys=True) for p in params}),
            'global_fixed':selections[str(s)]['global'],
            'per_input_fixed_counts':counts(list(selections[str(s)]['per_instance'].values())),
            'methods':{},
        }
        records = {}
        for mi,m in enumerate(METHODS):
            rows = [raw[trial_key({**j,'policy':j['method_policies'][m]})] for j in sjobs]
            assert len(rows)==100*protocol['repetitions_per_case']
            outcomes = [r['outcome'] for r in rows]
            actions, times, residual_ratios, pairs, growth_high, contraction_low = [],[],[],[],[],[]
            first, tail, switches = [], [], []
            for o in outcomes:
                aa = np.asarray(o['cycle_actions'],dtype=float)
                tt = np.asarray(o['cycle_times'],dtype=float)
                rr = np.asarray(o['cycle_residuals'],dtype=float)
                assert len(aa)==len(tt)==len(rr)
                ratios = rr / np.r_[1.,rr[:-1]]
                actions.extend(aa.tolist()); times.extend(tt.tolist()); residual_ratios.extend(ratios.tolist())
                if len(aa):
                    first.append(float(aa[0])); tail.append(float(aa[-1]))
                if len(aa)>1:
                    switches.extend((np.diff(aa)!=0).tolist())
                    for k in range(1,len(aa)):
                        if aa[k-1]>=2.5 and aa[k]<=1.5:
                            pairs.append(float(ratios[k-1]*ratios[k]))
                growth_high.extend(ratios[aa>=2.5].tolist())
                contraction_low.extend(ratios[aa<=1.5].tolist())
            cases_stats=[]
            for c in case_ids:
                cr=[r for r in rows if r['case_id']==c]
                cases_stats.append({
                    'case_id':c, 'native_sec':mean([r['native_continuation_sec'] for r in cr]),
                    'primary_cycles':mean([len(r['outcome']['cycle_actions']) for r in cr]),
                    'primary_sec':mean([sum(r['outcome']['cycle_times']) for r in cr]),
                    'recovery_sec':mean([r['outcome'].get('fallback_setup_runtime',0)+r['outcome'].get('fallback_solve_runtime',0) for r in cr]),
                    'fallbacks':sum(bool(r['outcome'].get('fallback_used')) for r in cr),
                    'repetition_cv':float(np.std([r['native_continuation_sec'] for r in cr],ddof=1)/np.mean([r['native_continuation_sec'] for r in cr])),
                })
            records[m]=cases_stats
            total = sum(r['native_continuation_sec'] for r in rows)/protocol['repetitions_per_case']
            np.testing.assert_allclose(total,costs[si,mi,:,0].sum(),rtol=1e-12)
            seed['methods'][m]={
                'native_sec':total,
                'native_total_sec':float(costs[si,mi,:,2].sum()),
                'reduction_pct':float(gains[si,mi]),
                'primary_cycles':describe([len(o['cycle_actions']) for o in outcomes]),
                'cycle_ms':mean(times)*1000,
                'recovery_sec':sum(x['recovery_sec'] for x in cases_stats),
                'recovered_cases':sum(x['fallbacks']>0 for x in cases_stats),
                'recovered_trials':sum(bool(o.get('fallback_used')) for o in outcomes),
                'failures':sum(not r['success'] for r in rows),
                'action_counts':counts([round(a,4) for a in actions]),
                'first_action_counts':counts(first), 'last_action_counts':counts(tail),
                'switch_fraction':mean(switches),
                'high_weight_fraction':mean(np.asarray(actions)>=2.5),
                'growth_cycle_fraction':mean(np.asarray(residual_ratios)>1),
                'high_weight_growth_fraction':mean(np.asarray(growth_high)>1),
                'median_high_low_pair_residual_factor':float(np.median(pairs)) if pairs else None,
                'high_low_pairs':len(pairs),
                'median_repetition_cv':float(np.median([x['repetition_cv'] for x in cases_stats])),
            }
        seed['rl_vs_periodic']={
            'reduction_pct':100*(1-seed['methods']['rl']['native_sec']/seed['methods']['periodic']['native_sec']),
            'faster_inputs':sum(a['native_sec']<b['native_sec'] for a,b in zip(records['rl'],records['periodic'])),
            'fewer_cycle_inputs':sum(a['primary_cycles']<b['primary_cycles'] for a,b in zip(records['rl'],records['periodic'])),
            'same_cycle_inputs':sum(a['primary_cycles']==b['primary_cycles'] for a,b in zip(records['rl'],records['periodic'])),
        }
        for ci,c in enumerate(case_ids):
            paired.append({'seed':s,'case_id':c,'params':cases[s][c]['params'],
                           'methods':{m:records[m][ci] for m in METHODS}})
        result['seeds'][s]=seed
    result['same_setup_across_seed_pairs']={f'{a}-{b}':sum(cases[a][c]['params']==cases[b][c]['params'] for c in case_ids)
                                           for a in seeds for b in seeds if a<b}
    return result,paired

if __name__=='__main__':
    for name,root in ROOTS.items():
        result,paired=analyze(root)
        (OUT/f'{name}_module2.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (OUT/f'{name}_paired_cases.json').write_text(json.dumps(paired,indent=2,allow_nan=False)+'\n')
        print(name, flush=True)
        for s,v in result['seeds'].items():
            print('seed',s,'fixed',v['global_fixed'],'aggressive_levels',v['setup_counts']['agg_num_levels'],flush=True)
            print('method | native sec | cycles | ms/cycle | recovered cases | reduction %',flush=True)
            for m,r in v['methods'].items():
                print(m,*(round(r[k],3) for k in ('native_sec',)),round(r['primary_cycles']['mean'],3),
                      round(r['cycle_ms'],3),r['recovered_cases'],round(r['reduction_pct'],2),flush=True)
            print('rl vs periodic',v['rl_vs_periodic'],flush=True)
