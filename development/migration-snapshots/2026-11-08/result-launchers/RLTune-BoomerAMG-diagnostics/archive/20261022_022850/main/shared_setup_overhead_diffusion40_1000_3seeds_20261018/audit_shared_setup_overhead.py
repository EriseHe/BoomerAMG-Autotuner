import json
import math
from pathlib import Path
import numpy as np

root=Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/shared_setup_overhead_diffusion40_1000_3seeds_20261018')
methods=('default_setup_default_solve','default_setup_lstdq','bandit_default','bandit_lstdq')
rl_methods=(methods[1],methods[3])
audit={'passed':True,'seeds':{},'cost_rule':'native cycle + current encoding + current selection + preceding retained update time','first_update_charge_sec':0.0}
metrics={method:[] for method in methods}
for seed in (1,2,3):
    stage=root/f'seed_{seed}'
    result=json.loads((stage/'result.json').read_text())
    assert result['complete'] and result['passed'] and result['cases']==1000
    checks={}
    records={}
    for method in methods:
        rows=[json.loads(line) for line in (stage/'trajectories'/f'{method}.jsonl').read_text().splitlines()]
        records[method]=rows
        assert len(rows)==1000
        metrics[method].append(result['methods'][method])
        if method not in rl_methods:
            assert all(not row['rl_active'] for row in rows)
            continue
        last=0.0
        retained=0
        executed=0
        rejected_inputs=0
        for row in rows:
            initial_last=last
            case_retained=0
            for attempt in row['outcome']['primary_attempts']:
                prior=last
                times=attempt['cycle_times']
                names=('cycle_assigned_costs','cycle_feature_times','cycle_decision_times','cycle_update_time_estimates','cycle_update_times')
                assert all(len(attempt[n])==len(times) for n in names)
                for native,cost,features,selection,estimate,update in zip(times,*(attempt[n] for n in names)):
                    assert math.isclose(estimate,last,rel_tol=1e-12,abs_tol=1e-15),(seed,method,row['problem_index'],estimate,last)
                    assert math.isclose(cost,native+features+selection+estimate,rel_tol=1e-12,abs_tol=1e-15)
                    assert all(math.isfinite(v) and v>=0 for v in (native,cost,features,selection,estimate,update))
                    last=update
                    executed+=1
                if attempt['rl_episode_recorded']:
                    case_retained+=len(times)
                else:
                    last=prior
            if row['outcome']['controller_update_committed']:
                retained+=case_retained
            else:
                last=initial_last
                rejected_inputs+=1
        summary=result['audits'][method]['controller']
        assert summary['steps']==retained==summary['update_runtime_count']
        assert summary['estimated_update_runtime_sec']==last
        with np.load(stage/'checkpoints'/f'{method}_final.npz',allow_pickle=False) as saved:
            assert int(saved['update_runtime_count'])==retained
            assert float(saved['last_update_runtime_sec'])==last
            assert int(saved['checkpoint_version'])==3
        checks[method]={'executed_cycles':executed,'retained_cycles':retained,'rolled_back_inputs':rejected_inputs,'final_carried_update_sec':last}
    reference=records[methods[0]]
    for method,rows in records.items():
        assert all((r['mkw'],r['context'])==(b['mkw'],b['context']) for r,b in zip(rows,reference))
    assert all(a['params']==b['params'] for a,b in zip(reference,records[methods[1]]))
    checks['inputs_per_method']=1000
    checks['paired_inputs_and_fixed_default_setup']=True
    audit['seeds'][str(seed)]=checks
baseline=metrics[methods[0]]
summary=[]
for method,values in metrics.items():
    reductions=[100*(1-v['total_sec']/d['total_sec']) for v,d in zip(values,baseline)]
    summary.append({'method':method,'inputs_per_seed':1000,'seeds':3,
        'mean_total_sec':float(np.mean([v['total_sec'] for v in values])),
        'mean_native_sec':float(np.mean([v['native_sec'] for v in values])),
        'total_reduction_pct_per_seed':reductions,
        'mean_paired_total_reduction_pct':float(np.mean(reductions)),
        'total_reduction_pct_sample_sd':float(np.std(reductions,ddof=1)),
        'final_failures':sum(v['failures'] for v in values),
        'first_attempt_failures':sum(v['first_failures'] for v in values),
        'fallbacks':sum(v['fallbacks'] for v in values),
        'evaluated_inputs':3000})
(root/'shared_encoding_cost_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(root/'total_time_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'audit':audit,'summary':summary},indent=2))
