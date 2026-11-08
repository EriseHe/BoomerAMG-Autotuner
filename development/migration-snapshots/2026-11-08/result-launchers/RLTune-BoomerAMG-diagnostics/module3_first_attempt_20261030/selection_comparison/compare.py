from pathlib import Path
import json,statistics,math
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
OLD=BASE/'modules_2_3_20261027/study/module2/run'
NEW=ROOT/'diffusion/run'
SEEDS=(1,2,3)

def read_scan(root):
    selected=defaultdict(list);first={}
    for path in sorted((root/'raw/scan').glob('worker_*.jsonl')):
        for line in path.open():
            row=json.loads(line)
            if row['seed'] not in SEEDS:continue
            outcome=row['outcome'];key=(row['seed'],row['case_id'],row['policy'])
            native=float(outcome['primary_solve_runtime'])
            selected[key].append(native)
            if row['repeat']==0:
                first[key]=dict(cycles=outcome['primary_cycles'],status=outcome['primary_status'],residual=outcome.get('primary_residual_norm'))
    return {k:statistics.mean(v) for k,v in selected.items()},first

old_means,old_first=read_scan(OLD)
new_means,new_first=read_scan(NEW)
assert set(old_means)==set(new_means) and len(new_means)==12300
cycles=[k for k in old_first if old_first[k]['cycles']!=new_first[k]['cycles']]
statuses=[k for k in old_first if old_first[k]['status']!=new_first[k]['status']]
residuals=[k for k in old_first if old_first[k]['residual']!=new_first[k]['residual']]
old_choices=json.loads((OLD/'selections.json').read_text());new_choices=json.loads((NEW/'selections.json').read_text())
rows=[]
for seed in SEEDS:
    before=old_choices[str(seed)];after=new_choices[str(seed)]
    changed=[case for case in range(100) if before['per_instance'][str(case)]!=after['per_instance'][str(case)]]
    weights=sorted({key[2] for key in old_means if key[0]==seed})
    old_first_global=min(weights,key=lambda w:(sum(old_means[seed,c,w] for c in range(100)),w))
    old_first_per={str(c):min(weights,key=lambda w:(old_means[seed,c,w],w)) for c in range(100)}
    objective_only_changes=sum(before['per_instance'][str(c)]!=old_first_per[str(c)] for c in range(100))
    penalties=[];examples=[];same_selected_cycles=0;diffs=[]
    for case in changed:
        old_w=before['per_instance'][str(case)];new_w=after['per_instance'][str(case)]
        old_time=new_means[seed,case,old_w];new_time=new_means[seed,case,new_w]
        penalties.append(100*(old_time/new_time-1))
        same_selected_cycles+=new_first[seed,case,old_w]['cycles']==new_first[seed,case,new_w]['cycles']
        detail=dict(case_id=case,old_weight=old_w,new_weight=new_w,new_scan_old_weight_ms=old_time*1000,new_scan_selected_weight_ms=new_time*1000,old_weight_cycles=new_first[seed,case,old_w]['cycles'],new_weight_cycles=new_first[seed,case,new_w]['cycles'])
        diffs.append(detail)
        if len(examples)<5:examples.append(detail)
    rows.append(dict(seed=seed,global_old=before['global'],global_new=after['global'],per_problem_unchanged=100-len(changed),per_problem_changed=len(changed),old_data_first_attempt_global=old_first_global,old_data_first_attempt_objective_changed_choices=objective_only_changes,changed_cases_with_equal_selected_cycle_count=same_selected_cycles,changed_choice_median_new_scan_advantage_pct=statistics.median(penalties) if penalties else 0,changed_choice_max_new_scan_advantage_pct=max(penalties) if penalties else 0,examples=examples,all_changes=diffs))
output=dict(diffusion=rows,all_fixed_weights_checked=len(old_first),cycle_count_differences=cycles,primary_status_differences=statuses,primary_residual_differences=residuals,notes=dict(old_scan_repetitions='One repetition on 90 cases and three on 10 cases',new_scan_repetitions='Three repetitions on all 100 cases',old_selection='Success-constrained native continuation cost',new_selection='First-attempt native solve time, no success filtering'))
(ROOT/'selection_comparison/results.json').write_text(json.dumps(output,indent=2)+'\n')
print('Matched numerical primary solves:',len(old_first),'cycle differences',len(cycles),'status differences',len(statuses),'residual differences',len(residuals))
for r in rows:print({k:v for k,v in r.items() if k not in ('examples','all_changes')})
