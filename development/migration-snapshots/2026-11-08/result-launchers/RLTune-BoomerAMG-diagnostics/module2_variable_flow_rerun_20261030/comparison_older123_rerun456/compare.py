from pathlib import Path
import json, csv, math, hashlib, statistics
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent
OLD1 = RESULTS/'modules_2_3_20261027/study/module3/variable_flow_60_batch1'
OLD2 = RESULTS/'modules_2_3_20261027/study/module3/variable_flow_60_batch2'
NEW = ROOT/'run'
OUT = Path(__file__).resolve().parent
METHODS = ['default_setup_default_solve','default_setup_lstdq','bandit_default','bandit_lstdq']
LABELS = ['Default','LSTDQ','LinUCB','LinUCB–LSTDQ']

def summaries(folder):
    with (folder/'summary.csv').open() as f:
        return {(int(r['seed']),r['method']):r for r in csv.DictReader(f) if r['window']=='full'}

old1, old2, new = map(summaries, (OLD1,OLD2,NEW))
checks=[]; rows=[]
for folder in (OLD1, NEW):
    assert json.loads((folder/'status.json').read_text())['status']=='complete'
    assert json.loads((folder/'completion_audit.json').read_text())['passed']
a=json.loads((OLD1/'source_manifest.json').read_text())
b=json.loads((NEW/'source_manifest.json').read_text())
assert set(a)==set(b)
changed=[n for n in a if a[n]!=b[n]]
assert sorted(changed)==['experiments/paper_final/plot_online.py','experiments/paper_final/run_online.py']
assert json.loads((OLD1/'protocol.json').read_text())['native_sha256']==json.loads((NEW/'protocol.json').read_text())['native_sha256']
for seed in range(1,7):
    folder=OLD1 if seed<=3 else NEW
    source=old1 if seed<=3 else new
    values={}
    for method,label in zip(METHODS,LABELS):
        path=folder/f'seed_{seed}/trajectories/{method}.jsonl'
        h=hashlib.sha256(); ids=set(); inputs=[]
        native=[]; inclusive=[]; failed=0; first_failed=0
        with path.open('rb') as handle:
            for line in handle:
                h.update(line);r=json.loads(line);o=r['outcome'];idx=r['problem_index']
                assert idx not in ids;ids.add(idx)
                assert r['method']==method
                inputs.append((idx,r['mkw']))
                native.append(o['runtime']);inclusive.append(o['end_to_end_runtime'])
                failed+=bool(o['failed'])
                first_failed+=o.get('first_primary_status',o.get('primary_status'))!='success'
        assert ids==set(range(1,5001))
        native=math.fsum(native);inclusive=math.fsum(inclusive)
        s=source[seed,method]
        assert math.isclose(native,float(s['native_sec']),rel_tol=1e-10)
        assert math.isclose(inclusive,float(s['total_sec']),rel_tol=1e-10)
        assert failed==int(s['failures'])
        assert first_failed==int(s['first_failures'])
        if method==METHODS[0]: paired_inputs=inputs
        else:assert inputs==paired_inputs
        row=dict(seed=seed,source='older' if seed<=3 else 'rerun',method=method,label=label,cases=len(ids),native_sec=native,total_sec=inclusive,final_failures=failed,first_attempt_failures=first_failed,native_ms_per_problem=native/5,total_ms_per_problem=inclusive/5,source_directory=str(folder))
        values[method]=row
        checks.append(dict(seed=seed,method=method,records=len(ids),sha256=h.hexdigest(),summary_matches=True,paired_input_order_matches=True))
    for method in METHODS:
        row=values[method]
        row['native_reduction_pct']=100*(1-row['native_sec']/values[METHODS[0]]['native_sec'])
        row['total_reduction_pct']=100*(1-row['total_sec']/values[METHODS[0]]['total_sec'])
        s=source[seed,method]
        assert math.isclose(row['total_reduction_pct'],float(s['inclusive_reduction_pct']),abs_tol=1e-8)
        rows.append(row)
    print('Verified seed',seed,flush=True)
with (OUT/'six_seed_results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
aggregate=[]
for method,label in zip(METHODS,LABELS):
    group=[r for r in rows if r['method']==method]
    aggregate.append(dict(method=method,label=label,total_reduction_mean_pct=statistics.mean(r['total_reduction_pct'] for r in group),total_reduction_sample_sd_pct=statistics.stdev(r['total_reduction_pct'] for r in group),native_reduction_mean_pct=statistics.mean(r['native_reduction_pct'] for r in group),native_reduction_sample_sd_pct=statistics.stdev(r['native_reduction_pct'] for r in group),final_failures=sum(r['final_failures'] for r in group),first_attempt_failures=sum(r['first_attempt_failures'] for r in group),cases=sum(r['cases'] for r in group)))
deltas=[]
for seed in (4,5,6):
    for method,label in zip(METHODS,LABELS):
        older=old2[seed,method];current=next(r for r in rows if r['seed']==seed and r['method']==method)
        deltas.append(dict(seed=seed,label=label,old_total_reduction_pct=float(older['inclusive_reduction_pct']),new_total_reduction_pct=current['total_reduction_pct'],change_percentage_points=current['total_reduction_pct']-float(older['inclusive_reduction_pct']),old_final_failures=int(older['failures']),new_final_failures=current['final_failures']))
output=dict(rows=rows,aggregate=aggregate,rerun_vs_original=deltas,validation=dict(records_checked=120000,all_summaries_match_raw_trajectories=True,all_problem_ids_complete_and_unique=True,paired_input_order_matches_across_branches=True,source_code_differences_only_report_module_numbers=changed,native_build_matches=True,trajectory_checks=checks),definitions=dict(total='All setup and solve attempts plus measured learner/controller overhead; excludes matrix assembly and file I/O',native='All native setup and solve attempts, including recovery/fallback; excludes learner/controller overhead',reduction='100*(1-method_summed_time/paired_default_summed_time), computed separately per seed',aggregate='Unweighted mean and sample standard deviation of six seed-level reductions; final failure counts are summed',final_failure='Problem remains unsolved after all permitted attempts',population='5000 online problems per seed; older seeds 1–3 and rerun seeds 4–6; no success filtering'))
(OUT/'results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(aggregate=aggregate,rerun_vs_original=deltas),indent=2))
