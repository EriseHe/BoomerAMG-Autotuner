"""Audit saved Advection 60 trajectories without executing or changing learners."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from development.tests.test_shared_setup_features import encoders

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.with_name('RLTune-BoomerAMG')
CURRENT = Path(json.loads((ROOT/'results/latest_shared_setup_features_run.json').read_text())['output'])
ORIGINAL = MAIN/'data/official/module1/batches/batch_01/advection_60'
OUTPUT = CURRENT/'trajectory_diagnosis'
BASELINE = '24a6b21cfe7607d64510b4a93d15006eb90d5681'
NUMERIC = ('strong_threshold', 'max_row_sum', 'trunc_factor', 'P_max_elmts', 'agg_num_levels')
CENTERS = np.array([.25, .9, 0., 4., 0.])
SCALES = np.array([.25, .1, .2, 4., 1.])


def read(path):
    return json.loads(path.read_text())


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {}
    assert np.all(np.isfinite(values))
    return dict(zip(('min', 'median', 'p95', 'p99', 'max'),
                    np.quantile(values, [0, .5, .95, .99, 1]).tolist()))


def load(stage, expected_hash):
    path = stage/'trajectories/bandit_lstdq.jsonl'
    digest = hashlib.sha256()
    rows = []
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            o = row['outcome']
            attempts = o['primary_attempts']
            # Retain attempt traces, not the duplicated top-level trace/metadata.
            rows.append(dict(index=row['problem_index'], mkw=row['mkw'], context=row['context'],
                attempts=attempts, setup=o['setup_runtime'], solve=o['native_solve_runtime'],
                native=o['runtime'], overhead=o['infer_runtime']+o['bandit_overhead_runtime'],
                fallback=o['fallback_used'], failed=o['unrecovered_failure'],
                fallback_setup=o['fallback_setup_runtime'], fallback_solve=o['fallback_solve_runtime']))
    assert digest.hexdigest() == expected_hash
    assert [r['index'] for r in rows] == list(range(1, 5001))
    for r in rows:
        np.testing.assert_allclose(r['native'], r['setup']+r['solve'], rtol=1e-9, atol=1e-9)
    return rows, digest.hexdigest()


def summarize(rows, encoder):
    attempts = [a for r in rows for a in r['attempts']]
    cycles = sum(a['cycles'] for a in attempts)
    first = [r['attempts'][0] for r in rows]
    retry = [a for r in rows for a in r['attempts'][1:]]
    result = dict(cases=len(rows), attempts=len(attempts), cycles=cycles,
        first_failures=sum(a['status']!='success' for a in first),
        final_failures=sum(r['failed'] for r in rows), fallbacks=sum(r['fallback'] for r in rows),
        **{k:sum(r[k] for r in rows) for k in ('setup','solve','native','overhead')},
        retry_setup=sum(a['setup_runtime'] for a in retry),
        retry_solve=sum(a['solve_runtime'] for a in retry),
        failed_attempt_setup=sum(a['setup_runtime'] for a in attempts if a['status']!='success'),
        failed_attempt_solve=sum(a['solve_runtime'] for a in attempts if a['status']!='success'))
    result['cycle_ms'] = 1000*result['solve']/cycles
    for param in (*NUMERIC, 'coarsen_type','interp_type','agg_interp_type'):
        result[param+'_counts'] = dict(Counter(str(a['params'][param]) for a in first))
    top = Counter(tuple(a['params'][p] for p in (*NUMERIC,'coarsen_type','interp_type','agg_interp_type')) for a in first)
    result['top_setups'] = [dict(params=dict(zip((*NUMERIC,'coarsen_type','interp_type','agg_interp_type'),p)), count=n)
                            for p,n in top.most_common(8)]
    result['unique_first_setups'] = len(top)
    numeric = np.array([[a['params'][p] for p in NUMERIC] for a in attempts])
    numeric = (numeric-CENTERS)/SCALES
    result['old_numeric_cap_attempts'] = dict(zip(NUMERIC,(np.abs(numeric)>2).sum(axis=0).tolist()))
    result['old_numeric_cap_any_attempts'] = int((np.abs(numeric)>2).any(axis=1).sum())
    full = np.array([encoder.encode_setup(a['params']) for a in attempts])
    result['shared_setup_max_absolute_feature'] = quantiles(np.abs(full).max(axis=1))
    result['shared_setup_norm'] = quantiles(np.linalg.norm(full,axis=1))
    states_g, states_d, times, time_ratios, actions, explored, uncertainties, qs, scores, td = ([] for _ in range(10))
    observed_steps=0
    for a in attempts:
        residuals = np.asarray(a['cycle_residuals'],dtype=float)
        cycle_times = np.asarray(a['cycle_times'],dtype=float)
        n=len(residuals)
        if not n:
            continue
        assert n==len(cycle_times)==len(a['cycle_actions'])==a['cycles']
        # A decision sees only preceding residuals; the final residual is terminal.
        # The native adapter normalizes the initial residual to exactly one.
        current_r = np.r_[1.,residuals[:-1]]
        previous_r = np.r_[1.,current_r[:-1]]
        assert np.all(current_r>0) and np.all(np.isfinite(current_r))
        states_g.extend(np.log(current_r/1e-6)/np.log(1e6))
        states_d.extend(np.log((current_r+1e-30)/(previous_r+1e-30))/5.)
        times.extend(np.r_[0.,cycle_times[:-1]])
        time_ratios.extend(np.r_[0.,cycle_times[:-1]/cycle_times[0]])
        actions.extend(a['cycle_actions'])
        explored.extend(a['cycle_explored'])
        uncertainties.extend(a['cycle_selected_uncertainties'])
        qs.extend(a['cycle_selected_q_values'])
        scores.extend(a['cycle_selection_scores'])
        td.extend(a['td_errors'])
        observed_steps+=n
    g,d,t,q=np.asarray(states_g),np.asarray(states_d),np.asarray(times),np.asarray(time_ratios)
    result['decision_states']=observed_steps
    result['old_dynamic_cap_events']=dict(residual_gap_upper=int((g>1.5).sum()),
        residual_gap_lower=int((g<0).sum()),residual_change=int((np.abs(d)>1).sum()),
        cycle_time=int((t>.01).sum()))
    result['residual_gap']=quantiles(g)
    result['old_scaled_residual_change']=quantiles(d)
    result['first_cycle_time_ratio']=quantiles(q)
    result['native_cycle_observation']=quantiles(t)
    result['selected_uncertainty']=quantiles(uncertainties)
    result['selected_mean']=quantiles(qs)
    result['absolute_td_error']=quantiles(np.abs(td))
    result['zero_score_fraction']=float(np.mean(np.asarray(scores)<=1e-14))
    result['negative_mean_fraction']=float(np.mean(np.asarray(qs)<0))
    result['random_action_fraction']=float(np.mean(explored))
    result['mean_weight']=float(np.mean(actions))
    result['weight_counts']=dict(Counter(str(w) for w in actions))
    result['greedy_weight_counts']=dict(Counter(str(w) for w,e in zip(actions,explored) if not e))
    return result


def paired(original,current):
    same=[(a,b) for a,b in zip(original,current) if a['attempts'][0]['params']==b['attempts'][0]['params']]
    result=dict(same_initial_setup_cases=len(same))
    for label,rows in [('original',[a for a,b in same]),('current',[b for a,b in same])]:
        result[label]=dict(first_cycles=sum(r['attempts'][0]['cycles'] for r in rows),
            first_setup=sum(r['attempts'][0]['setup_runtime'] for r in rows),
            first_solve=sum(r['attempts'][0]['solve_runtime'] for r in rows),
            first_failures=sum(r['attempts'][0]['status']!='success' for r in rows))
    result['largest_native_regressions']=[dict(index=a['index'],extra_native=b['native']-a['native'],
        original_params=a['attempts'][0]['params'],current_params=b['attempts'][0]['params'],
        original_cycles=sum(t['cycles'] for t in a['attempts']),current_cycles=sum(t['cycles'] for t in b['attempts']),
        original_native=a['native'],current_native=b['native'],original_attempts=len(a['attempts']),current_attempts=len(b['attempts']))
        for a,b in sorted(zip(original,current),key=lambda ab:ab[1]['native']-ab[0]['native'],reverse=True)[:5]]
    return result


def main():
    OUTPUT.mkdir(exist_ok=True)
    raw=read(ROOT/'data/raw_files.json')['files']
    _,encoder=encoders()
    result=dict(original=str(ORIGINAL),current=str(CURRENT),native_solves_executed=0,seeds=[],checks={})
    for seed in (1,2,3):
        old_stage=ORIGINAL/f'seed_{seed}'
        new_stage=CURRENT/f'seed_{seed}'
        path=old_stage/'trajectories/bandit_lstdq.jsonl'
        old,old_hash=load(old_stage,raw[str(path.relative_to(MAIN))]['sha256'])
        new,new_hash=load(new_stage,read(new_stage/'result.json')['audits']['bandit_lstdq']['trajectory_sha256'])
        assert all(a['mkw']==b['mkw'] and a['context']==b['context'] for a,b in zip(old,new))
        a=np.load(old_stage/'aot_candidate_schedules/expanded_no_endpoints/structured512/candidate_ids.npy',mmap_mode='r')
        b=np.load(new_stage/'aot_candidate_schedules/expanded_no_endpoints/structured512/candidate_ids.npy',mmap_mode='r')
        matches=np.all(a==b[:len(a)],axis=1)
        entry=dict(seed=seed,paired_inputs_identical=True,hashes=dict(original=old_hash,current=new_hash),
            candidate_schedule=dict(original_rows=len(a),current_rows=len(b),identical_rows=int(matches.sum()),
                first_different_row=int(np.flatnonzero(~matches)[0]) if not np.all(matches) else None),
            first_different_initial_setup=next((a['index'] for a,b in zip(old,new) if a['attempts'][0]['params']!=b['attempts'][0]['params']),None),
            windows={})
        for name,start,end in [('full',0,5000),*[(f'{i+1}_{i+1000}',i,i+1000) for i in range(0,5000,1000)]]:
            entry['windows'][name]=dict(original=summarize(old[start:end],encoder),
                current=summarize(new[start:end],encoder),paired=paired(old[start:end],new[start:end]))
        entry['controller_audit']=read(new_stage/'result.json')['audits']['bandit_lstdq']['controller']
        result['seeds'].append(entry)
        print('Audited seed',seed,flush=True)
    old_manifest=read(ROOT/'data/source_manifest.json')
    current_manifest=read(CURRENT/'source_manifest.json')
    result['source_changes']={k:dict(original=v,current=current_manifest.get(k)) for k,v in old_manifest.items() if v!=current_manifest.get(k)}
    result['checks']['native_libraries_unchanged']=read(ROOT/'data/module1/online/protocol.json')['native_sha256']==read(CURRENT/'protocol.json')['native_sha256']
    for name in ['solve/controllers/common/state_encoder.py','setup/space.py']:
        source=subprocess.check_output(['git','show',f'{BASELINE}:{name}'],cwd=ROOT)
        assert hashlib.sha256(source).hexdigest()==old_manifest[name]
        (OUTPUT/('original_'+Path(name).name)).write_bytes(source)
    result['checks']['original_source_verified']=True
    (OUTPUT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(OUTPUT/'diagnosis.json')


if __name__=='__main__':
    main()
