"""Compare saved official and current advection records; execute no solves."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.with_name('RLTune-BoomerAMG')
CURRENT = ROOT/'results/module1_remaining_20261020_015153/runs'
OUTPUT = ROOT/'results/advection_regression_diagnosis_20261020'
FIELDS = ('setup_runtime', 'native_solve_runtime', 'native_runtime',
          'infer_runtime', 'bandit_overhead_runtime', 'end_to_end_runtime')


def read(path):
    return json.loads(path.read_text())


def extract(path):
    digest = hashlib.sha256()
    records = []
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            o = row['outcome']
            attempts = o['primary_attempts']
            first = attempts[0]
            records.append(dict(
                index=row['problem_index'], mkw=row['mkw'], params=first['params'],
                arm=first['arm_index'], cycles=sum(a['cycles'] for a in attempts),
                first_failed=first['status'] != 'success',
                attempts=len(attempts), fallback=o['fallback_used'],
                primary_solve=first['solve_runtime'], primary_setup=first['setup_runtime'],
                retry_solve=sum(a['solve_runtime'] for a in attempts[1:]),
                retry_setup=sum(a['setup_runtime'] for a in attempts[1:]),
                first_actions=first['cycle_actions'],
                **{field:o[field] for field in FIELDS}))
    return records, digest.hexdigest()


def summarize(rows):
    out = {field:sum(r[field] for r in rows) for field in
           (*FIELDS, 'cycles', 'primary_solve', 'primary_setup', 'retry_solve', 'retry_setup')}
    out.update(first_failures=sum(r['first_failed'] for r in rows),
               fallbacks=sum(r['fallback'] for r in rows),
               attempts=sum(r['attempts'] for r in rows),
               solve_seconds_per_cycle=out['native_solve_runtime']/out['cycles'],
               numeric_descriptor_exceeds_old_cap=sum(
                   abs((r['params']['max_row_sum']-.9)/.1)>2 or
                   abs((r['params']['strong_threshold']-.25)/.25)>2 or
                   abs(r['params']['trunc_factor']/.2)>2 or
                   abs((r['params']['P_max_elmts']-4)/4)>2 for r in rows))
    for parameter in ('coarsen_type','interp_type','agg_num_levels','max_row_sum','P_max_elmts'):
        out[parameter+'_counts'] = dict(Counter(str(r['params'][parameter]) for r in rows))
    return out


def main():
    OUTPUT.mkdir(exist_ok=True)
    result = dict(native_solves_executed=0, comparisons=[], trajectories=[], checks={})
    raw_manifest = read(ROOT/'data/raw_files.json')['files']
    for grid, seeds in ((60,range(1,4)),(80,range(1,7))):
        old = np.load(ROOT/f'data/module1/online/arrays/advection_{grid}.npz')['timings_sec']
        old_frozen = np.load(ROOT/f'data/module1/frozen/arrays/advection_{grid}.npz')['timings_sec']
        batches = [1] if grid==60 else [1,2]
        current = np.concatenate([np.load(CURRENT/f'advection_{grid}_batch{batch}/paper_figures/plot_data.npz')['timings_sec'] for batch in batches])
        for window, a, b in [('learning',old[:len(seeds)],current[:,:,:5000]),
                             ('frozen',old_frozen[:len(seeds)],current[:,:,5000:])]:
            a=a.sum(axis=2); b=b.sum(axis=2)
            for method,index in [('default',0),('lstdq',1),('linucb',2),('joint',3)]:
                normalized_old=(100*a[:,index]/a[:,0,5,None]).mean(axis=0)
                normalized_new=(100*b[:,index]/b[:,0,5,None]).mean(axis=0)
                result['comparisons'].append(dict(grid=grid,seeds=list(seeds),window=window,method=method,
                    original_mean_seconds=dict(zip(FIELDS,a[:,index].mean(axis=0).tolist())),
                    current_mean_seconds=dict(zip(FIELDS,b[:,index].mean(axis=0).tolist())),
                    original_cost_percent_of_default=dict(zip(FIELDS,normalized_old.tolist())),
                    current_cost_percent_of_default=dict(zip(FIELDS,normalized_new.tolist())),
                    cost_change_percentage_points=dict(zip(FIELDS,(normalized_new-normalized_old).tolist()))))
    for seed in (1,2,3):
        original_stage = ORIGINAL/f'data/official/module1/batches/batch_01/advection_60/seed_{seed}'
        current_stage = CURRENT/f'advection_60_batch1/seed_{seed}'
        a = np.load(original_stage/'aot_candidate_schedules/expanded_no_endpoints/structured512/candidate_ids.npy',mmap_mode='r')
        b = np.load(current_stage/'aot_candidate_schedules/expanded_no_endpoints/structured512/candidate_ids.npy',mmap_mode='r')
        equal_rows = np.all(a == b[:len(a)], axis=1)
        changed_rows = np.flatnonzero(~equal_rows)
        first_changed_row = int(changed_rows[0]) if len(changed_rows) else None
        old,old_sha = extract(original_stage/'trajectories/bandit_lstdq.jsonl')
        new,new_sha = extract(current_stage/'trajectories/bandit_lstdq.jsonl')
        assert len(old)==5000 and len(new)==6000
        old_relative=str((original_stage/'trajectories/bandit_lstdq.jsonl').relative_to(ORIGINAL))
        assert old_sha==raw_manifest[old_relative]['sha256']
        assert new_sha==read(current_stage/'result.json')['audits']['bandit_lstdq']['trajectory_sha256']
        assert all(a['mkw']==b['mkw'] for a,b in zip(old,new))
        entry=dict(seed=seed,candidate_identical_rows=int(equal_rows.sum()),
                   candidate_original_rows=len(a), first_changed_candidate_row=first_changed_row,
                   first_potentially_affected_input=first_changed_row//3+1 if first_changed_row is not None else None,
                   paired_inputs_identical=True,
                   original_trajectory_sha256=old_sha,current_trajectory_sha256=new_sha,
                   first_different_setup_input=next((a['index'] for a,b in zip(old,new) if a['params']!=b['params']),None),
                   identical_setup_fraction=sum(a['params']==b['params'] for a,b in zip(old,new))/5000,
                   first_problem_actions=dict(original=old[0]['first_actions'],current=new[0]['first_actions']),
                   original=summarize(old),current=summarize(new[:5000]),
                   windows=[dict(start=start+1,end=start+1000,original=summarize(old[start:start+1000]),current=summarize(new[start:start+1000])) for start in range(0,5000,1000)])
        result['trajectories'].append(entry)
        print(json.dumps({k:entry[k] for k in ('seed','first_different_setup_input','identical_setup_fraction')}),flush=True)
    baseline='24a6b21cfe7607d64510b4a93d15006eb90d5681'
    manifest=read(ROOT/'data/source_manifest.json')
    pinned=read(ROOT/'results/module1_remaining_20261020_015153/source_manifest.json')
    result['source_changes']={k:dict(original=sha,current=pinned.get(k)) for k,sha in manifest.items() if sha!=pinned.get(k)}
    result['checks']['bandit_learner_source_unchanged'] = manifest['setup/learners/linucb/learner.py']==pinned['setup/learners/linucb/learner.py']
    result['checks']['problem_context_source_unchanged'] = manifest['problems/amg.py']==pinned['problems/amg.py']
    result['checks']['native_libraries_unchanged'] = read(ROOT/'data/module1/online/protocol.json')['native_sha256']==read(CURRENT/'advection_60_batch1/protocol.json')['native_sha256']
    result['baseline_source_commit']=baseline
    for name in ('solve/controllers/common/state_encoder.py','solve/core/episode.py','solve/controllers/recursive_lstdq/controller.py','setup/learners/common/action_features.py'):
        assert hashlib.sha256(subprocess.check_output(['git','show',f'{baseline}:{name}'],cwd=ROOT)).hexdigest()==manifest[name]
    # The shared-descriptor refactor must leave the bandit's full feature map intact.
    from experiments.paper_final.online.controllers import make_setup_feature_encoder
    from experiments.paper_final.online.configuration import parse_joint_experiment_config, runtime_config_from_spec
    from experiments.paper_final.online.methods import setup_configuration_spaces_from_args
    from setup.learners.common.aot_candidates import CompactActionCatalog
    original_features=types.ModuleType('original_action_features_audit')
    sys.modules[original_features.__name__]=original_features
    exec(subprocess.check_output(['git','show',f'{baseline}:setup/learners/common/action_features.py'],text=True,cwd=ROOT),original_features.__dict__)
    args=runtime_config_from_spec(parse_joint_experiment_config(read(ROOT/'configs/module1/advection_60/seed_1.json')))
    encoder=make_setup_feature_encoder(configuration_space=setup_configuration_spaces_from_args(args)['expanded_no_endpoints'],parameter_resolution=20)
    original_encoder=original_features.GenericActionFeatureEncoder(encoder.parameter_spec)
    catalog=CompactActionCatalog(encoder.parameter_spec)
    rng=np.random.default_rng(718)
    errors=[float(np.max(np.abs(encoder.encode_action(catalog[int(index)])-original_encoder.encode_action(catalog[int(index)]))))
            for index in [catalog.default_arm_index]+rng.integers(0,len(catalog),1000).tolist()]
    result['checks']['bandit_action_feature_equivalence']=dict(actions_checked=len(errors),maximum_absolute_difference=max(errors))
    result['encoder_dimensions']={}
    for directory,version in [(ORIGINAL/'data/official/module1/batches/batch_01/advection_60/seed_1','original'),(CURRENT/'advection_60_batch1/seed_1','current')]:
        metadata=read(directory/'initial_controllers/bandit_lstdq.npz.encoder.json')
        result['encoder_dimensions'][version]=dict(state=metadata['feature_dim'],state_action=metadata['feature_dim']*9,
            setup=metadata.get('setup_descriptor_dim',len(metadata['setup_observed_keys'])))
    (OUTPUT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',OUTPUT/'diagnosis.json',flush=True)


if __name__=='__main__':
    main()
