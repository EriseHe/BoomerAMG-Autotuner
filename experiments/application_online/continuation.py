"""Continue a completed campaign from its exact setup/controller checkpoints."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.application_online.run import Study, METHODS, TOL, balanced_orders, create_env, configure_smoother_profile
from problems.application_context import APPLICATION_CONTEXT_FIELDS


def additional_scenarios(application, seed):
    if application=='spe10':
        choices=[dict(viscosity_ratio=r,schedule=s) for r in (1.5,3.,4.,7.5,15.)
                 for s in ('constant','pulsed')]
    else:
        regimes=((.25,.05,.1),(.5,.2,.1),(.2,.05,.1),(.5,.1,.075),(.5,.1,.15))
        choices=[dict(kappa=k,alpha=a,dt=dt,hot_amplitude=h)
                 for k,a,dt in regimes for h in (1.,4.)]
    order=np.random.default_rng(seed+100994).permutation(10)
    return [dict(scenario=i+1,**choices[int(j)]) for i,j in enumerate(order)]


def restore(study, prefix):
    """Restore learned statistics, RNGs, counters, candidate prefix and cost carry."""
    prefix=Path(prefix);result={}
    tick=time.perf_counter();previous={}
    for line in (prefix/'records.jsonl').read_text().splitlines():
        row=json.loads(line)
        if 'bandit_timing' in row.get('raw',{}):
            previous[row['method']]=row['raw']['bandit_timing']['update_sec']
    prefix_read_sec=time.perf_counter()-tick
    for name,branch in study.branches.items():
        tick=time.perf_counter();model=branch.policy.model;schedule=study.rows[name].base
        source=np.load(prefix/name/'candidates'/schedule._ids_path.name,mmap_mode='r')
        target=np.load(schedule._ids_path,mmap_mode='r+')
        assert len(target)>=len(source) and target.shape[1:]==source.shape[1:]
        target[:len(source)]=source;target.flush()
        assert np.array_equal(schedule._ids[:len(source)],source)
        model.load_mutable_state(prefix/f'{name}_setup.npz')
        with np.load(prefix/f'{name}_setup.npz') as checkpoint:
            for key,value in (('A_inv',model.A_inv),('b',model.b),('retry_failure_b',model.failure_b)):
                assert np.array_equal(checkpoint[key],value),key
            assert model.t==int(checkpoint['t'])
            assert model.rng.bit_generator.state==json.loads(str(checkpoint['rng_state']))
        study.previous_update[name]=previous.get(name,0.)
        study.initialization_by_method[name]+=time.perf_counter()-tick+prefix_read_sec
        result[name]=dict(setup_observations=model.t,candidate_cursor=schedule.cursor,
            original_candidate_rows_preserved=len(source),exact_statistics_restored=True,
            previous_update_sec=study.previous_update[name],prefix_record_read_sec=prefix_read_sec)
    for name,bundle in study.bundles.items():
        tick=time.perf_counter();bundle.load(prefix/f'{name}_controller.npz')
        with np.load(prefix/f'{name}_controller.npz') as checkpoint:
            for key in ('theta','a_matrix','a_inverse','b'):
                assert np.array_equal(checkpoint[key],getattr(bundle.controller,key)),key
            assert bundle.controller.steps==int(checkpoint['steps'])
            assert bundle.controller.episodes==int(checkpoint['episodes'])
            assert bundle.controller.rng.bit_generator.state==json.loads(str(checkpoint['rng_state']))
        study.initialization_by_method[name]+=time.perf_counter()-tick
        result.setdefault(name,{})['controller']=bundle.summary()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--additional-cases',type=int,default=1000)
    args=parser.parse_args();prefix=args.prefix.resolve();out=args.output.resolve()
    prior=json.loads((prefix/'protocol.json').read_text())
    assert json.loads((prefix/'summary.json').read_text())['complete']
    assert prior['max_cycles']==100
    offset=prior.get('case_offset',0)+prior['cases'];steps=prior['steps_per_scenario']
    assert offset%steps==0 and 1<=args.additional_cases<=10*steps
    out.mkdir(parents=True,exist_ok=False)
    application=prior['application'];seed=prior['seed'];target=offset+args.additional_cases
    mode='spe10_pressure' if application=='spe10' else 'mfem_heat'
    configure_smoother_profile('l1_jacobi_direct_coarse')
    tick=time.perf_counter()
    with create_env(matrix=csr_matrix([[1.]]),rhs=np.ones(1)):pass
    common_init=time.perf_counter()-tick
    tick=time.perf_counter();study=Study(mode,out,target,seed,100)
    restoration=restore(study,prefix);startup=time.perf_counter()-tick
    campaign=additional_scenarios(application,seed)
    assert not ({json.dumps({k:v for k,v in s.items() if k!='scenario'},sort_keys=True) for s in campaign}
                & {json.dumps({k:v for k,v in s.items() if k!='scenario'},sort_keys=True) for s in prior['campaign']})
    (out/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    if application=='heat':
        (out/'campaign.tsv').write_text(''.join(' '.join(str(s[k]) for k in
            ('kappa','alpha','dt','hot_amplitude'))+'\n' for s in campaign))
    paths=set(prior['source_sha256'])
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'experiments/application_online').iterdir() if p.is_file())
    protocol=dict(prior,cases=args.additional_cases,case_offset=offset,target_cases=target,
        prefix=str(prefix),campaign=campaign,initialization_sec=startup,
        common_initialization_sec=common_init,initialization_by_method=study.initialization_by_method,
        restoration=restoration,scope='continued online learning from the completed prefix',
        native_hierarchy_restart='native baseline hierarchies rebuilt after process restart; construction cost charged',
        context_fields=APPLICATION_CONTEXT_FIELDS[mode],
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in protocol['source_sha256']:archive.write(ROOT/p,p)
        archive.write(out/'protocol.json','protocol.json')
    log=(out/'application.log').open('w')
    if application=='spe10':
        executable=ROOT/'results/application_default_100/octave_env/bin/octave-cli'
        env=dict(os.environ,OCTAVE_HOME=str(executable.parent.parent),OCTAVE_EXEC_HOME=str(executable.parent.parent))
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        expression=f"addpath({quote(ROOT/'experiments/application_online')});spe10_stream({quote(ROOT)},{quote(out)},{args.additional_cases},{steps});"
        command=[str(executable),'--no-gui','--quiet','--eval',expression]
    else:
        env=os.environ.copy()
        command=[str(ROOT/'results/application_default_100/heat_stream'),
            str(ROOT/'results/application_default_100/heat.mesh'),str(out),str(args.additional_cases),str(steps),str(out/'campaign.tsv')]
    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    records=[];previous=None;started=time.perf_counter()
    orders=balanced_orders(target,seed,methods=METHODS)[offset:]
    try:
        for local in range(args.additional_cases):
            index=offset+local
            while not (out/'ready').exists():
                if process.poll() is not None:raise RuntimeError(f'Application exited {process.returncode}')
                time.sleep(.05)
            with (out/'request.bin').open('rb') as f:
                n,nnz=np.fromfile(f,np.int32,2);ia=np.fromfile(f,np.int32,n+1)
                ja=np.fromfile(f,np.int32,nnz);values=np.fromfile(f,np.float64,nnz)
                b=np.fromfile(f,np.float64,n)
                x0=np.fromfile(f,np.float64,n) if application=='heat' else (np.zeros(n) if local%steps==0 else previous)
            A=csr_matrix((values,ja,ia),shape=(n,n))
            metadata=json.loads((out/'context.json').read_text())
            assert metadata['scenario']==local//steps+1 and metadata['scenario_step']==local%steps+1
            assert x0.shape==(n,) and np.isfinite(b).all()
            assert np.max(np.abs((A-A.T).data),initial=0.)<1e-11*np.max(abs(A.data))
            digest=hashlib.sha256((out/'request.bin').read_bytes()).hexdigest()
            for position,method in enumerate(orders[local]):
                row=study.solve(method,index,A,b,x0,metadata['context'],metadata['context_sec'])
                row.update(n=int(n),nnz=int(nnz),input_sha256=digest,position=position,context=metadata['context'],
                    scenario=offset//steps+metadata['scenario'],scenario_step=metadata['scenario_step'],max_cycles=100)
                with (out/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                records.append(row)
                print(json.dumps({k:row[k] for k in ('case','method','total_wall_sec','rescue_used','final_converged')}),flush=True)
                if not row['final_converged']:raise RuntimeError('Rescue failed; workload incomplete')
            reference,check=study.reference.solve(A,b,x0,tol=1e-10,cap=100)
            if not np.isfinite(check['true_residual']) or check['true_residual']>1e-9:
                raise RuntimeError(f'Reference failed: {check}')
            with (out/'reference.jsonl').open('a') as f:f.write(json.dumps(dict(case=index+1,**check))+'\n')
            previous=reference;reference.tofile(out/'response.bin')
            (out/'ready').unlink();(out/'response.ready').touch()
            progress=dict(completed_systems=index+1,total_systems=target,segment_completed=local+1,
                method_cases=(index+1)*len(METHODS),elapsed_sec=time.perf_counter()-started,
                scenario=offset//steps+metadata['scenario'],scenario_step=metadata['scenario_step'],
                rescue_calls=sum(r['rescue_used'] for r in records),final_failures=sum(not r['final_converged'] for r in records))
            (out/'progress.tmp').write_text(json.dumps(progress)+'\n');(out/'progress.tmp').replace(out/'progress.json')
            if (local+1)%steps==0:
                checkpoint=out/f'checkpoint_{index+1:04d}';checkpoint.mkdir()
                for name,bundle in study.bundles.items():bundle.save(checkpoint/f'{name}_controller.npz')
                for name,branch in study.branches.items():branch.policy.model.save_mutable_state(checkpoint/f'{name}_setup.npz')
        if process.wait()!=0:raise RuntimeError('Physical application failed after final response')
        summary=dict(complete=True,case_offset=offset,cases=args.additional_cases,target_cases=target,
                     initialization_sec=startup,elapsed_sec=time.perf_counter()-started,methods={})
        for method in METHODS:
            rows=[r for r in records if r['method']==method]
            summary['methods'][method]=dict(cases=len(rows),
                mean_total_wall_sec=sum(r['total_wall_sec'] for r in rows)/len(rows),
                total_wall_sec=sum(r['total_wall_sec'] for r in rows),
                primary_failures=sum(not r['primary_converged'] for r in rows),
                first_attempt_failures=sum(r['first_attempt_failed'] for r in rows),
                rescues=sum(r['rescue_used'] for r in rows),final_failures=sum(not r['final_converged'] for r in rows),
                hierarchy_builds=sum(r['builds'] for r in rows))
            summary['methods'][method]['cold_start_total_sec']=(summary['methods'][method]['total_wall_sec']+
                common_init+study.initialization_by_method[method])
        for name,bundle in study.bundles.items():
            bundle.save(out/f'{name}_controller.npz');summary['methods'][name]['controller']=bundle.summary()
        for name,branch in study.branches.items():
            branch.policy.model.save_mutable_state(out/f'{name}_setup.npz')
            summary['methods'][name]['setup_observations']=branch.policy.model.t
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:
        if process.poll() is None:process.terminate();process.wait()
        study.close();log.close()


if __name__=='__main__':main()
