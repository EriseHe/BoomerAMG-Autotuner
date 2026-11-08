"""Audit completed application pilots and retained setup-learning observations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from setup.space import SetupConfigurationSpace,DEFAULT_SETUP_PARAMS
from experiments.paper_final.online.controllers import make_setup_feature_encoder


def audit(path):
    path=Path(path)
    protocol=json.loads((path/'protocol.json').read_text())
    summary=json.loads((path/'summary.json').read_text())
    records=[json.loads(line) for line in (path/'records.jsonl').read_text().splitlines()]
    cases=protocol['cases'];methods=protocol['methods']
    assert summary['complete'] and len(records)==cases*len(methods)
    current_source_matches=True
    with zipfile.ZipFile(path/'source_snapshot.zip') as archive:
        for relative,sha in protocol['source_sha256'].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest()==sha,relative
            current_source_matches &= (hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==sha)
    for case in range(1,cases+1):
        rows=[r for r in records if r['case']==case]
        assert len(rows)==len(methods) and {r['method'] for r in rows}==set(methods)
        assert len({r['input_sha256'] for r in rows})==1
        assert [r['position'] for r in rows]==list(range(len(methods)))
        assert all(r['final_converged'] and r['true_residual']<=protocol['tolerance'] for r in rows)
        assert all(np.isfinite(r['total_wall_sec']) and r['total_wall_sec']>=0 for r in rows)
        if protocol['max_cycles']==100:
            for row in rows:
                if row.get('skipped_already_converged'):continue
                raw=row['raw']
                if 'primary_attempts' in raw:
                    for attempt in raw['primary_attempts']:
                        assert attempt.get('iterations',attempt.get('cycles',0))<=100
                    assert raw.get('fallback_cycles',0)<=100
                else:assert raw['iterations']<=100
                if row['rescue_used']:assert row['rescue']['iterations']<=100
    refs=[json.loads(line) for line in (path/'reference.jsonl').read_text().splitlines()]
    assert len(refs)==cases and max(r['true_residual'] for r in refs)<=1e-9
    if protocol['max_cycles']==100:assert max(r['iterations'] for r in refs)<=100
    application=np.loadtxt(path/'application.csv',delimiter=',',ndmin=2)
    assert np.array_equal(application[:,0],np.arange(1,cases+1))
    accuracy={}
    if protocol['application']=='heat':
        groups=application[:,6] if application.shape[1]>6 else np.ones(len(application))
        drifts=[]
        for group in np.unique(groups):
            masses=application[groups==group,4]
            drifts.append(float(np.max(abs(masses-masses[0]))/abs(masses[0])))
        accuracy['relative_mass_drift']=max(drifts)
        assert accuracy['relative_mass_drift']<1e-9
    else:
        assert np.min(application[:,3])>=-1e-10 and np.max(application[:,4])<=1+1e-10
        accuracy['saturation_min']=float(np.min(application[:,3]))
        accuracy['saturation_max']=float(np.max(application[:,4]))
    config=json.loads((ROOT/'data/module1/online/configurations/diffusion_60/seed_1.json').read_text())
    name,raw=next(iter(config['setup']['configuration_spaces'].items()))
    encoder=make_setup_feature_encoder(configuration_space=SetupConfigurationSpace.from_mapping(name,raw),parameter_resolution=20)
    models={}
    for method in ('linucb','joint'):
        features,targets,failures=[],[],[]
        for row in (r for r in records if r['method']==method):
            if row.get('skipped_already_converged'):continue
            context=np.asarray(row['context']);raw=row['raw']
            def observation(params,cost,failed):
                g=encoder.encode_action(params)
                features.append(np.concatenate([context,g,*[v*g for v in context[1:]]]))
                targets.append(cost);failures.append(float(failed))
            for attempt in raw['primary_attempts']:
                observation(attempt['params'],attempt['learning_cost'],attempt['status']!='success')
            if raw.get('fallback_learning_cost') is not None:
                observation(DEFAULT_SETUP_PARAMS,raw['fallback_learning_cost'],False)
        phi=np.asarray(features)
        with np.load(path/f'{method}_setup.npz') as saved:
            assert int(saved['t'])==len(features)==summary['methods'][method]['setup_observations']
            expected_inverse=np.linalg.inv(np.eye(phi.shape[1])+phi.T@phi)
            assert np.allclose(saved['A_inv'],expected_inverse,rtol=1e-7,atol=1e-9)
            assert np.allclose(saved['b'],phi.T@targets,rtol=1e-10,atol=1e-8)
            assert np.allclose(saved['retry_failure_b'],phi.T@failures,rtol=1e-10,atol=1e-8)
            models[method]=dict(observations=len(features),failure_observations=int(sum(failures)),
                               inverse_max_abs_error=float(np.max(abs(saved['A_inv']-expected_inverse))))
    for method in methods:
        rows=[r for r in records if r['method']==method]
        assert abs(sum(r['total_wall_sec'] for r in rows)-summary['methods'][method]['total_wall_sec'])<1e-9
    result=dict(passed=True,cases=cases,method_cases=len(records),paired_inputs=True,
                producing_source_snapshot_verified=True,current_source_hashes_match=bool(current_source_matches),
                all_final_residuals_pass=True,
                accuracy=accuracy,setup_models=models)
    (path/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path,nargs='+')
    for path in parser.parse_args().results:audit(path)
