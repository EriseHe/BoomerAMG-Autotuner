"""Offline numerical diagnosis on a copied matrix; no online learner updates."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import json
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from hypre.bindings import create_env
from hypre.bindings.config import configure_smoother_profile
from setup.space import DEFAULT_SETUP_PARAMS
directory=Path(__file__).resolve().parent
snapshot=json.loads((directory/'snapshot.json').read_text())
with Path(snapshot['path']).open('rb') as f:
    n,nnz=np.fromfile(f,np.int32,2)
    ia=np.fromfile(f,np.int32,n+1);ja=np.fromfile(f,np.int32,nnz)
    values=np.fromfile(f,np.float64,nnz);b=np.fromfile(f,np.float64,n);x0=np.fromfile(f,np.float64,n)
    assert not f.read(1)
A=csr_matrix((values,ja,ia),shape=(n,n));bnorm=np.linalg.norm(b)
assert n==220881 and np.isfinite(values).all() and x0.shape==(n,)
recent=None
with (directory.parent/'records.jsonl').open() as f:
    for line in f:
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if row['method']=='linucb' and row['case']<=snapshot['case']:
            recent=row
params=DEFAULT_SETUP_PARAMS
probes=[
    ('default_standalone_l1',dict(params,relax_type=18),False),
    ('default_standalone_l1_weight1.6',dict(params,relax_type=18,relax_wt=1.6),False),
    ('default_standalone_symmetric_gs',dict(params,relax_type=6),False),
    ('default_pcg_l1',dict(params,relax_type=18),True),
    ('default_pcg_symmetric_gs',dict(params,relax_type=6),True),
    ('wider_interpolation_standalone_l1',dict(params,relax_type=18,P_max_elmts=16),False),
    ('coarsen6_wider_interpolation_standalone_l1',dict(params,relax_type=18,coarsen_type=6,P_max_elmts=16),False),
    ('recent_linucb_standalone_l1',dict(recent['raw']['primary_attempts'][0]['params'],relax_type=18),False),
]
configure_smoother_profile('l1_jacobi_direct_coarse')
results=[]
for name,config,pcg in probes:
    x=np.empty_like(x0);tick=time.perf_counter()
    try:
        with create_env(matrix=A,rhs=b,initial_guess=x0,solution_out=x) as env:
            if pcg:
                prep=env.prepare_rl(config)
                result=env.solve_pcg(env,tol=1e-6,max_iter=100)
                result['hierarchy_setup_sec']=prep.setup_runtime_sec
            else:
                native=env.solve(params=config,tol=1e-6,max_iter=100)
                result=dict(iterations=native.iterations,hierarchy_setup_sec=native.setup_runtime_sec,
                            solve_sec=native.solve_runtime_sec,native_status=native.status.name)
        residual=float(np.linalg.norm(b-A@x)/bnorm)
        result.update(true_residual=residual,converged=bool(np.isfinite(residual) and residual<=1e-6))
    except Exception as error:
        result=dict(error=repr(error),converged=False)
    result.update(name=name,params=config,pcg=pcg,wall_sec=time.perf_counter()-tick)
    results.append(result)
    print(json.dumps(result),flush=True)
    (directory/f"probe_case_{snapshot['case']}.json").write_text(json.dumps(dict(snapshot=snapshot,
        protocol='offline diagnosis; 100 iterations/cycles, true residual 1e-6; not added to online results',
        results=results),indent=2)+'\n')
