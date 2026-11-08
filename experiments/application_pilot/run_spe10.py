import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from bridge import ROOT,solve

p=argparse.ArgumentParser();p.add_argument('--cases',type=int,default=100);args=p.parse_args()
out=ROOT/'results/application_default_100/spe10';out.mkdir(parents=True,exist_ok=True)
assert not (out/'records.jsonl').exists(), 'Use a fresh results directory'
octave=ROOT/'results/application_default_100/octave_env/bin/octave-cli'
os.environ['OCTAVE_HOME']=str(octave.parent.parent)
os.environ['OCTAVE_EXEC_HOME']=str(octave.parent.parent)
expression=f"addpath('{ROOT}/experiments/application_pilot'); spe10_run('{ROOT}',{args.cases});"
log=(out/'octave.log').open('w')
process=subprocess.Popen([str(octave),'--no-gui','--quiet','--eval',expression],stdout=log,stderr=subprocess.STDOUT)
previous=None
try:
  for case in range(1,args.cases+1):
    while not (out/'ready').exists():
      if process.poll() is not None: raise RuntimeError(f'Octave exited {process.returncode}; inspect '+str(out/'octave.log'))
      time.sleep(.05)
    with (out/'request.bin').open('rb') as f:
      n,nnz=np.fromfile(f,np.int32,2)
      ia=np.fromfile(f,np.int32,n+1);ja=np.fromfile(f,np.int32,nnz)
      av=np.fromfile(f,np.float64,nnz);b=np.fromfile(f,np.float64,n)
    A=csr_matrix((av,ja,ia),shape=(n,n))
    assert n==198000 and np.all(A.diagonal()>0)
    assert np.max(np.abs((A-A.T).data),initial=0)<1e-12*max(abs(A.data))
    x,measurement=solve(A,b,previous,maxit=500)
    measurement['max_cycles']=500
    # Accurate reference trajectory is independent of benchmark completion.
    reference,ref=solve(A,b,previous,tol=1e-10,maxit=1000,reference=True)
    # PCG's recursive residual and independently recomputed residual differ
    # slightly at this accuracy with absolute reservoir pressures in pascals.
    # Require an external 1e-9 bound (1000x tighter than the timed solve).
    ref['external_acceptance_tolerance']=1e-9
    ref['externally_accepted']=bool(ref['interface_code']==0 and ref['true_residual']<=1e-9)
    if not ref['externally_accepted']: raise RuntimeError(('reference failed',ref))
    row=dict(case=case,physical_time_years=(case-1)*.1,**measurement,reference=ref)
    with (out/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    reference.tofile(out/'response.bin'); previous=reference
    (out/'ready').unlink(); (out/'response.ready').touch()
  code=process.wait()
  if code:raise RuntimeError(f'Octave exit {code}')
finally:
  if process.poll() is None:process.terminate();process.wait()
  log.close()
