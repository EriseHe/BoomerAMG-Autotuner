"""Default AMG for application CSR matrices, with independently checked residuals."""
import ctypes as ct
from pathlib import Path
import time
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
lib = ct.CDLL(str(ROOT/'results/application_default_100/libpilot.dylib'))
ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
lib.pilot_default.argtypes = [ct.c_int,ip,ip,dp,dp,dp,ct.c_double,ct.c_int,ct.c_int,dp]
lib.pilot_default.restype = ct.c_int

def solve(A, b, x0=None, tol=1e-6, maxit=50, reference=False):
    start=time.perf_counter()
    A=sparse.csr_matrix(A); A.sort_indices()
    ia=np.asarray(A.indptr,dtype=np.int32)
    ja=np.asarray(A.indices,dtype=np.int32)
    a=np.asarray(A.data,dtype=np.float64)
    b=np.ascontiguousarray(b,dtype=np.float64)
    x=np.zeros(len(b)) if x0 is None else np.array(x0,dtype=np.float64,copy=True)
    s=np.zeros(6)
    code=lib.pilot_default(len(b),ia,ja,a,b,x,tol,maxit,int(reference),s)
    residual=float(np.linalg.norm(b-A@x)/max(np.linalg.norm(b),np.finfo(float).tiny))
    result=dict(n=len(b),nnz=A.nnz,import_sec=s[0],setup_sec=s[1],solve_sec=s[2],
                setup_solve_sec=s[1]+s[2],reported_residual=s[3],iterations=int(s[4]),
                native_error=int(s[5]),interface_code=code,true_residual=residual,
                converged=bool(code==0 and np.isfinite(residual) and residual<=tol),
                wall_with_import_and_check_sec=time.perf_counter()-start)
    return x,result

if __name__=='__main__':
    from scipy.sparse import diags,kronsum
    t=diags([-np.ones(11),2*np.ones(12),-np.ones(11)],[-1,0,1],format='csr')
    A=kronsum(kronsum(t,t),t).tocsr()
    truth=np.random.default_rng(20261104).normal(size=A.shape[0]);b=A@truth
    for ref in [False,True]:
        x,r=solve(A,b,tol=1e-10,maxit=200,reference=ref)
        assert r['converged'],r
        assert np.linalg.norm(x-truth)/np.linalg.norm(truth)<1e-8
        print(ref,r,flush=True)
