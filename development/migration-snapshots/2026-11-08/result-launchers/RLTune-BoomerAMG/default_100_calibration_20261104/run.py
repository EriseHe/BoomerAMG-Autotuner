"""Default-only timing calibration on the current pinned diffusion-60 stream."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from hypre.bindings import solve, SolveStatus
from hypre.bindings.config import configure_smoother_profile, augment_setup_params
from setup.space import DEFAULT_SETUP_PARAMS

config_path = ROOT / 'configs/module1/diffusion_60/seed_1.json'
config = json.loads(config_path.read_text())
stream_hash = config['stream']['expected_sha256']
stream_path = ROOT / 'data/module1/online/inputs' / (stream_hash + '.jsonl.gz')
payload = gzip.decompress(stream_path.read_bytes())
assert hashlib.sha256(payload).hexdigest() == stream_hash
inputs = [json.loads(line) for line in payload.splitlines()][:100]
assert len(inputs) == 100
configure_smoother_profile(config['solve']['smoother_profile'])
params = augment_setup_params(dict(DEFAULT_SETUP_PARAMS))
tol = config['solve']['tolerance']
max_iter = config['solve']['max_cycles']

def execute(row):
    started = time.perf_counter()
    r = solve(params=params, tol=tol, max_iter=max_iter, **row['mkw'])
    return dict(setup_sec=r.setup_runtime_sec, solve_sec=r.solve_runtime_sec,
                setup_solve_sec=r.runtime_sec, wall_with_assembly_sec=time.perf_counter()-started,
                iterations=r.iterations, residual=r.residual_norm,
                converged=r.status is SolveStatus.CONVERGED, status=r.status.name)

warmup_start = time.perf_counter()
warmups = [execute(row) for row in inputs[:3]]
warmup_wall = time.perf_counter()-warmup_start
rows = []
started = time.perf_counter()
with (OUT/'cases.jsonl').open('w') as f:
    for index, row in enumerate(inputs, 1):
        result = dict(case=index, mkw=row['mkw'], **execute(row))
        rows.append(result)
        f.write(json.dumps(result)+'\n')
        f.flush()
        if index % 20 == 0:
            print(f'{index}/100 cases; cumulative setup+solve {sum(r["setup_solve_sec"] for r in rows):.3f} s', flush=True)
wall = time.perf_counter()-started
summary = dict(
    workload='existing structured diffusion_60; NOT SPE10 or MFEM',
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    working_tree_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
    config=str(config_path.relative_to(ROOT)), stream_sha256=stream_hash,
    cases=len(rows), unknowns=60**3, ranks=1, threads=1, learning=False,
    default_params=params, smoother_profile=config['solve']['smoother_profile'],
    tolerance=tol, max_cycles=max_iter, initial_guess='zero', rhs='pinned random RHS per case',
    hierarchy_policy='fresh hierarchy every case', attempts_per_case=1, recovery=False,
    warmup_cases=3, warmup_wall_sec=warmup_wall, warmup_records=warmups,
    failures=sum(not r['converged'] for r in rows), elapsed_sec=wall,
    python=sys.version, platform=platform.platform(),
    total_setup_solve_sec=sum(r['setup_solve_sec'] for r in rows),
    timing_note='Native setup+solve excludes matrix/RHS assembly, destruction and Python/file I/O. Wall-with-assembly includes solve() entry through return. Three warmups excluded and recorded separately.',
)
for name in ['setup_sec','solve_sec','setup_solve_sec','wall_with_assembly_sec','iterations','residual']:
    values=[r[name] for r in rows]
    summary[name] = dict(mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values))
summary['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [config_path,ROOT/'hypre/interfaces/libamg_runtime.dylib',ROOT/'hypre/install/lib/libHYPRE.dylib',
              ROOT/'hypre/bindings/boomeramg.py',ROOT/'hypre/bindings/config.py',ROOT/'setup/space.py',
              *sorted((ROOT/'hypre/interfaces').glob('*.c')),*sorted((ROOT/'hypre/interfaces').glob('*.h'))]}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
