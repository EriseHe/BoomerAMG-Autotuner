"""Run the two prepared Module 3 cases serially with frozen source and inputs."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
PYTHON='/Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python'
FAMILIES=('diffusion','variable_flow')

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(state):
    path=ROOT/'status.json';temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)

state=dict(status='running',started_at=now(),pid=os.getpid(),current=None,completed=[])
try:
    env=dict(os.environ,PYTHONPATH=str(SOURCE))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    for family in FAMILIES:
        output=ROOT/family/'run'
        if json.loads((output/'status.json').read_text())['status']!='prepared':
            raise ValueError(f'{family}: expected a new prepared run')
        protocol=json.loads((output/'protocol.json').read_text())
        if protocol['timing_protocol']!='first_attempt_native_solve_v1' or protocol['recovery_enabled']:
            raise ValueError('Wrong first-attempt timing protocol')
        state.update(current=family,updated_at=now());save(state)
        print(f'{now()} Starting {family}: fresh scan, frozen choices, independent test',flush=True)
        with (ROOT/family/'supervisor.log').open('x') as log:
            subprocess.run([PYTHON,'-u','-m','experiments.paper_final.run_policy','run','--output',str(output)],
                           cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if json.loads((output/'status.json').read_text())['status']!='complete':
            raise RuntimeError(f'{family}: missing completed audit')
        state['completed'].append(family);save(state)
        print(f'{now()} Completed {family}',flush=True)
    state.update(status='complete',current=None,finished_at=now());save(state)
except BaseException:
    state.update(status='failed',updated_at=now(),traceback=traceback.format_exc());save(state)
    raise
