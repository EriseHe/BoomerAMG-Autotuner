"""Complete Module 3 with the requested old/new selection check before each test."""
from pathlib import Path
import datetime,json,os,subprocess,traceback
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
PYTHON='/Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(state):
    temp=ROOT/'status.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'status.json')
state=dict(status='running',started_at=now(),pid=os.getpid(),current=None,stage=None,completed=[],test_restarted_fresh_after_hold=True)
env=dict(os.environ,PYTHONPATH=str(SOURCE))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
try:
    for family in ('diffusion','variable_flow'):
        output=ROOT/family/'run'
        if json.loads((output/'status.json').read_text())['status']=='complete':
            state['completed'].append(family);continue
        state.update(current=family,stage='scan',updated_at=now());save(state)
        with (ROOT/family/'supervisor.log').open('a') as log:
            subprocess.run([PYTHON,'-u','-m','experiments.paper_final.run_policy','scan','--output',str(output)],cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            state.update(stage='compare_seeds_1_3',updated_at=now());save(state)
            subprocess.run([PYTHON,str(ROOT/'selection_comparison/compare_before_test.py'),'--family',family],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            state.update(stage='independent_test',updated_at=now());save(state)
            subprocess.run([PYTHON,'-u','-m','experiments.paper_final.run_policy','run','--output',str(output)],cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if json.loads((output/'status.json').read_text())['status']!='complete':raise RuntimeError('Missing completion audit')
        state['completed'].append(family);save(state)
    state.update(status='complete',current=None,stage=None,finished_at=now());save(state)
except BaseException:
    state.update(status='failed',updated_at=now(),traceback=traceback.format_exc());save(state);raise
