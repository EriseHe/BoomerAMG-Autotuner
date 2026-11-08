"""Prepare a two-method measurement of the verified original algorithm."""
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path.cwd()
pointer = root/'results/latest_original_two_branch_run.json'
if pointer.exists():
    raise RuntimeError(f'Inspect the existing original two-branch run before preparing another: {pointer.read_text()}')

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

comparison = Path(read(root/'results/latest_cycle_time_comparison_run.json')['path'])
reference = comparison/'paper_original'
latest = Path(read(root/'results/latest_quadratic_context_run.json')['run'])
assert read(reference/'status.json')['passed'] and read(latest/'status.json')['passed']
out = root/'results'/('original_two_branch_advection_40_seed1_5000_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir()
source = out/'source'
shutil.copytree(comparison/'paper_checkout',source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
(source/'development').mkdir(exist_ok=True)
shutil.copy2(root/'development/run_original_joint.py',source/'development/run_original_joint.py')
shutil.copy2(__file__,out/'prepare.py')
manifest = read(reference/'source_manifest.json')
assert len(manifest)==151
for name,digest in manifest.items():
    assert sha(source/name)==digest, name

original = read(reference/'seed_1/experiment_config.json')
raw = read(reference/'seed_1/experiment_config.json')
methods = ['default_setup_default_solve','bandit_lstdq']
raw['methods'] = [m for m in raw['methods'] if m['id'] in methods]
raw['reporting']['progress_every'] = 1000
raw['description'] = 'Original algorithm: Advection 40, seed 1, Joint and Default, 5,000 learning inputs.'
raw['output_dir'] = str(out/'run/seed_1')
assert [m['id'] for m in raw['methods']] == methods
dump(out/'config.json',raw)
dump(out/'suite.json',dict(name='Original algorithm, Joint and Default: Advection 40, seed 1',
    seeds=[dict(label='1',config=str(out/'config.json'))]))

environment=os.environ.copy()
environment['PYTHONPATH']=str(source)
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    environment[key]='1'
for name,cases in (('run',5000),('validation_run',5)):
    subprocess.run([sys.executable,'-m','development.run_original_joint','prepare',
        '--suite',str(out/'suite.json'),'--output',str(out/name),
        '--reference',str(reference),'--cases',str(cases)],cwd=source,env=environment,check=True)

protocol=read(out/'run/protocol.json')
old=read(reference/'protocol.json')
current=read(latest/'protocol.json')
assert protocol['stages']['1']['inputs_sha256']==old['stages']['1']['inputs_sha256']==current['stages']['1']['inputs_sha256']
assert protocol['stages']['1']['full_stream_sha256']==old['stages']['1']['full_stream_sha256']==current['stages']['1']['full_stream_sha256']
assert protocol['native_sha256']==old['native_sha256']==current['native_sha256']
assert protocol['source_commit']=='6736efd8c1dfd55082f5cb30833093678048b758'
captured=read(out/'run/source_manifest.json')
assert set(captured)-set(manifest)=={'development/run_original_joint.py'}
assert all(captured[name]==digest for name,digest in manifest.items())
for data in (original,raw):
    data['methods']=[m for m in data['methods'] if m['id'] in methods]
    data['reporting'].pop('progress_every')
    data.pop('description',None)
    data.pop('output_dir')
assert original==raw

# The timed execution loop is byte-for-byte the same Python function as the
# completed two-method comparisons; its backend is entirely original code.
def worker_ast(path):
    return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name=='worker'))
assert worker_ast(source/'development/run_original_joint.py')==worker_ast(latest/'source/development/run_joint_context.py')

with (out/'native_validation.log').open('w') as log:
    subprocess.run([sys.executable,'-m','development.run_original_joint','run','--output',str(out/'validation_run')],
        cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
audit=read(out/'validation_run/completion_audit.json')
assert audit['passed'] and audit['measured_method_cases']==10
initial=read(out/'validation_run/seed_1/initial_audit.json')
assert initial['passed'] and initial['exact_original_initial_checkpoints']
assert initial['candidates']['bandit_lstdq']==read(latest/'seed_1/initial_audit.json')['candidates']['bandit_lstdq']
current_events=[readline for readline in (json.loads(line) for line in (latest/'seed_1/execution.jsonl').read_text().splitlines())]
validation_events=[json.loads(line) for line in (out/'validation_run/seed_1/execution.jsonl').read_text().splitlines()]
for a,b in zip(validation_events,current_events):
    assert (a['input_index'],a['position'],a['method'])==(b['input_index'],b['position'],b['method'])
dump(out/'validation.json',dict(passed=True,original_source_files_verified=151,
    original_source_commit=protocol['source_commit'],algorithm_source_files_changed=[],
    adapter_added='development/run_original_joint.py',native_source_launch=True,
    measured_method_cases=10,measured_methods=methods,original_state_entries=35,
    original_state_action_entries=315,original_setup_descriptors=8,
    exact_initial_checkpoint_arrays=initial['exact_original_initial_checkpoints'],
    exact_original_encoder_metadata=True,paired_static_candidates=True,
    unchanged_timed_execution_loop=True,matched_native_libraries=True,
    matched_full_input_stream=True,matched_two_method_order_prefix=True,
    measurement_changes=['two-method roster','balanced two-method ordering','progress every 1000'],
    latest_reference=str(latest),original_reference=str(reference),no_frozen_phase=True))
dump(pointer,dict(path=str(out),run=str(out/'run'),source=str(source),status='prepared',
    seed=1,cases=5000,problem='advection_40',methods=methods,
    reference_original=str(reference),reference_quadratic=str(latest),source_commit=protocol['source_commit']))
print('PREPARED_VALIDATED',str(out),flush=True)
