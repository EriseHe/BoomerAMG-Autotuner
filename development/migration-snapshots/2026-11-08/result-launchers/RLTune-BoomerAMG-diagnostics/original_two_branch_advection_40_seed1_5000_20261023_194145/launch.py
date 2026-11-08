"""Launch the validated, frozen-source original-algorithm experiment once."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path.cwd()
pointer = root/'results/latest_original_two_branch_run.json'
details = json.loads(pointer.read_text())
out = Path(details['path'])
run = Path(details['run'])
source = Path(details['source'])
assert details['status'] == 'prepared'
assert json.loads((run/'status.json').read_text())['status'] == 'prepared'
assert json.loads((out/'validation.json').read_text())['passed']
assert json.loads((out/'pairing_audit.json').read_text())['passed']
branch = subprocess.check_output(['git','branch','--show-current'],text=True).strip()
assert branch == 'codex/development', branch
processes = subprocess.check_output(['ps','-axo','pid=,ppid=,%cpu=,%mem=,comm='],text=True)
commands = subprocess.check_output(['ps','-axo','pid=,command='],text=True)
competing = [line for line in commands.splitlines() if 'python' in line.lower() and
    any(module in line for module in (' -m development.run_', ' -m experiments.paper_final.run_'))]
assert not competing, competing
preflight = dict(at=datetime.now(timezone.utc).isoformat(), branch=branch,
    processes=processes, competing_experiments=competing,
    memory_pressure=subprocess.check_output(['memory_pressure','-Q'],text=True),
    free_disk_bytes=shutil.disk_usage(root).free,
    git_status=subprocess.check_output(['git','status','--short'],text=True))
(out/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
shutil.copy2(__file__,out/'launch.py')
environment = os.environ.copy()
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    environment[key] = '1'
environment['PYTHONPATH'] = str(source)
with (out/'runner.log').open('w') as log:
    process = subprocess.Popen([sys.executable,'-m','development.run_original_joint','run','--output',str(run)],
        cwd=source,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True)
details.update(status='launched',pid=process.pid,launched_at=datetime.now(timezone.utc).isoformat())
pointer.write_text(json.dumps(details,indent=2)+'\n')
(out/'launch.json').write_text(json.dumps(details,indent=2)+'\n')
print(json.dumps(details,indent=2))
