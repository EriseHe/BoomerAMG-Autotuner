"""Inspect paused experiment state; resume only on an explicit --resume request."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess

directory=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
args=parser.parse_args();path=directory/'pause.json';manifest=json.loads(path.read_text())
if manifest['status']!='paused':raise SystemExit(f"Current state is {manifest['status']}")
for process in manifest['paused_processes']:
    try:
        text=subprocess.check_output(['ps','-p',str(process['pid']),'-o','state=','-o','command='],text=True).strip()
    except subprocess.CalledProcessError:
        raise SystemExit('A saved process is gone; recover from the durable checkpoint instead.')
    state,command=text.split(None,1)
    if not state.startswith('T') or command!=process['command']:
        raise SystemExit(f"Process {process['pid']} does not match the saved stopped process; no signals sent.")
print(f"Verified paused at MFEM {manifest['completed_mfem']}; next case {manifest['next_mfem_case']}.")
if args.resume:
    base=Path(manifest['original_batch']);extension=Path(manifest['extension_batch'])
    batch=json.loads((base/'status.json').read_text());queued=json.loads((extension/'status.json').read_text())
    parent=batch['pid'];worker=batch['applications']['heat']['pid'];waiting=queued['pid']
    resumed=datetime.now(timezone.utc)
    manifest.update(status='resumed',resumed_utc=resumed.isoformat(),
        pause_duration_sec=(resumed-datetime.fromisoformat(manifest['paused_utc'])).total_seconds())
    batch.update(status='running',resumed_utc=manifest['resumed_utc'])
    batch['applications']['heat']['status']='running'
    queued.update(status='waiting_for_prefix',resumed_utc=manifest['resumed_utc'])
    (base/'status.json').write_text(json.dumps(batch,indent=2)+'\n')
    (extension/'status.json').write_text(json.dumps(queued,indent=2)+'\n')
    # Wake the physical driver first, then the solver and its supervisors.
    for process in manifest['paused_processes']:
        if process['pid'] not in (parent,worker,waiting):os.kill(process['pid'],signal.SIGCONT)
    for pid in (worker,parent,waiting):os.kill(pid,signal.SIGCONT)
    helpers=[]
    if Path('/usr/bin/caffeinate').exists():
        for pid in (parent,waiting):
            helper=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(pid)],
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            helpers.append(helper.pid)
    manifest['resume_sleep_assertion_pids']=helpers
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Resumed existing learners and simulation at the next system.')
