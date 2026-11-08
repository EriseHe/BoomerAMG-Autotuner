"""Wait for the current batch, then continue both learned campaigns to 2,000."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    prefix=args.prefix.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting_for_prefix',prefix=str(prefix),target_cases_per_application=2000,
        max_cycles=100,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),applications={})
    active=None
    def save():
        (out/'status.tmp').write_text(json.dumps(state,indent=2)+'\n');(out/'status.tmp').replace(out/'status.json')
    def interrupt(signum,frame):
        if active is not None and active.poll() is None:active.send_signal(signal.SIGINT)
        state['status']='interrupted';save();raise KeyboardInterrupt
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt);save()
    try:
        while True:
            previous=json.loads((prefix/'status.json').read_text())
            if previous['status']=='complete':break
            if previous['status'] in ('failed','audit_failed','interrupted'):
                state.update(status='prefix_incomplete',prefix_status=previous['status']);save();return 1
            time.sleep(3)
        for app in ('spe10','heat'):
            stage=out/app;combined=out/f'{app}_combined';state.update(status='running',active_application=app)
            command=[sys.executable,str(ROOT/'experiments/application_online/continuation.py'),
                '--prefix',str(prefix/app),'--output',str(stage),'--additional-cases','1000']
            with (out/f'{app}.log').open('x') as log:
                active=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state['applications'][app]=dict(status='running',pid=active.pid,continuation=str(stage),combined=str(combined));save()
                code=active.wait();active=None
            if code:
                state.update(status='failed',exit_code=code);state['applications'][app]['status']='failed';save();return code
            state['applications'][app]['status']='combining_and_auditing';save()
            command=[sys.executable,str(ROOT/'experiments/application_online/merge_continuation.py'),
                     str(prefix/app),str(stage),str(combined)]
            with (out/f'{app}_audit.log').open('x') as log:
                active=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                code=active.wait();active=None
            if code:
                state.update(status='audit_failed',exit_code=code);save();return code
            state['applications'][app]['status']='complete';save()
        (out/'summary.json').write_text(json.dumps({app:json.loads((out/f'{app}_combined'/'summary.json').read_text())
                                                 for app in ('spe10','heat')},indent=2)+'\n')
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());state.pop('active_application',None);save()
        return 0
    except KeyboardInterrupt:
        if active is not None:active.wait()
        return 130
    except BaseException as error:
        state.update(status='failed',error=repr(error));save()
        if active is not None and active.poll() is None:active.send_signal(signal.SIGINT);active.wait()
        raise


if __name__=='__main__':sys.exit(main())
