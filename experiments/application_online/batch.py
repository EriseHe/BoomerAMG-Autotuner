"""Run both 1,000-system campaigns sequentially and audit each on completion."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',cases_per_application=1000,max_cycles=100,
               pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),applications={})
    active=None

    def save():
        temporary=output/'status.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(output/'status.json')

    def interrupt(signum,frame):
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGINT)
        state['status']='interrupted';save()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGTERM,interrupt)
    save()
    try:
        for application in ('spe10','heat'):
            state['active_application']=application
            stage=output/application
            with (output/f'{application}.log').open('x') as log:
                command=[sys.executable,str(ROOT/'experiments/application_online/run.py'),
                    '--application',application,'--cases','1000','--max-cycles','100',
                    '--steps-per-scenario','100','--output',str(stage)]
                active=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state['applications'][application]=dict(status='running',pid=active.pid,output=str(stage))
                save()
                code=active.wait();active=None
            if code:
                state['status']='failed'
                state['applications'][application].update(status='failed',exit_code=code)
                save();return code
            state['applications'][application]['status']='auditing';save()
            with (output/f'{application}_audit.log').open('x') as log:
                active=subprocess.Popen([sys.executable,str(ROOT/'experiments/application_online/audit.py'),str(stage)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                code=active.wait();active=None
            if code:
                state['status']='audit_failed'
                state['applications'][application].update(status='audit_failed',exit_code=code)
                save();return code
            state['applications'][application]['status']='complete';save()
        summaries={app:json.loads((output/app/'summary.json').read_text()) for app in ('spe10','heat')}
        (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        state['status']='complete';state.pop('active_application',None)
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        return 0
    except KeyboardInterrupt:
        if active is not None:
            active.wait()
        return 130
    except BaseException as error:
        state.update(status='failed',error=repr(error));save()
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGINT);active.wait()
        raise


if __name__=='__main__':sys.exit(main())
