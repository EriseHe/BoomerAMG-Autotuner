"""Execute a declared Module 1 sequence through the existing prepare/run CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def write_status(path, value):
    value['at'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    directory = args.plan.parent
    status = directory / 'status.json'
    state = dict(status='starting', pid=os.getpid(), completed=[], total=len(plan['tasks']))
    guard = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())])
    try:
        for task in plan['tasks']:
            state.update(status='running', current=task['name'], phase=task['phase'])
            write_status(status, state)
            for command in ('prepare', 'run'):
                argv = [sys.executable, '-u', '-m', 'experiments.paper_final.run_online', command,
                        '--output', task['output']]
                if command == 'prepare':
                    argv.extend(['--suite', task['suite']])
                with (directory / (task['name'] + '.log')).open('a') as log:
                    subprocess.run(argv, cwd=plan['execution_root'], check=True,
                                   stdout=log, stderr=subprocess.STDOUT)
            state['completed'].append(task['name'])
            write_status(status, state)
        state.update(status='complete', current=None, phase=None)
        write_status(status, state)
    except BaseException:
        state.update(status='failed', traceback=traceback.format_exc())
        write_status(status, state)
        raise
    finally:
        guard.terminate()
        guard.wait()


if __name__ == '__main__':
    main()
