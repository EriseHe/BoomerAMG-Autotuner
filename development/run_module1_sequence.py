"""Execute a declared Module 1 sequence through the existing prepare/run CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_status(path, value):
    value['at'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


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
        for number in range(len(plan['tasks'])):
            plan = json.loads(args.plan.read_text())
            task = plan['tasks'][number]
            output_status = Path(task['output']) / 'status.json'
            existing = json.loads(output_status.read_text()) if output_status.exists() else {}
            if existing.get('status') == 'complete':
                state['completed'].append(task['name'])
                continue
            while task.get('gate') and not Path(task['gate']).exists():
                state.update(status='waiting_for_runner_update', current=task['name'], phase=task['phase'])
                write_status(status, state)
                time.sleep(5)
                plan = json.loads(args.plan.read_text())
                task = plan['tasks'][number]
            if existing.get('status') in ('running', 'initializing'):
                state.update(status='running', current=task['name'], phase=task['phase'], adopted_pid=existing['pid'])
                write_status(status, state)
                while True:
                    existing = json.loads(output_status.read_text())
                    if existing.get('status') == 'complete':
                        break
                    if existing.get('status') == 'failed' or not process_exists(existing['pid']):
                        raise RuntimeError(f"Existing suite stopped before completion: {task['name']}")
                    time.sleep(5)
                state.pop('adopted_pid', None)
                state['completed'].append(task['name'])
                write_status(status, state)
                continue
            if existing:
                raise RuntimeError(f"Refusing to overwrite an existing suite: {task['name']}")
            state.update(status='running', current=task['name'], phase=task['phase'])
            write_status(status, state)
            for command in ('prepare', 'run'):
                argv = [sys.executable, '-u', '-m', 'experiments.paper_final.run_online', command,
                        '--output', task['output']]
                if command == 'prepare':
                    argv.extend(['--suite', task['suite']])
                with (directory / (task['name'] + '.log')).open('a') as log:
                    environment = dict(os.environ, PYTHONPATH=task.get('execution_root', plan['execution_root']))
                    subprocess.run(argv, cwd=environment['PYTHONPATH'], env=environment,
                                   check=True, stdout=log, stderr=subprocess.STDOUT)
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
