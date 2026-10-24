"""Run prepared cycle-time variants and an archived paper baseline serially."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from experiments.runtime import configure_single_thread, single_thread_environment, prevent_sleep, stop_sleep_prevention
configure_single_thread()
from experiments.paper_final.common.artifacts import dump, read, now, write_csv


def sequence(output):
    plan = read(output/'plan.json')
    with (output/'sequence.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    state = dict(status='running', pid=os.getpid(), completed=[], current=None, started_at=now(), executions=[])
    child = None
    guard = prevent_sleep()
    try:
        for item in plan['runs']:
            name = item['name']
            path = output/name
            if read(path/'status.json')['status'] != 'prepared':
                raise ValueError('Refusing to restart existing or partial experiment: '+name)
            state.update(current=name, at=now())
            execution = dict(name=name, start_monotonic=time.monotonic(), started_at=now())
            environment = single_thread_environment()
            environment['PYTHONPATH'] = item['source']
            with (output/f'{name}.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', item['module'], 'run', '--output', str(path)],
                    cwd=item['source'], env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['child_pid'] = child.pid
                dump(output/'status.json', state)
                code = child.wait()
            if code:
                raise RuntimeError(f'{name} failed with exit code {code}; inspect its log')
            child = None
            execution.update(end_monotonic=time.monotonic(), finished_at=now())
            if not read(path/'status.json').get('passed'):
                raise RuntimeError(name+' did not pass completion checks')
            if read(path/'seed_1/status.json').get('processed') != plan['cases']:
                raise RuntimeError(name+' did not complete all requested inputs')
            state['executions'].append(execution)
            state['completed'].append(name)
            state.pop('child_pid', None)
            dump(output/'status.json', state)
        rows=[]
        for item in plan['runs']:
            for row in read(output/item['name']/'summary.json')['rows']:
                if row['window']=='full':
                    rows.append(dict(variant=item['name'], **row))
        # Current/ratio candidate preparation and input pairing must remain identical.
        first, second = (output/item['name'] for item in plan['runs'][:2])
        schedules=[]
        import hashlib
        for path in (first, second):
            directory = path/'seed_1/aot_candidate_schedules'
            files = {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in directory.rglob('*') if p.is_file() and p.suffix in ('.npy','.npz')}
            if not files:
                raise ValueError('Candidate schedule arrays are missing')
            schedules.append(files)
        if schedules[0] != schedules[1]:
            raise ValueError('Current and ratio candidate schedules differ')
        executions=state['executions']
        if any(a['end_monotonic']>b['start_monotonic'] for a,b in zip(executions,executions[1:])):
            raise ValueError('Measurements overlapped')
        dump(output/'comparison.json', dict(passed=True, rows=rows,
             caveat='One seed and fixed serial order; original paper differs in more than cycle-time context.'))
        write_csv(output/'comparison.csv', rows)
        dump(output/'completion_audit.json', dict(passed=True, all_runs_complete=True,
            nonoverlapping_execution=True, paired_current_ratio_candidate_arrays=len(schedules[0]),
            executions=executions))
        lines=['# Cycle-time context and original paper comparison','',
               'Advection 40, seed 1, 5,000 learning inputs, four branches; variants measured serially.','',
               '| Variant | Method | Native saving (%) | Saving including overhead (%) |',
               '|---|---|---:|---:|']
        for row in rows:
            lines.append(f"| {row['variant']} | {row['method']} | {row['native_reduction_pct']:.2f} | {row['inclusive_reduction_pct']:.2f} |")
        lines.extend(['','Each variant uses its own measured Default. The current/ratio pair differs only in the timing feature; the original paper is a full historical algorithm baseline. One seed and serial order do not establish general superiority.'])
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='complete', passed=True, current=None, at=now())
        dump(output/'status.json',state)
    except BaseException:
        state.update(status='failed', at=now(), traceback=traceback.format_exc())
        dump(output/'status.json',state)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        stop_sleep_prevention(guard)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    def interrupted(signum,_frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    sequence(args.output.resolve())


if __name__=='__main__':
    main()
