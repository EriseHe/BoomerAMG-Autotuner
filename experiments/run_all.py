"""Run Modules 1, 2 and 3 in order using their existing experiment commands."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.runtime import prevent_sleep, single_thread_environment, stop_sleep_prevention

ROOT = Path(__file__).resolve().parents[1]
MODULE1_CASES = ('advection_60', 'advection_40', 'advection_80',
                 'diffusion_40', 'diffusion_60', 'diffusion_80')


def plan(output: Path, module2_source: str):
    """Bind Module 2 to this invocation's completed Module 1 seed directories."""
    stages = []
    for module, cases in ((1, MODULE1_CASES), (3, ('variable_flow_60',))):
        for case in cases:
            for batch in (1, 2):
                name = f'{case}_batch{batch}'
                destination = output/f'module{module}'/name
                for command in ('prepare', 'run'):
                    args = ['-m', 'experiments.paper_final.run_online', command,
                            '--output', str(destination)]
                    if command == 'prepare':
                        args += ['--suite', str(ROOT/f'configs/module{module}/suites/{name}.json')]
                    stages.append((f'module{module}_{name}_{command}', args))
        if module == 1:
            config = output/'module2/configuration.json'
            destination = output/'module2/run'
            args = ['-m', 'experiments.paper_final.run_policy', 'init',
                    '--config', str(config), '--random-seed', '20261019']
            for seed in range(1, 7):
                batch = 1 if seed <= 3 else 2
                source = output/'module1'/f'{module2_source}_batch{batch}'/f'seed_{seed}'
                args += ['--training-run', f'{seed}={source}']
            stages.append(('module2_init', args))
            for command in ('validate', 'prepare', 'run'):
                args = ['-m', 'experiments.paper_final.run_policy', command]
                if command in ('validate', 'prepare'):
                    args += ['--config', str(config)]
                if command in ('prepare', 'run'):
                    args += ['--output', str(destination)]
                stages.append((f'module2_{command}', args))
    return stages


def source_identity():
    """Prevent source or configuration changes between experiment stages."""
    from experiments.paper_final.run_online import source_hashes
    hashes = source_hashes()
    for path in (ROOT/'configs').rglob('*.json'):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes['experiments/run_all.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for the complete study')
    parser.add_argument('--module2-source', choices=MODULE1_CASES, default='diffusion_60',
                        help='Module 1 family/grid supplying all six trained joint models (default: diffusion_60)')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without creating files or running solvers')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    stages = plan(output, args.module2_source)
    if args.dry_run:
        print(json.dumps([dict(stage=name, argv=[sys.executable, *argv]) for name, argv in stages], indent=2))
        return
    if output.exists():
        parser.error('Use a new output directory; existing experiments are never overwritten or resumed')
    environment = single_thread_environment()
    environment['PYTHONPATH'] = str(ROOT)
    # Configure the parent before source inspection imports numerical libraries.
    from experiments.runtime import configure_single_thread
    configure_single_thread()
    identity = source_identity()
    output.mkdir(parents=True)
    logs = output/'logs'
    logs.mkdir()
    (output/'plan.json').write_text(json.dumps(dict(module2_source=args.module2_source,
        source_sha256=identity, stages=[dict(name=name, argv=[sys.executable, *argv])
                                      for name, argv in stages]), indent=2)+'\n')
    state = dict(status='running', completed_commands=0, total_commands=len(stages), current=None)
    guard = prevent_sleep()
    try:
        for index, (name, argv) in enumerate(stages):
            if source_identity() != identity:
                raise RuntimeError('Source or configuration changed during the study')
            state['current'] = name
            (output/'status.json').write_text(json.dumps(state, indent=2)+'\n')
            print(f'[{index+1}/{len(stages)}] {name}', flush=True)
            with (logs/f'{index+1:02d}_{name}.log').open('x') as log:
                subprocess.run([sys.executable, *argv], cwd=ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            state['completed_commands'] = index+1
        state.update(status='complete', current=None)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        (output/'status.json').write_text(json.dumps(state, indent=2)+'\n')
        stop_sleep_prevention(guard)


if __name__ == '__main__':
    main()
