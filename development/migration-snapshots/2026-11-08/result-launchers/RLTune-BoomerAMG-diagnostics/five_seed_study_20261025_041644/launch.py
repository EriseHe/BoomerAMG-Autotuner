"""Local five-seed study using the unchanged published experiment runners."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))
from experiments import run_all

CASES = ('advection_80', 'diffusion_80', 'advection_60',
         'diffusion_60', 'advection_40', 'diffusion_40')
original_source_identity = run_all.source_identity


def plan(output, module2_source):
    if output != ROOT / 'study' or module2_source != 'diffusion_60':
        raise ValueError('This local launcher uses its prepared study directory and Diffusion 60 models')
    stages = []
    for module, cases in ((1, CASES), (3, ('variable_flow_60',))):
        for case in cases:
            destination = output / f'module{module}' / case
            for command in ('prepare', 'run'):
                argv = ['-m', 'experiments.paper_final.run_online', command,
                        '--output', str(destination)]
                if command == 'prepare':
                    argv += ['--suite', str(ROOT / 'configurations' / f'{case}.json')]
                stages.append((f'module{module}_{case}_{command}', argv))
        if module == 1:
            for command in ('validate', 'prepare', 'run'):
                argv = ['-m', 'experiments.paper_final.run_policy', command]
                if command in ('validate', 'prepare'):
                    argv += ['--config', str(ROOT / 'configurations/module2.json')]
                if command in ('prepare', 'run'):
                    argv += ['--output', str(output / 'module2/run')]
                stages.append((f'module2_{command}', argv))
    return stages


def source_identity():
    hashes = original_source_identity()
    for path in [Path(__file__), *sorted((ROOT / 'configurations').glob('*.json'))]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


if __name__ == '__main__':
    run_all.plan = plan
    run_all.source_identity = source_identity
    sys.argv[1:1] = ['--output', str(ROOT / 'study')]
    run_all.main()
