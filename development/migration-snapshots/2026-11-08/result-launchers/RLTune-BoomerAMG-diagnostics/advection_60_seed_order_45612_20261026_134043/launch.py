"""Local Module 1 rerun using the previously tested source and experiment runner."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG-diagnostics/results/five_seed_study_20261025_041644/source')
sys.path.insert(0, str(SOURCE))
from experiments import run_all

CASES = ('advection_60', 'advection_40', 'diffusion_40')
original_source_identity = run_all.source_identity


def plan(output, module2_source):
    if output != ROOT / 'study':
        raise ValueError('Use the prepared local study directory')
    stages = []
    for case in CASES:
        destination = output / 'module1' / case
        for command in ('prepare', 'run'):
            argv = ['-m', 'experiments.paper_final.run_online', command,
                    '--output', str(destination)]
            if command == 'prepare':
                argv += ['--suite', str(ROOT / 'configurations' / f'{case}.json')]
            stages.append((f'module1_{case}_{command}', argv))
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
