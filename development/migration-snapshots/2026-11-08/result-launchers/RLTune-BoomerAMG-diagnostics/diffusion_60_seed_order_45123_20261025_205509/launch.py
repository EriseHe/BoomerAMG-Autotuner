"""Local Diffusion 60 rerun using the existing Module 1 experiment runner."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG-diagnostics/results/five_seed_study_20261025_041644/source')
sys.path.insert(0, str(SOURCE))
from experiments import run_all

original_source_identity = run_all.source_identity


def plan(output, module2_source):
    if output != ROOT / 'study':
        raise ValueError('Use the prepared local study directory')
    destination = output / 'module1' / 'diffusion_60'
    return [
        ('module1_diffusion_60_prepare',
         ['-m', 'experiments.paper_final.run_online', 'prepare',
          '--output', str(destination), '--suite', str(ROOT / 'suite.json')]),
        ('module1_diffusion_60_run',
         ['-m', 'experiments.paper_final.run_online', 'run',
          '--output', str(destination)]),
    ]


def source_identity():
    hashes = original_source_identity()
    for path in (Path(__file__), ROOT / 'suite.json'):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


if __name__ == '__main__':
    run_all.plan = plan
    run_all.source_identity = source_identity
    sys.argv[1:1] = ['--output', str(ROOT / 'study')]
    run_all.main()
