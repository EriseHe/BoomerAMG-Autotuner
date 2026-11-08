"""Local Modules 2 and 3 launcher using unchanged published runners."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))
from experiments import run_all

original_plan = run_all.plan
original_source_identity = run_all.source_identity


def plan(output, module2_source):
    if output != ROOT / 'study' or module2_source != 'diffusion_60':
        raise ValueError('Use the prepared study directory and Diffusion 60 models')
    sources = json.loads((ROOT / 'training_sources.json').read_text())
    stages = []
    for name, argv in original_plan(output, module2_source):
        if not name.startswith(('module2_', 'module3_')):
            continue
        if name == 'module2_init':
            for index in range(1, len(argv)):
                if argv[index - 1] == '--training-run':
                    seed = argv[index].split('=', 1)[0]
                    argv[index] = f'{seed}={sources[seed]}'
        stages.append((name, argv))
    return stages


def source_identity():
    hashes = original_source_identity()
    for path in [Path(__file__), ROOT / 'training_sources.json', ROOT / 'training_source_hashes.json']:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected in json.loads((ROOT / 'training_source_hashes.json').read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Training checkpoint changed: {name}')
    return hashes


if __name__ == '__main__':
    run_all.plan = plan
    run_all.source_identity = source_identity
    sys.argv[1:1] = ['--output', str(ROOT / 'study')]
    run_all.main()
