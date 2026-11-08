"""Exercise the normal public four-branch commands from a clean checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
audit = root.parent
original = Path('/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG-diagnostics')
for name in ('hypre/install', 'hypre/interfaces/libamg_runtime.dylib'):
    destination = root/name
    assert not destination.exists()
    destination.symlink_to(original/name, target_is_directory=(original/name).is_dir())
environment = os.environ.copy()
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    environment[key] = '1'
environment['PYTHONPATH'] = str(root)
checks = {}
for family in ('diffusion_40', 'advection_40', 'variable_flow_60'):
    module = 'module3' if family == 'variable_flow_60' else 'module1'
    raw = json.loads((root/'configs'/module/family/'seed_1.json').read_text())
    raw.pop('frozen_stream', None)
    folder = audit/'normal_commands'/family
    folder.mkdir(parents=True)
    config = folder/'config.json'
    config.write_text(json.dumps(raw, indent=2)+'\n')
    suite = folder/'suite.json'
    suite.write_text(json.dumps(dict(name='Public runner validation '+family,
        seeds=[dict(label='1', config=str(config))]), indent=2)+'\n')
    output = folder/'run'
    commands = [
        [sys.executable, '-m', 'experiments.paper_final.run_online', 'prepare',
         '--suite', str(suite), '--output', str(output), '--cases', '5'],
        [sys.executable, '-m', 'experiments.paper_final.run_online', 'run', '--output', str(output)],
    ]
    for command in commands:
        with (folder/(command[3]+'.log')).open('w') as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
    status = json.loads((output/'status.json').read_text())
    audit_result = json.loads((output/'completion_audit.json').read_text())
    assert status['status'] == 'complete' and audit_result['passed']
    assert audit_result['measured_method_cases'] == 20
    result = json.loads((output/'seed_1/result.json').read_text())
    assert result['complete'] and result['passed']
    initial = json.loads((output/'seed_1/initial_audit.json').read_text())
    checks[family] = dict(passed=True, measured_method_cases=20,
        ordinary_prepare_and_run_commands=True, initial_audit_passed=initial['passed'])
    print(json.dumps({family:checks[family]}), flush=True)
assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).splitlines() == ['?? hypre/install']
assert (root/'hypre/install').is_symlink()
assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=root, text=True).strip()
assert not (root/'development').exists()
checks['clean_checkout'] = True
(audit/'clean_checkout_validation.json').write_text(json.dumps(checks, indent=2)+'\n')
