"""Check the production cleanup against the tested encoding and native runner."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from experiments.runtime import single_thread_environment
from experiments.paper_final.common.artifacts import dump, read, now
from experiments.paper_final.run_online import prepare
from experiments.paper_final.online.factorial import audit_execution
from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.space import SetupConfigurationSpace
from solve.controllers.common.state_encoder import SolveStateEncoder, STATE_ENCODING_VERSION

REFERENCE = REPO / 'results/rl_cycle_bin_ablation_20261018/native_1000_3seeds/seed_1_without_bins'
reference_path = REFERENCE / 'source/solve/controllers/common/state_encoder.py'
spec = importlib.util.spec_from_file_location('tested_encoder', reference_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raw = read(REFERENCE / 'seed_1/experiment_config.json')
space = SetupConfigurationSpace.from_mapping(
    'expanded', raw['setup']['configuration_spaces']['expanded_no_endpoints'])
kwargs = dict(tol=1e-6, max_cycles=50, c_max=1000., mode='setup_full',
    setup_obs_encoder=make_setup_feature_encoder(configuration_space=space, parameter_resolution=20),
    problem_context_mode='diffusion3d', weight_bounds=(1., 3.))
original, current = module.SolveStateEncoder(**kwargs), SolveStateEncoder(**kwargs)
assert current.feature_dim == original.feature_dim == 29
count = 0
for method in ('default_setup_lstdq', 'bandit_lstdq'):
    with (REFERENCE / f'seed_1/trajectories/{method}.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            for attempt in row['outcome']['primary_attempts']:
                if not attempt.get('rl_episode_recorded'):
                    continue
                residuals = [1., *attempt['cycle_residuals']]
                times = [0., *attempt['cycle_times']]
                weights = [row['outcome']['initial_environment_weight'], *attempt['cycle_actions']]
                for k in range(len(attempt['cycle_actions'])):
                    obs = dict(mkw=row['mkw'], setup_params=attempt['params'],
                        problem_context=row['context'], initial_residual=1., residual=residuals[k],
                        previous_residual=residuals[max(0,k-1)], cycle=k,
                        last_weight=weights[k], last_cycle_time=times[k])
                    np.testing.assert_array_equal(current.encode(**obs), original.encode(**obs))
                    count += 1
encoder_result = dict(passed=True, states_checked=count,
    identical_to_tested_reduced_encoding=True, cycle_time_retained=True,
    state_features=current.feature_dim, encoding_version=STATE_ENCODING_VERSION)
dump(HERE / 'encoding_equivalence.json', encoder_result)
print(json.dumps(encoder_result), flush=True)

raw['solve']['state_encoding'] = STATE_ENCODING_VERSION
raw['reporting']['progress_every'] = 10
raw['name'] = 'Production cycle feature cleanup smoke test'
config = HERE / 'smoke_config.json'
dump(config, raw)
suite = HERE / 'smoke_suite.json'
dump(suite, dict(name='Cycle feature cleanup native validation',
    seeds=[dict(label='1', config=str(config))]))
output = HERE / 'native_smoke'
prepare(suite, output, cases=10)
shutil.copytree(REFERENCE / 'seed_1/aot_candidate_schedules',
                output / 'seed_1/aot_candidate_schedules')
dump(output / 'release.json', dict(at=now()))
with (output / 'seed_1/worker.log').open('w') as log:
    subprocess.run([sys.executable, '-m', 'experiments.paper_final.run_online', 'worker',
        '--output', str(output), '--label', '1'], cwd=output / 'source',
        env=single_thread_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
audit_execution(output)
result = read(output / 'seed_1/result.json')
assert result['passed'] and result['complete'] and result['cases'] == 10
for method in ('default_setup_lstdq', 'bandit_lstdq'):
    metadata = read(output / f'seed_1/checkpoints/{method}_final.npz.encoder.json')
    assert metadata['encoding_version'] == STATE_ENCODING_VERSION
    assert metadata['feature_dim'] == 29
    assert result['audits'][method]['controller']['joint_feature_dim'] == 261
summary = dict(passed=True, encoder=encoder_result,
    native=dict(inputs_per_branch=10, branches=4,
                methods=result['methods'], state_features=29, joint_features=261))
dump(HERE / 'validation_summary.json', summary)
print(json.dumps(summary), flush=True)
