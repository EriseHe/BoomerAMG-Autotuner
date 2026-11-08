"""Validate the temporary launcher in a separate, deliberately tiny study."""
from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

LAUNCH = Path(__file__).resolve().parents[1]
namespace = runpy.run_path(str(LAUNCH / 'launch.py'))
runner = namespace['run_all']
from experiments.runtime import configure_single_thread
configure_single_thread()
from experiments.paper_final import run_online as online


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    base = Path(tempfile.mkdtemp(prefix='transition_smoke_', dir=LAUNCH / 'validation'))
    output = base / 'study'
    suites = {}
    for batch in (1, 2):
        source_suite = LAUNCH / 'source/configs/module3/suites' / f'variable_flow_60_batch{batch}.json'
        suite = read(source_suite)
        for item in suite['seeds']:
            raw = read((source_suite.parent / item['config']).resolve())
            raw['problem']['grid'] = [12, 12, 12]
            raw['stream'].update(cases=2, online_cases=2, group_take=2, expected_sha256='')
            spec = online.parse_joint_experiment_config(raw)
            args = online.runtime_config_from_spec(spec)
            online.configure_paired_environment(args)
            stream, manifest = online.build_paired_instance_stream(args)
            assert len(stream) == 2
            raw['stream']['expected_sha256'] = manifest['sha256']
            config = base / 'configs' / f"seed_{item['label']}.json"
            write(config, raw)
            item['config'] = str(config)
        suites[batch] = base / 'configs' / f'batch{batch}.json'
        write(suites[batch], suite)

    stages = namespace['plan'](LAUNCH / 'study', 'diffusion_60')
    for name, argv in stages:
        for index, value in enumerate(argv):
            if value.startswith(str(LAUNCH / 'study')):
                argv[index] = value.replace(str(LAUNCH / 'study'), str(output), 1)
        if name == 'module2_init':
            argv += ['--cases', '2']
        if name.endswith('_prepare') and name.startswith('module3_'):
            batch = 1 if 'batch1' in name else 2
            argv[argv.index('--suite') + 1] = str(suites[batch])

    actual_run = subprocess.run
    completed = []

    def execute(command, **kwargs):
        result = actual_run(command, **kwargs)
        if command[1:4] == ['-m', 'experiments.paper_final.run_policy', 'init']:
            config = Path(command[command.index('--config') + 1])
            cfg = read(config)
            training_config = Path(cfg['training_runs'][0]['run_dir']) / 'experiment_config.json'
            cfg.update(weights=[1.0, 2.0], scan_extra_repetitions=0, scan_repeat_fraction=0.0,
                       evaluation_repetitions=1,
                       problem=dict(read(training_config)['problem'], grid=[12, 12, 12]))
            write(config, cfg)
        completed.append(command)
        return result

    begin = time.monotonic()
    with patch.object(runner, 'plan', return_value=stages), \
         patch.object(runner, 'source_identity', side_effect=namespace['source_identity']), \
         patch.object(runner.subprocess, 'run', side_effect=execute), \
         patch('sys.argv', ['run_all', '--output', str(output)]):
        runner.main()

    status = read(output / 'status.json')
    assert status['status'] == 'complete' and status['completed_commands'] == 8
    assert len(completed) == 8
    module2 = read(output / 'module2/run/status.json')
    assert module2['status'] == 'complete'
    summary = read(output / 'module2/run/analysis/summary.json')
    assert summary['audit']['exact_coverage']
    for batch in (1, 2):
        path = output / 'module3' / f'variable_flow_60_batch{batch}'
        assert read(path / 'status.json')['status'] == 'complete'
        labels = ['1', '2', '3'] if batch == 1 else ['4', '5', '6']
        assert read(path / 'protocol.json')['seeds'] == labels
        for label in labels:
            result = read(path / f'seed_{label}/result.json')
            assert result['passed'] and result['cases'] == 2
            assert len(result['methods']) == 4

    # Verify a failed Module 2 command prevents any Module 3 command from running.
    failure_output = base / 'failure_stop_check'
    failure_stages = [(name, list(argv)) for name, argv in stages]
    failed_calls = []

    def fail_module2(command, **kwargs):
        failed_calls.append(command)
        if command[1:4] == ['-m', 'experiments.paper_final.run_policy', 'run']:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    with patch.object(runner, 'plan', return_value=failure_stages), \
         patch.object(runner, 'source_identity', return_value={'smoke': 'unchanged'}), \
         patch.object(runner, 'prevent_sleep', return_value=None), \
         patch.object(runner.subprocess, 'run', side_effect=fail_module2), \
         patch('sys.argv', ['run_all', '--output', str(failure_output)]):
        try:
            runner.main()
        except subprocess.CalledProcessError:
            pass
        else:
            raise AssertionError('Module 2 failure was swallowed')
    failed = read(failure_output / 'status.json')
    assert failed['status'] == 'failed' and failed['current'] == 'module2_run'
    assert len(failed_calls) == 4
    assert not any('experiments.paper_final.run_online' in argv for argv in failed_calls)
    assert not (LAUNCH / 'study').exists(), 'The full experiment must remain unstarted'
    report = dict(passed=True, full_experiment_started=False, source=str(LAUNCH / 'source'),
                  study=str(output), seconds=time.monotonic()-begin,
                  actual_production_stage_order=[name for name, _ in stages],
                  module2_seeds=[1, 2, 3, 4, 5, 6], module2_cases_per_seed=2,
                  module2_weights=[1.0, 2.0], grid=[12, 12, 12],
                  module3_seeds=[1, 2, 3, 4, 5, 6], module3_cases_per_seed=2,
                  module3_branches=4, module2_failure_blocks_module3=True)
    write(LAUNCH / 'validation/transition_smoke_result.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
