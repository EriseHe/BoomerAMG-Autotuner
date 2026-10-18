"""Paired, cold-start, 100-input native checks of the controller optimization."""
from pathlib import Path
import json
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from experiments.runtime import single_thread_environment
from experiments.paper_final.common.artifacts import dump, read, file_hash, now
from experiments.paper_final.run_online import prepare
from experiments.paper_final.online.factorial import METHODS, audit_execution

SOURCE_CONFIG = REPO / 'results/shared_setup_tolerance_diffusion40_5000_3seeds_20261018/seed_1/experiment_config.json'
BASELINE = HERE.parent / 'baseline'
CHANGED = ['solve/controllers/recursive_lstdq/controller.py',
           'solve/controllers/recursive_lstdq/common.py']
ORDER = [(1, 'before'), (1, 'after'), (2, 'after'), (2, 'before')]
dump(HERE / 'suite.json', dict(name='100-input diffusion40 controller optimization check',
     seeds=[dict(label='1', config=str(SOURCE_CONFIG))]))
dump(HERE / 'comparison_protocol.json', dict(
     prepared_at=now(), inputs_per_branch=100, methods=METHODS, seed='1',
     repetitions=2, run_order=ORDER, learning='Cold start, learning enabled in every run',
     pairing='Same first 100 inputs, candidate schedule, method order, and RNG seeds',
     timing='Recorded native setup/solve time plus measured setup and solve learner overhead',
     baseline_source=str(BASELINE), changed_source_files=CHANGED,
     native_execution='One serial single-thread worker per run; no concurrent benchmark workers'))

results = []
reference_manifest = None
reference_inputs = None
for repetition, version in ORDER:
    output = HERE / f'round_{repetition}_{version}'
    print(f'Preparing round {repetition} {version}', flush=True)
    protocol = prepare(HERE / 'suite.json', output, cases=100)
    manifest = read(output / 'source_manifest.json')
    if reference_manifest is None:
        reference_manifest = dict(manifest)
        reference_inputs = protocol['stages']['1']['inputs_sha256']
    assert manifest == reference_manifest
    assert protocol['stages']['1']['inputs_sha256'] == reference_inputs
    if version == 'before':
        for name in CHANGED:
            shutil.copy2(BASELINE / Path(name).name, output / 'source' / name)
            manifest[name] = file_hash(output / 'source' / name)
        assert {name for name in manifest if manifest[name] != reference_manifest[name]} == set(CHANGED)
        dump(output / 'source_manifest.json', manifest)
    dump(output / 'benchmark_version.json', dict(version=version, repetition=repetition,
         baseline_captured_before_optimization=True, source_manifest_sha256=file_hash(output / 'source_manifest.json')))
    dump(output / 'release.json', dict(at=now(), serial_single_worker=True))
    print(f'Running round {repetition} {version}: four branches x 100 inputs', flush=True)
    with (output / 'seed_1/worker.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'experiments.paper_final.run_online', 'worker',
                        '--output', str(output), '--label', '1'], cwd=output / 'source',
                       env=single_thread_environment(), stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    audit_execution(output)
    result = read(output / 'seed_1/result.json')
    assert result['passed'] and result['complete'] and result['cases'] == 100
    for method in METHODS:
        metrics = dict(result['methods'][method])
        rows = [json.loads(line) for line in (output / f'seed_1/trajectories/{method}.jsonl').read_text().splitlines()]
        metrics['rl_overhead_sec'] = sum(row['outcome']['infer_runtime'] for row in rows)
        metrics['setup_overhead_sec'] = sum(row['outcome']['bandit_overhead_runtime'] for row in rows)
        assert abs(metrics['total_sec'] - metrics['native_sec'] - metrics['rl_overhead_sec'] - metrics['setup_overhead_sec']) < 1e-8
        results.append(dict(repetition=repetition, version=version, method=method, **metrics))
    dump(output / 'status.json', dict(status='complete', passed=True, at=now()))
    dump(HERE / 'raw_results.json', results)
    print(json.dumps(dict(repetition=repetition, version=version, methods=result['methods'])), flush=True)

summary = []
for method in METHODS:
    entries = {version: [r for r in results if r['method'] == method and r['version'] == version]
               for version in ('before', 'after')}
    means = {version: {field: statistics.mean(r[field] for r in rows)
                      for field in ('total_sec', 'native_sec', 'rl_overhead_sec', 'setup_overhead_sec', 'cycles')}
             for version, rows in entries.items()}
    summary.append(dict(method=method, mean_per_100_inputs=means,
          total_reduction_pct=100*(1-means['after']['total_sec']/means['before']['total_sec']),
          rl_overhead_reduction_pct=(100*(1-means['after']['rl_overhead_sec']/means['before']['rl_overhead_sec'])
                                     if means['before']['rl_overhead_sec'] else None),
          paired_total_reduction_pct=[100*(1-entries['after'][i]['total_sec']/entries['before'][i]['total_sec']) for i in range(2)],
          final_failures={version: sum(r['failures'] for r in rows) for version, rows in entries.items()}))
dump(HERE / 'summary.json', dict(passed=True, inputs_per_branch=100, seed='1', repetitions=2,
     results=summary, interpretation='Short online-learning test; measured timing feeds back into both learners, so selected configurations and cycle counts may differ.'))
print('FINAL SUMMARY ' + json.dumps(summary), flush=True)
