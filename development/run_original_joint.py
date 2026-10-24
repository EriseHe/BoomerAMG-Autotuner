"""Run the archived original algorithm with only Joint and Default measured.

Execute this adapter from an isolated original-source checkout. All algorithm,
feature, recovery and timing-accounting functions come from that checkout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from experiments.runtime import configure_single_thread, THREAD_KEYS
configure_single_thread()
import numpy as np

from experiments.paper_final import run_04_factorial as runner

METHODS = ('default_setup_default_solve', 'bandit_lstdq')
ORIGINAL_COMMIT = '6736efd8c1dfd55082f5cb30833093678048b758'
ADAPTER = 'development/run_original_joint.py'
SOURCE_HASHES = runner.source_hashes


def source_hashes():
    hashes = SOURCE_HASHES()
    hashes[ADAPTER] = runner.file_hash(runner.ROOT/ADAPTER)
    return hashes


def validate_methods(spec):
    by_name = {m.name: m for m in spec.methods}
    if set(by_name) != set(METHODS):
        raise ValueError('This diagnostic measures exactly Joint and Default')
    for name, kinds in zip(METHODS, (('default','default'),('linucb','recursive_lstdq'))):
        method = by_name[name]
        if (method.setup_kind, method.solve_kind) != kinds or method.seed_offset != 0:
            raise ValueError('Incorrect diagnostic method specification')
        if method.solve_tolerance not in (None, spec.solve.state.tol):
            raise ValueError('Methods must share the configured tolerance')
    joint = by_name[METHODS[1]]
    if joint.candidate_sampling != 'structured512' or joint.solve_activation_case != 0:
        raise ValueError('Use the original cold-start candidate protocol')
    if spec.solve.state.encoding_version != 'space_aware_v2':
        raise ValueError('Use the original solve encoder')
    return by_name


def verify_initial_states(stage, reference):
    checked = {}
    for directory in ('initial_bandit_states','initial_controllers'):
        relative = f'{directory}/bandit_lstdq.npz'
        with np.load(stage/relative, allow_pickle=False) as actual, np.load(reference/relative, allow_pickle=False) as expected:
            if set(actual.files) != set(expected.files):
                raise ValueError('Initial checkpoint fields differ from the original run')
            for name in expected.files:
                if actual[name].dtype != expected[name].dtype:
                    raise ValueError(f'Initial checkpoint dtype differs: {relative}: {name}')
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=f'{relative}: {name}')
            checked[relative] = len(expected.files)
    relative = 'initial_controllers/bandit_lstdq.npz.encoder.json'
    if runner.read(stage/relative) != runner.read(reference/relative):
        raise ValueError('Original encoder metadata changed')
    return checked


def initialize(output, label):
    stage = output/f'seed_{label}'
    protocol = runner.read(output/'protocol.json')
    checks = protocol['stages'][label]
    reference = Path(protocol['original_reference'])/f'seed_{label}'
    if runner.read(stage/'status.json')['status'] != 'prepared':
        raise ValueError('Refusing to resume partial learned state')
    if runner.file_hash(stage/'experiment_config.json') != checks['config_sha256']:
        raise ValueError('Prepared configuration changed')
    root = runner.decode(runner.read(stage/'experiment_config.json'), stage)
    if runner.digest(runner.input_rows(root, checks['cases'])) != checks['inputs_sha256']:
        raise ValueError('Input pairing changed')
    study = runner.resolve_composable_study(root.args)
    setup = runner.build_named_setup_branches(root.args, study=study, warmup_instances=(),
                                              stream_hash=root.manifest['sha256'])
    solve = runner.build_composable_solve_runtime(root.args, specs=root.spec.methods)
    if set(setup.branches) != {METHODS[1]} or set(solve.controller_bundles) != {METHODS[1]}:
        raise ValueError('Only Joint should own learners')
    runtimes = {}
    for method in root.spec.methods:
        runtime = SimpleNamespace(**vars(root), name=method.name, method=method,
                                  bundle=solve.controller_bundles.get(method.name))
        if method.name in setup.branches:
            runtime.branch = setup.branches[method.name]
            model = runner.observed_model(runner.use_retry_rule(runtime.branch.policy.model, 'failure_model_adjacent'))
            runtime.branch.policy.model = model
            runtime.rows = runner.CandidateRows(model, cases=len(root.stream), attempts=3, directory=stage/'unused')
            path = stage/'initial_bandit_states'/f'{method.name}.npz'
            path.parent.mkdir(exist_ok=True)
            model.save_mutable_state(path)
        if runtime.bundle is not None:
            path = stage/'initial_controllers'/f'{method.name}.npz'
            path.parent.mkdir(exist_ok=True)
            runtime.bundle.save(path)
        runtimes[method.name] = runtime
    joint = runtimes[METHODS[1]]
    if joint.branch.policy.model.t or joint.bundle.controller.steps or joint.bundle.controller.episodes:
        raise ValueError('Learners trained before measurement')
    if (joint.bundle.encoder.feature_dim, joint.bundle.controller.joint_dim) != (35,315):
        raise ValueError('Expected the original 35 state entries and 315 coefficients')
    checked = verify_initial_states(stage, reference)
    features = runner.factorial.audit_features(joint)
    candidate_hash = runner.file_hash(Path(joint.rows.base._ids.filename))
    if candidate_hash != runner.read(reference/'initial_audit.json')['candidates'][METHODS[1]]:
        raise ValueError('Initial candidate schedule differs from the original')
    runner.dump(stage/'initial_audit.json', dict(passed=True, measured_methods=METHODS,
        features={METHODS[1]:features}, independent_joint_learners=True,
        exact_original_initial_checkpoints=checked,
        controller_metadata=joint.bundle.protocol_metadata()['state_encoder'],
        candidates={METHODS[1]:candidate_hash}, inputs_sha256=checks['inputs_sha256']))
    return runtimes


def worker(output, label):
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError('One MPI rank required')
    stage = output/f'seed_{label}'
    handles = {}
    try:
        runner.check_frozen(output)
        runtimes = initialize(output, label)
        protocol = runner.read(output/'protocol.json')
        cases = protocol['stages'][label]['cases']
        default = runtimes[METHODS[0]]
        runner.dump(stage/'environment.json', dict(pid=os.getpid(), executable=__import__('sys').executable,
            qos=runner.screen.request_performance_qos(), mpi=MPI.Get_library_version(),
            threads={key:os.environ.get(key) for key in THREAD_KEYS}))
        warmup = [runner.sweep.execute_default_case(default, i) for i in range(min(3,cases))]
        runner.dump(stage/'native_warmup.json', dict(measured=False, learner_updates=0, records=warmup))
        runner.dump(stage/'status.json', dict(status='ready',pid=os.getpid(),processed=0,cases=cases))
        while not (output/'release.json').exists():
            time.sleep(.2)
        rows = {name:[] for name in METHODS}
        previous = dict.fromkeys(METHODS, 0.)
        orders = runner.factorial.balanced_orders(cases, default.args.method_order_seed, methods=METHODS)
        (stage/'trajectories').mkdir()
        handles = {name:(stage/'trajectories'/f'{name}.jsonl').open('w') for name in METHODS}
        started = time.perf_counter()
        with (stage/'execution.jsonl').open('w') as schedule:
            for index,order in enumerate(orders):
                for rank,name in enumerate(order):
                    begin = time.monotonic_ns()
                    row, previous[name] = runner.factorial.execute(runtimes[name], index, previous[name])
                    end = time.monotonic_ns()
                    row.update(execution_rank=rank,execution=dict(pid=os.getpid(),position=rank))
                    rows[name].append(row)
                    handles[name].write(json.dumps(runner.ready(row),allow_nan=False)+'\n')
                    schedule.write(json.dumps(dict(input_index=index,method=name,position=rank,
                        start_ns=begin,end_ns=end,pid=os.getpid()))+'\n')
                if (index+1)%default.args.progress_every==0 or index+1==cases:
                    for handle in handles.values():handle.flush()
                    schedule.flush()
                    runner.dump(stage/'status.json',dict(status='running',pid=os.getpid(),processed=index+1,
                        cases=cases,elapsed_sec=time.perf_counter()-started,
                        methods={name:runner.sweep.metrics(records) for name,records in rows.items()}))
                    print(f'seed {label}: {index+1}/{cases}',flush=True)
                if (index+1)%1000==0:
                    runner.save_checkpoints(stage,runtimes,str(index+1))
        for handle in handles.values():handle.close()
        audits={}
        for name,runtime in runtimes.items():
            inputs=[dict(problem_index=row['problem_index'],mkw=row['mkw'],context=row['context']) for row in rows[name]]
            if runner.digest(inputs)!=protocol['stages'][label]['inputs_sha256']:
                raise ValueError('Measured input mismatch')
            controller=runner.controller_summary(runtime)
            if runtime.bundle is not None:runner.audit_controller_counts(rows[name],controller)
            model=runner.screen.audit_final_model(runtime,rows[name]) if hasattr(runtime,'branch') else None
            audits[name]=dict(controller=controller,model=model,cases=len(rows[name]),
                trajectory_sha256=runner.file_hash(stage/'trajectories'/f'{name}.jsonl'))
        runner.save_checkpoints(stage,runtimes,'final')
        runner.check_frozen(output)
        checkpoints={str(p.relative_to(stage)):runner.file_hash(p) for directory in ('checkpoints','final_bandit_states')
                     for p in (stage/directory).iterdir() if p.is_file()}
        runner.dump(stage/'result.json',dict(complete=True,passed=True,cases=cases,audits=audits,
            checkpoint_sha256=checkpoints,elapsed_sec=time.perf_counter()-started,
            methods={name:runner.sweep.metrics(records) for name,records in rows.items()}))
        runner.dump(stage/'status.json',dict(status='complete',processed=cases,cases=cases,
            elapsed_sec=time.perf_counter()-started))
    except BaseException:
        runner.dump(stage/'status.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
    finally:
        for handle in handles.values():handle.close()


def analyze(output):
    protocol=runner.read(output/'protocol.json')
    table=[]
    seeds={}
    for label in protocol['seeds']:
        stage=output/f'seed_{label}'
        result=runner.read(stage/'result.json')
        if not result.get('complete') or not result.get('passed'):
            raise ValueError('Incomplete or unaudited seed')
        config=runner.read(stage/'experiment_config.json')
        if runner.file_hash(stage/'experiment_config.json')!=protocol['stages'][label]['config_sha256']:
            raise ValueError('Configuration changed')
        for relative,digest in result['checkpoint_sha256'].items():
            if runner.file_hash(stage/relative)!=digest:raise ValueError('Checkpoint changed')
        records={}
        for name in METHODS:
            p=stage/'trajectories'/f'{name}.jsonl'
            if runner.file_hash(p)!=result['audits'][name]['trajectory_sha256']:
                raise ValueError('Trajectory changed')
            records[name]=list(runner.read_records(p))
        if {p.stem for p in (stage/'trajectories').glob('*.jsonl')}!=set(METHODS):
            raise ValueError('Unexpected measured branch')
        cases=result['cases']
        orders=runner.factorial.balanced_orders(cases,config['seeds']['method_order'],methods=METHODS)
        events=list(runner.read_records(stage/'execution.jsonl'))
        if len(events)!=2*cases:raise ValueError('Expected exactly two physical measurements per input')
        previous_end=0
        for index,order in enumerate(orders):
            for position,name in enumerate(order):
                event=events[2*index+position]
                if (event['input_index'],event['position'],event['method'])!=(index,position,name):
                    raise ValueError('Execution order mismatch')
                if not previous_end<=event['start_ns']<=event['end_ns']:
                    raise ValueError('Measurements overlapped')
                previous_end=event['end_ns']
        windows={'full':(0,cases),'first_1000':(0,min(1000,cases))}
        if cases>=1000:windows['last_1000']=(cases-1000,cases)
        for window,(start,stop) in windows.items():
            baseline=runner.sweep.metrics(records[METHODS[0]][start:stop])
            for name in METHODS:
                metrics=runner.sweep.metrics(records[name][start:stop])
                table.append(dict(seed=label,window=window,method=name,**metrics,
                    native_reduction_pct=100*(1-metrics['native_sec']/baseline['native_sec']),
                    inclusive_reduction_pct=100*(1-metrics['total_sec']/baseline['total_sec'])))
        seeds[label]=dict(passed=True,measured_method_cases=len(events),serial_execution=True,
                          methods=METHODS,worker_pids=sorted({e['pid'] for e in events}))
    runner.dump(output/'summary.json',dict(passed=True,protocol=protocol,rows=table))
    runner.write_csv(output/'summary.csv',table)
    runner.dump(output/'completion_audit.json',dict(passed=True,seed_audits=seeds,
        measured_method_cases=sum(s['measured_method_cases'] for s in seeds.values()),
        measured_methods=METHODS))
    title='the original paper feature map'
    text=[f'# Current Joint context with {title}','','Native and inclusive savings use the independently measured Default.','',
          '| Seed | Method | Native seconds | Native saving (%) | Inclusive saving (%) | Final failures |',
          '|---|---|---:|---:|---:|---:|']
    for row in table:
        if row['window']=='full':
            text.append(f"| {row['seed']} | {row['method']} | {row['native_sec']:.3f} | {row['native_reduction_pct']:.2f} | {row['inclusive_reduction_pct']:.2f} | {row['failures']} |")
    text+=['', 'Only Joint and Default are constructed and measured. The original 35-entry state, 315 coefficients, cycle bins, clipping, scalar setup descriptors, native-cycle reward, retry policy and fallback learning behavior are retained.', '', 'The full input stream and two-method execution order match the latest cycle-bin and quadratic comparisons. All 151 recorded original source files are unchanged; only this measurement adapter is added.']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    return table

class WorkerCommands:
    def __getattr__(self,name):return getattr(subprocess,name)
    def Popen(self,argv,*args,**kwargs):
        command=list(argv)
        if command[1:4]==['-m','experiments.paper_final.run_04_factorial','worker']:
            command[2]='development.run_original_joint'
        return subprocess.Popen(command,*args,**kwargs)


@contextmanager
def configured_runner():
    with patch.object(runner,'source_hashes',source_hashes), \
         patch.object(runner.factorial,'METHODS',METHODS), \
         patch.object(runner.factorial,'ORDERS',tuple(itertools.permutations(METHODS))), \
         patch.object(runner.factorial,'validate_methods',validate_methods), \
         patch.object(runner,'analyze',analyze), \
         patch.object(runner,'subprocess',WorkerCommands()):
        yield


def prepare(suite, output, *, reference, cases=5000):
    reference = reference.resolve()
    original = runner.read(reference/'source_manifest.json')
    if len(original) != 151 or SOURCE_HASHES() != original:
        raise ValueError('This adapter requires the exact recorded original source')
    if runner.read(reference/'protocol.json')['source_commit'] != ORIGINAL_COMMIT:
        raise ValueError('Incorrect original source identity')
    with configured_runner():
        protocol = runner.prepare(suite,output,cases=cases)
    protocol['enclosing_worktree_head'] = protocol['source_commit']
    protocol.update(source_commit=ORIGINAL_COMMIT, original_reference=str(reference),
        source_identity='All 151 original source files unchanged; two-method measurement adapter added',
        measured_learning_inputs=[1,cases], measured_frozen_inputs=[],
        learning_start='One setup learner and one LSTDQ controller, both cold from input 1',
        concurrency='One single-thread worker per seed; Joint and Default serially in balanced permutation blocks',
        solve_context=dict(state_entries=35,state_action_entries=315,setup_descriptors=8,
            original_feature_map=True,cycle_bins=[1,2,3,5,8,12],original_clipping=True,
            rl_reward='native_cycle',default_fallback_learning='original'),
        diagnostic_scope='Measurement roster and scheduling only; no algorithm changes')
    protocol.pop('fixed_rl_attempts',None)
    runner.dump(output/'protocol.json',protocol)
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','worker'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--cases',type=int,default=5000)
    parser.add_argument('--label')
    args=parser.parse_args();output=args.output.resolve()
    def interrupted(signum,_frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    if args.command=='prepare':
        if args.suite is None or args.reference is None:
            parser.error('prepare requires --suite and --reference')
        prepare(args.suite.resolve(),output,reference=args.reference,cases=args.cases)
    else:
        if runner.read(output/'protocol.json')['source_commit'] != ORIGINAL_COMMIT:
            raise ValueError('Incorrect prepared source identity')
        with configured_runner():
            if args.command=='worker':
                if args.label is None:parser.error('worker requires --label')
                worker(output,args.label)
            else:runner.launch(output)


if __name__=='__main__':main()
