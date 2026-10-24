"""Measure Joint and Default with a prepared compact solve basis."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
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

from development.compact_context import CompactSolveStateEncoder
from development.cycle_bins import CycleBinSolveStateEncoder, CYCLE_BOUNDS
from development.quadratic_context import QuadraticSolveStateEncoder
from development.setup_normalization import make_range_normalized_setup_encoder
from experiments.paper_final import run_online as runner
from experiments.paper_final.online import methods
from solve.controllers.common import config as state_config

METHODS = ('default_setup_default_solve', 'bandit_lstdq')
SOURCE_FILES = ('development/compact_context.py', 'development/cycle_bins.py',
                'development/quadratic_context.py',
                'development/setup_normalization.py', 'development/run_joint_context.py')
SOURCE_HASHES = runner.source_hashes
CONTEXT_ENCODERS = {'log_time': CompactSolveStateEncoder,
                    'cycle_bins': CycleBinSolveStateEncoder,
                    'quadratic': QuadraticSolveStateEncoder}
ADDITIONAL_CONTEXT_ENTRIES = {'log_time': 0, 'cycle_bins': 14, 'quadratic': 2}


def prepared_context(protocol):
    context = protocol['solve_context']
    variant = context['variant']
    # Earlier Advection snapshots omit these fields and use six problem entries
    # and nine action basis functions. Diffusion retains its configured context.
    state_entries = (7 + context.get('problem_entries', 6) + context['setup_descriptors']
                     + ADDITIONAL_CONTEXT_ENTRIES[variant])
    expected = (state_entries, state_entries * context.get('action_basis_entries', 9))
    if (context['state_entries'], context['state_action_entries']) != expected:
        raise ValueError('Prepared solve context dimensions do not match its basis')
    return variant


def source_hashes():
    hashes = SOURCE_HASHES()
    hashes.update({name: runner.file_hash(runner.ROOT/name) for name in SOURCE_FILES})
    return hashes


def validate_methods(spec):
    by_name = {m.name: m for m in spec.methods}
    if set(by_name) != set(METHODS):
        raise ValueError('This diagnostic measures exactly Joint and Default')
    for name, kinds in zip(METHODS, (('default', 'default'), ('linucb', 'recursive_lstdq'))):
        method = by_name[name]
        if (method.setup_kind, method.solve_kind) != kinds or method.seed_offset != 0:
            raise ValueError('Incorrect diagnostic method specification')
        if method.solve_tolerance not in (None, spec.solve.state.tol):
            raise ValueError('Methods must share the configured tolerance')
    joint = by_name[METHODS[1]]
    if joint.candidate_sampling != 'structured512' or joint.solve_activation_case != 0:
        raise ValueError('Use cold learning from input one and the standard candidate schedule')
    if spec.solve.state.encoding != runner.STATE_ENCODING:
        raise ValueError('Use the current shared setup encoding')
    if spec.frozen_stream is not None or spec.evaluation is not None:
        raise ValueError('This diagnostic measures learning inputs only')
    return by_name


def build_solve_runtime(args, *, specs, context_variant='cycle_bins'):
    with patch.object(state_config, 'SolveStateEncoder', CONTEXT_ENCODERS[context_variant]), \
         patch.object(methods, 'make_setup_feature_encoder', make_range_normalized_setup_encoder):
        built = methods.build_composable_solve_runtime(args, specs=specs)
    bundles = {}
    for name, bundle in built.controller_bundles.items():
        metadata = bundle.protocol_metadata()
        metadata['state_encoder'].update(bundle.encoder.metadata())
        bundles[name] = replace(bundle, _protocol_metadata=metadata)
    return methods.ComposableSolveRuntime(controller_bundles=bundles)


def initialize(output, label):
    stage = output/f'seed_{label}'
    protocol = runner.read(output/'protocol.json')
    context_variant = prepared_context(protocol)
    checks = protocol['stages'][label]
    if runner.read(stage/'status.json')['status'] != 'prepared':
        raise ValueError('Refusing to resume partial learned state')
    if runner.file_hash(stage/'experiment_config.json') != checks['config_sha256']:
        raise ValueError('Prepared configuration changed')
    root = runner.decode(runner.read(stage/'experiment_config.json'), stage)
    if runner.digest(runner.input_rows(root, checks['cases'])) != checks['inputs_sha256']:
        raise ValueError('Input pairing changed')
    if root.args.evaluation is not None or root.freeze_after is not None:
        raise ValueError('Learning-only diagnostic required')
    study = runner.resolve_composable_study(root.args)
    setup = runner.build_named_setup_branches(root.args, study=study)
    if set(setup) != {METHODS[1]}:
        raise ValueError('Only Joint should own a setup learner')
    solve = build_solve_runtime(root.args, specs=root.spec.methods, context_variant=context_variant)
    if set(solve.controller_bundles) != {METHODS[1]}:
        raise ValueError('Only Joint should own an RL controller')
    runtimes = {}
    for method in root.spec.methods:
        runtime = SimpleNamespace(**vars(root), name=method.name, method=method,
                                  bundle=solve.controller_bundles.get(method.name))
        if method.name in setup:
            runtime.branch = setup[method.name]
            model = runner.observed_model(runner.FailureAwareRetryLinUCB.from_runtime_model(runtime.branch.policy.model))
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
    expected = (protocol['solve_context']['state_entries'],
                protocol['solve_context']['state_action_entries'])
    if (joint.bundle.encoder.feature_dim, joint.bundle.controller.joint_dim) != expected:
        raise ValueError(f'Expected {expected[0]} state entries and {expected[1]} state-action coefficients')
    if joint.bundle.encoder.setup_obs_encoder.descriptor_dim != 19:
        raise ValueError('Expected 19 basic normalized setup descriptors')
    features = runner.factorial.audit_features(joint)
    runner.dump(stage/'initial_audit.json', dict(passed=True, measured_methods=METHODS,
        features={METHODS[1]:features}, independent_joint_learners=True,
        controller_metadata=joint.bundle.protocol_metadata()['state_encoder'],
        candidates={METHODS[1]:runner.file_hash(Path(joint.rows.base._ids.filename))},
        inputs_sha256=checks['inputs_sha256']))
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
            qos=runner.request_performance_qos(), mpi=MPI.Get_library_version(),
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
    context_variant=prepared_context(protocol)
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
    title={'log_time':'logarithmic cycle time', 'quadratic':'quadratic cycle-progress basis',
           'cycle_bins':'cycle bins'}[context_variant]
    text=[f'# Current Joint context with {title}','','Native and inclusive savings use the independently measured Default.','',
          '| Seed | Method | Native seconds | Native saving (%) | Inclusive saving (%) | Final failures |',
          '|---|---|---:|---:|---:|---:|']
    for row in table:
        if row['window']=='full':
            text.append(f"| {row['seed']} | {row['method']} | {row['native_sec']:.3f} | {row['native_reduction_pct']:.2f} | {row['inclusive_reduction_pct']:.2f} | {row['failures']} |")
    basis_description={
        'log_time':f"The compact state has {protocol['solve_context']['state_entries']} entries, with no cycle bins or added quadratic terms.",
        'quadratic':'The only added state terms are (k/K_max)^2 and (k/K_max) times the normalized residual gap; no cycle bins are used.',
        'cycle_bins':'Seven cycle indicators and their seven residual-gap products are added.',
    }[context_variant]
    text+=['',f'Only Joint and Default are constructed and measured. {basis_description} The 19 normalized setup descriptors, uncapped tolerance residual features and absolute log time are retained; no setup-model predictions are used.', '', 'Earlier four-branch runs used a different within-input measurement schedule. Comparisons with those results include timing and run-order variation.']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    return table


class WorkerCommands:
    def __getattr__(self,name):return getattr(subprocess,name)
    def Popen(self,argv,*args,**kwargs):
        command=list(argv)
        if command[1:4]==['-m','experiments.paper_final.run_online','worker']:
            command[2]='development.run_joint_context'
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


def prepare(suite,output,cases=5000,*,context_variant='cycle_bins'):
    dimensions = None
    for item in runner.read(suite)['seeds']:
        config_path = Path(item['config'])
        if not config_path.is_absolute():
            config_path = suite.parent/config_path
        spec = runner.parse_joint_experiment_config(runner.read(config_path))
        validate_methods(spec)
        args = runner.runtime_config_from_spec(spec)
        bundle = build_solve_runtime(args, specs=spec.methods,
            context_variant=context_variant).controller_bundles[METHODS[1]]
        current = (bundle.encoder.feature_dim, bundle.controller.joint_dim,
                   bundle.encoder.problem_feature_dim,
                   bundle.controller.joint_dim // bundle.encoder.feature_dim)
        if dimensions is not None and dimensions != current:
            raise ValueError('One suite must use the same solve context dimensions')
        dimensions = current
    if dimensions is None:
        raise ValueError('Suite needs at least one seed')
    state_entries, state_action_entries, problem_entries, action_basis_entries = dimensions
    with configured_runner():
        protocol=runner.prepare(suite,output,cases=cases)
    protocol.pop('fixed_rl_attempts', None)
    protocol.update(measured_learning_inputs=[1,cases],measured_frozen_inputs=[],
        solve_context=dict(variant=context_variant,state_entries=state_entries,
            state_action_entries=state_action_entries,setup_descriptors=19,
            problem_entries=problem_entries,action_basis_entries=action_basis_entries,
            numerical_setup_normalization='fixed allowed range [-1,1]',categorical='reference one-hot',
            residual_normalization='log initial residual over tolerance',residual_clipping=False,
            cycle_time='log1p(last_native_cycle_seconds / 0.001)',cycle_time_clipping=False,
            setup_model_predictions=False,rl_reward='native_cycle'),
        execution='Only Joint and Default; balanced serial order within each input',
        concurrency='One single-thread worker per seed; Joint and Default serially in balanced permutation blocks',
        learning_start='One setup learner and one LSTDQ controller, both cold from input 1')
    if context_variant=='cycle_bins':
        protocol['solve_context'].update(cycle_bins=list(CYCLE_BOUNDS),bin_indicators=7,bin_gap_interactions=7)
    elif context_variant=='quadratic':
        protocol['solve_context'].update(cycle_bins=False,quadratic_degree=2,
            additional_state_terms=['cycle_fraction_squared','cycle_fraction_times_residual_gap'])
    else:
        protocol['solve_context'].update(cycle_bins=False,additional_state_terms=[])
    prepared_context(protocol)
    runner.dump(output/'protocol.json',protocol)
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','worker'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',type=Path)
    parser.add_argument('--cases',type=int,default=5000)
    parser.add_argument('--label')
    parser.add_argument('--context',choices=tuple(CONTEXT_ENCODERS))
    args=parser.parse_args();output=args.output.resolve()
    def interrupted(signum,_frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    if args.command=='prepare':
        if args.suite is None:parser.error('prepare requires --suite')
        prepare(args.suite.resolve(),output,args.cases,context_variant=args.context or 'cycle_bins')
    else:
        if args.context is not None:
            parser.error('--context is selected only during preparation')
        prepared_context(runner.read(output/'protocol.json'))
        with configured_runner():
            if args.command=='worker':
                if args.label is None:parser.error('worker requires --label')
                worker(output,args.label)
            else:runner.launch(output)


if __name__=='__main__':main()
