"""Shared-state continuation and unchanged no-recovery solve-RL targets."""
from __future__ import annotations

from functools import partial
import time

from experiments.paper_final.online.configuration import evaluation_index_offset
from experiments.paper_final.online.case_loop import report_online_outcome
from joint.execution import EpisodeRecorder
from joint.recovery import run_deferred_case
from experiments.paper_final.recovery import metrics as sweep
from problems.registry import context_for_setup_method


def execute_learned(runtime, index, previous_update, *, feedback_mode, activation_case=1000):
    from hypre.bindings import augment_setup_params
    from experiments.paper_final.recovery.default_fallback import run_default_fallback
    from solve.core.episode import run_td_episode
    from experiments.paper_final.online.native_evaluation import solve_no_rl_case

    started=time.perf_counter()
    args=runtime.args
    evaluation = getattr(args, "evaluation", None)
    learn = evaluation is None
    schedule_index = index + evaluation_index_offset(args)
    mkw,context=runtime.stream[index]
    active=runtime.bundle is not None and index>=activation_case
    recorder,snapshot_runtime=None,0.0
    if active:
        tick=time.perf_counter()
        recorder=EpisodeRecorder(runtime.bundle.controller, learn=learn)
        snapshot_runtime=time.perf_counter()-tick
    traces=[]
    def native(params):
        return solve_no_rl_case(params=dict(params),mkw=dict(mkw),solver_tol=args.tol,
            solver_max_iter=args.max_cycles,augment_params=augment_setup_params)
    def solve(params):
        if recorder is None:
            raw=native(params)
        else:
            raw=run_td_episode(mkw=dict(mkw),params=dict(params),controller=recorder,
                encoder=runtime.bundle.encoder,problem_context=context,
                solve_tol=args.tol,solve_max_cycles=args.max_cycles,learn=learn,
                explore=True if learn else evaluation.explore,
                record_action_metadata=True,fallback_attempt=None,failure_penalty_sec=0.0)
            raw["recovery_protocol_applied"]=False
        traces.append(raw)
        return raw
    learner_context=context_for_setup_method(problem_kind=args.problem,
        setup_kind=runtime.method.setup_kind,setup_context=runtime.method.setup_context,
        matrix_kwargs=mkw,stream_context=context)
    model=runtime.branch.policy.model
    diagnostic_start=len(model.selection_diagnostics)
    result=run_deferred_case(policy=runtime.branch.policy,
        parameter_space=runtime.branch.parameter_space,context=learner_context,
        solver=solve,fallback_solver=partial(run_default_fallback, mkw=mkw,
            solver_tol=args.tol, solver_max_iter=args.max_cycles),variant=sweep.VARIANT,
        previous_update=previous_update,before_attempt=lambda attempt:runtime.rows.before_attempt(schedule_index,attempt),
        feedback_mode=feedback_mode,recorder=recorder,snapshot_runtime=snapshot_runtime,learn=learn)
    runtime.rows.finish_case(schedule_index)
    for attempt,raw in zip(result.outcome["primary_attempts"],traces):
        attempt.update({key:raw.get(key,[]) for key in sweep.TRACE_FIELDS})
    out=report_online_outcome(result.outcome,bandit_timing=result.timing)
    out["method_wall_runtime"]=time.perf_counter()-started
    row=dict(problem_index=index+1,stream_index=index,method=runtime.name,mkw=mkw,
        context=context.tolist(),params=result.params,outcome=out,rl_active=active,
        shared_prefix=index<activation_case,selection_diagnostics=model.selection_diagnostics[diagnostic_start:])
    beta=None if runtime.bundle is None else runtime.bundle.controller.spec.uncertainty_beta
    sweep.audit_case(row,beta,variant=sweep.VARIANT,feedback_mode=feedback_mode,learn=learn)
    if len(row["selection_diagnostics"])!=out["primary_attempt_count"]:
        raise ValueError("Missing setup selection diagnostic")
    return row,result.update_runtime_sec
