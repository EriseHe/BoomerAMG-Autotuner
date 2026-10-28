"""Coordinate learned setup selection, solves, retries, and feedback."""
from __future__ import annotations

from dataclasses import dataclass
import time

from hypre.bindings import RecoveryOutcome, execute_attempt
from setup.learners.linucb.feedback import DeferredCostLinUCB, setup_cost_target
from setup.space import DEFAULT_SETUP_PARAMS


@dataclass(frozen=True)
class SetupReselectionResult:
    params: dict[str, object]
    outcome: dict[str, object]
    timing: dict[str, float]
    fallback_used: int
    update_runtime_sec: float


def run_deferred_case(*, policy, parameter_space, context, solver,
                      fallback_solver, variant, previous_update, before_attempt,
                      feedback_mode, recorder=None, snapshot_runtime=0.0, learn=True):
    """Resolve adjacent or full remaining costs once, after the case finishes.

    Successful attempts receive their own cost. Default fallback time also
    belongs to the preceding cost sequence. A successful fallback adds one
    setup observation; a failed fallback adds none.
    In adjacent mode, failed attempts accumulate costs through the next
    attempt whose construction succeeds, or through the final fallback.
    All valid setup observations survive final case failure. Optional solve RL
    keeps its existing attempt costs, with no recovery augmentation or replay;
    only RL learning is rolled back when the entire case fails.
    """
    model = policy.model
    if not isinstance(model, DeferredCostLinUCB):
        raise TypeError("Delayed feedback requires a deferred-cost setup learner")
    if feedback_mode not in ("adjacent_delayed", "suffix_delayed"):
        raise ValueError("Unknown delayed-cost rule")
    if variant.rl_recovery_cost or not variant.default_fallback or variant.attempts < 1:
        raise ValueError("Expected learned attempts plus default, without RL recovery cost")
    if recorder is None and snapshot_runtime:
        raise ValueError("A snapshot cost requires an RL recorder")
    attempts, params, arms, selections, costs, observations = [], [], [], [], [], []
    raw_attempts = []
    evaluations, updates = [], []
    fallback_update_runtime = 0.0
    fallback = None
    fallback_observation = None
    try:
        for index in range(variant.attempts):
            tick = time.perf_counter()
            before_attempt(index)
            if index == 0:
                choice, _ = policy.select(context=context, parameter_space=parameter_space)
            else:
                choice, _ = model.select_recovery(context, excluded_arms=arms)
            selections.append(time.perf_counter()-tick)
            arm = int(model._last_arm)
            if arm in arms:
                raise AssertionError("Same failed setup selected twice on one problem")
            arms.append(arm)
            params.append(dict(choice))
            if recorder is not None:
                recorder.attempt_index, recorder.current = index, None
            attempt = execute_attempt(lambda: solver(choice), require_valid_observation=True)
            attempts.append(attempt)
            raw_attempts.append(dict(attempt.result))
            tick = time.perf_counter()
            costs.append(attempt.end_to_end_runtime_sec + selections[-1] + previous_update)
            evaluations.append(time.perf_counter()-tick)
            tick = time.perf_counter()
            if learn:
                observations.append(model.defer_cost(attempt.succeeded))
                updates.append(time.perf_counter()-tick)
            else:
                model.cancel_pending()
                updates.append(0.0)
            if attempt.succeeded:
                break
        if not attempts[-1].succeeded:
            fallback = execute_attempt(fallback_solver, require_valid_observation=True)
        recovery = RecoveryOutcome(primary_attempts=tuple(attempts), fallback=fallback)
        tick = time.perf_counter()
        sequence = costs + ([] if fallback is None else [fallback.end_to_end_runtime_sec])
        targets = []
        if learn:
            # Keep shared sequence preparation in the first target's interval.
            for i in range(len(attempts)):
                targets.append(setup_cost_target(attempts, sequence, i, feedback_mode))
                end = time.perf_counter()
                evaluations[i] += end-tick
                tick = end
            evaluations[-1] += time.perf_counter()-tick
        else:
            targets = [None]*len(attempts)
        tick = time.perf_counter()
        for i, (observation, target) in enumerate(zip(observations, targets)):
            model.resolve_cost(observation, target)
            end = time.perf_counter()
            updates[i] += end-tick
            tick = end
        if learn and fallback is not None and fallback.succeeded:
            fallback_observation = model.update_default_fallback(
                context, DEFAULT_SETUP_PARAMS, fallback.end_to_end_runtime_sec,
            )
        if learn:
            finished = time.perf_counter()
            # Keep the full original interval, including trailing loop/branch
            # work, with the last update (the fallback's when it was observed).
            if fallback_observation is not None:
                fallback_update_runtime = finished-tick
            else:
                updates[-1] += finished-tick
        finalization_runtime = 0.0
        if recorder is not None:
            tick = time.perf_counter()
            additions, replayed = recorder.finalize(attempts, fallback, include_recovery=False)
            finalization_runtime = time.perf_counter()-tick
            if replayed or any(additions):
                raise AssertionError("Recovery expense entered the no-recovery RL target")
    except Exception:
        model.cancel_pending()
        if recorder is not None and recorder.snapshot is not None:
            recorder.controller.restore_learning_state(recorder.snapshot)
        raise

    outcome = recovery.to_result()
    controller_extra = float(snapshot_runtime + finalization_runtime)
    for field in ("runtime", "solve_runtime", "infer_runtime"):
        outcome[field] += controller_extra
    outcome.update(failure_feedback_mode=feedback_mode, failure_penalty_sec=0.0,
        bandit_update_committed=learn,
        bandit_observation_count=(len(attempts) + int(fallback_observation is not None)) if learn else 0,
        bandit_learning_cost=targets[-1],
        fallback_update_runtime=fallback_update_runtime,
        fallback_learning_cost=None if fallback_observation is None else fallback_observation.loss,
        fallback_arm_index=None if fallback_observation is None else fallback_observation.arm_index,
        controller_update_committed=bool(not outcome["failed"] and recorder is not None
            and any(e.committed for e in recorder.episodes)),
        case_learning_runtime=controller_extra, target_replay_runtime=0.0, replayed_transitions=0,
        selected_arm_index=arms[-1])
    for i, row in enumerate(outcome["primary_attempts"]):
        row.update(arm_index=arms[i], params=params[i], selection_runtime=selections[i],
            loss_eval_runtime=evaluations[i], update_runtime=updates[i],
            attempt_cost=costs[i], learning_cost=targets[i],
            rl_episode_recorded=bool(recorder is not None and any(
                e.committed and e.attempt_index == i for e in recorder.episodes)),
            rl_recovery_cost=None if outcome["failed"] and recorder is not None else 0.0,
            cycle_actions=raw_attempts[i].get("cycle_actions", []),
            cycle_times=raw_attempts[i].get("cycle_times", []),
            cycle_residuals=raw_attempts[i].get("cycle_residuals", []))
    select_runtime = sum(selections)
    loss_runtime = sum(evaluations)
    update_runtime = sum(updates) + fallback_update_runtime
    timing = dict(select_sec=select_runtime, loss_eval_sec=loss_runtime, update_sec=update_runtime,
                  overhead_sec=select_runtime+loss_runtime+update_runtime)
    return SetupReselectionResult(params[-1], outcome, timing, int(recovery.fallback_used), update_runtime)
