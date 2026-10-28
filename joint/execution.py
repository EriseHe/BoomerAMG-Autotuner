"""Coordinate solve-learning state and bounded fixed-setup retries."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from hypre.bindings import RecoveryOutcome, execute_attempt


@dataclass
class RecordedEpisode:
    attempt_index: int
    initial_weight: float | None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    update_runtimes: list[float] = field(default_factory=list)
    committed: bool = False


class EpisodeRecorder:
    """Record executed updates so delayed targets replace, rather than duplicate, data."""

    def __init__(self, controller, *, learn=True):
        self.controller = controller
        self.learning_enabled = learn
        self.snapshot = controller.snapshot_learning_state() if learn else None
        self.episodes: list[RecordedEpisode] = []
        self.attempt_index = 0
        self.current = None

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def start_episode(self, *, initial_environment_weight=None):
        self.current = RecordedEpisode(self.attempt_index, initial_environment_weight)
        self.episodes.append(self.current)
        return self.controller.start_episode(
            initial_environment_weight=initial_environment_weight,
        )

    def update(self, **kwargs):
        result = self.controller.update(**kwargs)
        self.current.transitions.append({
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in kwargs.items()
        })
        return result

    def finish_episode(self, *, learned):
        result = self.controller.finish_episode(learned=learned)
        self.current.committed = bool(learned)
        return result

    def record_update_runtime(self, duration):
        self.controller.record_update_runtime(duration)
        self.current.update_runtimes.append(float(duration))

    def restore_learning_state(self, state):
        # run_td_episode may reject an exceptional attempt independently.
        if self.current is not None:
            self.current.committed = False
        return self.controller.restore_learning_state(state)

    def finalize(self, attempts, fallback, *, include_recovery):
        if not self.learning_enabled:
            return [0.0] * len(attempts), 0
        success = attempts[-1].succeeded or (fallback is not None and fallback.succeeded)
        additions = [0.0] * len(attempts)
        if not success:
            self.controller.restore_learning_state(self.snapshot)
            return additions, 0
        episodes = [e for e in self.episodes if e.committed]
        observed = {e.attempt_index for e in episodes}
        suffix = 0.0 if fallback is None else fallback.native_runtime_sec
        for i in range(len(attempts) - 1, -1, -1):
            if not attempts[i].succeeded and i in observed:
                additions[i] = float(suffix) if include_recovery else 0.0
            suffix += attempts[i].native_runtime_sec
        if not any(additions[e.attempt_index] for e in episodes):
            return additions, 0

        # Preserve actual action-selection state and consumed RNG draws. The
        # learning snapshot deliberately excludes the RNG, as in Module 1.
        flags = (self.controller._default_first_action_pending,
                 self.controller.forced_initial_action_count)
        self.controller.restore_learning_state(self.snapshot)
        replayed = 0
        for episode in episodes:
            self.controller.start_episode(initial_environment_weight=episode.initial_weight)
            if len(episode.transitions) != len(episode.update_runtimes):
                raise ValueError("Missing measured runtime for a recorded RL update")
            for transition, duration in zip(episode.transitions, episode.update_runtimes):
                target = dict(transition)
                if target["terminal"]:
                    target["cost"] += additions[episode.attempt_index]
                self.controller.update(**target)
                # Retain the executed update measurements once. Replay replaces
                # the targets; its own bookkeeping time is evaluation overhead.
                self.controller.record_update_runtime(duration)
                replayed += 1
            self.controller.finish_episode(learned=True)
        (self.controller._default_first_action_pending,
         self.controller.forced_initial_action_count) = flags
        return additions, replayed


def run_fixed_case(solver, recorder, snapshot_runtime, include_recovery, *, fallback_solver=None,
                   max_attempts=3):
    """A bounded number of fresh fixed-setup RL solves, then default fallback."""
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("Fixed-setup attempt budget must be a positive integer")
    attempts, raw_attempts = [], []
    fallback = None
    try:
        for index in range(max_attempts):
            recorder.attempt_index, recorder.current = index, None
            attempt = execute_attempt(solver, require_valid_observation=True)
            attempts.append(attempt)
            raw_attempts.append(dict(attempt.result))
            if attempt.succeeded:
                break
        if not attempts[-1].succeeded and fallback_solver is not None:
            fallback = execute_attempt(fallback_solver, require_valid_observation=True)
        recovery = RecoveryOutcome(primary_attempts=tuple(attempts), fallback=fallback)
        started = time.perf_counter()
        additions, replayed = recorder.finalize(attempts, fallback, include_recovery=include_recovery)
        finalization = time.perf_counter() - started
    except BaseException:
        if recorder.snapshot is not None:
            recorder.controller.restore_learning_state(recorder.snapshot)
        raise
    extra = snapshot_runtime + finalization
    out = recovery.to_result()
    for field in ("runtime", "solve_runtime", "infer_runtime"):
        out[field] += extra
    out.update(
        bandit_update_committed=False, bandit_observation_count=0,
        controller_update_committed=bool(not out["failed"] and any(e.committed for e in recorder.episodes)),
        failure_feedback_mode="rollback_unrecovered", failure_penalty_sec=0.0,
        case_learning_runtime=extra, target_replay_runtime=finalization if replayed else 0.0,
        replayed_transitions=replayed, fixed_setup_retries=len(attempts) - 1,
        learned_reselection_count=0,
    )
    for index, row in enumerate(out["primary_attempts"]):
        row.update(
            rl_episode_recorded=any(e.committed and e.attempt_index == index for e in recorder.episodes),
            rl_recovery_cost=additions[index] if not out["failed"] else None,
            cycle_actions=raw_attempts[index].get("cycle_actions", []),
            cycle_times=raw_attempts[index].get("cycle_times", []),
            cycle_residuals=raw_attempts[index].get("cycle_residuals", []),
        )
    return out
