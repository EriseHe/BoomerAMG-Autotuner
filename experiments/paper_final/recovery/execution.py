"""Candidate scheduling for paper runs and compatibility execution imports."""
from __future__ import annotations

from pathlib import Path

from setup.learners.common import AOTCandidateSchedule
from joint.execution import RecordedEpisode, EpisodeRecorder, run_fixed_case
from .protocol import EXTRA_CANDIDATE_SEED_OFFSET


class CandidateRows:
    """Preserve Module 1's first three candidate rows on every problem."""

    def __init__(self, model, *, cases: int, attempts: int, directory: Path):
        self.model = model
        self.base = model._candidate_schedule
        self.extra = None
        if attempts > 3:
            source = self.base
            self.extra = AOTCandidateSchedule(
                directory=directory, catalog=source.catalog,
                rounds=cases * 2, pool_size=source.pool_size,
                seed=source.seed + EXTRA_CANDIDATE_SEED_OFFSET,
                chunk_rounds=source.chunk_rounds,
                sampling_method=source.sampling_method,
                factorized_cache=source.factorized_cache,
                **{key: getattr(source, key) for key in (
                    "structured_anchor_size", "structured_global_size",
                    "structured_local_size", "structured_sobol_size",
                )},
            )

    def before_attempt(self, case_index: int, attempt_index: int):
        if attempt_index < 3:
            schedule, row = self.base, case_index * 3 + attempt_index
        else:
            if self.extra is None or attempt_index >= 5:
                raise ValueError("Attempt exceeds the prepared candidate budget")
            schedule, row = self.extra, case_index * 2 + attempt_index - 3
        self.model._candidate_schedule = schedule
        self.model.set_candidate_schedule_cursor(row)

    def finish_case(self, case_index: int):
        self.model._candidate_schedule = self.base
        self.model.set_candidate_schedule_cursor((case_index + 1) * 3)
