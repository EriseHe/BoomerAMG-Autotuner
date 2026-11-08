"""Recovery settings shared by the retained paper workflows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryVariant:
    name: str
    label: str
    attempts: int
    unified: bool
    rl_recovery_cost: bool
    default_fallback: bool = False


EXTRA_CANDIDATE_SEED_OFFSET = 600006
