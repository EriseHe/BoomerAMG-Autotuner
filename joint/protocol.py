"""Attempt budgets and recovery settings for joint execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryVariant:
    name: str
    label: str
    attempts: int
    unified: bool
    rl_recovery_cost: bool
    default_fallback: bool = False
