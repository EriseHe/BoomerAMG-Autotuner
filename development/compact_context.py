"""Solve context using normalized setup descriptors and absolute log time."""
from __future__ import annotations

import math

import numpy as np

from solve.controllers.common.state_encoder import SolveStateEncoder


TIME_REFERENCE_SEC = 0.001


def log_cycle_time(seconds: float) -> float:
    """Natural log of native cycle seconds relative to a fixed 1 ms unit."""
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("The preceding native cycle time must be finite and nonnegative")
    return math.log1p(seconds / TIME_REFERENCE_SEC)


class CompactSolveStateEncoder:
    """Use the basic state with a fixed logarithmic cycle-time scale."""

    def __init__(self, *, cycle_time_mode="log", **kwargs):
        if cycle_time_mode not in ("log", "first_cycle_ratio"):
            raise ValueError("Unknown cycle-time context")
        self.cycle_time_mode = cycle_time_mode
        self.base = SolveStateEncoder(**kwargs)
        self.feature_dim = self.base.feature_dim

    def __getattr__(self, name):
        return getattr(self.base, name)

    def encode(self, *, cycle, last_cycle_time, first_cycle_time, setup_params=None, **kwargs):
        # Reuse the existing residual, weight, problem and setup construction.
        # A unit divisor passes the log-time value through without using the
        # attempt's first measured cycle or modifying any reward calculation.
        if self.cycle_time_mode == "first_cycle_ratio":
            return self.base.encode(cycle=cycle, last_cycle_time=last_cycle_time,
                                    first_cycle_time=first_cycle_time,
                                    setup_params=setup_params, **kwargs)
        timing = 0.0 if cycle == 0 else log_cycle_time(last_cycle_time)
        base = self.base.encode(cycle=cycle, last_cycle_time=timing,
                                first_cycle_time=1.0, setup_params=setup_params, **kwargs)
        return base

    def constant_value_parameters(self, value):
        parameters = np.zeros(self.feature_dim, dtype=float)
        parameters[0] = float(value)
        return parameters

    def metadata(self):
        logarithmic = self.cycle_time_mode == "log"
        return dict(
            context_variant="normalized_setup_cycle_time",
            cycle_time_mode=self.cycle_time_mode,
            cycle_time_normalization="log1p_native_seconds_over_fixed_reference" if logarithmic else "first_native_cycle_per_attempt",
            cycle_time_reference_sec=TIME_REFERENCE_SEC if logarithmic else None,
            cycle_time_logarithm="natural" if logarithmic else None, cycle_time_clipping=False,
            initial_cycle_time_feature=0.0, feature_dim=self.feature_dim,
        )
