from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np
from problems.heterogeneous_3d import (
    STRESS_CONTEXT_FIELDS, context_from_matrix_kwargs, validate_context,
)

from problems.amg import (
    COMPACT_DIFFUSION_CONTEXT_INDICES,
    DIFFUSION_ADVECTION_CONTEXT_DIM,
    DIFFUSION_ADVECTION_CONTEXT_FIELDS,
    DIFFUSION_ADVECTION_NO_MEAN_CONTEXT,
    DIFFUSION_ADVECTION_NO_MEAN_INDICES,
    build_context_diffusion_advection_from_matrix_kwargs,
    compact_diffusion_context,
    diffusion_advection_context_without_mean,
    normalize_diffusion_advection_context as normalize_problem_context,
)


CANONICAL_PROBLEM_CONTEXT = "canonical"
LEGACY_DIFFUSION_ONLY_CONTEXT = "legacy"
PROBLEM_CONTEXT_MODES = (
    *STRESS_CONTEXT_FIELDS,
    CANONICAL_PROBLEM_CONTEXT,
    LEGACY_DIFFUSION_ONLY_CONTEXT,
    DIFFUSION_ADVECTION_NO_MEAN_CONTEXT,
    *COMPACT_DIFFUSION_CONTEXT_INDICES,
)
STATE_ENCODING_VERSION = "shared_setup_one_hot_tolerance_residuals_v3"
STATE_ENCODING_VERSIONS = (STATE_ENCODING_VERSION,)


class SolveStateEncoder:
    """Cycle observations with the setup learner's shared configuration features."""

    def __init__(
        self,
        *,
        tol: float,
        max_cycles: int,
        c_max: float,
        mode: str = "full",
        setup_obs_encoder: Any | None = None,
        problem_context_mode: str = CANONICAL_PROBLEM_CONTEXT,
        encoding_version: str = STATE_ENCODING_VERSION,
        weight_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.tol = float(tol)
        if not math.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("Solve tolerance must be finite and positive")
        self._log_tolerance = math.log(self.tol)
        self.max_cycles = int(max_cycles)
        self.c_max = float(c_max)
        self.mode = str(mode).strip().lower()
        self.setup_obs_encoder = setup_obs_encoder
        self.encoding_version = str(encoding_version)
        if self.encoding_version not in STATE_ENCODING_VERSIONS:
            raise ValueError(
                f"encoding_version must be one of {STATE_ENCODING_VERSIONS}"
            )
        self.weight_bounds = weight_bounds
        if weight_bounds is None or len(weight_bounds) != 2:
            raise ValueError("The solve encoder requires actual action weight bounds")
        low, high = map(float, weight_bounds)
        if not all(math.isfinite(v) for v in (low, high)) or high < low:
            raise ValueError("action weight bounds must be finite and ordered")
        self.weight_bounds = (low, high)
        self._weight_center = (low + high) / 2.0
        self._weight_scale = (high - low) / 2.0 if high > low else 1.0
        self.problem_context_mode = str(problem_context_mode).strip().lower()
        if self.problem_context_mode not in PROBLEM_CONTEXT_MODES:
            raise ValueError(
                f"problem_context_mode must be one of {PROBLEM_CONTEXT_MODES}"
            )
        if self.problem_context_mode in STRESS_CONTEXT_FIELDS:
            self.problem_context_fields = STRESS_CONTEXT_FIELDS[self.problem_context_mode]
            self.problem_feature_dim = len(self.problem_context_fields) - 1
        elif self.problem_context_mode == CANONICAL_PROBLEM_CONTEXT:
            self.problem_context_fields = DIFFUSION_ADVECTION_CONTEXT_FIELDS
            self.problem_feature_dim = DIFFUSION_ADVECTION_CONTEXT_DIM - 1
        elif self.problem_context_mode == DIFFUSION_ADVECTION_NO_MEAN_CONTEXT:
            self.problem_context_fields = tuple(
                DIFFUSION_ADVECTION_CONTEXT_FIELDS[index]
                for index in DIFFUSION_ADVECTION_NO_MEAN_INDICES
            )
            self.problem_feature_dim = len(self.problem_context_fields) - 1
        elif self.problem_context_mode in COMPACT_DIFFUSION_CONTEXT_INDICES:
            self.problem_context_fields = tuple(
                DIFFUSION_ADVECTION_CONTEXT_FIELDS[index]
                for index in COMPACT_DIFFUSION_CONTEXT_INDICES[
                    self.problem_context_mode
                ]
            )
            self.problem_feature_dim = len(self.problem_context_fields) - 1
        else:
            self.problem_context_fields = (
                "c_x",
                "c_y",
                "c_z",
                "diffusion_spread",
            )
            self.problem_feature_dim = 4
        self._problem_offset = 7
        self._setup_offset = self._problem_offset + self.problem_feature_dim
        self._cached_problem_key: tuple[float, ...] | None = None
        self._cached_problem_features = np.zeros(
            self.problem_feature_dim,
            dtype=float,
        )
        self._cached_initial_residual: float | None = None
        self._cached_initial_gap = 1.0
        self._cached_setup_key: tuple[Any, ...] | None = None
        self._cached_setup_features = np.empty(0, dtype=float)
        if self.mode in {"full", "setup_full"}:
            setup_dim = 0
            if self.mode == "setup_full":
                if self.setup_obs_encoder is None:
                    raise ValueError(
                        "setup_full mode requires a setup observation encoder"
                    )
                setup_dim = self.setup_obs_encoder.descriptor_dim
            self.feature_dim = self._setup_offset + setup_dim
        else:
            raise ValueError(f"Unknown solve state mode: {self.mode}")

    def encode(
        self,
        *,
        mkw: Dict[str, Any],
        initial_residual: float,
        residual: float,
        previous_residual: float,
        cycle: int,
        last_weight: float,
        last_cycle_time: float,
        first_cycle_time: float | None,
        setup_params: Dict[str, Any] | None = None,
        problem_context: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        eps = 1.0e-30
        initial_residual_value = float(initial_residual)
        if self._cached_initial_residual != initial_residual_value:
            if not math.isfinite(initial_residual_value) or initial_residual_value <= self.tol:
                raise ValueError("An active solve requires initial_residual > tolerance")
            initial_gap = math.log(initial_residual_value) - self._log_tolerance
            if initial_gap <= 0.0:
                raise ValueError("Initial log-residual gap must be positive")
            self._cached_initial_gap = initial_gap
            self._cached_initial_residual = initial_residual_value
        initial_gap = self._cached_initial_gap
        if any(not math.isfinite(value) or value <= 0.0
               for value in (float(residual), float(previous_residual))):
            raise ValueError("Residual state observations must be finite and positive")
        log_residual = math.log(float(residual))
        log_previous_residual = math.log(float(previous_residual))
        # Both features use the required log reduction G = log(r0 / tol).
        # With no clipping, d_k = g_k - g_(k-1): contraction is negative.
        gap_fraction = (log_residual - self._log_tolerance) / initial_gap
        log_ratio = (log_residual - log_previous_residual) / initial_gap
        cycle_fraction = min(
            max(float(cycle) / max(1.0, float(self.max_cycles)), 0.0),
            1.0,
        )
        # The first native cycle sets a fixed reference for this attempt.
        # Before it executes, no timing observation is available.
        if cycle == 0:
            cycle_time = 0.0
        else:
            if (first_cycle_time is None
                    or not math.isfinite(float(first_cycle_time))
                    or float(first_cycle_time) <= 0.0):
                raise ValueError("The first native cycle time must be finite and positive")
            previous_time = float(last_cycle_time)
            if not math.isfinite(previous_time) or previous_time < 0.0:
                raise ValueError("The preceding native cycle time must be finite and nonnegative")
            cycle_time = previous_time / float(first_cycle_time)
            if not math.isfinite(cycle_time):
                raise ValueError("The native cycle-time ratio must be finite")
        weight = min(
            max((float(last_weight) - self._weight_center) / self._weight_scale, -1.0),
            1.0,
        )

        if self.problem_context_mode in STRESS_CONTEXT_FIELDS:
            resolved_context = validate_context(
                context_from_matrix_kwargs(mkw) if problem_context is None else problem_context,
                self.problem_context_mode,
            )
            problem_key = tuple(float(value) for value in resolved_context)
            if self._cached_problem_key != problem_key:
                self._cached_problem_key = problem_key
                self._cached_problem_features[:] = resolved_context[1:]
        elif self.problem_context_mode in {
            CANONICAL_PROBLEM_CONTEXT,
            DIFFUSION_ADVECTION_NO_MEAN_CONTEXT,
            *COMPACT_DIFFUSION_CONTEXT_INDICES,
        }:
            canonical_context = (
                self._context_from_matrix_kwargs(mkw)
                if problem_context is None
                else normalize_problem_context(problem_context)
            )
            resolved_context = canonical_context
            if self.problem_context_mode in COMPACT_DIFFUSION_CONTEXT_INDICES:
                resolved_context = compact_diffusion_context(
                    canonical_context, mode=self.problem_context_mode
                )
            elif self.problem_context_mode == DIFFUSION_ADVECTION_NO_MEAN_CONTEXT:
                resolved_context = diffusion_advection_context_without_mean(
                    canonical_context
                )
            problem_key = tuple(float(value) for value in resolved_context)
            if self._cached_problem_key != problem_key:
                self._cached_problem_key = problem_key
                self._cached_problem_features[:] = resolved_context[1:]
        else:
            problem_key = (
                float(mkw["k"]),
                float(mkw["c"]),
                float(mkw["a0"]),
            )
            if self._cached_problem_key != problem_key:
                self._cached_problem_key = problem_key
                c_denom = max(math.log(max(1.0, self.c_max)), eps)
                coeffs = np.asarray(
                    [math.log(max(value, eps)) / c_denom for value in problem_key],
                    dtype=float,
                )
                coeffs = np.clip(coeffs, -1.0, 1.5)
                self._cached_problem_features[:3] = coeffs
                self._cached_problem_features[3] = float(
                    np.max(coeffs) - np.min(coeffs)
                )

        features = np.zeros(self.feature_dim, dtype=float)
        features[:7] = (
            1.0,
            cycle_fraction,
            gap_fraction,
            gap_fraction * gap_fraction,
            log_ratio,
            cycle_time,
            weight,
        )
        features[self._problem_offset : self._setup_offset] = (
            self._cached_problem_features
        )
        if self.mode == "setup_full":
            assert self.setup_obs_encoder is not None
            params = {} if setup_params is None else setup_params
            setup_key = tuple(
                params.get(param.name, param.default)
                for param in self.setup_obs_encoder.parameter_spec.parameters
            )
            if self._cached_setup_key != setup_key:
                setup_features = np.asarray(
                    self.setup_obs_encoder.encode_setup(params),
                    dtype=float,
                )
                self._cached_setup_features = setup_features
                self._cached_setup_key = setup_key
            features[self._setup_offset :] = self._cached_setup_features
        if features.size != self.feature_dim:
            raise RuntimeError(
                f"Expected {self.feature_dim} state features, got {features.size}"
            )
        return features

    def constant_value_parameters(self, value: float) -> np.ndarray:
        parameters = np.zeros(self.feature_dim, dtype=float)
        parameters[0] = float(value)
        return parameters

    def _context_from_matrix_kwargs(
        self,
        mkw: Dict[str, Any],
    ) -> np.ndarray:
        """Compatibility path for callers that predate explicit context wiring."""

        nx = int(mkw.get("nx", 1))
        ny = int(mkw.get("ny", 1))
        nz = int(mkw.get("nz", 1))
        advection = tuple(float(mkw.get(key, 0.0)) for key in ("a1", "a2", "a3"))
        return build_context_diffusion_advection_from_matrix_kwargs(
            mkw,
            grid_norm_div=float(max(nx, ny, nz, 1)),
            c_norm_div=float(self.c_max),
            a_norm_div=float(
                max(self.c_max, *(abs(value) for value in advection), 1.0)
            ),
        )


__all__ = [
    "CANONICAL_PROBLEM_CONTEXT",
    "LEGACY_DIFFUSION_ONLY_CONTEXT",
    "PROBLEM_CONTEXT_MODES",
    "SolveStateEncoder",
    "normalize_problem_context",
]
