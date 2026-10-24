"""Default setup and fixed relaxation shared by all recovery paths."""
from __future__ import annotations

from typing import Any, Mapping

from hypre.bindings import augment_setup_params
from setup.space import DEFAULT_SETUP_PARAMS
from experiments.paper_final.online.native_evaluation import solve_no_rl_case


def run_default_fallback(
    *, mkw: Mapping[str, Any], solver_tol: float, solver_max_iter: int,
) -> dict[str, Any]:
    """Execute one fresh Default attempt with the experiment's solve limits."""
    return solve_no_rl_case(
        params=dict(DEFAULT_SETUP_PARAMS), mkw=dict(mkw),
        solver_tol=solver_tol, solver_max_iter=solver_max_iter,
        augment_params=augment_setup_params,
    )
