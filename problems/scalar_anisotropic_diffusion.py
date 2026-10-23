from __future__ import annotations

"""Scalar anisotropic diffusion views of the shared DifConv problem."""

from typing import Any

import numpy as np

from .amg import (
    DIFFUSION_ADVECTION_CONTEXT_DIM,
    DIFFUSION_ADVECTION_CONTEXT_FIELDS,
    build_context_diffusion_advection_from_matrix_kwargs,
    stencil_0_difconv_rl,
)


SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_FIELDS = (
    DIFFUSION_ADVECTION_CONTEXT_FIELDS
)
SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM = (
    DIFFUSION_ADVECTION_CONTEXT_DIM
)


def stencil_0_scalar_anisotropic_diffusion_rl(
    *, rng: np.random.Generator, **extras: Any
):
    """Sample scalar diffusion and expose the canonical zero-advection context."""

    scalar_extras = dict(extras)
    scalar_extras.update({"ax": 0.0, "ay": 0.0, "az": 0.0})
    matrix_kwargs, _legacy_context, metadata = stencil_0_difconv_rl(
        rng=rng,
        **scalar_extras,
    )
    context = build_context_diffusion_advection_from_matrix_kwargs(
        matrix_kwargs,
        grid_norm_div=float(
            scalar_extras.get("grid_norm_div", metadata["n_max"])
        ),
        c_norm_div=float(
            scalar_extras.get("c_norm_div", metadata["c_max"])
        ),
        a_norm_div=1.0,
    )
    return matrix_kwargs, context, metadata


__all__ = [
    "SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM",
    "SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_FIELDS",
    "stencil_0_scalar_anisotropic_diffusion_rl",
]
