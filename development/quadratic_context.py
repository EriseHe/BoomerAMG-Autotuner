"""Complete the quadratic cycle-progress and residual-gap basis."""
from __future__ import annotations

import numpy as np

from development.compact_context import CompactSolveStateEncoder


class QuadraticSolveStateEncoder(CompactSolveStateEncoder):
    """Add u squared and u times g to the existing 1, u, g, g squared terms."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = self.base.feature_dim + 2

    def encode(self, *, cycle, **kwargs):
        state = super().encode(cycle=cycle, **kwargs)
        offset = self.base._setup_offset
        progress, gap = state[1:3]
        quadratic = (progress * progress, progress * gap)
        return np.concatenate((state[:offset], quadratic, state[offset:]))

    def metadata(self):
        metadata = super().metadata()
        metadata.update(
            context_variant="normalized_setup_quadratic_progress",
            feature_dim=self.feature_dim,
            additional_state_terms=["cycle_fraction_squared", "cycle_fraction_times_residual_gap"],
            quadratic_variables=["cycle / max_cycles", "normalized_log_residual_gap"],
            quadratic_degree=2,
            cycle_bins=False,
            residual_gap_clipping=False,
        )
        return metadata
