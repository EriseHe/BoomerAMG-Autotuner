"""Cycle-bin indicators and residual-gap interactions for the current solve state."""
from __future__ import annotations

import numpy as np

from development.compact_context import CompactSolveStateEncoder


CYCLE_BOUNDS = (1, 2, 3, 5, 8, 12)


class CycleBinSolveStateEncoder(CompactSolveStateEncoder):
    """Keep every current feature and insert the original fourteen bin terms."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = self.base.feature_dim + 2 * (len(CYCLE_BOUNDS) + 1)

    def encode(self, *, cycle, **kwargs):
        state = super().encode(cycle=cycle, **kwargs)
        offset = self.base._setup_offset
        indicators = np.zeros(len(CYCLE_BOUNDS) + 1, dtype=float)
        indicators[np.searchsorted(CYCLE_BOUNDS, int(cycle), side='right')] = 1.
        return np.concatenate((state[:offset], indicators, state[2] * indicators, state[offset:]))

    def metadata(self):
        metadata = super().metadata()
        metadata.update(context_variant='normalized_setup_cycle_bins', feature_dim=self.feature_dim,
                        cycle_bin_bounds=list(CYCLE_BOUNDS), cycle_bin_count=len(CYCLE_BOUNDS)+1,
                        cycle_bin_gap_interactions=True, residual_gap_clipping=False)
        return metadata
