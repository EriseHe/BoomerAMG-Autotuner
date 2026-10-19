from __future__ import annotations

import numpy as np


class _FactorizedLstdqScoring:
    """Contract joint-feature quadratics through the small action basis."""

    def __init__(
        self,
        *,
        action_basis: np.ndarray,
        feature_dim: int,
    ) -> None:
        self.action_basis = np.asarray(action_basis, dtype=float)
        self.feature_dim = int(feature_dim)
        self.basis_dim = int(self.action_basis.shape[1])
        self.joint_dim = int(self.basis_dim * self.feature_dim)

    def prepare_state(self, features: np.ndarray) -> np.ndarray:
        state = np.asarray(features, dtype=float)
        if state.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected {self.feature_dim} state features, got {state.shape}"
            )
        return state

    def means(self, theta: np.ndarray, state: np.ndarray) -> np.ndarray:
        state_parameters = np.asarray(theta, dtype=float).reshape(
            self.basis_dim,
            self.feature_dim,
        ) @ state
        return self.action_basis @ state_parameters

    def action_quadratic(self, reduced_covariance: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced_covariance, dtype=float)
        if reduced.shape != (self.basis_dim, self.basis_dim):
            raise ValueError(
                "Reduced covariance must match the action-basis dimension"
            )
        return np.sum(
            (self.action_basis @ reduced) * self.action_basis,
            axis=1,
        )


__all__ = ["_FactorizedLstdqScoring"]
