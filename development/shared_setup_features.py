"""Expose the shared setup-action feature structure to the solve controller."""
from development.setup_normalization import make_bandit_scaled_setup_encoder as make_setup_feature_encoder
from development.setup_normalization import make_range_normalized_setup_encoder


class SharedSetupActionEncoder:
    """Adapt the existing action encoder to the solve-state interface.

    The supplied encoder owns scaling, category ordering and conditional masks.
    Only the names below are assembled for checkpoint metadata.
    """
    def __init__(self, action_encoder):
        self.action_encoder = action_encoder
        self.parameter_spec = action_encoder.parameter_spec
        numeric = action_encoder.numeric_names
        categorical = action_encoder.descriptor_names[action_encoder.numeric_dim:]
        self.descriptor_names = (
            numeric
            + tuple(f"{name}^2" for name in numeric)
            + tuple(f"{left}*{right}" for i, left in enumerate(numeric) for right in numeric[i+1:])
            + categorical
            + tuple(f"{category}*{name}" for category in categorical for name in numeric)
        )
        self.descriptor_dim = action_encoder.feature_dim
        if len(self.descriptor_names) != self.descriptor_dim:
            raise ValueError("Setup feature names do not match the bandit's action feature dimension")

    def encode_setup(self, params):
        return self.action_encoder.encode_action(params)


def make_shared_setup_action_encoder(**kwargs):
    return SharedSetupActionEncoder(make_setup_feature_encoder(**kwargs))


def make_range_normalized_setup_action_encoder(**kwargs):
    """Normalize numerical inputs before building squares and interactions.

    Reuse the earlier 19-feature experiment's fixed range mapping. Numerical
    inputs span [-1, 1], so every term in the 104-feature expansion is bounded
    by one in magnitude. Categories and their conditional masks are unchanged.
    """
    return SharedSetupActionEncoder(make_range_normalized_setup_encoder(**kwargs))
