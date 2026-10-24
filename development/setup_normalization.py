"""Development comparisons of bandit scaling and the published RL descriptors."""
from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.learners.common import GenericActionFeatureEncoder
from setup.space import build_setup_parameter_spec


def make_range_normalized_setup_encoder(**kwargs):
    """Use the numerical range normalization now owned by the standard factory."""
    return make_setup_feature_encoder(**kwargs)


def make_bandit_scaled_setup_encoder(*, configuration_space=None, parameter_resolution=None):
    """Retain the original bandit scaling solely for development comparisons."""
    spec, _ = build_setup_parameter_spec(
        tune_dim=7, tune7_variant='categorical', parameter_resolution=parameter_resolution,
        coarsen_type_values=configuration_space.coarsen_types if configuration_space else None,
        interp_type_values=configuration_space.interp_types if configuration_space else None,
        agg_interp_type_values=configuration_space.agg_interp_types if configuration_space else None,
        numeric_grids=configuration_space.numeric_grids if configuration_space else None,
    )
    return GenericActionFeatureEncoder(spec)
