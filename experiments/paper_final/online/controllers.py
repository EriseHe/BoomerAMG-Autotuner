"""Build solve-context setup descriptors from the shared parameter specification."""

from __future__ import annotations
from setup.learners.common import GenericActionFeatureEncoder
from setup.space import (
    SetupConfigurationSpace,
    build_setup_parameter_spec,
)


def make_setup_feature_encoder(
    *,
    configuration_space: SetupConfigurationSpace | None = None,
    parameter_resolution: int | None = None,
) -> GenericActionFeatureEncoder:
    parameter_spec, _fixed_params = build_setup_parameter_spec(
        tune_dim=7,
        tune7_variant="categorical",
        parameter_resolution=parameter_resolution,
        coarsen_type_values=(
            configuration_space.coarsen_types if configuration_space else None
        ),
        interp_type_values=(
            configuration_space.interp_types if configuration_space else None
        ),
        agg_interp_type_values=(
            configuration_space.agg_interp_types if configuration_space else None
        ),
        numeric_grids=configuration_space.numeric_grids if configuration_space else None,
    )
    return GenericActionFeatureEncoder(parameter_spec)
