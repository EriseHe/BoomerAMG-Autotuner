"""Build solve-context setup descriptors from the shared parameter specification."""

from __future__ import annotations
from dataclasses import replace
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
    # Preserve categorical one-hot entries and numerical distances. The setup
    # learner owns its original encoder; only solve-context scaling changes.
    parameters = []
    for parameter in parameter_spec.parameters:
        if parameter.kind in ("continuous", "integer"):
            values = (*parameter.values, parameter.default)
            low, high = min(values), max(values)
            parameter = replace(parameter, center=(low + high) / 2.,
                                scale=(high - low) / 2. if high > low else 1.)
        parameters.append(parameter)
    return GenericActionFeatureEncoder(replace(parameter_spec, parameters=tuple(parameters)))
