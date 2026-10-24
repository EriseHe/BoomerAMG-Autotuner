"""Fixed parameter-range scaling for the solve controller's setup descriptors."""
from dataclasses import replace

from experiments.paper_final.online.controllers import make_setup_feature_encoder
from setup.learners.common import GenericActionFeatureEncoder


def make_range_normalized_setup_encoder(**kwargs):
    """Keep reference one-hot categories; map each numerical range to [-1, 1].

    Bounds include the allowed grid and Default. Discretized numerical parameters
    retain their numerical distances. Constant parameters map to zero. This
    creates a separate RL encoder and never mutates the setup learner's encoder.
    """
    shared = make_setup_feature_encoder(**kwargs)
    parameters = []
    for parameter in shared.parameter_spec.parameters:
        if parameter.kind in ("continuous", "integer"):
            values = (*parameter.values, parameter.default)
            low, high = min(values), max(values)
            parameter = replace(parameter, center=(low + high) / 2.,
                                scale=(high - low) / 2. if high > low else 1.)
        parameters.append(parameter)
    return GenericActionFeatureEncoder(replace(shared.parameter_spec, parameters=tuple(parameters)))
