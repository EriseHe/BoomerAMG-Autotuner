"""Declared pre-solve contexts for the application pilot (no fitted scaling)."""
import numpy as np

APPLICATION_CONTEXT_FIELDS = {
    "spe10_pressure": ("bias", "log_viscosity_ratio_over_log20", "saturation_mean",
        "saturation_std", "saturation_x_low", "saturation_x_high",
        "saturation_y_low", "saturation_y_high", "injected_pore_volumes", "injection_rate_multiplier"),
    "mfem_heat": ("bias", "log_dt", "log_kappa", "log_alpha",
        "temperature_mean", "temperature_std", "temperature_min", "temperature_max"),
}


def validate_application_context(context, mode):
    values = np.asarray(context, dtype=float)
    if (values.shape != (len(APPLICATION_CONTEXT_FIELDS[mode]),)
            or not np.isfinite(values).all() or values[0] != 1.0):
        raise ValueError(f"Invalid pre-solve application context for {mode}")
    return values
