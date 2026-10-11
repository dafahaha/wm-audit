"""Diagnostic battery: perceptual, control-relevant, and uncertainty."""
from .perceptual import (
    mse,
    ssim,
    gaussian_nll,
    lpips_distance,
    frechet_distance,
    FixedFilterFeatures,
)
from .control import (
    regression_ece,
    calibration_curve,
    counterfactual_kendall_tau,
)
from .uncertainty import (
    ensemble_dispersion,
    uncertainty_error_alignment,
    uncertainty_ece,
)
from .fidelity_utility import audit_correlations, prereg_decision

__all__ = [
    "mse", "ssim", "gaussian_nll", "lpips_distance", "frechet_distance",
    "FixedFilterFeatures",
    "regression_ece", "calibration_curve", "counterfactual_kendall_tau",
    "ensemble_dispersion", "uncertainty_error_alignment", "uncertainty_ece",
    "audit_correlations", "prereg_decision",
]
