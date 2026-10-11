"""Control-relevant diagnostics: return calibration and counterfactual ranking.

These target what the planner actually consumes:
  * ``regression_ece`` -- calibration of imagined return vs real return.
  * ``counterfactual_kendall_tau`` -- whether the imagined model ranks two
    candidate actions the same way the (oracle) environment does.
"""
from __future__ import annotations

import numpy as np

from ..stats.rank import kendall_tau


def regression_ece(predicted, actual, n_bins: int = 10) -> float:
    """Expected calibration error for a scalar regression target.

    Predictions are binned by *quantile* (so bins are never empty), and ECE is
    the bin-frequency-weighted gap between mean predicted and mean realized
    value per bin. Perfect calibration (predicted == actual) -> 0.0.
    """
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    n = predicted.size
    if n == 0:
        return float("nan")
    n_bins = max(1, min(n_bins, n))
    # quantile edges guarantee non-empty bins
    edges = np.quantile(predicted, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.digitize(predicted, edges[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        ece += (count / n) * abs(np.mean(predicted[mask]) - np.mean(actual[mask]))
    return float(ece)


def calibration_curve(predicted, actual, n_bins: int = 10):
    """Return (bin_centers_predicted, bin_means_actual) for plotting."""
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    n = predicted.size
    n_bins = max(1, min(n_bins, n))
    edges = np.quantile(predicted, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.digitize(predicted, edges[1:-1])
    centers, realized = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if np.sum(mask) == 0:
            continue
        centers.append(float(np.mean(predicted[mask])))
        realized.append(float(np.mean(actual[mask])))
    return np.asarray(centers), np.asarray(realized)


def counterfactual_kendall_tau(oracle_rewards, imagined_rewards) -> float:
    """Rank agreement between oracle and imagined action values.

    Inputs are aligned arrays over ``(state, action)`` entries collected from a
    counterfactual dataset (both actions stepped in the real env once, and once
    imagined through the world model). On a perfect world model the ordering
    is preserved -> tau ~= 1.0.
    """
    return kendall_tau(oracle_rewards, imagined_rewards)
