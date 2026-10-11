"""Uncertainty diagnostics: does ensemble dispersion track the true error?"""
from __future__ import annotations

import numpy as np

from ..stats.rank import spearmanr


def ensemble_dispersion(preds) -> np.ndarray:
    """Per-sample predictive variance across ensemble members.

    ``preds``: list of E arrays, each shape (N, ...). Returns a (N,) array of
    mean predictive variance -- higher = more disagreement across the ensemble.
    """
    P = np.stack([np.asarray(p, dtype=float) for p in preds], axis=0)  # (E,N,...)
    var = np.var(P, axis=0)
    while var.ndim > 1:
        var = var.mean(axis=-1)
    return var


def uncertainty_error_alignment(dispersion, squared_error) -> float:
    """Spearman rho between ensemble dispersion and realized squared error.

    +1.0 means uncertainty perfectly ranks which predictions are wrong. This is
    the preregistered RQ2(c) diagnostic.
    """
    rho, _ = spearmanr(dispersion, squared_error)
    return rho


def uncertainty_ece(dispersion, squared_error, n_bins: int = 5) -> float:
    """Calibration of uncertainty: bin by dispersion, compare mean predicted
    variance to mean realized squared error. Lower = better calibrated."""
    dispersion = np.asarray(dispersion, dtype=float).reshape(-1)
    squared_error = np.asarray(squared_error, dtype=float).reshape(-1)
    n = dispersion.size
    if n == 0:
        return float("nan")
    n_bins = max(1, min(n_bins, n))
    edges = np.quantile(dispersion, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.digitize(dispersion, edges[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        ece += (count / n) * abs(np.mean(dispersion[mask]) - np.mean(squared_error[mask]))
    return float(ece)
