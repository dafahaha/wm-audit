"""Bootstrap confidence intervals for the audit's rank correlations."""
from __future__ import annotations

from typing import Callable
import numpy as np


def bootstrap_ci(
    x,
    y,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
):
    """Percentile bootstrap CI of ``stat_fn(x, y)``.

    Samples paired indices with replacement (preserving the (x,y) pairing).
    Bootstrap replicates whose statistic is undefined (e.g. a resample where
    one variable is constant) are dropped rather than crashing, and the
    interval is taken over the finite replicates.

    Returns ``(observed, lo, hi)``. If too few finite replicates exist,
    ``lo``/``hi`` are ``NaN``.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    observed = stat_fn(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat_fn(x[idx], y[idx])
        if np.isfinite(v):
            boots.append(v)
    if not boots:
        return observed, float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boots, alpha * 100.0))
    hi = float(np.percentile(boots, (1.0 - alpha) * 100.0))
    return observed, lo, hi
