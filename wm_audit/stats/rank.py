"""Rank-correlation utilities (Spearman rho, Kendall tau-b).

Implemented from scratch on numpy so the audit's inferential layer is
explicit, reviewable, and has no hidden scipy magic. Ties receive average
ranks. Constant inputs yield ``NaN`` (undefined correlation) rather than a
silent ``0`` -- the caller must decide how to treat it.
"""
from __future__ import annotations

import math
import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform, ascending, 1-based. Ties get the mean rank."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        # tie block i..j -> average of the 1-based ranks i+1..j+1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def spearmanr(x, y):
    """Spearman rank correlation.

    Returns ``(rho, p)``. ``rho`` is the Pearson correlation of the average
    ranks; ``p`` is a two-sided p-value via the Fisher-z normal approximation
    (``NaN`` when n < 4 or correlation undefined). Constant/degenerate input
    returns ``(nan, nan)`` -- never a silent 0.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.size
    if n < 2 or y.size != n:
        return float("nan"), float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho, spearman_pvalue(rho, n)


def spearman_pvalue(rho: float, n: int) -> float:
    """Two-sided p-value for Spearman rho using the Fisher-z transform."""
    if n < 4:
        return float("nan")
    if rho is None or not np.isfinite(rho):
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0
    z = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
    se = 1.0 / math.sqrt(n - 3)
    # two-sided normal tail: 2 * Phi(-|z/se|) = 1 - erf(|z/se|/sqrt(2))
    return float(1.0 - math.erf(abs(z / se) / math.sqrt(2.0)))


def kendall_tau(x, y):
    """Kendall's tau-b (tie-aware). Higher score means stronger agreement in
    ordering between ``x`` and ``y``; perfect agreement -> 1.0."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.size
    if n < 2 or y.size != n:
        return float("nan")
    conc = 0
    disc = 0
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        s = np.sign(dx) * np.sign(dy)
        conc += int(np.sum(s > 0))
        disc += int(np.sum(s < 0))
    # tie group counts for normalization (tau-b)
    def _tie_sum(v):
        _, counts = np.unique(v, return_counts=True)
        return float(np.sum(counts * (counts - 1) / 2.0))
    n0 = n * (n - 1) / 2.0
    n1 = _tie_sum(x)
    n2 = _tie_sum(y)
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return float("nan")
    return (conc - disc) / denom
