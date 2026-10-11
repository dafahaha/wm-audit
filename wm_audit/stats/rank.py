"""Rank-correlation utilities (Spearman rho, Kendall tau-b).

Implemented from scratch on numpy so the audit's inferential layer is
explicit, reviewable, and has no hidden scipy magic. Ties receive average
ranks. Constant inputs yield ``NaN`` (undefined correlation) rather than a
silent ``0`` -- the caller must decide how to treat it.

Spearman p-value convention
---------------------------
We use the *Student-t* approximation used by ``scipy.stats.spearmanr``: under
H0, ``t = rho * sqrt((n-2)/(1-rho^2))`` follows a Student-t distribution with
``df = n-2``, and the two-sided p-value is ``2 * P(T >= |t|)``. Using the
standard beta/t identity this is

    p = I_{ df / (df + t^2) }(df/2, 1/2),

i.e. the regularized incomplete beta function, which we evaluate ourselves
(Lentz continued fraction, Numerical Recipes ``betacf``). This is the default
scipy approximation and -- unlike the old Fisher-z normal approximation -- it
is honest at small n. ``rho = 0`` gives ``p = 1`` and ``|rho| = 1`` gives
``p = 0`` exactly; degenerate inputs stay ``NaN``.
"""
from __future__ import annotations

import math
import numpy as np


# --- regularized incomplete beta (numpy/math only) ---------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta integral (NR betacf).

    Computes the continued fraction used to evaluate B(x; a, b); callers must
    supply the prefactor themselves via ``_betainc``.
    """
    maxit = 200
    eps = 3e-14
    fmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fmin:
        d = fmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fmin:
            d = fmin
        c = 1.0 + aa / c
        if abs(c) < fmin:
            c = fmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fmin:
            d = fmin
        c = 1.0 + aa / c
        if abs(c) < fmin:
            c = fmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) = B(x; a, b) / B(a, b), 0<=x<=1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    # symmetry: evaluate the continued fraction on the side closer to the median
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


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
    ranks; ``p`` is the two-sided p-value from the Student-t approximation
    ``t = rho*sqrt((n-2)/(1-rho^2))`` with ``df = n-2``
    (``p = I_{df/(df+t^2)}(df/2, 1/2)``, matching ``scipy.stats.spearmanr``;
    ``NaN`` when n < 4 or the correlation is undefined). Constant/degenerate
    input returns ``(nan, nan)`` -- never a silent 0.
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
    """Two-sided p-value for Spearman rho via the Student-t approximation.

    Under H0, ``t = rho * sqrt((n-2)/(1-rho^2))`` ~ Student-t(df=n-2); the
    two-sided p is ``2*P(T>=|t|) = I_{df/(df+t^2)}(df/2, 1/2)``, evaluated
    through the regularized incomplete beta (``_betainc``). This matches
    ``scipy.stats.spearmanr``'s default t approximation and is honest at small
    n (the previous Fisher-z normal approximation was far too aggressive).

    Boundaries: ``rho == 0`` -> ``p == 1``; ``|rho| == 1`` -> ``p == 0``;
    ``n < 4`` or non-finite/degenerate rho -> ``NaN``.
    """
    if n < 4:
        return float("nan")
    if rho is None or not np.isfinite(rho):
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0
    df = n - 2
    t = rho * math.sqrt(df / (1.0 - rho * rho))
    x = df / (df + t * t)
    return float(_betainc(df / 2.0, 0.5, x))


def kendall_tau(x, y):
    """Kendall's tau-b (tie-aware). Higher score means stronger agreement in
    ordering between ``x`` and ``y``; perfect agreement -> 1.0.

    We return only the tau-b statistic (scipy's ``tau_b`` variant,
    ``sqrt((n0-n1)(n0-n2))`` normalization) -- no asymptotic p-value: the
    audit's counterfactual_tau consumes just the magnitude, so a normal /
    permutation approximation for it is deliberately not taken.
    """
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
