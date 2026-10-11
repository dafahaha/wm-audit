"""Multiple-comparison correction (Holm-Bonferroni step-down).

Used so that the many diagnostic-vs-return correlations are not
cherry-picked: every reported diagnostic gets an adjusted p-value under the
preregistered family.
"""
from __future__ import annotations

import numpy as np


def holm_correction(pvals):
    """Return Holm-adjusted p-values aligned to the input order.

    ``NaN`` p-values (undefined tests) are treated as non-significant (1.0)
    for the correction. The step-down monotonicity constraint is enforced so
    adjusted p-values are non-decreasing along the sorted order.
    """
    p = np.asarray(pvals, dtype=float).copy()
    bad = ~np.isfinite(p)
    p[bad] = 1.0
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running_min = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        val = min(val, 1.0)
        val = max(val, running_min)
        adj[idx] = val
        running_min = val
    return adj
