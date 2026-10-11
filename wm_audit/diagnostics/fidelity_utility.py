"""The core audit: correlate every diagnostic with real closed-loop return.

Given a table of rows, each row = one (architecture, capacity, seed, [compression])
configuration holding a vector of diagnostic values plus the measured real
closed-loop return ``R_real``, compute for each diagnostic:

    corr(diagnostic, R_real) = Spearman rho + 95% percentile-bootstrap CI
    + raw p-value, then Holm-adjust all p-values over the diagnostic family.

This is exactly the RQ1/RQ2 inferential step in the preregistration.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..stats.rank import spearmanr, spearman_pvalue
from ..stats.bootstrap import bootstrap_ci
from ..stats.multiple import holm_correction


def _spearman_only(x, y) -> float:
    rho, _ = spearmanr(x, y)
    return rho


def audit_correlations(
    rows: List[dict],
    diagnostics: List[str],
    return_key: str = "R_real",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, dict]:
    """Correlate each diagnostic column with the closed-loop return column.

    ``rows``: list of dicts; each must contain every key in ``diagnostics`` and
    ``return_key``. Returns an ordered dict diagnostic ->
    ``{rho, ci_lo, ci_hi, p_raw, p_adj, n}``.
    """
    returns = np.asarray([r[return_key] for r in rows], dtype=float)
    n = returns.size
    out: Dict[str, dict] = {}
    raw_p = []
    for d in diagnostics:
        vals = np.asarray([r[d] for r in rows], dtype=float)
        rho = _spearman_only(vals, returns)
        _, lo, hi = bootstrap_ci(vals, returns, _spearman_only,
                                 n_boot=n_boot, ci=ci, seed=seed)
        p = spearman_pvalue(rho, n)
        out[d] = {"rho": rho, "ci_lo": lo, "ci_hi": hi, "p_raw": p, "n": n}
        raw_p.append(p)
    adj = holm_correction(raw_p)
    for d, a in zip(diagnostics, adj):
        out[d]["p_adj"] = float(a)
    return out


def prereg_decision(rho_pixel: float, threshold: float = 0.4) -> dict:
    """Preregistered H1 verdict: pixel fidelity weakly predicts control utility.

    H1 predicts ``|rho(pixel, R_real)| < threshold``. We report direction
    honestly -- a null/bounded result (pixel fidelity *does* predict control) is
    still a publishable finding.
    """
    support = bool(np.isfinite(rho_pixel) and abs(rho_pixel) < threshold)
    return {
        "hypothesis": "H1: |rho(pixel_fidelity, R_real)| < 0.4",
        "observed_rho": rho_pixel,
        "threshold": threshold,
        "supports_H1": support,
        "note": ("H1 supported (pixel fidelity weakly tracks control utility)."
                 if support else
                 "H1 NOT supported: pixel fidelity tracks control utility more "
                 "than preregistered -- report as a bounded/null-result."),
    }
