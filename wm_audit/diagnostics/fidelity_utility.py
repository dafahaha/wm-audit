"""The core audit: correlate every diagnostic with real closed-loop return.

Given a table of rows, each row = one (architecture, capacity, seed, [compression])
configuration holding a vector of diagnostic values plus the measured real
closed-loop return ``R_real``, compute for each diagnostic:

    corr(diagnostic, R_real) = Spearman rho + 95% percentile-bootstrap CI
    + raw p-value, then Holm-adjust all p-values over the diagnostic family.

This is exactly the RQ1/RQ2 inferential step in the preregistration.

Composite diagnostics (H1 / H2)
-------------------------------
The preregistration locks H1 to a *composite* of the five pixel diagnostics
(mse / ssim / nll / lpips / fvd), not to pixel_mse alone. The five quantities
have mixed units and mixed directions: mse / nll / lpips / fvd are error-type
metrics (lower = better fidelity), while ssim is a quality-type metric
(higher = better fidelity). To combine them into one comparable score we use a
direction-unified rank composite (the exact recipe below is mirrored verbatim
in ``configs/preregistration.yaml``):

  1. Within the configuration set (one score per row), rank each diagnostic
     ascending with average ranks for ties (see ``stats.rank.rankdata``).
  2. Flip the rank of every error-type diagnostic so that a HIGHER rank always
     means BETTER fidelity. Quality-type diagnostics keep ascending ranks.
  3. ``pixel_composite`` = mean of the five oriented ranks.

The same recipe applies to the H2 control battery (return_ece /
counterfactual_tau / uncertainty_rho), where return_ece is the only error-type
metric and is flipped.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np

from ..stats.rank import rankdata, spearmanr, spearman_pvalue
from ..stats.bootstrap import bootstrap_ci
from ..stats.multiple import holm_correction


# --- Preregistered diagnostic batteries -------------------------------------

#: Five pixel diagnostics locked by preregistration H1.
PIXEL_DIAGNOSTICS: Sequence[str] = (
    "pixel_mse", "pixel_ssim", "pixel_nll", "pixel_lpips", "rollout_fvd",
)

#: True => "lower value is BETTER fidelity" (error metric);
#: False => "higher value is BETTER fidelity" (quality metric).
_PIXEL_LOWER_IS_BETTER: Dict[str, bool] = {
    "pixel_mse": True,
    "pixel_nll": True,
    "pixel_lpips": True,
    "rollout_fvd": True,
    "pixel_ssim": False,   # ssim: higher = better
}

#: Control-relevant diagnostics locked by preregistration H2.
CONTROL_DIAGNOSTICS: Sequence[str] = (
    "return_ece", "counterfactual_tau", "uncertainty_rho",
)

_CONTROL_LOWER_IS_BETTER: Dict[str, bool] = {
    "return_ece": True,        # calibration error: lower = better
    "counterfactual_tau": False,
    "uncertainty_rho": False,
}

#: Row key under which a compression lever (int8 / channel_prune / distill) is
#: recorded. Absent or falsy => the float baseline row.
COMPRESSION_KEY = "compression"


def _oriented_ranks(values: Sequence[float], lower_is_better: bool) -> np.ndarray:
    """Ascending ranks flipped so a HIGHER rank means BETTER fidelity/control.

    Error-type diagnostics (lower value = better) get flipped; quality-type
    diagnostics (higher value = better) keep ascending ranks. Ties use average
    ranks (see ``rankdata``).
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    r = rankdata(v)  # ascending: smallest value -> rank 1
    if lower_is_better:
        n = v.size
        r = (n + 1.0) - r  # smallest (best) -> largest rank
    return r


def composite_score(
    rows: List[dict],
    diagnostics: Sequence[str],
    lower_is_better_map: Dict[str, bool],
) -> np.ndarray:
    """Direction-unified rank composite -> one score per row.

    Higher score = better fidelity (pixel) / better control relevance.
    """
    n = len(rows)
    score = np.zeros(n, dtype=float)
    for d in diagnostics:
        vals = [r[d] for r in rows]
        score += _oriented_ranks(vals, lower_is_better_map[d])
    return score / float(len(diagnostics))


def pixel_composite_scores(rows: List[dict]) -> np.ndarray:
    """Preregistered H1 composite: mean of the five direction-unified pixel ranks."""
    return composite_score(rows, PIXEL_DIAGNOSTICS, _PIXEL_LOWER_IS_BETTER)


def control_composite_scores(rows: List[dict]) -> np.ndarray:
    """Preregistered H2 composite: mean of the three direction-unified control ranks."""
    return composite_score(rows, CONTROL_DIAGNOSTICS, _CONTROL_LOWER_IS_BETTER)


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


# --- Preregistered verdicts --------------------------------------------------

def prereg_decision_h1(rho_pixel: float, threshold: float = 0.4) -> dict:
    """H1: pixel fidelity weakly predicts closed-loop control utility.

    Supports H1 iff ``|rho(pixel_composite, R_real)| < threshold``. Reported
    honestly -- a null/bounded result (pixel fidelity *does* predict control) is
    still publishable.
    """
    support = bool(np.isfinite(rho_pixel) and abs(rho_pixel) < threshold)
    return {
        "hypothesis": "H1: |rho(pixel_composite, R_real)| < 0.4",
        "observed_rho": float(rho_pixel),
        "threshold": threshold,
        "supports_H1": support,
        "note": ("H1 supported (pixel fidelity weakly tracks control utility)."
                 if support else
                 "H1 NOT supported: pixel fidelity tracks control utility more "
                 "than preregistered -- report as a bounded/null-result."),
    }


def prereg_decision_h2(rho_control: float, threshold: float = 0.6) -> dict:
    """H2: control-relevant diagnostics strongly predict closed-loop utility.

    Supports H2 iff ``rho(control_composite, R_real) > threshold`` on the
    direction-unified control composite (return_ece flipped so higher = better
    calibration; counterfactual_tau and uncertainty_rho ascending).
    """
    support = bool(np.isfinite(rho_control) and rho_control > threshold)
    return {
        "hypothesis": "H2: rho(control_composite, R_real) > 0.6",
        "observed_rho": float(rho_control),
        "threshold": threshold,
        "supports_H2": support,
        "note": ("H2 supported: control-relevant diagnostics strongly predict "
                 "closed-loop return."
                 if support else
                 "H2 NOT supported: the control composite does not track "
                 "closed-loop return above the preregistered threshold."),
    }


def prereg_decision_h3(rows: List[dict],
                       compression_key: str = COMPRESSION_KEY) -> dict:
    """H3: under compression, pixel fidelity degrades slowly while return
    calibration and uncertainty alignment degrade faster.

    If the run contains no compression rows, the verdict is honestly
    ``not_assessable`` -- we never fabricate a compression comparison.
    """
    comp_rows = [r for r in rows if r.get(compression_key)]
    base = {
        "hypothesis": ("H3: under compression (int8/channel_prune/distill), "
                       "pixel fidelity degrades slowly while return-calibration "
                       "and uncertainty alignment degrade faster"),
    }
    if not comp_rows:
        base["status"] = "not_assessable"
        base["reason"] = (
            "No compression configurations present in this run; the pixel-vs-"
            "control degradation comparison is undefined in this config "
            "(smoke runs float baselines only)."
        )
        return base

    pixel_delta: List[float] = []
    control_delta: List[float] = []
    for r in comp_rows:
        # matched float baseline = same (hidden, seed), uncompressed
        b = next((x for x in rows
                  if not x.get(compression_key)
                  and x.get("hidden") == r.get("hidden")
                  and x.get("seed") == r.get("seed")), None)
        if b is None:
            continue
        pixel_delta.append(abs(r["pixel_mse"] - b["pixel_mse"])
                           / (abs(b["pixel_mse"]) + 1e-12))
        control_delta.append(
            abs(r["return_ece"] - b["return_ece"])
            / (abs(b["return_ece"]) + 1e-12)
        )
    if not pixel_delta:
        base["status"] = "not_assessable"
        base["reason"] = (
            "Compression rows exist but none matched a float baseline by "
            "(hidden, seed); degradation cannot be computed."
        )
        return base

    pixel_deg = float(np.mean(pixel_delta))
    control_deg = float(np.mean(control_delta))
    support = bool(control_deg > pixel_deg)
    base.update({
        "status": "assessable",
        "compression_lever": comp_rows[0].get(compression_key),
        "pixel_relative_degradation": pixel_deg,
        "control_relative_degradation": control_deg,
        "supports_H3": support,
        "note": ("H3 supported: control/calibration degradation exceeds pixel "
                 "degradation under compression."
                 if support else
                 "H3 NOT supported: pixel degradation is not slower than "
                 "control/calibration degradation under compression."),
    })
    return base
