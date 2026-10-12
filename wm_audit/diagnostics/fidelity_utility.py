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


def _insertion_rank_oriented(x: float, ref_values: Sequence[float],
                             lower_is_better: bool) -> float:
    """Competition / insertion rank of ONE value against a FIXED reference ladder.

    H3 is a *paired* test (each compressed row vs its own float baseline). The
    rank-normalisation scale must therefore NOT move when we add/remove other
    compression levers -- it is locked to the float baseline set, which is the
    capacity ladder. For each diagnostic we take the float rows' values as the
    fixed reference distribution ``ref`` (size ``n``), and ask where a compressed
    row's value ``x`` would slot in on that SAME ladder:

        ins_asc(x) = #{s in ref : s < x} + 0.5 * #{s in ref : s == x} + 0.5

    This is the average-rank / midrank insertion convention:
      * x strictly above every ref value  -> n + 0.5 (best possible compressed),
      * x strictly below every ref value  -> 0.5    (worst possible compressed),
      * x tying a ref tie-block           -> reproduces that block's ``rankdata``
        average rank exactly, so a compressed row identical to its float
        baseline scores the SAME as the baseline on every diagnostic (paired
        degradation 0.0).

    ``lower_is_better`` flips the ascending insertion just like
    ``_oriented_ranks`` flips ascending ranks, so a HIGHER number always means
    BETTER fidelity/control. Crucially this depends ONLY on the fixed float
    reference set -- never on how many compressed rows are present -- so adding
    or deleting another lever cannot shift a shared pair's degradation.
    """
    ref = np.asarray(ref_values, dtype=float).reshape(-1)
    n = ref.size
    k = float(np.sum(ref < x))     # reference values strictly below x
    m = float(np.sum(ref == x))    # reference values tied with x
    ins_asc = k + 0.5 * m + 0.5
    if lower_is_better:
        return (n + 1.0) - ins_asc
    return ins_asc


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


def h3_pair_degradations(
    rows: List[dict],
    compression_key: str = COMPRESSION_KEY,
) -> Dict[str, object]:
    """Per-pair H3 degradation on the FIXED float reference ladder.

    Returns ``{"float_rows": [...], "pairs": [ {pair dict}, ... ]}``. Each pair
    dict carries the matched ``(hidden, embed, seed)`` triple, the lever tag,
    the float baseline composite, the inserted compressed composite, and the
    paired degradation on the pixel and control sides. Pairs are listed in
    compression-row order; unmatched compression rows are simply absent.

    The reference ladder is ``float_rows`` ONLY, so a given compressed row's
    inserted composite -- and therefore its paired degradation -- does not
    change when other compression levers are added or removed (the H3
    invariance property this module is locked around).
    """
    comp_rows = [r for r in rows if r.get(compression_key)]
    float_rows = [r for r in rows if not r.get(compression_key)]

    pairs: List[dict] = []
    if comp_rows and float_rows:
        float_pixel = composite_score(float_rows, PIXEL_DIAGNOSTICS,
                                      _PIXEL_LOWER_IS_BETTER)
        float_control = composite_score(float_rows, CONTROL_DIAGNOSTICS,
                                        _CONTROL_LOWER_IS_BETTER)
        float_lookup: Dict[tuple, tuple] = {}
        for fr, ps, cs in zip(float_rows, float_pixel, float_control):
            key = (fr.get("hidden"), fr.get("embed"), fr.get("seed"))
            float_lookup.setdefault(key, (float(ps), float(cs)))

        def _inserted_composite(row: dict, diagnostics: Sequence[str],
                                lower_is_better_map: Dict[str, bool]) -> float:
            total = 0.0
            for d in diagnostics:
                ref = [fr[d] for fr in float_rows]
                total += _insertion_rank_oriented(row[d], ref,
                                                  lower_is_better_map[d])
            return total / float(len(diagnostics))

        for r in comp_rows:
            key = (r.get("hidden"), r.get("embed"), r.get("seed"))
            if key not in float_lookup:
                continue   # no exact triple match -> dropped, no wider-key fallback
            fps, fcs = float_lookup[key]
            cps = _inserted_composite(r, PIXEL_DIAGNOSTICS, _PIXEL_LOWER_IS_BETTER)
            ccs = _inserted_composite(r, CONTROL_DIAGNOSTICS, _CONTROL_LOWER_IS_BETTER)
            pairs.append({
                "hidden": r.get("hidden"), "embed": r.get("embed"),
                "seed": r.get("seed"), "compression": r.get(compression_key),
                "float_pixel_composite": fps, "float_control_composite": fcs,
                "compressed_pixel_composite": cps,
                "compressed_control_composite": ccs,
                "pixel_degradation": fps - cps,
                "control_degradation": fcs - ccs,
            })
    return {"float_rows": float_rows, "pairs": pairs}


def prereg_decision_h3(rows: List[dict],
                       compression_key: str = COMPRESSION_KEY) -> dict:
    """H3: under compression, pixel fidelity degrades slowly while return
    calibration and uncertainty alignment degrade faster.

    Operationalization (faithful to ``configs/preregistration.yaml``):

    Fixed reference ladder (invariance-critical)
    --------------------------------------------
    H3 is a *paired* test -- each compressed row is judged against its own float
    baseline on the exact ``(hidden, embed, seed)`` triple. The rank scale we
    measure degradation on MUST be fixed, so it is locked to the float baseline
    set (the capacity ladder), NOT to the float+compressed union:

      1. Split ``rows`` into ``float_rows`` (no ``compression`` key) and
         ``comp_rows``. The float rows alone form the fixed reference ladder.
      2. Compute the H1/H2 direction-unified rank composites ON ``float_rows``
         (``composite_score`` -> ``rankdata`` oriented ranks) to score every
         float baseline.
      3. For every compressed row, score it on that SAME ladder by INSERTING
         each of its diagnostic values into the fixed float reference
         distribution (``_insertion_rank_oriented``), then averaging the
         inserted ranks into a pixel composite and a control composite. Because
         the reference set is the fixed float ladder, the inserted rank of a
         given compressed value does not depend on how many other compressed
         rows exist -- adding/deleting a lever cannot move it.
      4. Match the compressed row to its float baseline by the EXACT triple
         ``(hidden, embed, seed)`` (first float row in row order wins; there is
         NO wider-key fallback -- an unmatched compression row is dropped).
      5. Paired degradation = float_composite - compressed_composite on each
         side (both composites are "higher is better", so positive = the
         compressed model scored worse / degraded). Average over all matched
         pairs.

      * H3 is supported iff the mean control-composite degradation strictly
        exceeds the mean pixel-composite degradation. No post-hoc threshold.

    Self-consistency: a compressed row identical to its float baseline on every
    diagnostic inserts at exactly the baseline's own ladder rank -> 0.0
    degradation on both sides.

    If the run contains no compression rows, no float rows, or no compression
    row matches a float baseline, the verdict is honestly ``not_assessable``.
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

    ladder = h3_pair_degradations(rows, compression_key)
    float_rows = ladder["float_rows"]
    pairs = ladder["pairs"]

    if not float_rows:
        base["status"] = "not_assessable"
        base["reason"] = (
            "Compression rows exist but no float baseline rows are present to "
            "form the fixed reference ladder; degradation cannot be computed."
        )
        return base

    if not pairs:
        base["status"] = "not_assessable"
        base["reason"] = (
            "Compression rows exist but none matched a float baseline by the "
            "exact (hidden, embed, seed) triple (no wider-key fallback is "
            "performed); degradation cannot be computed."
        )
        return base

    pixel_delta = [p["pixel_degradation"] for p in pairs]
    control_delta = [p["control_degradation"] for p in pairs]
    pixel_deg = float(np.mean(pixel_delta))
    control_deg = float(np.mean(control_delta))
    support = bool(control_deg > pixel_deg)
    base.update({
        "status": "assessable",
        "compression_lever": comp_rows[0].get(compression_key),
        "n_pairs": len(pixel_delta),
        "reference_ladder_size": len(float_rows),
        "pixel_composite_degradation": pixel_deg,
        "control_composite_degradation": control_deg,
        "supports_H3": support,
        "note": ("H3 supported: control/calibration composite degradation "
                 "exceeds pixel composite degradation under compression."
                 if support else
                 "H3 NOT supported: pixel composite degradation is not slower "
                 "than control/calibration composite degradation under "
                 "compression."),
    })
    return base
