"""Unit tests for the preregistered composite diagnostics and H1/H2/H3 verdicts.

These are pure (no training, no torch): synthetic rows with controlled values
pin down (a) the direction-unification math, (b) the H1/H2 threshold logic,
and (c) H3's honest not_assessable branch.
"""
import math

import numpy as np
import pytest

from wm_audit.diagnostics import (
    audit_correlations,
    control_composite_scores,
    pixel_composite_scores,
    prereg_decision_h1,
    prereg_decision_h2,
    prereg_decision_h3,
)


# --- synthetic rows ---------------------------------------------------------

def _pixel_row(mse, ssim, nll, lpips, fvd, **extra):
    row = {
        "pixel_mse": mse, "pixel_ssim": ssim, "pixel_nll": nll,
        "pixel_lpips": lpips, "rollout_fvd": fvd,
    }
    row.update(extra)
    return row


# --- composite direction unification -----------------------------------------

def test_pixel_composite_orients_error_and_quality_diagnostics():
    # 3 rows, each uniformly best / mid / worst on ALL five pixel diagnostics.
    rows = [
        _pixel_row(1.0, 0.9, 1.0, 0.1, 0.1),   # best fidelity
        _pixel_row(2.0, 0.5, 2.0, 0.2, 0.2),    # mid
        _pixel_row(3.0, 0.1, 3.0, 0.3, 0.3),     # worst fidelity
    ]
    scores = pixel_composite_scores(rows)
    # On n=3, uniform orientation gives the best row the top rank (3.0) and the
    # worst row the bottom rank (1.0), in the same order as R_real below.
    assert scores[0] == pytest.approx(3.0)
    assert scores[1] == pytest.approx(2.0)
    assert scores[2] == pytest.approx(1.0)
    assert scores[0] > scores[1] > scores[2]


def test_pixel_composite_flips_ssim_relative_to_error_metrics():
    # mse says row0 is best (lowest), ssim says row0 is worst (lowest). After
    # direction unification the composite must reflect BOTH consistently:
    # row0 = best-on-error (rank-flipped up) AND worst-on-ssim (no flip).
    rows = [
        _pixel_row(mse=1.0, ssim=0.1, nll=1.0, lpips=0.1, fvd=0.1),  # best error, worst ssim
        _pixel_row(mse=3.0, ssim=0.9, nll=3.0, lpips=0.3, fvd=0.3),  # worst error, best ssim
    ]
    scores = pixel_composite_scores(rows)
    # n=2 -> ranks are {1,2}; row0 is best on the 4 flipped diagnostics and
    # worst on ssim, row1 the mirror. Mean of 4*2 and 1 = 4.5 vs mean of 1*4
    # and 2 = 3.0; row0 wins (4 error diagnostics outweigh 1 ssim).
    assert scores[0] == pytest.approx((4 * 2.0 + 1.0) / 5.0)
    assert scores[1] == pytest.approx((4 * 1.0 + 2.0) / 5.0)
    assert scores[0] > scores[1]


def test_control_composite_flips_return_ece():
    # return_ece is error-type (lower=better), tau/unc_rho quality-type.
    rows = [
        {"return_ece": 1.0, "counterfactual_tau": 0.9, "uncertainty_rho": 0.9},
        {"return_ece": 3.0, "counterfactual_tau": 0.1, "uncertainty_rho": 0.1},
    ]
    scores = control_composite_scores(rows)
    # row0: best on ece (flipped -> rank 2), best on tau (ascending -> rank 2),
    # best on unc_rho (ascending -> rank 2) => 2.0. row1 => 1.0.
    assert scores[0] == pytest.approx(2.0)
    assert scores[1] == pytest.approx(1.0)


# --- H1 verdict --------------------------------------------------------------

def test_h1_supported_below_threshold():
    d = prereg_decision_h1(0.3, threshold=0.4)
    assert d["supports_H1"] is True
    assert d["observed_rho"] == pytest.approx(0.3)


def test_h1_not_supported_above_threshold_and_negative():
    assert prereg_decision_h1(0.6, threshold=0.4)["supports_H1"] is False
    # direction-blind: a strong negative correlation also falsifies H1
    assert prereg_decision_h1(-0.6, threshold=0.4)["supports_H1"] is False


def test_h1_nan_is_not_supported():
    assert prereg_decision_h1(float("nan"), threshold=0.4)["supports_H1"] is False


# --- H2 verdict --------------------------------------------------------------

def test_h2_supported_above_threshold():
    d = prereg_decision_h2(0.7, threshold=0.6)
    assert d["supports_H2"] is True


def test_h2_not_supported_below_threshold():
    d = prereg_decision_h2(0.5, threshold=0.6)
    assert d["supports_H2"] is False


def test_h2_negative_rho_not_supported():
    # a negative control composite correlation is not "strong positive"
    assert prereg_decision_h2(-0.7, threshold=0.6)["supports_H2"] is False


# --- H3 verdict ---------------------------------------------------------------

def _full_row(hidden, seed, compression=None, *,
              mse, ssim, nll, lpips, fvd, ece, tau, unc_rho):
    """A complete diagnostics row (all 5 pixel + all 3 control keys) so the
    H1/H2 rank composites are well-defined."""
    row = {
        "hidden": hidden, "seed": seed,
        "pixel_mse": mse, "pixel_ssim": ssim, "pixel_nll": nll,
        "pixel_lpips": lpips, "rollout_fvd": fvd,
        "return_ece": ece, "counterfactual_tau": tau,
        "uncertainty_rho": unc_rho,
    }
    if compression is not None:
        row["compression"] = compression
    return row


def test_h3_not_assessable_without_compression_rows():
    rows = [
        _full_row(16, 0, mse=0.01, ssim=0.9, nll=0.1, lpips=0.1, fvd=0.1,
                  ece=1.0, tau=0.9, unc_rho=0.9),
        _full_row(32, 1, mse=0.02, ssim=0.8, nll=0.2, lpips=0.2, fvd=0.2,
                  ece=2.0, tau=0.8, unc_rho=0.8),
    ]
    d = prereg_decision_h3(rows)
    assert d["status"] == "not_assessable"
    assert "compression" in d["reason"].lower() or "undefined" in d["reason"].lower()


def test_h3_not_assessable_when_compression_row_has_no_matched_baseline():
    # comp row on hidden=64, but no float baseline at (hidden=64, seed=0).
    rows = [
        _full_row(16, 0, mse=1.0, ssim=0.9, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=1.0, tau=0.9, unc_rho=0.9),
        _full_row(64, 0, compression="int8",
                  mse=1.0, ssim=0.9, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=3.0, tau=0.1, unc_rho=0.1),
    ]
    d = prereg_decision_h3(rows)
    assert d["status"] == "not_assessable"
    assert "baseline" in d["reason"].lower()


def test_h3_assessable_control_degrades_faster():
    # Float F is best-on-everything. Compressed C keeps pixel IDENTICAL to F
    # (no pixel degradation) but worsens all three control diagnostics.
    rows = [
        _full_row(16, 0, mse=1.0, ssim=0.9, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=1.0, tau=0.9, unc_rho=0.9),                 # F float
        _full_row(16, 0, compression="int8",
                  mse=1.0, ssim=0.9, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=3.0, tau=0.1, unc_rho=0.1),               # C compressed
    ]
    d = prereg_decision_h3(rows)
    assert d["status"] == "assessable"
    assert d["n_pairs"] == 1
    # pixel composites tied (identical) -> pixel degradation 0.0
    assert d["pixel_composite_degradation"] == pytest.approx(0.0)
    # control: F rank 2 on all three, C rank 1 on all three -> deg = 2.0-1.0 = 1.0
    assert d["control_composite_degradation"] == pytest.approx(1.0)
    assert d["supports_H3"] is True


def test_h3_not_supported_when_pixel_degrades_faster():
    # C worsens ALL five pixel diagnostics but keeps control identical to F.
    rows = [
        _full_row(16, 0, mse=1.0, ssim=0.9, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=1.0, tau=0.9, unc_rho=0.9),                 # F float
        _full_row(16, 0, compression="int8",
                  mse=3.0, ssim=0.1, nll=3.0, lpips=0.3, fvd=0.3,
                  ece=1.0, tau=0.9, unc_rho=0.9),               # C compressed
    ]
    d = prereg_decision_h3(rows)
    assert d["status"] == "assessable"
    # control tied -> 0.0; pixel F=2.0 C=1.0 -> deg 1.0
    assert d["control_composite_degradation"] == pytest.approx(0.0)
    assert d["pixel_composite_degradation"] == pytest.approx(1.0)
    assert d["supports_H3"] is False


def test_h3_control_side_includes_uncertainty_rho_and_tau_not_just_ece():
    # return_ece is IDENTICAL between F and C (ece-only would show zero control
    # degradation), but uncertainty_rho and counterfactual_tau degrade. The H2
    # control composite must still register a control degradation.
    rows = [
        _full_row(16, 0, mse=1.0, ssim=0.9, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=1.0, tau=0.9, unc_rho=0.9),                 # F float
        _full_row(16, 0, compression="int8",
                  mse=1.0, ssim=0.9, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=1.0, tau=0.1, unc_rho=0.1),               # C: ece same, tau/unc worse
    ]
    d = prereg_decision_h3(rows)
    assert d["status"] == "assessable"
    assert d["pixel_composite_degradation"] == pytest.approx(0.0)
    # ece tied -> rank 1.5 each; tau/unc F=2 C=1. control F=(1.5+2+2)/3, C=(1.5+1+1)/3
    expected = ((1.5 + 2 + 2) - (1.5 + 1 + 1)) / 3.0
    assert d["control_composite_degradation"] == pytest.approx(expected)
    assert d["control_composite_degradation"] > 0.0   # rho/tu NOT dropped
    assert d["supports_H3"] is True


def test_h3_pixel_side_uses_composite_not_pixel_mse_alone():
    # mSE alone says C degraded (1.0 -> 1.1), but ssim compensates (0.5 -> 0.9)
    # and nll/lpips/fvd are tied. The pixel COMPOSITE is unchanged -> pixel
    # degradation 0.0. Control is tied -> control degradation 0.0.
    rows = [
        _full_row(16, 0, mse=1.0, ssim=0.5, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=1.0, tau=0.9, unc_rho=0.9),                 # F float
        _full_row(16, 0, compression="int8",
                  mse=1.1, ssim=0.9, nll=1.0, lpips=0.1, fvd=0.1,
                  ece=1.0, tau=0.9, unc_rho=0.9),               # C: mse worse, ssim better
    ]
    d = prereg_decision_h3(rows)
    assert d["status"] == "assessable"
    assert d["pixel_composite_degradation"] == pytest.approx(0.0)
    assert d["control_composite_degradation"] == pytest.approx(0.0)
    assert d["supports_H3"] is False


# --- composite flows through audit_correlations ------------------------------

def test_composite_column_correlates_with_return():
    # Perfect monotonicity: better fidelity -> better return.
    rows = [
        _pixel_row(3.0, 0.1, 3.0, 0.3, 0.3, return_ece=3.0,
                   counterfactual_tau=0.1, uncertainty_rho=0.1,
                   R_real=-100.0),
        _pixel_row(2.0, 0.5, 2.0, 0.2, 0.2, return_ece=2.0,
                   counterfactual_tau=0.5, uncertainty_rho=0.5,
                   R_real=-50.0),
        _pixel_row(1.0, 0.9, 1.0, 0.1, 0.1, return_ece=1.0,
                   counterfactual_tau=0.9, uncertainty_rho=0.9,
                   R_real=-10.0),
    ]
    scores = pixel_composite_scores(rows)
    for r, s in zip(rows, scores):
        r["pixel_composite"] = float(s)
    res = audit_correlations(rows, ["pixel_composite"], return_key="R_real",
                             n_boot=20)
    # Monotone increasing fidelity vs increasing return -> rho = +1.
    assert res["pixel_composite"]["rho"] == pytest.approx(1.0)
