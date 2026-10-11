"""test_calibration: perfect calibration -> ECE ~0; monotone offset -> ECE grows.

This is one of the two mutation-guard tests (see README 'mutation checks').
"""
import numpy as np

from wm_audit.diagnostics.control import regression_ece


def test_perfect_calibration_zero():
    rng = np.random.default_rng(0)
    pred = rng.uniform(-1, 1, size=200)
    actual = pred.copy()           # imagined return == real return everywhere
    assert regression_ece(pred, actual, n_bins=10) < 1e-6


def test_offset_increases_ece_monotonically():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0.0, 1.0, size=500)
    actual = pred.copy()
    base = regression_ece(pred, actual)
    eces = [base]
    for shift in (0.1, 0.2, 0.4):
        shifted = pred + shift     # systematic over-prediction
        e = regression_ece(shifted, actual, n_bins=10)
        eces.append(e)
    # ECE must grow as the systematic offset grows
    assert eces[1] > eces[0]
    assert eces[2] > eces[1]
    assert eces[3] > eces[2]
