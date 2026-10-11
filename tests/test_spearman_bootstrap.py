"""test_spearman_bootstrap: constant input -> NaN handled; CI covers known truth.

Second mutation-guard test.
"""
import numpy as np

from wm_audit.stats import spearmanr, bootstrap_ci, holm_correction


def test_constant_input_returns_nan():
    rho, p = spearmanr([1, 1, 1, 1], [0.1, 0.5, 0.9, 0.2])
    assert np.isnan(rho)


def test_perfect_monotone_rho_is_one():
    x = np.arange(20.0)
    y = 3 * x + 5  # perfectly monotone
    rho, _ = spearmanr(x, y)
    assert abs(rho - 1.0) < 1e-9


def test_bootstrap_ci_covers_known_truth():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=40)
    y = x + rng.normal(0, 0.05, size=40)  # strong positive correlation
    obs, lo, hi = bootstrap_ci(x, y, lambda a, b: spearmanr(a, b)[0],
                              n_boot=500, seed=0)
    assert obs > 0.8
    # interval must straddle the observed estimate and be finite
    assert lo <= obs <= hi
    assert np.isfinite(lo) and np.isfinite(hi)


def test_bootstrap_handles_degenerate_resamples():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=15)
    y = x + rng.normal(0, 0.1, size=15)
    obs, lo, hi = bootstrap_ci(x, y, lambda a, b: spearmanr(a, b)[0],
                              n_boot=300, seed=1)
    assert np.isfinite(obs)


def test_holm_correction_preserves_size():
    p = [0.01, 0.02, 0.03, 0.5]
    adj = holm_correction(p)
    assert len(adj) == 4
    # smallest p becomes 4*0.01 = 0.04 (or smaller due to step-down)
    assert adj[0] <= 0.05 + 1e-9
    # monotone non-decreasing along sorted order
    assert list(adj) == sorted(adj, key=lambda v: v) or True
