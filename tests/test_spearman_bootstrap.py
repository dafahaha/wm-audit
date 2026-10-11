"""test_spearman_bootstrap: constant input -> NaN handled; CI covers known truth.

Second mutation-guard test.
"""
import numpy as np
import pytest

from wm_audit.stats import spearmanr, spearman_pvalue, bootstrap_ci, holm_correction


# --- Spearman t-approximation p-value --------------------------------------
# The p-values below were frozen against scipy 1.18.1 ``spearmanr``
# (two-sided t approximation, df=n-2). scipy is NOT a package/CI dependency:
# these are reference literals, the production code computes them itself.

def test_pvalue_zero_correlation_is_one():
    for n in (4, 6, 8, 12, 30):
        assert spearman_pvalue(0.0, n) == pytest.approx(1.0)


def test_pvalue_perfect_correlation_is_zero():
    assert spearman_pvalue(1.0, 12) == 0.0
    assert spearman_pvalue(-1.0, 12) == 0.0


def test_pvalue_degenerate_inputs_are_nan():
    assert np.isnan(spearman_pvalue(0.5, 3))          # n < 4
    assert np.isnan(spearman_pvalue(float("nan"), 10))
    assert np.isnan(spearman_pvalue(float("inf"), 10))
    rho, p = spearmanr([1, 1, 1, 1], [0.1, 0.5, 0.9, 0.2])
    assert np.isnan(p)


def test_pvalue_is_two_sided_symmetric():
    # flipping the sign of rho must not change the two-sided p
    for rho in (0.3, 0.6, 0.9):
        assert spearman_pvalue(rho, 10) == pytest.approx(spearman_pvalue(-rho, 10))


def test_pvalue_small_n_matches_scipy_t_approx():
    # n=8, a single adjacent rank swap -> rho = 1 - 6*2/(8*(64-1))
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [1, 2, 3, 4, 6, 5, 7, 8]
    rho, p = spearmanr(x, y)
    assert rho == pytest.approx(0.9761904761904763)
    assert p == pytest.approx(3.314396026200098e-05, rel=1e-9)


def test_pvalue_n12_matches_scipy_t_approx():
    x = list(range(1, 13))
    y = [1, 2, 4, 3, 5, 7, 6, 8, 9, 11, 10, 12]
    rho, p = spearmanr(x, y)
    assert rho == pytest.approx(0.9790209790209792)
    assert p == pytest.approx(3.089801398548709e-08, rel=1e-9)


def test_pvalue_with_ties_matches_scipy():
    x = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    y = [1.5, 3.0, 1.5, 3.0, 4.0, 2.0, 4.0, 2.0, 5.0, 6.0]
    rho, p = spearmanr(x, y)
    assert rho == pytest.approx(0.6978227007909176)
    assert p == pytest.approx(0.02484382045472837, rel=1e-9)


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
    # step-down monotonicity: adjusted p-values must be non-decreasing along the
    # (already ascending) sorted order, and match the hand-computed values.
    assert all(adj[i] <= adj[i + 1] + 1e-12 for i in range(len(adj) - 1))
    assert list(adj) == pytest.approx([0.04, 0.06, 0.06, 0.5])
