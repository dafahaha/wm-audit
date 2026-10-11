"""test_counterfactual: (s,a1,a2) ordering agrees with oracle on a perfect model."""
import numpy as np

from wm_audit.diagnostics.control import counterfactual_kendall_tau


def test_perfect_model_tau_is_one():
    rng = np.random.default_rng(0)
    oracle = rng.uniform(-2, 2, size=100)
    imagined = oracle.copy()        # perfect world model reproduces oracle values
    assert abs(counterfactual_kendall_tau(oracle, imagined) - 1.0) < 1e-6


def test_noisy_model_tau_partial():
    rng = np.random.default_rng(1)
    oracle = np.linspace(0, 1, 50)
    imagined = oracle + rng.normal(0, 0.05, size=50)   # small noise keeps ordering
    tau = counterfactual_kendall_tau(oracle, imagined)
    assert tau > 0.8


def test_reversed_model_tau_is_negative():
    oracle = np.arange(10.0)
    imagined = np.arange(10.0)[::-1]
    tau = counterfactual_kendall_tau(oracle, imagined)
    assert tau < -0.9
