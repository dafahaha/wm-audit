"""Statistical core for WM-Audit.

All functions here are dependency-light (numpy only), deterministic given an
explicit RNG seed, and independently unit-tested. They implement the
preregistered inferential layer (Spearman / Kendall / bootstrap CI / Holm)
in the spirit of the ``transit-truth`` audit methodology.
"""
from .rank import spearmanr, kendall_tau, rankdata, spearman_pvalue
from .bootstrap import bootstrap_ci
from .multiple import holm_correction

__all__ = [
    "spearmanr",
    "kendall_tau",
    "rankdata",
    "spearman_pvalue",
    "bootstrap_ci",
    "holm_correction",
]
