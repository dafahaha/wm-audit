"""--jobs is a wall-clock switch, not a results switch: the guard for that claim.

`experiments/run_audit.py` promises that the worker grid only changes
wall-clock. That promise holds at the level the audit actually depends on --
row order, and the rank ordering that Spearman turns into every reported rho --
but it does NOT hold bit-for-bit: each worker pins torch to one thread while
the serial path leaves thread count alone, so reduction order can move the last
few bits (measured: 1.5e-7 worst relative difference on the smoke-scale grid).

This test pins both halves of that sentence so neither can rot silently:

  1. the same rows come back in the same order with the same keys,
  2. ``R_real`` -- the closed-loop return the whole audit is built on --
     is bit-identical between the two paths,
  3. every other diagnostic agrees to 1e-5 relative (far tighter than any
     difference that could reorder a 15-point rank correlation),
  4. the correlation table computed from either set is identical to 1e-9.

Runtime is ~40 s on CI hardware: one serial pass plus one 2-worker pass over a
3-config tiny grid. It is deliberately not part of the smoke mode.
"""
from __future__ import annotations

import math

import pytest

from wm_audit.audit import evaluate_config
from wm_audit.diagnostics import audit_correlations

TINY = dict(
    buffer_capacity=500, collect_steps=400, data_seed=7,
    ensemble_size=2, train_steps=20, batch_size=32, lr=2e-3,
    calib_segments=6, calib_horizon=4, cf_pairs=10,
    cem_horizon=3, cem_candidates=16, cem_elites=4, cem_iters=1,
)

# Three configs: two seeds at one capacity plus one capacity tier, so the
# rank ordering has something to get wrong (a 2-row grid would be uninformative
# -- any permutation is a valid rank flip).
GRID = [(16, 16, 0), (16, 16, 1), (32, 16, 0)]

# Fields that must match exactly, and the tolerance for the rest.
EXACT_FIELDS = ("R_real",)
REL_TOL = 1e-5


def _eval_task(task):
    """Mirror of experiments.run_audit._eval_task (module-level: picklable)."""
    env, hidden, embed, seed, cfg = task
    import torch
    torch.set_num_threads(1)
    return evaluate_config(env, hidden=hidden, embed=embed, seed=seed, cfg=cfg)


def _serial(grid, cfg):
    rows = []
    for (hidden, embed, seed) in grid:
        rows.extend(evaluate_config("pixel_nav", hidden=hidden, embed=embed,
                                    seed=seed, cfg=cfg))
    return rows


def _parallel(grid, cfg, workers=2):
    from concurrent.futures import ProcessPoolExecutor

    tasks = [("pixel_nav", h, e, s, cfg) for (h, e, s) in grid]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for new_rows in ex.map(_eval_task, tasks):
            rows.extend(new_rows)
    return rows


@pytest.fixture(scope="module")
def two_runs():
    serial = _serial(GRID, TINY)
    parallel = _parallel(GRID, TINY, workers=2)
    return serial, parallel


def test_same_rows_in_same_order(two_runs):
    serial, parallel = two_runs
    assert len(serial) == len(parallel) == len(GRID)
    for a, b in zip(serial, parallel):
        assert sorted(a) == sorted(b), "row schema drifted between paths"


def test_closed_loop_return_is_bit_identical(two_runs):
    serial, parallel = two_runs
    for a, b in zip(serial, parallel):
        for k in EXACT_FIELDS:
            # Not approx(): a different R_real means the planner took a
            # different action, which is a real behavioural difference and
            # would invalidate the "wall-clock only" claim outright.
            assert a[k] == b[k], f"{k} differs: {a[k]!r} vs {b[k]!r}"


def test_diagnostics_match_within_tolerance(two_runs):
    serial, parallel = two_runs
    for a, b in zip(serial, parallel):
        for k in a:
            if isinstance(a[k], float) and isinstance(b[k], float):
                assert math.isclose(a[k], b[k], rel_tol=REL_TOL, abs_tol=1e-12), (
                    f"{k} drifted beyond {REL_TOL}: {a[k]!r} vs {b[k]!r}")


def test_correlation_table_identical(two_runs):
    serial, parallel = two_runs
    diags = ["pixel_mse", "return_ece", "counterfactual_tau"]
    res_s = audit_correlations(serial, diags, return_key="R_real", n_boot=50)
    res_p = audit_correlations(parallel, diags, return_key="R_real", n_boot=50)
    for d in diags:
        assert math.isclose(res_s[d]["rho"], res_p[d]["rho"],
                            rel_tol=1e-9, abs_tol=1e-9), (
            f"{d}: rho serial={res_s[d]['rho']!r} parallel={res_p[d]['rho']!r}")


def test_no_hidden_tagging_difference(two_runs):
    """Neither path may invent or drop a compression tag."""
    serial, parallel = two_runs
    tags_s = [r.get("compression", "float") for r in serial]
    tags_p = [r.get("compression", "float") for r in parallel]
    assert tags_s == tags_p
