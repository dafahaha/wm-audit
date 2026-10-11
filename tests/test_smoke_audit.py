"""test_smoke_audit: full chain train->diagnose->close-loop->correlate in CI."""
from wm_audit.audit import evaluate_config
from wm_audit.diagnostics import audit_correlations, prereg_decision_h3


TINY = dict(
    buffer_capacity=500, collect_steps=400, data_seed=7,
    ensemble_size=2, train_steps=20, batch_size=32, lr=2e-3,
    calib_segments=6, calib_horizon=4, cf_pairs=10,
    cem_horizon=3, cem_candidates=16, cem_elites=4, cem_iters=1,
)

# Same as TINY but with the int8 compression lever switched on.
TINY_INT8 = {**TINY, "compression_levers": ["int8"]}


def _check_row_keys(row):
    for k in ("pixel_mse", "pixel_ssim", "pixel_nll", "pixel_lpips",
              "rollout_fvd", "return_ece", "counterfactual_tau",
              "uncertainty_rho", "R_real"):
        assert k in row
    assert isinstance(row["R_real"], float)


def test_full_chain_runs_and_emits_row():
    rows = evaluate_config("pixel_nav", hidden=16, embed=16, seed=0, cfg=TINY)
    assert len(rows) == 1                       # no lever -> float row only
    row = rows[0]
    _check_row_keys(row)
    assert "compression" not in row              # float baseline is un-tagged


def test_correlation_runs_on_two_rows():
    rows = (
        evaluate_config("pixel_nav", hidden=16, embed=16, seed=0, cfg=TINY)
        + evaluate_config("pixel_nav", hidden=16, embed=16, seed=1, cfg=TINY)
    )
    res = audit_correlations(rows, ["pixel_mse", "return_ece"],
                             return_key="R_real", n_boot=50)
    assert "pixel_mse" in res
    assert "p_adj" in res["pixel_mse"]


def test_int8_lever_emits_tagged_row_with_same_triple():
    rows = evaluate_config("pixel_nav", hidden=16, embed=16, seed=0, cfg=TINY_INT8)
    assert len(rows) == 2
    float_row, int8_row = rows
    _check_row_keys(float_row)
    _check_row_keys(int8_row)
    # compressed row carries the tag and the SAME (hidden, embed, seed) triple,
    # so H3 can pair it with the float baseline.
    assert int8_row["compression"] == "int8"
    for k in ("hidden", "embed", "seed"):
        assert int8_row[k] == float_row[k]


def test_h3_assessable_on_real_compression_rows():
    # Deterministic end-to-end: one capacity x two seeds, each producing a float
    # and an int8 row. H3 must consume the REAL compression rows, pair them with
    # their float baselines on the exact triple, and return assessable (not a
    # hand-built row, not a tautology branch).
    rows = []
    for seed in (0, 1):
        rows += evaluate_config("pixel_nav", hidden=16, embed=16,
                                seed=seed, cfg=TINY_INT8)
    d = prereg_decision_h3(rows)
    assert d["status"] == "assessable"
    assert d["n_pairs"] == 2                    # one pair per seed
    assert d["compression_lever"] == "int8"
    # honest degradation numbers (may be ~0 for a tiny net, never fabricated)
    assert "pixel_composite_degradation" in d
    assert "control_composite_degradation" in d


def test_h3_not_assessable_without_compression_levers():
    # The no-compression path stays honestly not_assessable.
    rows = []
    for seed in (0, 1):
        rows += evaluate_config("pixel_nav", hidden=16, embed=16,
                                seed=seed, cfg=TINY)
    d = prereg_decision_h3(rows)
    assert d["status"] == "not_assessable"
