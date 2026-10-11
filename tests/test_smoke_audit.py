"""test_smoke_audit: full chain train->diagnose->close-loop->correlate in CI."""
from wm_audit.audit import evaluate_config
from wm_audit.diagnostics import audit_correlations


TINY = dict(
    buffer_capacity=500, collect_steps=400, data_seed=7,
    ensemble_size=2, train_steps=20, batch_size=32, lr=2e-3,
    calib_segments=6, calib_horizon=4, cf_pairs=10,
    cem_horizon=3, cem_candidates=16, cem_elites=4, cem_iters=1,
)


def test_full_chain_runs_and_emits_row():
    row = evaluate_config("pixel_nav", hidden=16, embed=16, seed=0, cfg=TINY)
    for k in ("pixel_mse", "pixel_ssim", "pixel_nll", "pixel_lpips",
              "rollout_fvd", "return_ece", "counterfactual_tau",
              "uncertainty_rho", "R_real"):
        assert k in row
    assert isinstance(row["R_real"], float)


def test_correlation_runs_on_two_rows():
    rows = [
        evaluate_config("pixel_nav", hidden=16, embed=16, seed=0, cfg=TINY),
        evaluate_config("pixel_nav", hidden=16, embed=16, seed=1, cfg=TINY),
    ]
    res = audit_correlations(rows, ["pixel_mse", "return_ece"],
                              return_key="R_real", n_boot=50)
    assert "pixel_mse" in res
    assert "p_adj" in res["pixel_mse"]
