# WM-Audit Leaderboard

> Which world-model diagnostics actually predict closed-loop control utility? Lower on pixel metrics = better fidelity.

_Status: smoke / [IN PREPARATION]. Main results pending multi-seed, multi-env runs._

| diagnostic | Spearman rho vs R_real | 95% CI | Holm-adj p |
|---|---|---|---|
| pixel_mse | 0.657 | [0.000, 1.000] | 1.000 |
| pixel_ssim | -0.371 | [-1.000, 0.800] | 1.000 |
| pixel_nll | 0.657 | [0.000, 1.000] | 1.000 |
| pixel_lpips | 0.314 | [-1.000, 1.000] | 1.000 |
| rollout_fvd | 0.143 | [-1.000, 1.000] | 1.000 |
| return_ece | 0.829 | [0.000, 1.000] | 0.363 |
| counterfactual_tau | 0.143 | [-0.800, 1.000] | 1.000 |
| uncertainty_rho | 0.657 | [0.000, 1.000] | 1.000 |
| uncertainty_ece | 0.657 | [0.000, 1.000] | 1.000 |
