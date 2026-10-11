# WM-Audit Leaderboard

> Which world-model diagnostics actually predict closed-loop control utility? Lower on pixel metrics = better fidelity.

_Status: smoke / [IN PREPARATION]. Main results pending multi-seed, multi-env runs._

| diagnostic | Spearman rho vs R_real | 95% CI | Holm-adj p |
|---|---|---|---|
| pixel_mse | 0.657 | [0.000, 1.000] | 1.000 |
| pixel_ssim | -0.371 | [-1.000, 0.800] | 1.000 |
| pixel_nll | 0.657 | [0.000, 1.000] | 1.000 |
| pixel_lpips† | 0.314 | [-1.000, 1.000] | 1.000 |
| rollout_fvd† | 0.143 | [-1.000, 1.000] | 1.000 |
| return_ece | 0.829 | [0.000, 1.000] | 0.457 |
| counterfactual_tau | 0.143 | [-0.800, 1.000] | 1.000 |
| uncertainty_rho | 0.657 | [0.000, 1.000] | 1.000 |
| uncertainty_ece | 0.657 | [0.000, 1.000] | 1.000 |
| pixel_composite | -0.580 | [-0.949, 0.869] | 1.000 |
| control_composite | -0.029 | [-0.936, 1.000] | 1.000 |

_† `pixel_lpips` / `rollout_fvd` are fixed-filter stand-ins (deterministic `FixedFilterFeatures` bank), not the learned VGG/AlexNet trunk behind real LPIPS / the feature distribution behind real FVD: real, reproducible distances, but not publication-grade LPIPS/FVD._
