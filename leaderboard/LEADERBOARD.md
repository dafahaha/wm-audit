# WM-Audit Leaderboard

> Which world-model diagnostics actually predict closed-loop control utility? Lower on pixel metrics = better fidelity.

_Status: smoke / [IN PREPARATION]. Main results pending multi-seed, multi-env runs._

## Diagnostics vs R_real

| diagnostic | Spearman rho vs R_real | 95% CI | Holm-adj p |
|---|---|---|---|
| pixel_mse | 0.643 | [0.190, 0.843] | 0.220 |
| pixel_ssim | -0.364 | [-0.886, 0.329] | 1.000 |
| pixel_nll | 0.643 | [0.190, 0.843] | 0.220 |
| pixel_lpips† | 0.287 | [-0.435, 0.850] | 1.000 |
| rollout_fvd† | 0.154 | [-0.595, 0.785] | 1.000 |
| return_ece | 0.755 | [0.241, 0.936] | 0.050 |
| counterfactual_tau | -0.007 | [-0.616, 0.719] | 1.000 |
| uncertainty_rho | 0.650 | [0.242, 0.905] | 0.220 |
| uncertainty_ece | 0.643 | [0.190, 0.843] | 0.220 |
| pixel_composite | -0.636 | [-0.836, 0.026] | 0.220 |
| control_composite | -0.046 | [-0.618, 0.697] | 1.000 |

## Rows (float baseline + compression variants)

| hidden | embed | seed | compression | R_real | pixel_mse | return_ece | counterfactual_tau |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 0 | float | -34.453 | 0.038 | 2.845 | 0.149 |
| 16 | 16 | 0 | int8 | -34.468 | 0.038 | 3.258 | 0.149 |
| 32 | 16 | 0 | float | -154.155 | 0.036 | 2.731 | 0.127 |
| 32 | 16 | 0 | int8 | -146.341 | 0.036 | 2.802 | 0.132 |
| 32 | 32 | 0 | float | -109.273 | 0.035 | 1.586 | 0.123 |
| 32 | 32 | 0 | int8 | -116.085 | 0.035 | 1.752 | 0.119 |
| 16 | 16 | 1 | float | -21.708 | 0.039 | 4.961 | 0.145 |
| 16 | 16 | 1 | int8 | -22.582 | 0.039 | 4.971 | 0.145 |
| 32 | 16 | 1 | float | -14.578 | 0.037 | 4.295 | 0.118 |
| 32 | 16 | 1 | int8 | -14.264 | 0.037 | 4.448 | 0.121 |
| 32 | 32 | 1 | float | -64.157 | 0.036 | 2.135 | 0.127 |
| 32 | 32 | 1 | int8 | -53.396 | 0.036 | 2.146 | 0.126 |

## Preregistered verdicts (smoke)

| hypothesis | status / result |
|---|---|
| H1 | observed=-0.636; supports_H1=False |
| H2 | observed=-0.046; supports_H2=False |
| H3 | assessable; n_pairs=6; supports_H3=False; pixel_deg=0.400, control_deg=0.222 |

_† `pixel_lpips` / `rollout_fvd` are fixed-filter stand-ins (deterministic `FixedFilterFeatures` bank), not the learned VGG/AlexNet trunk behind real LPIPS / the feature distribution behind real FVD: real, reproducible distances, but not publication-grade LPIPS/FVD._
