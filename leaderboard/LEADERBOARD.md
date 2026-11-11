# WM-Audit Leaderboard

> Which world-model diagnostics actually predict closed-loop control utility? Lower on pixel metrics = better fidelity.

## Main results — `pixel_nav` single-env subset (2026-11-07)

> **Status: single-env subset of the locked preregistered MAIN — NOT the full 2-env main.**
> The preregistration asks for 200000 train steps × 5 seeds ×
> 3 capacity tiers × 2 envs. This run executes `pixel_nav` at the locked
> scale; the second prereg env (`dm_control.cartpole_swingup`) is **not included** — MuJoCo/EGL unavailable in this Linux CPU container; not installed for this run.
>
> Kept exactly as preregistered: 200000 train steps, 100000 collect steps,
> seeds [0, 1, 2, 3, 4], capacity tiers [[16, 16], [32, 16], [32, 32]], ensemble_size=3,
> bootstrap=2000, INT8 lever, and the whole stats pipeline (float-only H1/H2,
> fixed-float-ladder H3, Spearman t-approx p + Holm). Device: CPU (torch 2.14.1+cpu, torch.cuda.is_available()==False); 4-core AMD EPYC, 8 GiB RAM, Ubuntu 22.04 container; 4 worker processes, torch 1 thread each.
>
> **Recurrent caveat (unchanged):** training resets the GRU state to zeros every batch
> (`wm_audit/train/train_wm.py`); sequences are never unrolled, so no temporal dependency
> is learned and this run is a per-transition stand-in. Its H1/H2/H3 conclusions describe
> only this stand-in pipeline and must NOT be extrapolated to a genuine recurrent world
> model with learned temporal dynamics.
>
> Reproduce: `python -m experiments.run_audit --mode main --config configs/audit/main_pixel_nav_full.yaml --jobs 4 --out leaderboard/results_main_pixelnav_200k.json`, then `python experiments/make_leaderboard.py`.

### Diagnostics vs R_real (float-only, n=15; percentile bootstrap=2000)

| diagnostic | Spearman rho vs R_real | 95% CI | Holm-adj p |
|---|---|---|---|
| pixel_mse | 0.061 | [-0.562, 0.612] | 1.000 |
| pixel_ssim | -0.043 | [-0.629, 0.596] | 1.000 |
| pixel_nll | 0.061 | [-0.562, 0.612] | 1.000 |
| pixel_lpips† | -0.154 | [-0.617, 0.389] | 1.000 |
| rollout_fvd† | -0.236 | [-0.669, 0.366] | 1.000 |
| return_ece | 0.021 | [-0.555, 0.576] | 1.000 |
| counterfactual_tau | 0.250 | [-0.274, 0.688] | 1.000 |
| uncertainty_rho | -0.218 | [-0.699, 0.366] | 1.000 |
| uncertainty_ece | 0.007 | [-0.640, 0.568] | 1.000 |
| **pixel_composite** | **0.021** | [-0.540, 0.622] | 1.000 |
| **control_composite** | **-0.102** | [-0.590, 0.436] | 1.000 |

### Rows (float baseline + INT8 variants)

| hidden | embed | seed | compression | R_real | pixel_mse | return_ece | counterfactual_tau |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 0 | float | -112.122 | 0.009 | 3.907 | 0.769 |
| 16 | 16 | 0 | int8 | -96.570 | 0.009 | 3.888 | 0.771 |
| 32 | 16 | 0 | float | -109.102 | 0.008 | 2.732 | 0.737 |
| 32 | 16 | 0 | int8 | -53.197 | 0.008 | 2.701 | 0.734 |
| 32 | 32 | 0 | float | -90.346 | 0.007 | 2.941 | 0.740 |
| 32 | 32 | 0 | int8 | -86.308 | 0.007 | 3.105 | 0.741 |
| 16 | 16 | 1 | float | -44.889 | 0.009 | 4.199 | 0.762 |
| 16 | 16 | 1 | int8 | -13.361 | 0.009 | 4.768 | 0.764 |
| 32 | 16 | 1 | float | -15.813 | 0.008 | 2.602 | 0.796 |
| 32 | 16 | 1 | int8 | -10.642 | 0.008 | 2.931 | 0.782 |
| 32 | 32 | 1 | float | -45.944 | 0.007 | 2.450 | 0.754 |
| 32 | 32 | 1 | int8 | -43.609 | 0.007 | 2.069 | 0.753 |
| 16 | 16 | 2 | float | -22.752 | 0.009 | 3.940 | 0.784 |
| 16 | 16 | 2 | int8 | -14.086 | 0.009 | 3.767 | 0.783 |
| 32 | 16 | 2 | float | -14.456 | 0.008 | 2.739 | 0.778 |
| 32 | 16 | 2 | int8 | -47.041 | 0.008 | 3.031 | 0.760 |
| 32 | 32 | 2 | float | -54.297 | 0.006 | 2.090 | 0.773 |
| 32 | 32 | 2 | int8 | -63.061 | 0.007 | 2.494 | 0.771 |
| 16 | 16 | 3 | float | -98.246 | 0.009 | 4.818 | 0.796 |
| 16 | 16 | 3 | int8 | -81.628 | 0.009 | 4.956 | 0.789 |
| 32 | 16 | 3 | float | -53.662 | 0.008 | 2.829 | 0.789 |
| 32 | 16 | 3 | int8 | -42.076 | 0.008 | 3.112 | 0.773 |
| 32 | 32 | 3 | float | -114.253 | 0.006 | 1.542 | 0.773 |
| 32 | 32 | 3 | int8 | -76.796 | 0.007 | 1.777 | 0.771 |
| 16 | 16 | 4 | float | -25.915 | 0.009 | 4.243 | 0.794 |
| 16 | 16 | 4 | int8 | -23.634 | 0.009 | 4.092 | 0.786 |
| 32 | 16 | 4 | float | -93.379 | 0.008 | 2.538 | 0.778 |
| 32 | 16 | 4 | int8 | -49.674 | 0.008 | 2.885 | 0.773 |
| 32 | 32 | 4 | float | -9.362 | 0.007 | 1.928 | 0.763 |
| 32 | 32 | 4 | int8 | -8.117 | 0.007 | 1.930 | 0.759 |

### Preregistered verdicts (main — pixel_nav single-env subset)

| hypothesis | threshold | observed rho | result |
|---|---|---|---|
| H1 | \|ρ(pixel_composite, R)\| < 0.4 | 0.021 | **THRESHOLD MET — weak / tentative exploratory signal only** (n=15; CI [-0.540, 0.622] crosses zero; p_raw=0.939, Holm-adj p=1.000 → **not significant; NOT an established finding**) |
| H2 | ρ(control_composite, R) > 0.6 | -0.102 | **NOT supported** (control composite below threshold) |
| H3 | control_deg > pixel_deg under INT8 | pixel_deg=0.780, control_deg=0.422 | **NOT supported** (n_pairs=15) |

Read honestly: the strongest absolute diagnostic-vs-return correlation on this run is
`counterfactual_tau` (rho=0.250; p_raw=0.368, Holm-adj p=1.000).
The H1 threshold firing (|rho|=0.021<0.4) remains a **weak, exploratory
label only** — CI [-0.540, 0.622] crosses zero and
Holm-adj p=1.000. **H2 is contradicted at this scale**:
rho(control_composite)=-0.102, far below the preregistered 0.6 threshold
(Holm-adj p=1.000). **H3 reverses**: under INT8 the pixel composite degrades by 0.780 while the control/calibration composite degrades by only 0.422. These are
bounded null/negative results per the preregistration's honesty clause, reported as such.

**Rank-collinearity caveat (n=15).** 2 diagnostics — `pixel_mse`, `pixel_nll` — share identical rho=0.061, p_raw=0.830 and the same CI: one rank pattern, not that many independent confirmations. Multi-diagnostic agreement must not be counted as cross-validation on this run.


## Reduced-pilot results — CPU (2026-10-13, historical)

> **Status: historical CPU REDUCED PILOT (3 seeds × 12k steps), kept for
> reproducibility. Superseded at scale by the main single-env subset above
> (200k steps × 5 seeds on `pixel_nav`); the full 2-env main remains
> unexecuted.**
> Env: built-in `pixel_nav` only. The 2nd prereg env `dm_control.cartpole_swingup`
> **cannot install on Windows**: `pip install dm_control` pulls `labmaze`, which has
> no cp3.13 wheel and requires `bazel` to build from source (`error: command 'bazel'
> failed`). Device: CPU (venv torch 2.14.1+cpu, `cuda_available=False`; a cu124 install
> was attempted but pip keeps the satisfied CPU wheel).
>
> What WAS kept exactly as preregistered: the 3 capacity tiers, `bootstrap_resamples=2000`,
> the INT8 lever, and the whole stats pipeline (float-only H1/H2, fixed-float-reference-ladder
> H3, Spearman t-approx p + Holm). What was reduced and labeled: **3 seeds (prereg goal 5),
> `train_steps=12000` (prereg 200000), `collect_steps=8000` (prereg 100000)** — the locked
> design was out of the original Windows session budget (~21 CPU-hours at the measured
> ~236 member-steps/s) and dm_control does not install there. No numbers were tuned or
> back-filled. The runnable env has since been executed at the locked scale in the main
> section above.
>
> **Recurrent caveat:** training resets the GRU state to zeros every batch
> (`wm_audit/train/train_wm.py`); sequences are never unrolled, so no temporal
> dependency is learned and this pilot is a per-transition stand-in. Its
> H1/H2/H3 conclusions describe only this stand-in pipeline and must NOT be
> extrapolated to a genuine recurrent world model with learned temporal
> dynamics.
>
> Reproduce this section: `python experiments/make_leaderboard.py` regenerates it from the
> committed `leaderboard/results_main.json`.

Correlations are computed on the **float baseline rows only** (n=9 = 3 seeds × 3 capacities):
INT8 rows are deployment perturbations of an already-trained model, not new capacity tiers, so
they belong only to the H3 paired test and never enter the H1/H2 correlation pool.

### Diagnostics vs R_real (float-only, n=9; percentile bootstrap=2000)

| diagnostic | Spearman rho vs R_real | 95% CI | Holm-adj p |
|---|---|---|---|
| pixel_mse | 0.317 | [-0.554, 0.947] | 1.000 |
| pixel_ssim | -0.350 | [-1.000, 0.554] | 1.000 |
| pixel_nll | 0.317 | [-0.554, 0.947] | 1.000 |
| pixel_lpips† | 0.317 | [-0.554, 0.947] | 1.000 |
| rollout_fvd† | 0.317 | [-0.554, 0.947] | 1.000 |
| return_ece | 0.183 | [-0.579, 0.892] | 1.000 |
| counterfactual_tau | 0.050 | [-0.843, 0.684] | 1.000 |
| uncertainty_rho | 0.700 | [0.078, 0.982] | 0.393 |
| uncertainty_ece | 0.317 | [-0.554, 0.947] | 1.000 |
| **pixel_composite** | **-0.317** | [-0.947, 0.554] | 1.000 |
| **control_composite** | **0.259** | [-0.498, 0.842] | 1.000 |

### Rows (float baseline + INT8 variants)

| hidden | embed | seed | compression | R_real | pixel_mse | return_ece | counterfactual_tau |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 0 | float | -41.587 | 0.010 | 2.313 | 0.677 |
| 16 | 16 | 0 | int8 | -74.919 | 0.010 | 2.232 | 0.675 |
| 32 | 16 | 0 | float | -43.725 | 0.008 | 3.297 | 0.679 |
| 32 | 16 | 0 | int8 | -59.778 | 0.008 | 3.298 | 0.676 |
| 32 | 32 | 0 | float | -107.628 | 0.007 | 2.828 | 0.597 |
| 32 | 32 | 0 | int8 | -74.683 | 0.007 | 2.518 | 0.592 |
| 16 | 16 | 1 | float | -7.329 | 0.011 | 3.365 | 0.667 |
| 16 | 16 | 1 | int8 | -1.353 | 0.011 | 3.341 | 0.661 |
| 32 | 16 | 1 | float | -89.635 | 0.008 | 3.092 | 0.689 |
| 32 | 16 | 1 | int8 | -80.596 | 0.008 | 3.098 | 0.681 |
| 32 | 32 | 1 | float | -79.017 | 0.006 | 3.245 | 0.586 |
| 32 | 32 | 1 | int8 | -49.060 | 0.006 | 3.349 | 0.578 |
| 16 | 16 | 2 | float | -41.086 | 0.012 | 2.520 | 0.607 |
| 16 | 16 | 2 | int8 | -12.732 | 0.012 | 2.664 | 0.603 |
| 32 | 16 | 2 | float | -85.485 | 0.008 | 3.022 | 0.661 |
| 32 | 16 | 2 | int8 | -90.592 | 0.008 | 2.834 | 0.652 |
| 32 | 32 | 2 | float | -34.766 | 0.006 | 3.033 | 0.618 |
| 32 | 32 | 2 | int8 | -37.177 | 0.006 | 3.080 | 0.619 |

### Preregistered verdicts (reduced pilot; NOT the locked MAIN run)

| hypothesis | threshold | observed rho | result |
|---|---|---|---|
| H1 | \|ρ(pixel_composite, R)\| < 0.4 | -0.317 | **THRESHOLD MET — weak / tentative exploratory signal only** (n=9; CI [-0.947, 0.554] crosses zero; p_raw=0.406, Holm-adj p=1.000 → **not significant; NOT an established finding**) |
| H2 | ρ(control_composite, R) > 0.6 | 0.259 | **NOT supported** (control composite below threshold) |
| H3 | control_deg > pixel_deg under INT8 | pixel_deg=0.433, control_deg=0.352 | **NOT supported** (n_pairs=9) |

Read honestly: on this reduced CPU pilot, the single strongest diagnostic-vs-return correlation
is `uncertainty_rho` (ρ=0.70, p_raw=0.036 but Holm-adj p=0.393 — not
significant at n=9). The H1 threshold being met (|ρ|=0.317<0.4) is a
**weak, exploratory signal only** — the CI [-0.947, 0.554] crosses zero and
Holm-adj p=1.000, so it is NOT significant and must not be read as an established
finding that pixel fidelity predicts control. H2/H3 are bounded null/negative results per the
preregistration's honesty clause, not falsifications.

**Rank-collinearity caveat (n=9).** Five diagnostics — `pixel_mse`, `pixel_nll`,
`pixel_lpips`†, `rollout_fvd`†, `uncertainty_ece` — share *identical* ρ=0.317,
p_raw=0.4064 and CI [-0.554, 0.947] in the correlation table
above: their 9-row rank vectors are exactly collinear. They are one rank pattern, not five
independent confirmations; the battery therefore contains far fewer independent signals than its 11
rows suggest, and multi-diagnostic agreement must not be counted as cross-validation on this pilot
(likely driven by n=9 plus deterministic, capacity-ordered stand-in models; to be
re-checked in the locked main run).

---

## SMOKE results — CI placeholder (downgraded, preserved)

> **This block is the old CI smoke run (n=6 float rows = 2 seeds × 3 caps, `bootstrap=200`,
> `train_steps=150`). It is a placeholder, NOT a main result.** Kept for history; do not cite as
> evidence. Generated by `python -m experiments.run_audit --mode smoke` → `results_smoke.json`.

### Diagnostics vs R_real (float-only, n=6)

| diagnostic | Spearman rho vs R_real (float-only) | 95% CI | Holm-adj p |
|---|---|---|---|
| pixel_mse | 0.657 | [0.000, 1.000] | 1.000 |
| pixel_ssim | -0.371 | [-1.000, 0.800] | 1.000 |
| pixel_nll | 0.657 | [0.000, 1.000] | 1.000 |
| pixel_lpips† | 0.314 | [-1.000, 1.000] | 1.000 |
| rollout_fvd† | 0.143 | [-1.000, 1.000] | 1.000 |
| return_ece | 0.771 | [0.000, 1.000] | 0.796 |
| counterfactual_tau | -0.086 | [-1.000, 1.000] | 1.000 |
| uncertainty_rho | 0.657 | [0.000, 1.000] | 1.000 |
| uncertainty_ece | 0.657 | [0.000, 1.000] | 1.000 |
| pixel_composite | -0.580 | [-0.949, 0.869] | 1.000 |
| control_composite | -0.086 | [-1.000, 1.000] | 1.000 |

### Rows (float baseline + compression variants)

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

### Preregistered verdicts (smoke)

| hypothesis | status / result |
|---|---|
| H1 | observed=-0.580; supports_H1=False |
| H2 | observed=-0.086; supports_H2=False |
| H3 | assessable; n_pairs=6; supports_H3=False; pixel_deg=0.200, control_deg=0.111 |

---

_† `pixel_lpips` / `rollout_fvd` are fixed-filter stand-ins (deterministic `FixedFilterFeatures`
bank), not the learned VGG/AlexNet trunk behind real LPIPS / the feature distribution behind real
FVD: real, reproducible distances, but not publication-grade LPIPS/FVD._
