# WM-Audit Leaderboard

> Which world-model diagnostics actually predict closed-loop control utility? Lower on pixel metrics = better fidelity.

## Main results — `pixel_nav` (2026-11-07)

> **Status: first of the two locked environments, executed at full preregistered scale.**
> The preregistration asks for 200000 train steps × 5 seeds ×
> 3 capacity tiers × 2 envs. This section executes `pixel_nav`; the
> second env (`dm_control.cartpole_swingup`) was executed separately on 2026-11-13 and is
> reported in the next section, so the locked 2-of-2 MAIN is complete.
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

### Preregistered verdicts (main — `pixel_nav`)

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


## Second environment — `dm_control.cartpole_swingup` (15 float rows + int8)

> **Status: second locked environment, executed at the full preregistered scale
> (2026-11-13). With it, the locked 2-of-2 2-env MAIN is complete.**
> Kept exactly as preregistered (see `configs/audit/main_cartpole_full.yaml`):
> 200000 train steps, 100000 collect steps, seeds [0, 1, 2, 3, 4], capacity tiers
> [[16, 16], [32, 16], [32, 32]], ensemble_size=3, bootstrap=2000, INT8 lever.
> Run as five seed shards on GitHub-hosted `ubuntu-latest` runners
> (MuJoCo 4.0.1 via `dm_control`, `MUJOCO_GL=osmesa`, CPU only); shard artifacts
> are committed under `leaderboard/ci/cartpole_seed{0..4}.json`.
>
> **Two diagnostics are not measurable in this environment and are reported as
> `null`, not as numbers.** Non-finite entries: `counterfactual_tau` in 15/15 float rows; `uncertainty_rho` in 3/15 float rows. `counterfactual_tau`
> compares oracle and imagined reward on two single 10 ms steps, where `cartpole_swingup`
> reward is effectively constant across the sampled action pairs, so the tau-b
> denominator vanishes; `uncertainty_rho` is undefined where the ensemble collapses
> onto identical predictions at the smallest capacity. Composite coverage
> (diagnostics actually averaged per row): pixel 5 of
> 5 everywhere; control 3 rows with 1 of 3, 12 rows with 2 of 3. **H2's control composite is therefore
> a degraded 1-2-of-3 mean, not the strict preregistered composite.**
>
> Reproduce: `python experiments/run_audit.py --config configs/audit/main_cartpole_full.yaml --mode main --resume --out leaderboard/results_main_cartpole_200k.json`, then `python experiments/make_leaderboard.py`.

### Diagnostics vs R_real (float-only, n=15; percentile bootstrap=2000)

| diagnostic | Spearman rho vs R_real | 95% CI | p (raw) | Holm-adj p | n used |
|---|---|---|---|---|---|
| pixel_mse | -0.160 | [-0.665, 0.420] | 0.568 | 1.000 | 15 |
| pixel_ssim | -0.061 | [-0.637, 0.574] | 0.828 | 1.000 | 15 |
| pixel_nll | -0.160 | [-0.665, 0.420] | 0.568 | 1.000 | 15 |
| pixel_lpips | -0.184 | [-0.674, 0.355] | 0.512 | 1.000 | 15 |
| rollout_fvd | -0.216 | [-0.690, 0.310] | 0.439 | 1.000 | 15 |
| return_ece | -0.525 | [-0.869, 0.004] | 0.044 | 0.443 | 15 |
| counterfactual_tau | undefined | [undefined, undefined] | undefined | undefined | 0 |
| uncertainty_rho | -0.327 | [-0.828, 0.307] | 0.300 | 1.000 | 12 |
| uncertainty_ece | -0.083 | [-0.579, 0.429] | 0.769 | 1.000 | 15 |
| pixel_composite | +0.130 | [-0.477, 0.685] | 0.645 | 1.000 | 15 |
| control_composite | +0.301 | [-0.182, 0.668] | 0.275 | 1.000 | 15 |

### Rows (float baseline + INT8 variants)

| hidden | embed | seed | compression | R_real | pixel_mse | return_ece | counterfactual_tau |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 0 | float | 22.414 | 0.00202 | 0.702 | undefined |
| 16 | 16 | 0 | int8 | 22.414 | 0.00202 | 0.702 | undefined |
| 32 | 16 | 0 | float | 20.026 | 0.00199 | 0.703 | undefined |
| 32 | 16 | 0 | int8 | 22.414 | 0.00206 | 0.703 | undefined |
| 32 | 32 | 0 | float | 22.414 | 0.00197 | 0.499 | undefined |
| 32 | 32 | 0 | int8 | 22.414 | 0.00200 | 0.500 | undefined |
| 16 | 16 | 1 | float | 5.796 | 0.00202 | 0.716 | undefined |
| 16 | 16 | 1 | int8 | 5.796 | 0.00202 | 0.716 | undefined |
| 32 | 16 | 1 | float | 4.035 | 0.00196 | 0.521 | undefined |
| 32 | 16 | 1 | int8 | 5.796 | 0.00205 | 0.521 | undefined |
| 32 | 32 | 1 | float | 5.796 | 0.00202 | 0.716 | undefined |
| 32 | 32 | 1 | int8 | 5.796 | 0.00202 | 0.716 | undefined |
| 16 | 16 | 2 | float | 4.302 | 0.00202 | 0.716 | undefined |
| 16 | 16 | 2 | int8 | 9.130 | 0.00202 | 0.716 | undefined |
| 32 | 16 | 2 | float | 158.788 | 0.00200 | 0.475 | undefined |
| 32 | 16 | 2 | int8 | 159.405 | 0.00203 | 0.481 | undefined |
| 32 | 32 | 2 | float | 25.303 | 0.00197 | 0.513 | undefined |
| 32 | 32 | 2 | int8 | 43.742 | 0.00199 | 0.514 | undefined |
| 16 | 16 | 3 | float | 22.176 | 0.00202 | 0.707 | undefined |
| 16 | 16 | 3 | int8 | 22.176 | 0.00202 | 0.707 | undefined |
| 32 | 16 | 3 | float | 22.176 | 0.00202 | 0.707 | undefined |
| 32 | 16 | 3 | int8 | 22.176 | 0.00204 | 0.707 | undefined |
| 32 | 32 | 3 | float | 22.176 | 0.00196 | 0.346 | undefined |
| 32 | 32 | 3 | int8 | 22.176 | 0.00206 | 0.360 | undefined |
| 16 | 16 | 4 | float | 18.318 | 0.00202 | 0.698 | undefined |
| 16 | 16 | 4 | int8 | 18.318 | 0.00202 | 0.698 | undefined |
| 32 | 16 | 4 | float | 18.318 | 0.00197 | 0.511 | undefined |
| 32 | 16 | 4 | int8 | 18.318 | 0.00202 | 0.511 | undefined |
| 32 | 32 | 4 | float | 18.318 | 0.00198 | 0.552 | undefined |
| 32 | 32 | 4 | int8 | 18.318 | 0.00202 | 0.554 | undefined |

### Preregistered verdicts (second env)

| hypothesis | threshold | observed | result |
|---|---|---|---|
| H1 | \|ρ(pixel_composite, R)\| < 0.4 | +0.130 | **THRESHOLD MET — weak / tentative only** (n=15; CI [-0.477, 0.685] crosses zero; p_raw=0.645, Holm-adj p=1.000 → not significant; NOT an established finding) |
| H2 | ρ(control_composite, R) > 0.6 | +0.301 | **NOT supported** (below threshold; strongest single control diagnostic `return_ece` = -0.525, raw p=0.044, Holm-adj p=0.443) |
| H3 | control_deg > pixel_deg under INT8 | pixel_deg=1.580, control_deg=1.367 | **NOT supported** (n_pairs=15; direction reverses again — pixel composite degrades *more*) |

Read honestly: this environment **repeats** the navigation result rather than
rescuing it. H1's rule fires again at a wide interval; H2 misses the 0.6 bar
again (+0.301); and H3's preregistered ordering fails in the same
direction as before (1.580 vs 1.367). `counterfactual_tau`
is unmeasurable here (0 usable pairs) and `uncertainty_rho` is measured
on 12 of 15 pairs; the Holm family shrinks to 10 tests accordingly.

**Outcome resolution.** The 15 float rows carry only 9 distinct
`R_real` values, and 2 of 5 seeds show a single value across all capacities (seed 3, 4), so the rank correlations above are orderings over
fewer effective levels than rows. `composite_coverage` and `outcome_resolution`
in the JSON state these limits numerically rather than leaving them to prose.

**Aggregation defect fixed before publishing this section.** The first aggregation
ranked the NaN columns anyway, which manufactured ρ=−0.043 for the all-NaN
diagnostic, moved H2 to +0.308, and scored unmeasurable compressed rows as the
worst value on the fixed ladder — reporting control degradation 4.311 and making
**H3 appear supported**. The numbers above come from the fixed pipeline, which
drops non-finite pairs, reports `n`/`n_dropped`, leaves unmeasurable diagnostics
out of the composite mean, and writes `null` instead of RFC-invalid `NaN`.


## Reduced-pilot results — CPU (2026-10-13, historical)

> **Status: historical CPU REDUCED PILOT (3 seeds × 12k steps), kept for
> reproducibility. Superseded at scale by the locked-scale runs above
> (200k steps × 5 seeds on `pixel_nav`, then on `dm_control.cartpole_swingup`);
> both preregistered environments are now executed.**
> Env: built-in `pixel_nav` only. The 2nd prereg env `dm_control.cartpole_swingup`
> **cannot install on Windows**: `pip install dm_control` pulls `labmaze`, which has
> no cp3.13 wheel and requires `bazel` to build from source (`error: command 'bazel'
> failed`); it was run on GitHub-hosted Linux runners instead (see the section above).
> Device: CPU (venv torch 2.14.1+cpu, `cuda_available=False`; a cu124 install
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
