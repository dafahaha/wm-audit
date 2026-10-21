# WM-Audit — Round 1 independent read-only audit (2026-10-22)

- **Auditor posture:** fresh, no prior context; read every source/test/config/doc
  before touching anything. Only P0/P1/P2 defects were fixed; P3 items are
  recorded, not changed.
- **Baseline reviewed:** `757e5b2` "fix(leaderboard): make MAIN reduced-pilot
  results reproducible from committed data" (pristine working tree at start).
- **Environment (local, not CI):** Windows, Python 3.13.16, torch 2.14.1+cpu
  (`cuda_available=False`), numpy 2.5.4, pytest 9.1.1; scipy 1.18.1 used only
  as an independent oracle (never imported by the package). No C compiler, no
  CUDA; `dm_control` not installed and never imported by any test.
- **Scope honored:** no D1–D10 decision items touched; no GPU/MAIN 200k-step
  experiment fabricated; reduced-pilot honesty wording (H1 supported, H2/H3 not
  supported, n=9, wide overlapping CIs) preserved verbatim.

---

## 1. Findings register

Grades: **P0** = wrong published number / integrity break; **P1** = wrong
computation on a tested path; **P2** = reproducibility or robustness defect that
breaks a documented command; **P3** = recording only.

### P0 — none found.

### P1 — none found.

The audited inferential core was independently re-derived from committed data and
cross-checked against scipy (see §4): Spearman ρ, Student-t p, percentile
bootstrap CIs, Holm adjustment, H1/H2 thresholds, and the H3 fixed-ladder paired
degradation all match the committed numbers to machine precision. No published
statistic needed correction.

### P2-1 — `make_leaderboard.py` crashes on a fresh clone (smoke JSON was gitignored)

- **Where:** `.gitignore:12-13` (before fix: `leaderboard/results_*.json` with
  only `!results_main.json` un-ignored); consumer `experiments/make_leaderboard.py:21`
  (`SMOKE_RES = .../results_smoke.json`); docstring at `make_leaderboard.py:4-8`
  claims "Two sections are emitted, each generated from its own committed
  machine-readable results."
- **Root cause:** the preserved SMOKE history block needs `results_smoke.json`,
  but the ignore rule excluded it, so on a fresh clone the documented quickstart
  command `python experiments/make_leaderboard.py` died with `FileNotFoundError`,
  contradicting both the docstring and the clean-render invariant.
- **Fix:** un-ignore the deterministic smoke placeholder (fixed-seed rerun is
  bit-identical, verified in §4.3) and commit it; no rendering code changed.
- **Commit:** `f0db23b` `fix(leaderboard): track smoke placeholder json so
  render works on clone`.
- **Red/green evidence:** defect is a missing-file crash, not a math path;
  verified by (a) absence of the file under the old ignore rule, (b) after the
  fix the render runs and `git status` stays clean (§4.2).

### P2-2 — Holm step-down monotonic floor was untested (mutation survived)

- **Where:** `wm_audit/stats/multiple.py:29` (`val = max(val, running_min)`);
  old test `tests/test_spearman_bootstrap.py::test_holm_correction_preserves_size`
  used `p=[0.01,0.02,0.03,0.5]`, whose plain products `(n-rank)*p` are already
  non-decreasing.
- **Root cause / defect:** the floor that enforces Holm's required
  non-decreasing adjusted p-values had no test that could catch its removal:
  deleting the floor stayed green (mutation S1 in §3). The production floor
  itself is correct; the *test coverage* was the gap.
- **Fix:** added `test_holm_floor_raises_dipping_step_down_values` with
  `p=[0.04,0.05,0.05,0.5]` where raw products dip `0.16 → 0.15 → 0.10`; the
  floor must raise them to `[0.16, 0.16, 0.16, 0.5]` and the sorted adjusted
  vector must be non-decreasing.
- **Commit:** `60717ae` `test(stats): pin Holm step-down monotonic floor with a
  dipping case`.
- **Red/green evidence:** with the floor deleted, the new test fails
  (`Obtained 0.15/0.10, Expected 0.16`); with source restored it passes.

### P3 — recorded, intentionally not changed

| # | Where | Observation |
|---|---|---|
| P3-1 | `notebooks/10min_audit.ipynb` cells 1–3 | Stale: imports nonexistent `prereg_decision`; `rows.append(evaluate_config(...))` nests a list instead of extending; cfg omits keys the pipeline reads. Notebook is not executed in CI; left as-is. |
| P3-2 | `pyproject.toml:8`, `wm_audit/__init__.py:1-2` | Description says "single-GPU" while the suite is a CPU single-machine audit; wording only, no behavior. |
| P3-3 | `experiments/run_audit.py:123-125` | Default `--out` is `results_smoke.json` even under `--mode main`; an operator who forgets `--out` overwrites the smoke placeholder. No main run was executed here. |
| P3-4 | `wm_audit/policy/cem.py:29` | CEM elite selection (`np.argsort(scores)[-k:]`) has no unit test; mutation S3 (pick worst elites) survived the smoke chain, which only checks row keys/finite values. |
| P3-5 | `wm_audit/diagnostics/perceptual.py:94` | `FixedFilterFeatures` seed/weights are not numerically pinned by any test; mutation S4 (seed 0→99) survived. Stand-in reproducibility rests on code inspection. |
| P3-6 | `wm_audit/envs/pixel_nav.py:39` | Only *relative* same-seed determinism between two fresh envs is tested; absolute seed→observation values and dynamics coefficients (0.9/0.15/0.08) are unpinned; mutation S2 (`seed+1`) survived. |
| P3-7 | `wm_audit/diagnostics/uncertainty.py:32-52` | `uncertainty_ece` bins dispersion and compares mean dispersion directly to mean squared error (different scales); `ensemble_dispersion`/`uncertainty_ece` lack direct unit tests. |
| P3-8 | `wm_audit/diagnostics/perceptual.py:160-170` | `frechet_distance` has a dead `c1.ndim==0` branch after `np.cov`; `rollout_fvd` pools validation frames rather than true rollout sequences (already labeled a fixed-filter stand-in). |
| P3-9 | `wm_audit/train/train_wm.py:38-39` | GRU state is zeroed per batch, so the model is effectively a one-step predictor, not a temporal rollout; deliberate stand-in simplification, undocumented in the code comment. |
| P3-10 | `wm_audit/diagnostics/fidelity_utility.py:380` | H3 `compression_lever` label takes `comp_rows[0]`; a future mixed-lever run would mislabel. Irrelevant to today's int8-only pilot. |
| P3-11 | row emission (`wm_audit/audit.py:166-187`) | No explicit finite-value assertion on emitted diagnostics; committed MAIN/smoke rows were verified all-finite (§4.4), but a future run could emit NaN/Inf silently. |

---

## 2. Platform note (pristine evidence)

- Pristine baseline (`git stash` of all round-1 changes, HEAD `757e5b2`):
  `python -m pytest -q` → **57 passed, 0 failed, 0 errored, 0 skipped**,
  exit 0 (log: `pristine_baseline.log`). No platform-specific failure occurred
  locally, so there was nothing to attribute to the changes.
- `dm_control.cartpole_swingup` is an extension path: `envs/dmc.py:19-26` raises a
  clear `ImportError` only on explicit construction; no test imports it, and CI
  runs on the built-in `pixel_nav`. This matches the leaderboard's documented
  Windows limitation (`labmaze` needs bazel, no cp3.13 wheel).

---

## 3. Mutation experiment (test effectiveness)

Method: inject one defect at a time into production source, run the targeted
tests, record FAIL/SURVIVE, revert via `git checkout`, re-run to green. Every
mutation was removed before the final suite.

| ID | Mutation (file) | Guarding test | Result |
|----|------------------|---------------|--------|
| M1 | ECE gap `abs(pred-act)` → `pred+act` (`diagnostics/control.py:39`) | `test_perfect_calibration_zero` | **KILLED** |
| M2 | Bootstrap lo/hi percentiles swapped (`stats/bootstrap.py:42-43`) | `test_bootstrap_ci_covers_known_truth` | **KILLED** |
| M3 | Spearman p halved (two-sided → one-sided) (`stats/rank.py:153`) | 4 frozen p-value tests | **KILLED** |
| M4 | Composite orientation condition inverted (`diagnostics/fidelity_utility.py:84`) | 7 composite/H3 tests | **KILLED** |
| M5 | H3 match key dropped `embed` (`fidelity_utility.py:262,275`) | embed-triple guards | **KILLED** |
| M6 | H3 support direction `>` → `<` (`fidelity_utility.py:377`) | 3 directional H3 tests | **KILLED** |
| M7 | Kendall τ conc/disc sign flipped (`stats/rank.py:188`) | all 3 counterfactual tests | **KILLED** |
| M8 | NLL dropped `log(2π)` (`diagnostics/perceptual.py:78`) | `test_gaussian_nll_known_mode` | **KILLED** |
| M9 | MSE `mean` → `sum` (`diagnostics/perceptual.py:22`) | `test_mse_known_value` | **KILLED** |
| M10 | Buffer capacity cap removed (`data/buffer.py:33`) | `test_buffer_caps_at_capacity` | **KILLED** |
| S1 | Holm monotonic floor removed (`stats/multiple.py:29`) | old suite → **SURVIVED**; fixed by P2-2 |
| S2 | env reset `seed+1` (`envs/pixel_nav.py:39`) | relative-determinism test only → **SURVIVED** (P3-6) |
| S3 | CEM picks worst elites (`policy/cem.py:29`) | smoke chain key check → **SURVIVED** (P3-4) |
| S4 | Filter bank seed 0→99 (`diagnostics/perceptual.py:94`) | no numerical pin → **SURVIVED** (P3-5) |

**Coverage verdict:** the published inferential layer (ECE, bootstrap CI,
Spearman p, composite orientation, H3 key/direction, Kendall sign, NLL/MSE
scaling, buffer cap) is mutation-robust. The survivors all live in
untested-by-design plumbing (planner, stand-in features, env dynamics) and are
listed as attack surface in §7.

---

## 4. Reduced-pilot reproduction from committed data

No training was re-run (12k steps × 9 configs × 3 members is the recorded
~21 CPU-hour pilot; this audit re-derives every statistic from committed rows,
which is exactly what commit `757e5b2` made reproducible).

### 4.1 Commands

```powershell
# (CI uses `pip install -e .`; local shim reuses the existing CPU site-packages)
$env:PYTHONPATH = ".venv\Lib\site-packages"

# Re-derive composites, Spearman+bootstrap(2000, seed=0), Holm, H1/H2/H3
# from leaderboard/results_main.json and compare to stored values (self-contained):
python -c "import json; from wm_audit.diagnostics import *; \
d=json.load(open('leaderboard/results_main.json',encoding='utf-8')); \
fr=[r for r in d['rows'] if not r.get('compression')]; \
ps=pixel_composite_scores(fr); cs=control_composite_scores(fr); \
[ r.update(pixel_composite=float(p),control_composite=float(c)) for r,p,c in zip(fr,ps,cs)]; \
res=audit_correlations(fr,list(d['correlations']),n_boot=2000,seed=0); \
print('H1',prereg_decision_h1(res['pixel_composite']['rho'],0.4)['supports_H1']); \
print('H2',prereg_decision_h2(res['control_composite']['rho'],0.6)['supports_H2']); \
print('H3',prereg_decision_h3(d['rows'])['status'],prereg_decision_h3(d['rows'])['n_pairs'])"
# expected: H1 True / H2 False / H3 assessable 9; rho/CI match JSON within 1e-6

# Re-render the public leaderboard; must leave git clean
python experiments\make_leaderboard.py
git status --porcelain          # empty

# CI end-to-end smoke step (overwrites gitignored/committed smoke json)
python -m experiments.run_audit --mode smoke    # exit 0

# Full suite
python -m pytest -q
```

Re-derivation used the committed pipeline
(`audit_correlations`, `pixel_composite_scores`, `control_composite_scores`,
`prereg_decision_h1/h2/h3`) on `float_rows = [r for r in rows if not
r.get("compression")]`, `n_boot=2000, ci=0.95, seed=0`.

### 4.2 Result: 178/178 checks OK, 0 FAIL

- MAIN: 18 rows (9 float + 9 int8), composites, all 11 diagnostic ρ/CI/p_raw/
  p_adj/n, H1, H2, H3 (n_pairs=9, pixel_deg, control_deg) match stored values.
- SMOKE: 12 rows (6 float + 6 int8), same checks at `n_boot=200` match.
- `make_leaderboard.py` regenerates `LEADERBOARD.md` byte-identical
  (`git diff` empty).
- Fresh `run_audit --mode smoke` reproduced `results_smoke.json`
  **bit-identical** (file hash equal) — fixed seeds hold on this stack.

### 4.3 Independent scipy oracle (does not use audited stats code)

| diagnostic (n=9 float) | scipy ρ | stored ρ | scipy p_raw | stored p_raw |
|---|---|---|---|---|
| pixel_composite | −0.3167 | −0.3167 | 0.4064 | 0.4064 |
| control_composite | +0.2594 | +0.2594 | 0.5003 | 0.5003 |
| uncertainty_rho | +0.7000 | +0.7000 | 0.0358 | 0.0358 |
| pixel_mse | +0.3167 | +0.3167 | 0.4064 | 0.4064 |
| return_ece | +0.1833 | +0.1833 | 0.6368 | 0.6368 |
| counterfactual_tau | +0.0500 | +0.0500 | 0.8984 | 0.8984 |

### 4.4 Leaderboard number checklist (committed vs re-derived)

| Item | Published | Re-derived | Match |
|---|---|---|---|
| H1 observed ρ (pixel_composite, R) | −0.317 (\|ρ\|≈0.32) | −0.3167 | yes; \|−0.317\|<0.4 → **SUPPORTED** |
| H2 observed ρ (control_composite, R) | 0.259 (≈0.26) | +0.2594 | yes; 0.259<0.6 → **NOT supported** |
| H3 pixel_deg / control_deg | 0.433 / 0.352, n_pairs=9 | 0.4333 / 0.3519, n=9 | yes; control≤pixel → **NOT supported** |
| uncertainty_rho | ρ=0.700, CI [0.078,0.982], p_adj=0.393 | 0.7000 / [0.0783,0.9817] / 0.3935 | yes; not significant at n=9 |
| pixel_mse / nll / lpips† / fvd† ρ | 0.317, CI [−0.554,0.947] | 0.3167, [−0.5536,0.9474] | yes |
| pixel_ssim ρ | −0.350, CI [−1.000,0.554] | −0.3500 | yes |
| return_ece ρ | 0.183, CI [−0.579,0.892] | 0.1833 | yes |
| counterfactual_tau ρ | 0.050, CI [−0.843,0.684] | 0.0500 | yes |
| Spot row (16,16,0,float) | R=−41.587, mse=0.010, ece=2.313, τ=0.677 | −41.5869/0.01039/2.3127/0.6766 | yes (.3f) |
| Spot row (32,32,2,int8) | R=−37.177, mse=0.006, ece=3.080, τ=0.619 | −37.1767/0.00630/3.0801/0.6190 | yes (.3f) |
| Finite values | — | all 18 MAIN + 12 SMOKE rows finite | yes |

**Honesty check (verbatim preserved):** H1 supported is a *weak/null* result
(\|ρ\|<0.4), not evidence that pixel fidelity predicts control; H2/H3
"not supported" is a bounded non-result at n=9 with wide overlapping CIs, not a
falsification; no main 200k×5-seed×2-env result is claimed anywhere in
README/LEADERBOARD/config.

---

## 5. Final test numbers

| Run | Collected | Passed | Failed | Error | Skipped |
|---|---|---|---|---|---|
| Pristine `757e5b2` (stash) | 57 | 57 | 0 | 0 | 0 |
| After P2-1/P2-2 fixes | 58 | 58 | 0 | 0 | 0 |

Plus CI end-to-end step `python -m experiments.run_audit --mode smoke` → exit 0.

---

## 6. Commits this round

1. `60717ae` test(stats): pin Holm step-down monotonic floor with a dipping case
2. `f0db23b` fix(leaderboard): track smoke placeholder json so render works on clone
3. `147bc84` docs(audit): round-1 independent read-only audit report

---

## 6b. CI verification (post-push)

- Pushed `757e5b2..147bc84` to `origin/main` over SSH (identity
  dafahaha / dafahaha@users.noreply.github.com).
- GitHub Actions **CI #12** on `147bc84`: **completed successfully** (3m33s),
  per the public Actions page. Per-step log pages are robots-blocked and the
  anonymous API is rate-limited; both CI steps were instead executed locally
  with exit 0 (`pytest -q` = 58 passed; `run_audit --mode smoke` = exit 0).
- No GitHub Pages / static site exists (no `gh-pages` branch, no HTML); the
  public leaderboard is `leaderboard/LEADERBOARD.md`, re-rendered byte-identical
  from committed JSON and confirmed matching on `main` raw.
- No "main run in progress" overclaim introduced; the locked 200k×5-seed×2-env
  main experiment remains "preregistered, pending Linux/CUDA" everywhere.
- **Honesty-wording alignment (follow-up):** the phrase "MAIN run executed as a
  CPU reduced pilot" (README status bullet, README references note,
  `configs/audit/default.yaml` main-block header, leaderboard section/verdict
  headers) could be read as the locked 200k×5-seed×2-env MAIN having run. It was
  reworded to state explicitly that the locked MAIN run has NOT been executed
  (requires Linux/CUDA) and that committed numbers are the preregistered CPU
  reduced pilot only. No statistic, table cell, or verdict changed; the
  leaderboard diff is prose-only (verified: no numeric line touched).

---

## 7. Attack surface for the next fresh adversarial reviewer

Prioritize these unverified/weak spots:

1. **CEM-MPC planner (`policy/cem.py`)** — elite selection, variance collapse,
   first-action caching, and `imagine()`'s zero-action GRU warm-up
   (`models/dreamer_lite/rssm.py:50-52`) have no unit test; S3 survived. A wrong
   planner silently changes every `R_real` and hence every correlation.
2. **FixedFilterFeatures stand-in (`diagnostics/perceptual.py:85-124`)** — filter
   bank, edge padding, energy pooling, L2 norm are unpinned (S4); check the
   batched-grayscale path `ndim==3` branch for a shape misread, and whether
   per-filter energy pooling makes lpips/fvd nearly collinear (notice in the
   pilot: mse/nll/lpips/fvd share ρ=0.317 exactly).
3. **pixel_nav dynamics & absolute seeds (`envs/pixel_nav.py`)** — only relative
   determinism is tested (S2); verify reward shaping, goal bonus, and clip
   boundaries against frozen numeric fixtures.
4. **Uncertainty battery (`diagnostics/uncertainty.py`)** — scale-mismatched ECE,
   dispersion mean-reduction; no direct tests; `uncertainty_rho=0.70` is the
   pilot's strongest correlation and deserves its own attack.
5. **Bootstrap implementation (`stats/bootstrap.py`)** — shared resample seed
   across diagnostics, undefined-replicate dropping, percentile interpolation;
   only the cover property is tested, not CI width/coverage calibration.
6. **Ensemble averaging convention (`audit.py:_perceptual_on_buffer`,
   `_return_calibration`, `_counterfactual_tau`)** — verify per-member averaging
   is truly identical across pixel/control/uncertainty and that compressed rows
   re-run the *same* battery (R_real from quantized lead member).
7. **H3 insertion-rank edge cases** — ties at boundaries, compressed value
   outside the ladder (0.5 / n+0.5), mixed levers mislabeling (P3-10), and the
   "first float row wins" `setdefault` collision.
8. **run_audit main-mode footgun (P3-3)** — default `--out` overwrites smoke
   data; audit the handoff path before any real main run writes the wrong file.
9. **Notebook drift (P3-1)** — stale imports/nesting; either execute-and-fix or
   remove so it cannot be cited as a working path.
10. **Non-finite emission guard (P3-11)** — assert every emitted diagnostic and
    R_real is finite before writing JSON; today all committed rows pass but the
    pipeline does not enforce it.
