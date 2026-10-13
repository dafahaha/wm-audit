# Which world models actually *control*? We measured.

**WM-Audit** is a preregistered, single-GPU audit suite that disentangles
world-model **perceptual fidelity** (next-frame MSE / SSIM / NLL, plus
fixed-filter stand-ins† for LPIPS / rollout FVD) from **control utility**
(does the model let a planner actually solve the task?).

> *Everyone is building bigger world models; almost no one is checking whether
> the ones we have actually control. We measured it.*

---

## Research question

World models are trained, selected, and reported on pixel fidelity (MSE / NLL /
FVD). But planners *consume* counterfactual action effects and reward
predictions. **Do the diagnostics we report predict closed-loop control?**

We run a battery of diagnostics on each world model, measure the *real*
closed-loop return, then correlate every diagnostic with the return across
seed/capacity — with percentile-bootstrap 95% CIs and Holm-corrected p-values,
all preregistered before seeing data.

| | Diagnostic | What it tests |
|---|---|---|
| Perceptual | `pixel_mse`, `pixel_ssim`, `pixel_nll`, `pixel_lpips`†, `rollout_fvd`† | next-frame / rollout image fidelity |
| Control | `return_ece`, `counterfactual_tau` | imagined-vs-real return calibration; (s,a1,a2) action ordering |
| Uncertainty | `uncertainty_rho`, `uncertainty_ece` | does ensemble dispersion track the true error? |

> † **`pixel_lpips` / `rollout_fvd` are fixed-filter stand-ins, not learned perceptual backbones.** They are computed by a deterministic, dependency-free bank of seeded random convolution filters (`wm_audit.diagnostics.perceptual.FixedFilterFeatures`) standing in for the pretrained VGG/AlexNet trunk behind real LPIPS and the feature distribution behind real FVD. The numbers are real and reproducible (no mock), but they are **not** publication-grade learned LPIPS/FVD — do not over-read them as such. Swap in the real `lpips` trunk for main-result numbers.

`counterfactual_tau` is operationalized as the **global pooled Kendall-τb**: at
each sampled state we step two candidate actions in the oracle env and through
the model (one-step rewards), then pool *all* `(state, action)` triples across
states into one two-sample series and compute a single tie-aware Kendall-τb
between oracle and imagined rewards. It is **not** a per-state sign-agreement
rate / fraction of states where the two actions are ordered correctly.

**Ensemble convention (consistent).** The pixel and uncertainty diagnostics
already averaged per-member values across the ensemble. The control battery
(`return_ece`, `counterfactual_tau`) is now on the same footing: each is computed
*per member* on the identical segments/states and averaged across members, rather
than silently using only the lead member.

**Which rows enter which test.** H1/H2 ask which diagnostics predict closed-loop
return *across model capacity*, so their oriented-rank composites and every
Spearman correlation are computed on the **float baseline rows only**
(`[r for r in rows if not r['compression']]`). Compression-tagged rows (e.g.
INT8) are the *same already-trained ensemble* under a deployment perturbation — a
near-duplicate of a float baseline, not a new capacity tier — so they belong only
to H3. They are still shown in the leaderboard Rows table, but never enter the
H1/H2 correlation pool. H3 pairs each INT8 row with its float baseline on the
exact `(hidden, embed, seed)` triple, and measures degradation on a **fixed
reference ladder** locked to the float baseline set: each compressed row's
diagnostic values are *inserted* into the fixed float reference distribution
(competition/insertion rank), so adding or removing another compression lever
cannot shift a shared pair's degradation.

## Preregistered hypotheses (see `configs/preregistration.yaml`)

- **H1** — pixel fidelity *weakly* predicts closed-loop return:
  `|ρ(pixel, R_real)| < 0.4`. (A falsification is still published as a bounded result.)
- **H2** — control-relevant diagnostics strongly predict return: `ρ > 0.6`.
- **H3** — compression (INT8 / prune / distill) degrades calibration and
  uncertainty alignment *faster* than raw pixel fidelity.

## Status (honest)

- ✅ **Implemented and CI-tested**: stats core (Spearman / Kendall / bootstrap /
  Holm), the diagnostic battery, a deterministic built-in env, replay buffer, a
  tiny CPU RSSM, CEM-MPC close-loop, and INT8 quantization parity.
- 🧪 **MAIN run executed as a CPU reduced pilot (2026-10-13), results in
  `leaderboard/LEADERBOARD.md`** — env `pixel_nav` only (the 2nd prereg env
  `dm_control.cartpole_swingup` cannot install on Windows: `labmaze` needs
  `bazel`); 3 seeds × 3 capacities (n=9 float rows), INT8 lever,
  `bootstrap=2000`. Verdicts: **H1 supported** (|ρ(pixel_composite,R)|=0.32<0.4),
  **H2 not supported** (ρ(control_composite,R)=0.26<0.6), **H3 not supported**
  (INT8 control-deg 0.35 ≤ pixel-deg 0.43, n_pairs=9). This is a *reduced pilot*,
  not the locked 200k-step × 5-seed × 2-env main: seeds 3 (goal 5),
  `train_steps=12000` (goal 200000) for CPU wall-clock; see the config comment block.
- 🚧 **Still in preparation**: the full preregistered 200k-step multi-env main on
  `dm_control` cartpole, public leaderboard numbers. No result counts are claimed
  beyond what is measured above.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .            # for CPU-only torch:
                            # pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

pytest                      # run the audit's unit + smoke tests
python -m experiments.run_audit        # end-to-end smoke (CPU, a few minutes)
python experiments\make_leaderboard.py  # render leaderboard/LEADERBOARD.md
```

The built-in `pixel_nav` env makes the whole chain run on CPU without
`dm_control`. Heavy envs (`dm_control.cartpole_swingup`, Atari) live behind
optional extras.

## Repository layout

```
configs/         preregistration.yaml + env / wm / audit configs
wm_audit/
  stats/         Spearman, Kendall tau-b, bootstrap CI, Holm (numpy, tested)
  diagnostics/   perceptual / control / uncertainty / fidelity_utility
  envs/          unified reset/step interface; built-in pixel_nav (+ optional dm_control)
  data/          replay buffer + intervention perturbations
  models/        dreamer_lite RSSM (CPU)
  train/         collection + ensemble training
  policy/        CEM-MPC close-loop
  deploy/        INT8 quantization parity (RQ3 axis)
experiments/     run_audit.py (entry), make_leaderboard.py
notebooks/       10min_audit.ipynb
tests/           unit + smoke (CI must pass)
leaderboard/     placeholder (LEADERBOARD.md)
```

## Mutation checks

Two tests are deliberately wired so that a broken implementation turns them red:
`test_calibration.py::test_offset_increases_ece_monotonically` (ECE must grow
with systematic offset) and `test_spearman_bootstrap.py::test_bootstrap_ci_covers_known_truth`.
We verified both go red when the underlying math is broken.

## References

WM-Audit is a measurement layer on top of the following world models and
recent audit-adjacent work. (IDs verified against arXiv abstract pages; the
numbers in this repo are an MVP smoke run, not a main result.)

- **DreamerV3** — D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap.
  *Mastering Diverse Domains through World Models.* arXiv:2301.04104.
- **TD-MPC2** — N. Hansen, H. Su, X. Wang.
  *TD-MPC2: Scalable, Robust World Models for Continuous Control.* arXiv:2310.16828.
- **Validate the Dream** — C. Oefinger et al.
  *Validate the Dream Before You Trust Its Verdict: Admissibility for World-Model
  Simulators.* RSS 2026 Workshop on Robot World Models. arXiv:2607.07196.
- **Biased Dreams** — J. Berger, B. Frauenknecht, S. Trimpe, B. Leibe.
  *Biased Dreams: Limitations to Epistemic Uncertainty Quantification in Latent
  Dynamics Models.* RLC 2026. arXiv:2604.25416.
- **stable-worldmodel** — L. Maes et al., incl. Y. LeCun, R. Balestriero.
  *stable-worldmodel: A Platform for Reproducible World Modeling Research and
  Evaluation.* arXiv:2605.21800.

## License

MIT — see [LICENSE](LICENSE).
