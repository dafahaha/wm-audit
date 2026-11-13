# What You Imagine Is Not What You Get

## A preregistered, reproducible null on whether world-model fidelity predicts control utility

**Daizhi Liao** — Guangzhou University
November 2026

---

### Abstract

World models are typically justified by two claims at once: that they predict future observations accurately, and that those predictions are useful for deciding what to do. The two are widely assumed to move together. We preregistered a test of that assumption — locked thresholds, a fixed Holm family, a fixed sample size — and ran it at a scale one person can execute on four CPU cores. The result is a null, and we report it as one: pixel fidelity does not track closed-loop return at this scale (Spearman ρ = 0.021, 95% CI [−0.540, 0.622], p = 0.939, Holm-adjusted p = 1.000, n = 15). The preregistered rule that counts |ρ| < 0.4 as support fires mechanically, but the interval is wide enough to contain both zero and a moderate correlation, so the data cannot distinguish "no relationship" from "a real relationship we are underpowered to see." Two further hypotheses failed in the informative direction: control-relevant diagnostics did not predict return (ρ = −0.102, against a preregistered threshold of 0.6), and INT8 compression degraded the pixel composite *more* than the control/calibration composite (0.780 vs 0.422), the reverse of the preregistered direction. A second locked environment, `dm_control.cartpole_swingup`, was then run at the same scale on GitHub-hosted runners and repeated the pattern rather than rescuing it: ρ = 0.130 for H1 (rule met, still uninformative), 0.301 for H2 (below 0.6), and the same reversed H3 ordering (1.580 vs 1.367). Two of the nine diagnostics are undefined in that environment and are reported as `null` rather than as numbers. We publish the full 30-row table for each environment, the correlation code, and the exact command. The contribution here is not a finding about world models; it is a demonstration that a null can be produced honestly, at low cost, and in a form someone else can re-run.

---

### 1. The question

A world model earns its place in a control stack by promising two things. First, it imagines the future well enough that the imagined frames resemble the real ones — measured by reconstruction loss, structural similarity, perceptual distance, or the distributional distance between imagined and real rollouts. Second, that imagination is useful: a planner that searches inside the model finds actions that also work in the environment.

Textbooks and papers tend to bundle these. A model that predicts better is assumed to plan better. If that assumption held unconditionally, model design would collapse into a single objective — minimize prediction error — and the field would not need separate diagnostics for "is the model accurate" and "is the model useful."

We wanted to know whether the assumption holds when the model is small, the environment is cheap, and the budget is one undergraduate's CPU time. Not whether it holds for DreamerV3 on DMControl, but whether it holds in the regime most people actually start in. That is a modest question, and the honest answer we got was also modest: at this scale, we cannot show that it holds, and two of our three probes came back pointing the other way.

A word on why we preregistered something this unglamorous. Negative results are cheap to produce and expensive to trust, because a null can always be explained away after the fact — the model was too small, the environment too easy, the metric too crude. Locking the thresholds, the sample size, the correction family, and the decision rules *before* running removes most of that discretion. What remains is a result that can be checked, and a record of what we predicted before we knew.

---

### 2. Preregistered design

The preregistration was committed before the main run and is unmodified. Every parameter below is the locked value; nothing was reduced after seeing results.

**Task and scale.** A locked main of 200,000 training steps × 5 seeds × 3 model capacities × 2 environments. The two environments were a self-built 16×16 pixel navigation task (`pixel_nav`) and `dm_control.cartpole_swingup`. Both halves are now executed at the full locked scale — 30 rows per environment, 15 float baselines plus 15 INT8-compressed counterparts — so the study is a 2-of-2 preregistered main rather than a subset. The first environment ran on a 4-core CPU container; the second, which could not install MuJoCo there, ran as five seed shards on GitHub-hosted runners. Verdicts differ by environment on H3 and agree on H1 and H2; §3 reports both, and the disagreements are not smoothed over.

**Model.** A tiny recurrent state-space model, trained from pixels: hidden size 16, embedding size 16, ensemble of 3 members, batch size 64, learning rate 0.002, buffer capacity 16,000, 100,000 collection steps per configuration. "Tiny" is not a hedge; these are the actual numbers, and they are the reason the whole study fits on four cores.

**Planning.** Closed-loop return is measured by cross-entropy-method planning inside the learned model: horizon 6, 48 candidate action sequences per iteration, 8 elites, 2 iterations. The return reported as `R_real` is the environment return actually obtained by executing the planned action, not the model's imagined return.

**Diagnostics.** Nine per configuration, in two families. Pixel fidelity: `pixel_mse`, `pixel_ssim`, `pixel_nll`, `pixel_lpips`, `rollout_fvd`. Control relevance: `return_ece` (imagined-versus-real return calibration), `counterfactual_tau` (a pooled Kendall-tau-b over state–action triples stepped both in the oracle environment and through the model), `uncertainty_rho`, `uncertainty_ece`.

Two composites are formed by an oriented-rank recipe fixed in advance: within the configuration set, rank each diagnostic, flip the ranks of error-type metrics so that higher always means better, and average. The five pixel diagnostics form `pixel_composite`; the control battery forms `control_composite`. The compression-tagged rows are excluded from every H1/H2 correlation by rule — they are the same trained model under a deployment perturbation, not a new capacity tier — and enter only the paired H3 test.

**Hypotheses and decision rules.**

- **H1** — pixel fidelity weakly predicts control utility. Supported iff |ρ(pixel_composite, R_real)| < 0.4.
- **H2** — control-relevant diagnostics strongly predict control utility. Supported iff ρ(control_composite, R_real) > 0.6.
- **H3** — under compression, pixel fidelity degrades slowly while calibration and uncertainty alignment degrade faster. Supported iff mean control-composite degradation strictly exceeds mean pixel-composite degradation.

Correction is Holm step-down over all reported statistics, family fixed at eleven (nine diagnostics plus two composites), α = 0.05, bootstrap CIs from 2,000 resamples.

Note the H1 rule: it tests for *absence* of a strong correlation. A model that predicts nothing at all also satisfies it. That is a design flaw we accepted knowingly, because the alternative — testing for the presence of a correlation — risks claiming success from noise at n = 15. The consequence is that H1 can never be evidence *for* the relationship; it can only fail to refute it.

---

### 3. Results

Float baselines only, n = 15. All eleven statistics, with bootstrap CIs and Holm-adjusted p-values:

| Statistic | ρ | 95% CI | p (raw) | p (Holm) |
|---|---:|---|---:|---:|
| `pixel_mse` | +0.0607 | [−0.562, +0.612] | 0.830 | 1.000 |
| `pixel_ssim` | −0.0429 | [−0.629, +0.596] | 0.880 | 1.000 |
| `pixel_nll` | +0.0607 | [−0.562, +0.612] | 0.830 | 1.000 |
| `pixel_lpips` | −0.1536 | [−0.617, +0.389] | 0.585 | 1.000 |
| `rollout_fvd` | −0.2357 | [−0.669, +0.366] | 0.398 | 1.000 |
| `return_ece` | +0.0214 | [−0.555, +0.576] | 0.940 | 1.000 |
| `counterfactual_tau` | +0.2504 | [−0.274, +0.688] | 0.368 | 1.000 |
| `uncertainty_rho` | −0.2179 | [−0.700, +0.366] | 0.435 | 1.000 |
| `uncertainty_ece` | +0.0071 | [−0.640, +0.568] | 0.980 | 1.000 |
| `pixel_composite` | +0.0215 | [−0.540, +0.622] | 0.939 | 1.000 |
| `control_composite` | −0.1021 | [−0.590, +0.436] | 0.717 | 1.000 |

**H1 — threshold met, nothing demonstrated.** |ρ| = 0.021 < 0.4, so the locked rule counts H1 as supported. We report it as supported *and* as uninformative, because both are true. The confidence interval spans −0.54 to +0.62: at n = 15 the design cannot separate "no relationship" from "a correlation of half a standard deviation's worth of rank ordering." The raw p is 0.939 and the Holm-adjusted p is 1.000. The rule fired by construction; the data did not speak.

**H2 — contradicted.** The control composite correlates −0.102 with realized return, against a preregistered threshold of +0.6. The single strongest control diagnostic is `counterfactual_tau` at +0.250, still far below the bar and far from significance. This is the one place where the null carries real information: the preregistered prediction was specific and directional, and it failed. A planner-side diagnostic built from counterfactual reward comparisons did not tell us which model would control better.

**H3 — reversed.** Under INT8 compression the pixel composite degrades by 0.780 rank units while the control/calibration composite degrades by 0.422. The preregistration expected the opposite ordering — that quantization would be felt more by calibration and uncertainty than by raw reconstruction. Fifteen pairs, and the direction came out backwards. This is a small, cheap, and genuinely surprising result: the thing we thought fragile under quantization was the robust one.

**What the diagnostics did to each other.** At the pilot scale of 9 rows, five diagnostics — four pixel metrics (`pixel_mse`, `pixel_nll`, `pixel_lpips`, `rollout_fvd`) and one control metric (`uncertainty_ece`) — had *identical* rank order (ρ = 0.3167 each), which collapsed the composite into a single metric wearing five names. At n = 15 the collapse relaxed but did not disappear: `pixel_mse` and `pixel_nll` are still exactly co-linear at ρ = +0.0607, while the other three separated. Two of five protocols are measuring the same rank ordering on this task. With 15 points and 11 tests, the Holm family is already optimistic — the effective number of independent probes is smaller than the nominal one.

**The second environment, measured.** The locked design named two environments; the first run executed one. The second — `dm_control.cartpole_swingup`, 64×64×3 observations, 200k training steps, 100k collection steps, seeds 0–4, three capacities, one INT8 lever — has now been run at the same locked scale (5 seed shards on GitHub-hosted `ubuntu-latest` runners, MuJoCo 4.0.1 through `dm_control` under `MUJOCO_GL=osmesa`). Thirty rows, n = 15 float baselines:

| Statistic | ρ | 95% CI | p (raw) | p (Holm) | n |
|---|---:|---|---:|---:|---:|
| `pixel_mse` | −0.1604 | [−0.665, +0.420] | 0.568 | 1.000 | 15 |
| `pixel_ssim` | −0.0613 | [−0.637, +0.574] | 0.828 | 1.000 | 15 |
| `pixel_nll` | −0.1604 | [−0.665, +0.420] | 0.568 | 1.000 | 15 |
| `pixel_lpips` | −0.1838 | [−0.674, +0.355] | 0.512 | 1.000 | 15 |
| `rollout_fvd` | −0.2162 | [−0.690, +0.310] | 0.439 | 1.000 | 15 |
| `return_ece` | −0.5253 | [−0.869, +0.004] | 0.0443 | 0.443 | 15 |
| `counterfactual_tau` | undefined | — | — | — | 0 |
| `uncertainty_rho` | −0.3269 | [−0.828, +0.307] | 0.300 | 1.000 | 12 |
| `uncertainty_ece` | −0.0829 | [−0.579, +0.429] | 0.769 | 1.000 | 15 |
| `pixel_composite` | +0.1297 | [−0.477, +0.685] | 0.645 | 1.000 | 15 |
| `control_composite` | +0.3014 | [−0.182, +0.668] | 0.275 | 1.000 | 15 |

**H1 — threshold met again, still nothing demonstrated.** |ρ| = 0.130 < 0.4, so the locked rule counts H1 as supported. The interval [−0.477, +0.685] is as wide as it was in the navigation task: a second environment does not add power, only a second sample of the same width. What it does add is replication of the *shape* — pixel fidelity carries no usable rank information about return in either task.

**H2 — contradicted again.** +0.301 against a threshold of +0.6. The strongest single control-side signal here is `return_ece` at −0.525 (raw p = 0.044, Holm p = 0.443): imagined-return calibration does order the models somewhat better than chance, in the direction that better calibration means higher realized return. That is the first control-side diagnostic in this study to come close to conventional significance, and it survives neither the correction nor the second environment's own threshold rule. It is a hint, not a result.

**H3 — contradicted in the same direction, a second time.** Under INT8 the pixel composite degrades by 1.580 rank units and the control/calibration composite by 1.367. The preregistration expected the opposite ordering (calibration and uncertainty more fragile than reconstruction). Both environments now disagree with the preregistered direction, which turns a single surprising reversal into a repeated one: in our models, quantization is felt more on the pixels than on the calibration and uncertainty diagnostics.

**Two diagnostics were not measurable here, and we report them as such.** `counterfactual_tau` is NaN in all 30 rows: `_counterfactual_series` steps two randomly drawn actions once each in the real environment and compares the oracle rewards with the imagined ones, and for `cartpole_swingup` the one-step reward is effectively constant across those 40 pairs, so the tie-corrected τb denominator vanishes. `uncertainty_rho` is NaN in 6 of 30 rows, all at the smallest capacity `(16,16)` on seeds 0, 3 and 4: the three ensemble members collapse onto the same prediction, ensemble dispersion becomes constant, and the Spearman correlation against realized error is undefined. Consequently `counterfactual_tau` is reported as `null` with `n = 0` rather than as a number, `uncertainty_rho` is computed on 12 of 15 pairs, and the H2 `control_composite` mixes 2 of 3 diagnostics in 12 rows and 1 of 3 in the three collapsed rows. The H2 number above is therefore a degraded composite, not the strict three-diagnostic one the preregistration names; treating it as the preregistered test would be overclaiming.

**A pipeline defect found while reading this data, and fixed before publishing.** The first aggregation of these 30 rows ranked the NaN columns anyway, because `rankdata` parks non-finite values at one end of the ladder instead of refusing them. That produced a plausible-looking ρ = −0.043 for an all-NaN diagnostic, dragged the H2 composite to +0.308, and — worse — scored unmeasurable compressed rows as the worst possible value on the fixed ladder, which reported a control-composite degradation of 4.311 and made **H3 look supported**. The fix drops non-finite pairs, reports `n` and `n_dropped` explicitly, emits `null` instead of `NaN` in the JSON, and leaves an unmeasurable diagnostic out of a row's composite mean. Numbers above are from the fixed pipeline; the buggy ones are recorded here so the size of the artifact is on the record. `tests/test_prereg_decisions.py` now pins that behaviour.

---

### 4. How to read a null like this

Three readings are available, and only one of them is honest.

**"Fidelity doesn't matter for control."** This reading overreaches. The interval is too wide, the environment too simple, and the models too small to support a claim about world models in general. A null at this scale is a statement about this scale.

**"The study was underpowered, so nothing can be concluded."** Also an overreach in the other direction. Underpowering does not erase H2's directional failure or H3's reversal; those are predictions that were specific, locked, and wrong.

**The defensible reading.** At 200k steps, on a 16×16 navigation task, with a 16-dimensional latent and 15 samples per condition, pixel fidelity carried no information about which model would control better, and the control-side diagnostics we designed carried no information either. Two of the three preregistered predictions were contradicted by the data. Whether this generalizes is an open question we have not answered — and saying so is the point.

There is a fourth reading we should name and reject explicitly: that the composite recipe rescued a weak signal. It did not. `pixel_composite` at +0.021 is closer to zero than every individual pixel diagnostic, and `control_composite` at −0.102 is weaker than `counterfactual_tau` alone at +0.250. Averaging did not concentrate signal; it diluted what little structure existed.

---

### 5. Threats to validity

**Environment.** `pixel_nav` is a 16×16 grid with a deterministic transition function and a dense reward. A world model on this task barely needs to model anything: the observation is small, the dynamics are near-Markov, and the horizon is short. `cartpole_swingup` is the cheapest DMControl task there is — a 64×64×3 render of two rigid bodies — so adding it widens the claim by one step, not by enough to speak about visual control in the round. Neither result says much about DreamerV3-scale tasks, ProcGen, or a real robot.

**Baselines.** There is no SOTA world model in the comparison — no DreamerV3, no TD-MPC2. Every number comes from our own 16-dimensional RSSM. This means the question "do these diagnostics predict control utility" has only been asked inside our own model family, where the variation we are correlating against is variation in capacity and seed, not variation in modeling approach. A diagnostic that fails to discriminate three tiny RSSMs may still discriminate a good model from a bad one.

**Sample size.** n = 15 with 11 reported statistics. The bootstrap intervals are honest about this — every one of them spans zero by a wide margin — but a reader looking for a "significant" correlation will not find one, and should not conclude that the absence is informative beyond its width.

**Hardware.** CPU only, four cores, no GPU. This bounds model size and training steps and is the direct reason the study is small. It is also the reason the study exists: most people who want to test a hypothesis about world models start here.

**Outcome resolution in the second environment.** The 15 `cartpole_swingup` float rows carry only 9 distinct `R_real` values, and 2 of the 5 seeds show a single value across all three capacities. One configuration returns 158.8 while the rest sit between 4.0 and 25.3. A rank correlation over 15 rows with 9 outcome levels — one of them a lone outlier — cannot distinguish a weak diagnostic from an uninformative one, and that is part of why the intervals above are as wide as they are. The pipeline now prints and stores this as `outcome_resolution`, so the limit is visible in the artifact rather than only in this paragraph.

**One design flaw, stated plainly.** H1's decision rule tests for the absence of a strong correlation, which a completely uninformative model also satisfies. We accepted this because the alternative risks a false positive at n = 15. Anyone reusing this design should know the rule cannot produce affirmative evidence.

---

### 6. Reproducibility

Everything needed to re-run the analysis is in the repository.

- Data: `leaderboard/results_main_pixelnav_200k.json` and `leaderboard/results_main_cartpole_200k.json` — 30 rows each, with all nine diagnostics (where measurable), both composites, `R_real`, and seed/capacity identifiers; each non-finite entry is `null` and carries the reason (`n`, `n_dropped`, `undefined`), and `composite_coverage` states how many diagnostics actually entered each row's composite. The navigation run's `run_meta` also records the machine, device, seeds, capacities, and the reason MuJoCo was unavailable in that container.
- Second-environment data provenance: `leaderboard/ci/cartpole_seed0..4.{json,partial.jsonl}` are the five per-seed shard artifacts exactly as the runners committed them; the merged file is their concatenation in seed order, aggregated with `--resume`.
- Code: `wm_audit/diagnostics/fidelity_utility.py` — the oriented-rank composite, Spearman correlations, bootstrap CIs, and Holm step-down are all in one module and are exercised by the test suite.
- Command: `python -m experiments.run_audit --config configs/audit/main_pixel_nav_full.yaml --mode main --out leaderboard/results_main_pixelnav_200k.json`
- Machine: 4-core AMD EPYC, 8 GiB RAM, Ubuntu 22.04 container, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`, four worker processes at one thread each.
- Determinism: fixed data seed 123, seeds 0–4, bootstrap resamples 2,000. Correlations recompute from the rows to machine precision — the reported ρ values reproduce to 1e-15 from the stored diagnostics, which is how we checked the analysis before writing this.
- Independently re-derived on a second machine: the committed 30-row partial was re-aggregated from a clean checkout of `main` on a separate Linux box and the output is **byte-identical** to `leaderboard/results_main_cartpole_200k.json` (129 leaf fields, zero mismatches), with the test suite green there (78 tests). The analysis does not depend on the machine that produced it.

The pilot at n = 9 is committed separately (`leaderboard/results_main.json`) and is retained on purpose. It is the run in which five pixel diagnostics collapsed to one rank ordering, and the run in which a 60k-step reward-scaling "win" looked like a 3× improvement before the preregistered 120k budget and a fixed-temperature control arm overturned it as seed noise. Keeping both runs visible is the cheapest available evidence that the analysis pipeline is not tuned to produce favorable output.

---

### 7. What comes next

Three things, in increasing cost.

**A counterfactual measurement that works on continuous control.** `counterfactual_tau` was the point of the control battery — "does the model rank two actions correctly?" — and it is undefined in `cartpole_swingup` because a single 10 ms step barely moves the pole, so the oracle rewards across 40 random action pairs are numerically identical. Measuring it on a multi-step discounted reward (or on a state distribution where the actions actually diverge) would restore one of the three control diagnostics. Until then, H2 is tested on two diagnostics out of the three the preregistration names.

**A real baseline.** The strongest criticism available against this study is that it compares tiny models to each other. Adding one external world model — a released DreamerV3 or TD-MPC2 checkpoint at a comparable parameter budget — would let the diagnostics be tested against variation in *approach* rather than variation in width and seed. This is the highest-leverage single change, and also the most expensive.

**More environments versus more seeds.** With 15 points and 11 tests, adding configurations is the only route to usable power. The preregistered design chose 2 environments × 3 capacities × 5 seeds, and both halves are now executed. If we were re-locking today, we would trade one capacity tier for two more environments: the capacity axis produced most of the co-linearity and little of the signal.

---

### 8. Closing

The paper this most resembles is not a positive result about world models. It is a record of a prediction made in advance, a run executed at the promised scale, and an outcome reported without renegotiation. The preregistered thresholds fired in ways we did not want: one rule met by accident, one prediction contradicted, one direction reversed. All three are in the artifact.

If there is a claim worth carrying forward, it is narrower than "fidelity doesn't predict control." It is that the coupling between what a model imagines and what a planner can do with it is not automatic at small scale, is not detectable with 15 samples, and is not delivered by averaging diagnostics that were already measuring the same thing. Determining whether it holds at large scale requires exactly what we did not have: a real baseline, a harder environment, and more than 15 points.

---

*Artifacts: `leaderboard/results_main_pixelnav_200k.json`, `leaderboard/results_main.json`, `configs/preregistration.yaml`, `configs/audit/main_pixel_nav_full.yaml`, `wm_audit/diagnostics/fidelity_utility.py`, `docs/audit_round1_1022.md`. Repository: https://github.com/dafahaha/wm-audit*
