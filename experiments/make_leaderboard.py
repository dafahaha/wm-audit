#!/usr/bin/env python
"""Render leaderboard/LEADERBOARD.md from committed machine-readable results.

Two sections are emitted, each generated from its own committed JSON so the
public leaderboard is fully reproducible from code + data in the repo:

* Reduced pilot (CPU, NOT the locked MAIN run) <- leaderboard/results_main.json (tracked in git)
* SMOKE CI placeholder <- leaderboard/results_smoke.json (preserved history, tracked in git)

Re-running this script must leave a clean `git status` (generated file == the
committed one). It never hand-writes numbers; every table cell is derived from
the two JSON files. The only static text is the explanatory prose / headers.
"""
from __future__ import annotations

import json
import os

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_RES = os.path.join(here, "leaderboard", "results_main.json")
# Locked-scale MAIN run restricted to the one env that is runnable here
# (pixel_nav). Kept as its own artifact so the reduced pilot stays reproducible
# and neither file can silently stand in for the other.
MAIN_FULL_RES = os.path.join(here, "leaderboard", "results_main_pixelnav_200k.json")
SMOKE_RES = os.path.join(here, "leaderboard", "results_smoke.json")
out_path = os.path.join(here, "leaderboard", "LEADERBOARD.md")

# Diagnostics whose value label carries the fixed-filter stand-in dagger.
_STANDIN = {"pixel_lpips", "rollout_fvd"}
# Composite rows bold their name + rho (they are the preregistered H1/H2 scores).
_COMPOSITE = {"pixel_composite", "control_composite"}


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _corr_table(data, *, main: bool):
    """Diagnostics-vs-R_real correlation table. Bold composite rows when main."""
    head = ("| diagnostic | Spearman rho vs R_real | 95% CI | Holm-adj p |"
            if main else
            "| diagnostic | Spearman rho vs R_real (float-only) | 95% CI | Holm-adj p |")
    lines = [head, "|---|---|---|---|"]
    for d, v in data["correlations"].items():
        label = d + ("†" if d in _STANDIN else "")
        rho = f"{v['rho']:.3f}"
        if main and d in _COMPOSITE:
            label = f"**{label}**"
            rho = f"**{rho}**"
        lines.append(f"| {label} | {rho} | "
                     f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] | "
                     f"{v['p_adj']:.3f} |")
    return "\n".join(lines)


def _rows_table(data, *, main: bool):
    title = "### Rows (float baseline + INT8 variants)" if main else \
            "### Rows (float baseline + compression variants)"
    lines = [title, "",
             "| hidden | embed | seed | compression | R_real | pixel_mse | "
             "return_ece | counterfactual_tau |",
             "|---|---|---|---|---|---|---|---|"]
    for r in data["rows"]:
        lines.append(
            f"| {r['hidden']} | {r['embed']} | {r['seed']} | "
            f"{r.get('compression', 'float')} | "
            f"{_fmt(r['R_real'])} | {_fmt(r['pixel_mse'])} | "
            f"{_fmt(r['return_ece'])} | {_fmt(r['counterfactual_tau'])} |")
    return "\n".join(lines)


def _main_verdicts(data, title="### Preregistered verdicts (reduced pilot; NOT the locked MAIN run)"):
    h1, h2, h3 = data["H1"], data["H2"], data["H3"]
    # S2 honesty alignment: the preregistered threshold firing (|rho| < 0.4) is a
    # mechanical label only. On the n=9 pilot the bootstrap CI crosses zero and
    # Holm-adjusted p = 1.0, so "threshold met" is NOT a significant finding. When
    # the pixel_composite row is available, render the qualified wording (numbers
    # derived from it, never hardcoded); fall back to the plain label only for
    # synthetic/unit-test inputs that carry no correlations block.
    pc = data.get("correlations", {}).get("pixel_composite")
    if h1["supports_H1"] and pc is not None:
        h1_res = (
            f"**THRESHOLD MET — weak / tentative exploratory signal only** "
            f"(n={pc['n']}; CI [{pc['ci_lo']:.3f}, {pc['ci_hi']:.3f}] crosses zero; "
            f"p_raw={pc['p_raw']:.3f}, Holm-adj p={pc['p_adj']:.3f} → "
            f"**not significant; NOT an established finding**)"
        )
    elif h1["supports_H1"]:
        # Composite stats are absent (pc is None). The threshold rule fired, but
        # there is no CI or p-value to quote — never render a bare "SUPPORTED".
        h1_res = ("**THRESHOLD MET — composite stats unavailable, so no CI/p "
                  "can be quoted; not an established finding**")
    else:
        h1_res = ("**NOT supported** (pixel fidelity tracks control "
                  "more than preregistered)")
    h2_res = ("**SUPPORTED** (control composite tracks closed-loop return)"
              if h2["supports_H2"] else
              "**NOT supported** (control composite below threshold)")
    # P3-1: H3 is only `assessable` when compression rows matched float
    # baselines; the not_assessable branch omits supports_H3 and the
    # degradation keys by design, so index them only when present (otherwise
    # rendering would KeyError instead of degrading to an em dash).
    if h3.get("status") == "assessable":
        h3_res = ("**SUPPORTED** (control/calibration degrades faster)"
                  if h3["supports_H3"] else
                  "**NOT supported** (n_pairs={})".format(h3.get("n_pairs", "-")))
        h3_observed = (
            f"pixel_deg={h3['pixel_composite_degradation']:.3f}, "
            f"control_deg={h3['control_composite_degradation']:.3f}"
        )
    else:
        h3_res = f"**not assessable** (status={h3.get('status', 'unknown')})"
        h3_observed = "pixel_deg=—, control_deg=—"
    lines = [
        title,
        "",
        "| hypothesis | threshold | observed rho | result |",
        "|---|---|---|---|",
        f"| H1 | \\|ρ(pixel_composite, R)\\| < 0.4 | {h1['observed_rho']:.3f} | {h1_res} |",
        f"| H2 | ρ(control_composite, R) > 0.6 | {h2['observed_rho']:.3f} | {h2_res} |",
        (f"| H3 | control_deg > pixel_deg under INT8 | "
         f"{h3_observed} | {h3_res} |"),
    ]
    return "\n".join(lines)


def _smoke_verdicts(data):
    h1, h2, h3 = data["H1"], data["H2"], data["H3"]
    # H3 is only `assessable` when compression rows matched float baselines;
    # the not_assessable branch (no / unmatched compression rows) omits the
    # degradation keys and `supports_H3` by design -- render an em dash instead
    # of indexing missing keys (which would KeyError).
    if h3.get("status") == "assessable":
        h3_cell = (
            f"{h3['status']}; n_pairs={h3.get('n_pairs', '-')}; "
            f"supports_H3={h3['supports_H3']}; "
            f"pixel_deg={h3['pixel_composite_degradation']:.3f}, "
            f"control_deg={h3['control_composite_degradation']:.3f}"
        )
    else:
        h3_cell = (
            f"{h3.get('status', 'unknown')}; "
            f"n_pairs={h3.get('n_pairs', '—')}; supports_H3=—; "
            "pixel_deg=—, control_deg=—"
        )
    lines = [
        "### Preregistered verdicts (smoke)",
        "",
        "| hypothesis | status / result |",
        "|---|---|",
        f"| H1 | observed={h1['observed_rho']:.3f}; supports_H1={h1['supports_H1']} |",
        f"| H2 | observed={h2['observed_rho']:.3f}; supports_H2={h2['supports_H2']} |",
        f"| H3 | {h3_cell} |",
    ]
    return "\n".join(lines)


def _collinear_group(corr, anchor="pixel_mse"):
    """Diagnostics whose (rho, CI, p_raw) equal the anchor's exactly.

    Identical values mean one shared rank vector, i.e. these diagnostics are not
    independent confirmations. Detected from the data so the disclosure follows
    the run actually being rendered instead of a remembered pilot shape.
    """
    base = corr.get(anchor)
    if base is None:
        return []
    tied = []
    for d, v in corr.items():
        if (abs(v["rho"] - base["rho"]) < 1e-12
                and abs(v["p_raw"] - base["p_raw"]) < 1e-9
                and abs(v["ci_lo"] - base["ci_lo"]) < 1e-9
                and abs(v["ci_hi"] - base["ci_hi"]) < 1e-9):
            tied.append(d)
    return tied


def _main_full_section(data):
    """Locked-scale MAIN section (single-env subset) from its own committed JSON."""
    meta = data.get("run_meta", {})
    corr = data["correlations"]
    n = next(iter(corr.values()))["n"]
    envs_run = meta.get("envs_run", ["pixel_nav"])
    envs_missing = meta.get("envs_missing", ["dm_control.cartpole_swingup"])
    missing_reason = meta.get("missing_reason", "MuJoCo/EGL unavailable")
    train_steps = meta.get("train_steps", "?")
    collect_steps = meta.get("collect_steps", "?")
    bootstrap = meta.get("bootstrap_resamples", "?")
    ensemble_size = meta.get("ensemble_size", "?")
    device = meta.get("device", "CPU")
    machine = meta.get("machine", "")
    seeds = meta.get("seeds", [])
    caps = meta.get("capacities", [])
    date = meta.get("date", "2026-11-07")

    scalar = [d for d in corr if d not in _COMPOSITE]
    strongest = max(scalar, key=lambda d: abs(corr[d]["rho"]))
    s = corr[strongest]
    pc = corr["pixel_composite"]
    cc = corr["control_composite"]
    h3 = data["H3"]
    pixel_deg = h3.get("pixel_composite_degradation")
    control_deg = h3.get("control_composite_degradation")

    tied = _collinear_group(corr)
    if len(tied) > 1:
        tied_txt = (
            f"**Rank-collinearity caveat (n={n}).** {len(tied)} diagnostics — "
            + ", ".join(f"`{d}`" for d in tied)
            + f" — share identical rho={corr[tied[0]]['rho']:.3f}, "
            f"p_raw={corr[tied[0]]['p_raw']:.3f} and the same CI: one rank pattern, "
            "not that many independent confirmations. Multi-diagnostic agreement "
            "must not be counted as cross-validation on this run."
        )
    else:
        tied_txt = (
            f"**Rank-collinearity check (n={n}).** No pair of diagnostics shares an "
            "identical rank vector on this run, so the battery's signals are not "
            "mechanically duplicated here."
        )

    deg_txt = ""
    if pixel_deg is not None and control_deg is not None:
        deg_txt = (f"pixel composite degrades by {pixel_deg:.3f} while the "
                   f"control/calibration composite degrades by only {control_deg:.3f}")

    return f"""## Main results — `{envs_run[0]}` single-env subset ({date})

> **Status: single-env subset of the locked preregistered MAIN — NOT the full 2-env main.**
> The preregistration asks for {train_steps} train steps × {len(seeds)} seeds ×
> {len(caps)} capacity tiers × 2 envs. This run executes `{envs_run[0]}` at the locked
> scale; the second prereg env (`{envs_missing[0]}`) is **not included** — {missing_reason}.
>
> Kept exactly as preregistered: {train_steps} train steps, {collect_steps} collect steps,
> seeds {seeds}, capacity tiers {caps}, ensemble_size={ensemble_size},
> bootstrap={bootstrap}, INT8 lever, and the whole stats pipeline (float-only H1/H2,
> fixed-float-ladder H3, Spearman t-approx p + Holm). Device: {device}; {machine}.
>
> **Recurrent caveat (unchanged):** training resets the GRU state to zeros every batch
> (`wm_audit/train/train_wm.py`); sequences are never unrolled, so no temporal dependency
> is learned and this run is a per-transition stand-in. Its H1/H2/H3 conclusions describe
> only this stand-in pipeline and must NOT be extrapolated to a genuine recurrent world
> model with learned temporal dynamics.
>
> Reproduce: `python -m experiments.run_audit --mode main --config configs/audit/main_pixel_nav_full.yaml --jobs 4 --out leaderboard/results_main_pixelnav_200k.json`, then `python experiments/make_leaderboard.py`.

### Diagnostics vs R_real (float-only, n={n}; percentile bootstrap={bootstrap})

{_corr_table(data, main=True)}

{_rows_table(data, main=True)}

{_main_verdicts(data, title="### Preregistered verdicts (main — pixel_nav single-env subset)")}

Read honestly: the strongest absolute diagnostic-vs-return correlation on this run is
`{strongest}` (rho={s['rho']:.3f}; p_raw={s['p_raw']:.3f}, Holm-adj p={s['p_adj']:.3f}).
The H1 threshold firing (|rho|={abs(pc['rho']):.3f}<0.4) remains a **weak, exploratory
label only** — CI [{pc['ci_lo']:.3f}, {pc['ci_hi']:.3f}] crosses zero and
Holm-adj p={pc['p_adj']:.3f}. **H2 is contradicted at this scale**:
rho(control_composite)={cc['rho']:.3f}, far below the preregistered 0.6 threshold
(Holm-adj p={cc['p_adj']:.3f}). **H3 reverses**: under INT8 the {deg_txt}. These are
bounded null/negative results per the preregistration's honesty clause, reported as such.

{tied_txt}
"""


def main():
    main_data = _load(MAIN_RES)
    smoke_data = _load(SMOKE_RES)
    main_full_data = _load(MAIN_FULL_RES) if os.path.exists(MAIN_FULL_RES) else None
    main_full_section = _main_full_section(main_full_data) if main_full_data else ""
    n_main = next(iter(main_data["correlations"].values()))["n"]
    n_smoke = next(iter(smoke_data["correlations"].values()))["n"]
    # P3-2: derive the strongest-correlation numbers from the JSON rather than
    # hardcoding rho=0.70 / p=0.393, so the prose cannot drift from the table.
    _unc = main_data["correlations"]["uncertainty_rho"]
    unc_rho = _unc["rho"]
    unc_p = _unc["p_adj"]
    unc_raw = _unc["p_raw"]
    # S2: pixel_composite row drives the qualified H1 prose; the collinear group
    # (pixel_mse == pixel_nll == pixel_lpips == rollout_fvd == uncertainty_ece
    # at n=9) shares one rank vector, so pixel_mse's numbers stand in for all five.
    pc = main_data["correlations"]["pixel_composite"]
    col = main_data["correlations"]["pixel_mse"]

    md = f"""# WM-Audit Leaderboard

> Which world-model diagnostics actually predict closed-loop control utility? Lower on pixel metrics = better fidelity.

{main_full_section}

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

Correlations are computed on the **float baseline rows only** (n={n_main} = 3 seeds × 3 capacities):
INT8 rows are deployment perturbations of an already-trained model, not new capacity tiers, so
they belong only to the H3 paired test and never enter the H1/H2 correlation pool.

### Diagnostics vs R_real (float-only, n={n_main}; percentile bootstrap=2000)

{_corr_table(main_data, main=True)}

{_rows_table(main_data, main=True)}

{_main_verdicts(main_data)}

Read honestly: on this reduced CPU pilot, the single strongest diagnostic-vs-return correlation
is `uncertainty_rho` (ρ={unc_rho:.2f}, p_raw={unc_raw:.3f} but Holm-adj p={unc_p:.3f} — not
significant at n={n_main}). The H1 threshold being met (|ρ|={abs(pc['rho']):.3f}<0.4) is a
**weak, exploratory signal only** — the CI [{pc['ci_lo']:.3f}, {pc['ci_hi']:.3f}] crosses zero and
Holm-adj p={pc['p_adj']:.3f}, so it is NOT significant and must not be read as an established
finding that pixel fidelity predicts control. H2/H3 are bounded null/negative results per the
preregistration's honesty clause, not falsifications.

**Rank-collinearity caveat (n={n_main}).** Five diagnostics — `pixel_mse`, `pixel_nll`,
`pixel_lpips`†, `rollout_fvd`†, `uncertainty_ece` — share *identical* ρ={col['rho']:.3f},
p_raw={col['p_raw']:.4f} and CI [{col['ci_lo']:.3f}, {col['ci_hi']:.3f}] in the correlation table
above: their {n_main}-row rank vectors are exactly collinear. They are one rank pattern, not five
independent confirmations; the battery therefore contains far fewer independent signals than its 11
rows suggest, and multi-diagnostic agreement must not be counted as cross-validation on this pilot
(likely driven by n={n_main} plus deterministic, capacity-ordered stand-in models; to be
re-checked in the locked main run).

---

## SMOKE results — CI placeholder (downgraded, preserved)

> **This block is the old CI smoke run (n={n_smoke} float rows = 2 seeds × 3 caps, `bootstrap=200`,
> `train_steps=150`). It is a placeholder, NOT a main result.** Kept for history; do not cite as
> evidence. Generated by `python -m experiments.run_audit --mode smoke` → `results_smoke.json`.

### Diagnostics vs R_real (float-only, n={n_smoke})

{_corr_table(smoke_data, main=False)}

{_rows_table(smoke_data, main=False)}

{_smoke_verdicts(smoke_data)}

---

_† `pixel_lpips` / `rollout_fvd` are fixed-filter stand-ins (deterministic `FixedFilterFeatures`
bank), not the learned VGG/AlexNet trunk behind real LPIPS / the feature distribution behind real
FVD: real, reproducible distances, but not publication-grade LPIPS/FVD._
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
