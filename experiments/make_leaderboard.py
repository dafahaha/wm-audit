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


def _main_verdicts(data):
    h1, h2, h3 = data["H1"], data["H2"], data["H3"]
    h1_res = ("**SUPPORTED** (pixel fidelity weakly tracks control)"
              if h1["supports_H1"] else
              "**NOT supported** (pixel fidelity tracks control more than preregistered)")
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
        "### Preregistered verdicts (reduced pilot; NOT the locked MAIN run)",
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


def main():
    main_data = _load(MAIN_RES)
    smoke_data = _load(SMOKE_RES)
    n_main = next(iter(main_data["correlations"].values()))["n"]
    n_smoke = next(iter(smoke_data["correlations"].values()))["n"]
    # P3-2: derive the strongest-correlation numbers from the JSON rather than
    # hardcoding rho=0.70 / p=0.393, so the prose cannot drift from the table.
    _unc = main_data["correlations"]["uncertainty_rho"]
    unc_rho = _unc["rho"]
    unc_p = _unc["p_adj"]

    md = f"""# WM-Audit Leaderboard

> Which world-model diagnostics actually predict closed-loop control utility? Lower on pixel metrics = better fidelity.

## Reduced-pilot results — CPU (2026-10-13)

> **Status: preregistered CPU REDUCED PILOT only. The locked 200k-step × 5-seed ×
> 2-env MAIN run has NOT been executed (requires Linux/CUDA); nothing below is a
> MAIN result.**
> Env: built-in `pixel_nav` only. The 2nd prereg env `dm_control.cartpole_swingup`
> **cannot install on Windows**: `pip install dm_control` pulls `labmaze`, which has
> no cp3.13 wheel and requires `bazel` to build from source (`error: command 'bazel'
> failed`). Device: CPU (venv torch 2.14.1+cpu, `cuda_available=False`; a cu124 install
> was attempted but pip keeps the satisfied CPU wheel).
>
> What WAS kept exactly as preregistered: the 3 capacity tiers, `bootstrap_resamples=2000`,
> the INT8 lever, and the whole stats pipeline (float-only H1/H2, fixed-float-reference-ladder
> H3, Spearman t-approx p + Holm). What was reduced and labeled: **3 seeds (prereg goal 5),
> `train_steps=12000` (prereg 200000), `collect_steps=8000` (prereg 100000)** — because the
> locked 200k×5-seed×3-cap×2-env design is ~21 CPU-hours (measured ~236 member-steps/s) and
> dm_control is unavailable. No numbers were tuned or back-filled.
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
is `uncertainty_rho` (ρ={unc_rho:.2f}, Holm-adj p={unc_p:.3f} — not significant at n={n_main}). The H1 null result
(pixel fidelity does *not* strongly track control) and the H2/H3 falsifications are publishable as
bounded/null findings per the preregistration's honesty clause.

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
