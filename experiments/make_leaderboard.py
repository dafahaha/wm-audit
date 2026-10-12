#!/usr/bin/env python
"""Render the leaderboard from results_smoke.json into markdown.

The public leaderboard (GitHub Pages / HF Space) will list, per model:
each diagnostic vs closed-loop return. For now this emits a static markdown
table from the latest smoke results, including the compression-tagged rows and
the preregistered H1/H2/H3 verdicts (smoke).
"""
from __future__ import annotations

import json
import os

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res_path = os.path.join(here, "leaderboard", "results_smoke.json")
out_path = os.path.join(here, "leaderboard", "LEADERBOARD.md")


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def main():
    if not os.path.exists(res_path):
        table = "_No results yet. Run `python -m experiments.run_audit` first._"
        rows_block = ""
        verdicts_block = ""
    else:
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        corr = data["correlations"]
        lines = ["| diagnostic | Spearman rho vs R_real (float-only) | 95% CI | Holm-adj p |",
                 "|---|---|---|---|"]
        for d, v in corr.items():
            # Mark the fixed-filter stand-ins so they are never read as real
            # learned LPIPS / FVD (see README diagnostic table footnote).
            label = d + ("†" if d in ("pixel_lpips", "rollout_fvd") else "")
            lines.append(f"| {label} | {v['rho']:.3f} | "
                         f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] | "
                         f"{v['p_adj']:.3f} |")
        table = "\n".join(lines)

        # Per-config rows, surfacing the compression tag and paired H3 evidence.
        rlines = ["| hidden | embed | seed | compression | R_real | pixel_mse | "
                  "return_ece | counterfactual_tau |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in data["rows"]:
            rlines.append(
                f"| {r['hidden']} | {r['embed']} | {r['seed']} | "
                f"{r.get('compression', 'float')} | "
                f"{_fmt(r['R_real'])} | {_fmt(r['pixel_mse'])} | "
                f"{_fmt(r['return_ece'])} | {_fmt(r['counterfactual_tau'])} |")
        rows_block = (
            "\n\n## Rows (float baseline + compression variants)\n\n"
            + "\n".join(rlines))

        vlines = ["| hypothesis | status / result |", "|---|---|"]
        for label in ("H1", "H2", "H3"):
            dec = data.get(label, {})
            if label == "H3":
                res = (f"{dec.get('status')}; n_pairs={dec.get('n_pairs', '-')}; "
                       f"supports_H3={dec.get('supports_H3', '-')}")
                if "pixel_composite_degradation" in dec:
                    res += (f"; pixel_deg={dec['pixel_composite_degradation']:.3f}, "
                            f"control_deg={dec['control_composite_degradation']:.3f}")
            else:
                key = f"supports_{label}"
                res = (f"observed={dec.get('observed_rho', float('nan')):.3f}; "
                       f"supports_{label}={dec.get(key, '-')}")
            vlines.append(f"| {label} | {res} |")
        verdicts_block = "\n\n## Preregistered verdicts (smoke)\n\n" + "\n".join(vlines)

    md = (
        "# WM-Audit Leaderboard\n\n"
        "> Which world-model diagnostics actually predict closed-loop control "
        "utility? Lower on pixel metrics = better fidelity.\n\n"
        "_Status: smoke / [IN PREPARATION]. Main results pending multi-seed, "
        "multi-env runs._\n\n"
        "> Correlations are computed on the **float baseline rows only** "
        "(n=" + str(next(iter(corr.values()))["n"]) + "): compression-tagged "
        "rows (int8) are deployment perturbations of an already-trained model, "
        "not new capacity tiers, so they belong only to the H3 paired test and "
        "never enter the H1/H2 correlation pool.\n\n"
        "## Diagnostics vs R_real\n\n"
        + table + rows_block + verdicts_block + "\n\n"
        "_† `pixel_lpips` / `rollout_fvd` are fixed-filter stand-ins (deterministic "
        "`FixedFilterFeatures` bank), not the learned VGG/AlexNet trunk behind "
        "real LPIPS / the feature distribution behind real FVD: real, "
        "reproducible distances, but not publication-grade LPIPS/FVD._\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
