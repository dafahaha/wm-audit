#!/usr/bin/env python
"""Render the leaderboard placeholder from results_smoke.json into markdown.

The public leaderboard (GitHub Pages / HF Space) will list, per model:
each diagnostic vs closed-loop return. For now this emits a static markdown
table from the latest smoke results.
"""
from __future__ import annotations

import json
import os

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res_path = os.path.join(here, "leaderboard", "results_smoke.json")
out_path = os.path.join(here, "leaderboard", "LEADERBOARD.md")


def main():
    if not os.path.exists(res_path):
        table = "_No results yet. Run `python -m experiments.run_audit` first._"
    else:
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        corr = data["correlations"]
        lines = ["| diagnostic | Spearman rho vs R_real | 95% CI | Holm-adj p |",
                 "|---|---|---|---|"]
        for d, v in corr.items():
            lines.append(f"| {d} | {v['rho']:.3f} | "
                         f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] | {v['p_adj']:.3f} |")
        table = "\n".join(lines)
    md = (
        "# WM-Audit Leaderboard\n\n"
        "> Which world-model diagnostics actually predict closed-loop control "
        "utility? Lower on pixel metrics = better fidelity.\n\n"
        "_Status: smoke / [IN PREPARATION]. Main results pending multi-seed, "
        "multi-env runs._\n\n"
        + table + "\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
