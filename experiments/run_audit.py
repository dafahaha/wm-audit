#!/usr/bin/env python
"""WM-Audit entrypoint: train -> diagnostics -> closed-loop -> correlate.

Runs the preregistered audit on a set of (capacity, seed) configurations and
prints the diagnostic-vs-closed-loop-return table with bootstrap CIs and Holm
adjusted p-values, then emits the preregistered H1/H2/H3 verdicts. The default
`smoke` mode finishes in minutes on CPU.

Usage:
    python -m experiments.run_audit                 # smoke
    python -m experiments.run_audit --mode smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

# make the package importable when run as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wm_audit.audit import evaluate_config              # noqa: E402
from wm_audit.diagnostics import (                     # noqa: E402
    audit_correlations,
    prereg_decision_h1, prereg_decision_h2, prereg_decision_h3,
    pixel_composite_scores, control_composite_scores,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "audit", "default.yaml"))
    ap.add_argument("--mode", default="smoke", choices=["smoke", "main"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_all = load_config(args.config)
    cfg = cfg_all[args.mode]
    env_name = cfg_all["env"]
    diagnostics = cfg_all["diagnostics"]

    print("=" * 72)
    print(f"WM-Audit | env={env_name} mode={args.mode}")
    print("=" * 72)

    rows = []
    for seed in cfg["seeds"]:
        for (hidden, embed) in cfg["capacities"]:
            print(f"[config] hidden={hidden} embed={embed} seed={seed} ...")
            row = evaluate_config(env_name, hidden=hidden, embed=embed,
                                 seed=seed, cfg=cfg)
            rows.append(row)
            print(f"    R_real={row['R_real']:.3f}  pixel_mse={row['pixel_mse']:.4f}  "
                  f"return_ece={row['return_ece']:.4f}  tau={row['counterfactual_tau']:.3f}")

    print("\nRows collected:")
    for r in rows:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.items()})

    # Preregistered composite scores (direction-unified rank composites).
    pixel_scores = pixel_composite_scores(rows)
    control_scores = control_composite_scores(rows)
    for r, ps, cs in zip(rows, pixel_scores, control_scores):
        r["pixel_composite"] = float(ps)
        r["control_composite"] = float(cs)

    corr_diagnostics = list(diagnostics) + ["pixel_composite", "control_composite"]

    print("\n" + "=" * 72)
    print("Diagnostic vs closed-loop return R_real  (Spearman rho + 95% bootstrap CI)")
    print("=" * 72)
    res = audit_correlations(rows, corr_diagnostics, return_key="R_real",
                             n_boot=cfg.get("bootstrap_resamples", 200),
                             ci=0.95, seed=0)
    print(f"{'diagnostic':<22}{'rho':>8}{'ci_lo':>9}{'ci_hi':>9}{'p_raw':>9}{'p_adj':>9}")
    for d, v in res.items():
        print(f"{d:<22}{v['rho']:>8.3f}{v['ci_lo']:>9.3f}{v['ci_hi']:>9.3f}"
              f"{v['p_raw']:>9.3f}{v['p_adj']:>9.3f}")

    # Preregistered verdicts.
    h1 = prereg_decision_h1(res["pixel_composite"]["rho"], threshold=0.4)
    h2 = prereg_decision_h2(res["control_composite"]["rho"], threshold=0.6)
    h3 = prereg_decision_h3(rows)

    print("\n" + "=" * 72)
    print("Preregistered verdicts")
    print("=" * 72)
    for label, dec in (("H1", h1), ("H2", h2), ("H3", h3)):
        print(f"[{label}]")
        print(json.dumps(dec, indent=2))

    out = {"rows": rows, "correlations": res, "H1": h1, "H2": h2, "H3": h3}
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "leaderboard", "results_smoke.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote results -> {out_path}")


if __name__ == "__main__":
    main()
