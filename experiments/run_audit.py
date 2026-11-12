#!/usr/bin/env python
"""WM-Audit entrypoint: train -> diagnostics -> closed-loop -> correlate.

Runs the preregistered audit on a set of (capacity, seed) configurations and
prints the diagnostic-vs-closed-loop-return table with bootstrap CIs and Holm
adjusted p-values, then emits the preregistered H1/H2/H3 verdicts. The default
`smoke` mode finishes in minutes on CPU.

Usage:
    python -m experiments.run_audit                 # smoke -> .tmp/results_smoke.json (gitignored)
    python -m experiments.run_audit --mode smoke
    python -m experiments.run_audit --out leaderboard/results_smoke.json
        # --out writes the committed/tracked artifact on purpose; without it
        # the default goes to the gitignored .tmp/ scratch path and never
        # overwrites tracked results_*.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

# make the package importable when run as a plain script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wm_audit.audit import evaluate_config              # noqa: E402
from wm_audit.diagnostics import (                     # noqa: E402
    audit_correlations,
    prereg_decision_h1, prereg_decision_h2, prereg_decision_h3,
    pixel_composite_scores, control_composite_scores,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _eval_task(task):
    """Worker entrypoint for one (seed, capacity) config.

    Each worker pinning torch to a single thread keeps N workers on N cores from
    oversubscribing (the tiny RSSMs are launch-bound, not compute-bound).
    """
    env_name, hidden, embed, seed, cfg = task
    import torch
    torch.set_num_threads(1)
    return evaluate_config(env_name, hidden=hidden, embed=embed, seed=seed, cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "audit", "default.yaml"))
    ap.add_argument("--mode", default="smoke", choices=["smoke", "main"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes for the (seed,capacity) grid; "
                         "1 keeps the original serial order")
    ap.add_argument("--resume", action="store_true",
                    help="reuse rows already written to <out>.partial.jsonl "
                         "and skip the (capacity, seed) configs they cover")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated subset of the config's seeds, e.g. "
                         "'2'. Lets independent hosts run one shard of the grid "
                         "each; the shards are concatenated in seed order "
                         "before the H1/H2/H3 aggregation.")
    args = ap.parse_args()

    cfg_all = load_config(args.config)
    cfg = cfg_all[args.mode]
    env_name = cfg_all["env"]
    diagnostics = cfg_all["diagnostics"]

    seeds = list(cfg["seeds"])
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(",") if s.strip()}
        unknown = keep - set(seeds)
        if unknown:
            raise SystemExit(f"--seeds {sorted(unknown)} not in config seeds {seeds}")
        seeds = [s for s in seeds if s in keep]

    print("=" * 72)
    print(f"WM-Audit | env={env_name} mode={args.mode}")
    print("=" * 72)

    # Grid = seeds x capacities. Order is preserved exactly as the serial
    # nested loop always did, so --jobs only changes wall-clock -- with one
    # caveat: each worker pins torch to one thread while the serial path does
    # not, so reduction order can move the last few bits (measured worst case
    # ~1.5e-7 relative on the smoke grid). Row order and R_real are unaffected,
    # and those are what the rank correlations -- hence every reported rho --
    # depend on. tests/test_parallel_consistency.py guards both halves.
    tasks = [(env_name, hidden, embed, seed, cfg)
             for seed in seeds for (hidden, embed) in cfg["capacities"]]

    rows = []
    partial_path = os.path.splitext(args.out)[0] + ".partial.jsonl"
    if args.resume and os.path.exists(partial_path):
        with open(partial_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        done = {(r["hidden"], r["embed"], r["seed"]) for r in rows}
        tasks = [t for t in tasks if (t[1], t[2], t[3]) not in done]
        print(f"[resume] {len(rows)} rows on disk, {len(tasks)} configs left",
              flush=True)
    else:
        open(partial_path, "w").close()

    def _append(new_rows):
        """Persist each finished config immediately.

        A config is ~hours of CPU on this host; without this, a single OOM or
        kill loses the whole run (it happened once at 16k rows x 3 workers).
        """
        with open(partial_path, "a") as fh:
            for r in new_rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"[parallel] {len(tasks)} configs on {args.jobs} workers")
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for (_, hidden, embed, seed, _), new_rows in zip(
                    tasks, ex.map(_eval_task, tasks)):
                print(f"[done] hidden={hidden} embed={embed} seed={seed}", flush=True)
                for row in new_rows:
                    tag = row.get("compression", "float")
                    print(f"    [{tag:>5}] R_real={row['R_real']:.3f}  "
                          f"pixel_mse={row['pixel_mse']:.4f}  "
                          f"return_ece={row['return_ece']:.4f}  "
                          f"tau={row['counterfactual_tau']:.3f}")
                rows.extend(new_rows)
                _append(new_rows)
    else:
        for (env_name_i, hidden, embed, seed, cfg_i) in tasks:
            print(f"[config] hidden={hidden} embed={embed} seed={seed} ...",
                  flush=True)
            # evaluate_config returns [float baseline, <compressed variant rows>...]
            new_rows = evaluate_config(env_name_i, hidden=hidden, embed=embed,
                                       seed=seed, cfg=cfg_i)
            for row in new_rows:
                tag = row.get("compression", "float")
                print(f"    [{tag:>5}] R_real={row['R_real']:.3f}  "
                      f"pixel_mse={row['pixel_mse']:.4f}  "
                      f"return_ece={row['return_ece']:.4f}  "
                      f"tau={row['counterfactual_tau']:.3f}")
            rows.extend(new_rows)
            _append(new_rows)

    print("\nRows collected:")
    for r in rows:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.items()})

    # H1/H2 ask "which diagnostics predict closed-loop return ACROSS MODEL
    # CAPACITY". The compression-tagged rows (int8 etc.) are the SAME trained
    # model under a deployment perturbation -- a near-duplicate of a float row,
    # not a new capacity tier -- so they belong ONLY to the H3 paired test and
    # must NOT pollute the H1/H2 composite/correlation pool. Compute composites
    # and Spearman correlations on the float baseline rows alone. Compression
    # rows are still displayed in the Rows table above; they just never enter
    # the H1/H2 correlation pool.
    float_rows = [r for r in rows if not r.get("compression")]

    # Preregistered composite scores (direction-unified rank composites) on the
    # float capacity ladder ONLY.
    pixel_scores = pixel_composite_scores(float_rows)
    control_scores = control_composite_scores(float_rows)
    for r, ps, cs in zip(float_rows, pixel_scores, control_scores):
        r["pixel_composite"] = float(ps)
        r["control_composite"] = float(cs)

    corr_diagnostics = list(diagnostics) + ["pixel_composite", "control_composite"]

    print("\n" + "=" * 72)
    print(f"Diagnostic vs closed-loop return R_real  (Spearman rho + 95% bootstrap CI)")
    print(f"POOL: float baseline rows only (n={len(float_rows)}); compression rows "
          f"excluded (H3-only)")
    print("=" * 72)
    res = audit_correlations(float_rows, corr_diagnostics, return_key="R_real",
                             n_boot=cfg.get("bootstrap_resamples", 200),
                             ci=0.95, seed=0)
    print(f"{'diagnostic':<22}{'rho':>8}{'ci_lo':>9}{'ci_hi':>9}{'p_raw':>9}{'p_adj':>9}")
    for d, v in res.items():
        print(f"{d:<22}{v['rho']:>8.3f}{v['ci_lo']:>9.3f}{v['ci_hi']:>9.3f}"
              f"{v['p_raw']:>9.3f}{v['p_adj']:>9.3f}")

    # Preregistered verdicts.
    h1 = prereg_decision_h1(res["pixel_composite"]["rho"], threshold=0.4,
                            n=len(float_rows))
    h2 = prereg_decision_h2(res["control_composite"]["rho"], threshold=0.6)
    h3 = prereg_decision_h3(rows)

    print("\n" + "=" * 72)
    print("Preregistered verdicts")
    print("=" * 72)
    for label, dec in (("H1", h1), ("H2", h2), ("H3", h3)):
        print(f"[{label}]")
        print(json.dumps(dec, indent=2))

    out = {"rows": rows, "correlations": res, "H1": h1, "H2": h2, "H3": h3}
    # Default output is a gitignored scratch path (.tmp/) so a bare
    # `python -m experiments.run_audit` NEVER overwrites the tracked,
    # committed leaderboard/results_smoke.json. Pass --out to write anywhere
    # else (e.g. to regenerate a committed artifact on purpose).
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(ROOT, ".tmp", f"results_{args.mode}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote results -> {out_path}")


if __name__ == "__main__":
    main()
