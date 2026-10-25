#!/usr/bin/env python
"""Render WM-Audit's method-overview figure (Apple-style, deterministic).

Output: docs/wm_audit_pipeline.png (reproducible; rerun to regenerate).
Requires: matplotlib (included in the [dev] extras).
"""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Apple palette
BG = "#fbfbfd"
INK = "#1d1d1f"
GRAY = "#6e6e73"
BLUE = "#0071e3"
BLUE_LT = "#e8f1fd"
GRAY_BLOCK = "#f0f0f2"
GRAY_EDGE = "#d2d2d7"
OK = "#34c759"

FIG_W, FIG_H = 15.0, 7.2
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.2)
ax.axis("off")


def block(x, y, w, h, fc, ec, title, lines, title_fs=12.5, body_fs=10.5,
          title_color=INK, rounded=0.18):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rounded}",
                         fc=fc, ec=ec, lw=1.4, mutation_aspect=1.0)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="top",
            fontsize=title_fs, fontweight="semibold", color=title_color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - i * 0.34, ln, ha="center", va="top",
                fontsize=body_fs, color=GRAY)


def arrow(x1, y1, x2, y2, color=BLUE, lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


# --- header --------------------------------------------------------------- #
ax.text(7.5, 6.78, "Which world models actually control? We measured.",
        ha="center", fontsize=17, fontweight="semibold", color=INK)
ax.text(7.5, 6.28, "Do the diagnostics we report predict closed-loop control?  "
                   "— preregistered before seeing data",
        ha="center", fontsize=11.5, color=GRAY)

# --- column A: world model ------------------------------------------------- #
ax.text(1.35, 5.72, "1 · World model", ha="center", fontsize=12.5,
        fontweight="semibold", color=INK)
block(0.45, 3.55, 1.8, 1.95, GRAY_BLOCK, GRAY_EDGE, "Model",
      ["RSSM", "(Dreamer-lite)", "trained ensemble", "CPU"], title_fs=11.5)

# --- column B: diagnostics ------------------------------------------------- #
ax.text(4.35, 5.72, "2 · Diagnostic battery", ha="center", fontsize=12.5,
        fontweight="semibold", color=INK)
block(2.9, 4.55, 2.9, 0.95, BLUE_LT, BLUE, "Perceptual fidelity",
      ["pixel_mse · ssim · nll", "lpips† · rollout_fvd†"], body_fs=9.8)
block(2.9, 3.35, 2.9, 0.95, BLUE_LT, BLUE, "Control + uncertainty",
      ["return_ece · counterfactual_tau", "uncertainty_rho · uncertainty_ece"], body_fs=9.8)

# --- column C: closed-loop ------------------------------------------------- #
ax.text(8.0, 5.72, "3 · Real closed-loop control", ha="center", fontsize=12.5,
        fontweight="semibold", color=INK)
block(7.1, 3.55, 1.8, 1.95, GRAY_BLOCK, GRAY_EDGE, "Ground truth",
      ["CEM-MPC planner", "real env", "closed-loop return"], title_fs=11.5)

# --- bottom: correlation & verdict ----------------------------------------- #
block(0.6, 0.15, 13.8, 1.95, "#ffffff", BLUE, "Correlate · calibrate · verdict",
      ["Spearman ρ and Kendall-τb across seed/capacity · percentile-bootstrap 95% CIs · Holm-corrected p",
       "H1 pixel fidelity weakly predicts return (|ρ|<0.4)   ·   H2 control diagnostics strongly predict (ρ>0.6)   ·   "
       "H3 compression degrades calibration faster than fidelity",
       "Reduced-pilot (2026-10-13, pixel_nav, n=9):  H1 threshold met (|ρ|=0.32) — weak/tentative, not significant   ·   "
       "H2 not supported (0.26)",
       "H3 not supported (0.35 ≤ 0.43)   —   locked MAIN run pending (Linux/CUDA)"],
      title_fs=12.5, body_fs=10.5)

# --- arrows ---------------------------------------------------------------- #
arrow(2.3, 4.55, 2.86, 4.62)           # A -> B perceptual
arrow(2.3, 4.15, 2.86, 4.05)           # A -> B control
arrow(5.85, 4.95, 7.06, 4.75)          # B perceptual -> C
arrow(5.85, 4.15, 7.06, 4.15)          # B control -> C
arrow(4.35, 4.48, 4.35, 3.0)           # B -> bottom
arrow(4.35, 2.85, 4.35, 2.15)          # B -> bottom (correlate)
arrow(8.0, 3.5, 8.0, 2.85)             # C -> bottom
arrow(8.0, 2.8, 8.0, 2.15)             # C -> bottom

# legend note
ax.text(7.5, -0.02, "† fixed-filter stand-ins for learned LPIPS / FVD — reproducible, not publication-grade; "
                   "swap in the real trunk for main-result numbers",
        ha="center", fontsize=8.6, color=GRAY, style="italic")

out = Path(__file__).resolve().parent / "wm_audit_pipeline.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"wrote {out}")
