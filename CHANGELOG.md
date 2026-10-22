# Changelog

All notable changes to WM-Audit are documented here. The format is loosely based
on Keep a Changelog; the project follows semantic versioning and is pre-1.0.

## [0.0.1] - 2026-10-22

### Fixed
- **Tutorial notebook (`notebooks/10min_audit.ipynb`) ran end-to-end.**
  - Imported the real `prereg_decision_h1/h2/h3` (nonexistent `prereg_decision`
    removed); `evaluate_config` rows are now accumulated with `extend` (it
    returns a list); verdicts call the per-hypothesis decision functions.
  - Notebook execution is now a CI regression step (`nbconvert --execute`).
- **`make_leaderboard._smoke_verdicts` no longer KeyErrors** when H3 is
  `not_assessable` (degradation keys / `supports_H3` absent by design); missing
  quantities render as `—`.
- **`run_audit.py` no longer overwrites the tracked `leaderboard/results_smoke.json`
  by default.** Bare runs write to the gitignored `.tmp/` scratch path; `--out`
  writes an explicit target.
- **`envs/dmc.py`** splits the `cartpole_swingup` label into dm_control's
  separate `domain_name`/`task_name` (previously passed the whole label as
  `task_name`, which `suite.load` rejects).
- **`ssim` raises `ValueError`** on images smaller than the 7×7 window or on
  mismatched shapes, instead of returning NaN from an empty window.

### Documented
- GRU recurrent state is reset to zeros every training batch; no temporal
  dependency is learned, and pilot conclusions do not extrapolate to a genuine
  recurrent world model (README + leaderboard + `train_wm.py`).
- `pixel_nll` uses one scalar global logvar shared by all pixels (weak,
  homoscedastic approximation); not a per-pixel likelihood.
- Dead documentation-only configs (`configs/env/*.yaml`, `configs/wm/*.yaml`) are
  labeled as placeholders; only `configs/audit/default.yaml` is loaded by code.
- References are cited by verified arXiv ID; unverified workshop/venue labels
  were removed.

### Notes
- Locked MAIN run (200k steps × 5 seeds × 2 envs, Linux/CUDA) remains
  unexecuted; committed numbers are the CPU reduced pilot / preserved smoke
  placeholder only.
