"""High-level orchestration: train one WM ensemble, compute the full diagnostic
battery, measure closed-loop return, and return one row (plus one row per
compression lever) for the fidelity-vs-utility correlation table."""
from __future__ import annotations

import numpy as np
import torch

from .envs import make_env
from .data import ReplayBuffer
from .train import collect_random_episodes, build_ensemble
from .policy import close_loop_episode
from .deploy import quantize_dynamic_int8
from .diagnostics import (
    mse, ssim, gaussian_nll, lpips_distance, frechet_distance,
    FixedFilterFeatures, regression_ece, counterfactual_kendall_tau,
    ensemble_dispersion, uncertainty_error_alignment, uncertainty_ece,
)


#: Compression levers the pipeline knows how to build post-hoc from a trained
#: float ensemble. ``evaluate_config`` trains the float ensemble once, then
#: produces one compressed variant per lever listed in ``cfg["compression_levers"]``
#: and re-runs the *exact same* rollout/diagnostic battery on it. A lever value
#: is stamped onto the compressed row as ``compression`` so H3 can pair it with
#: its float baseline on the exact (hidden, embed, seed) triple.
_COMPRESSION_BUILDERS = {
    "int8": quantize_dynamic_int8,
}


def _np_batch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs.reshape(obs.shape[0], -1), dtype=torch.float32)


def _perceptual_on_buffer(members, obs, action, next_obs) -> dict:
    """Next-frame pixel diagnostics, averaged over ensemble members."""
    B = obs.shape[0]
    fe = FixedFilterFeatures()
    msess, ssims, nlls, lpipss, fvds = [], [], [], [], []
    for m in members:
        h = m.init_state(B)
        with torch.no_grad():
            mean, _rew, _cont, _h = m.step(_np_batch(obs),
                                           torch.tensor(action, dtype=torch.float32), h)
        pred = mean.numpy().reshape(obs.shape)
        msess.append(mse(next_obs, pred))
        ssims.append(float(np.mean([ssim(next_obs[i], pred[i])
                                    for i in range(next_obs.shape[0])])))
        nlls.append(gaussian_nll(pred, float(m.logvar.detach().item()), next_obs))
        lpipss.append(lpips_distance(next_obs, pred, fe))
        fr = fe(next_obs)
        fg = fe(pred)
        fvds.append(frechet_distance(fr, fg))
    return {
        "pixel_mse": float(np.mean(msess)),
        "pixel_ssim": float(np.mean(ssims)),
        "pixel_nll": float(np.mean(nlls)),
        "pixel_lpips": float(np.mean(lpipss)),
        "rollout_fvd": float(np.mean(fvds)),
    }


def _uncertainty_on_buffer(members, obs, action, next_obs) -> dict:
    B = obs.shape[0]
    preds = []
    for m in members:
        h = m.init_state(B)
        with torch.no_grad():
            mean, _r, _c, _h = m.step(_np_batch(obs),
                                      torch.tensor(action, dtype=torch.float32), h)
        preds.append(mean.numpy().reshape(B, -1))
    disp = ensemble_dispersion(preds)  # (B,)
    target = next_obs.reshape(B, -1)
    mean_pred = np.mean(preds, axis=0)
    err = np.mean((mean_pred - target) ** 2, axis=1)
    rho = uncertainty_error_alignment(disp, err)
    uece = uncertainty_ece(disp, err)
    return {"uncertainty_rho": rho, "uncertainty_ece": float(uece)}


def _imagined_return_series(m, env, n_segments: int, horizon: int,
                            seed: int):
    """Imagined return (summed predicted reward over a rollout) vs real return,
    on matched segments driven by one fixed action sequence. Returns
    ``(imagined_rets, real_rets)`` arrays for ONE ensemble member."""
    imag_rets, real_rets = [], []
    for k in range(n_segments):
        seg_seed = seed * 1000 + k
        rng = np.random.default_rng(seg_seed)
        obs = env.reset(seed=seg_seed)
        actions = rng.uniform(-1, 1, size=(horizon, env.action_dim)).astype(np.float32)
        # real segment
        real_ret = 0.0
        o = obs
        for t in range(horizon):
            o, r, done, _ = env.step(actions[t])
            real_ret += r
            if done:
                break
        # imagined segment (replan-free: feed the same action sequence)
        h = m.init_state(1)
        cur = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
        imag_ret = 0.0
        for t in range(horizon):
            with torch.no_grad():
                mean, rew, _c, h = m.step(cur,
                                          torch.tensor(actions[t:t+1], dtype=torch.float32), h)
            imag_ret += float(rew.item())
            cur = mean
        imag_rets.append(imag_ret)
        real_rets.append(real_ret)
    return np.asarray(imag_rets), np.asarray(real_rets)


def _return_calibration(env, members, n_segments: int, horizon: int,
                        seed: int) -> dict:
    """Calibration ECE, computed per ensemble member on the SAME matched
    segments and then averaged across members -- the same ensemble-averaged
    convention as the pixel/uncertainty battery (previously this silently used
    only ``members[0]``, an asymmetry we remove)."""
    eces = []
    for m in members:
        imag_rets, real_rets = _imagined_return_series(m, env, n_segments,
                                                      horizon, seed)
        eces.append(regression_ece(imag_rets, real_rets))
    return {"return_ece": float(np.mean(eces))}


def _counterfactual_series(m, env, n_pairs: int, seed: int):
    """For each state, step two candidate actions in the real env (oracle
    reward) and through ONE member (imagined reward). Returns
    ``(oracle, imagined)`` pooled arrays."""
    oracle, imagined = [], []
    for k in range(n_pairs):
        rng = np.random.default_rng(seed + k)
        obs = env.reset(seed=seed + k)
        a1 = rng.uniform(-1, 1, size=env.action_dim).astype(np.float32)
        a2 = rng.uniform(-1, 1, size=env.action_dim).astype(np.float32)
        # oracle (real env) one-step reward
        _, r1, _, _ = env.step(a1)
        env.reset(seed=seed + k)
        _, r2, _, _ = env.step(a2)
        # imagined reward
        flat = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
        h = m.init_state(1)
        with torch.no_grad():
            _, p1, _, _ = m.step(flat, torch.tensor(a1[None, :], dtype=torch.float32), h)
            _, p2, _, _ = m.step(flat, torch.tensor(a2[None, :], dtype=torch.float32), h)
        oracle.extend([r1, r2])
        imagined.extend([float(p1.item()), float(p2.item())])
    return np.asarray(oracle), np.asarray(imagined)


def _counterfactual_tau(env, members, n_pairs: int, seed: int) -> float:
    """Global pooled Kendall-τb per ensemble member over the SAME counterfactual
    states, then averaged across members -- matching the ensemble-averaged
    pixel/uncertainty convention (previously only ``members[0]`` was used)."""
    taus = []
    for m in members:
        oracle, imagined = _counterfactual_series(m, env, n_pairs, seed)
        taus.append(counterfactual_kendall_tau(oracle, imagined))
    return float(np.mean(taus))


def _diagnostics_row(env, members, data, cfg: dict, seed: int) -> dict:
    """Run the identical rollout/diagnostic battery on ANY member list (float or
    compressed) and return the diagnostic dict."""
    out = {}
    out.update(_perceptual_on_buffer(members, data["obs"], data["action"],
                                     data["next_obs"]))
    out.update(_uncertainty_on_buffer(members, data["obs"], data["action"],
                                      data["next_obs"]))
    out.update(_return_calibration(env, members,
                                   n_segments=cfg["calib_segments"],
                                   horizon=cfg["calib_horizon"], seed=seed))
    out["counterfactual_tau"] = _counterfactual_tau(env, members,
                                                     n_pairs=cfg["cf_pairs"],
                                                     seed=seed)
    # closed-loop utility on the real env, planned with the lead ensemble member
    ep = close_loop_episode(env, members[0],
                            horizon=cfg["cem_horizon"],
                            candidates=cfg["cem_candidates"],
                            elites=cfg["cem_elites"], iters=cfg["cem_iters"],
                            seed=seed)
    out["R_real"] = ep["return"]
    return out


def evaluate_config(env_name: str, hidden: int, embed: int, seed: int,
                    cfg: dict) -> list:
    """Train one (capacity, seed) config and return a list of diagnostics rows.

    The first row is the float baseline (no ``compression`` key). For every lever
    in ``cfg.get("compression_levers", [])`` (e.g. ``"int8"``) we build a
    compressed variant of the SAME trained ensemble and re-run the identical
    rollout battery, stamping ``compression=<lever>`` while keeping the same
    ``(hidden, embed, seed)`` triple so H3 can pair it with the float row.
    """
    env = make_env(env_name)
    obs_shape = env.obs_shape
    action_dim = env.action_dim

    # fixed dataset shared across configs
    buffer = ReplayBuffer(cfg["buffer_capacity"], obs_shape, action_dim)
    collect_random_episodes(env, buffer, n_steps=cfg["collect_steps"],
                            seed=cfg["data_seed"])
    members = build_ensemble(obs_shape, action_dim, hidden, embed,
                             cfg["ensemble_size"], buffer,
                             steps=cfg["train_steps"],
                             batch_size=cfg["batch_size"], lr=cfg["lr"],
                             base_seed=seed)

    data = buffer.all()
    # embed is recorded alongside hidden/seed: the H3 compression-parity match
    # requires the exact (hidden, embed, seed) triple, because the capacity
    # tiers put two different embeds on the same hidden size.
    base = {"hidden": hidden, "embed": embed, "seed": seed}

    rows = []
    rows.append({**base, **_diagnostics_row(env, members, data, cfg, seed)})

    for lever in cfg.get("compression_levers", []) or []:
        builder = _COMPRESSION_BUILDERS.get(lever)
        if builder is None:
            raise ValueError(f"Unknown compression lever: {lever!r}; "
                             f"known: {sorted(_COMPRESSION_BUILDERS)}")
        compressed = [builder(m) for m in members]
        rows.append({**base, "compression": lever,
                     **_diagnostics_row(env, compressed, data, cfg, seed)})

    env.close()
    return rows
