"""Closed-loop control: CEM-MPC planned inside the learned world model and
executed on the *real* environment. This is where we measure R_real."""
from __future__ import annotations

import numpy as np
import torch


def cem_plan(wm, obs_flat: np.ndarray, horizon: int = 5, candidates: int = 64,
             elites: int = 10, iters: int = 2, action_dim: int = 2,
             rng: np.random.Generator = None) -> np.ndarray:
    """Plan the first action via cross-entropy method on imagined reward."""
    if rng is None:
        rng = np.random.default_rng(0)
    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    std = np.ones((horizon, action_dim), dtype=np.float32)
    obs_t = torch.tensor(obs_flat.reshape(1, -1), dtype=torch.float32)
    best_action = mean[0].copy()
    best_score = -np.inf
    for _ in range(iters):
        seqs = mean[None, :, :] + std[None, :, :] * rng.standard_normal(
            (candidates, horizon, action_dim))
        seqs = np.clip(seqs, -1.0, 1.0).astype(np.float32)
        with torch.no_grad():
            obs_b = obs_t.expand(candidates, -1)
            scores, _ = wm.imagine(obs_b, torch.tensor(seqs))
        scores = scores.numpy()
        k = min(elites, candidates)
        elite_idx = np.argsort(scores)[-k:]
        elite = seqs[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        if scores[elite_idx[-1]] > best_score:
            best_score = scores[elite_idx[-1]]
            best_action = mean[0].copy()
    return np.clip(best_action, -1.0, 1.0)


def close_loop_episode(env, wm, horizon: int = 5, candidates: int = 64,
                       elites: int = 10, iters: int = 2, seed: int = 0,
                       max_steps: int = None) -> dict:
    """Execute CEM-MPC on the real env. Returns total return and trajectory."""
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    total = 0.0
    frames = [obs]
    steps = 0
    done = False
    while not done:
        obs_flat = obs.reshape(-1)
        action = cem_plan(wm, obs_flat, horizon=horizon, candidates=candidates,
                          elites=elites, iters=iters, action_dim=env.action_dim,
                          rng=rng)
        obs, reward, done, _info = env.step(action)
        total += reward
        frames.append(obs)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {"return": float(total), "steps": steps, "frames": np.stack(frames)}
