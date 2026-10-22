"""Training loops for the world model (next-step prediction) and ensembles."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..models import TinyRSSM
from ..data.buffer import ReplayBuffer


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_wm(model: TinyRSSM, buffer: ReplayBuffer, steps: int = 200,
              batch_size: int = 32, lr: float = 1e-3, seed: int = 0,
              log_every: int = 0) -> TinyRSSM:
    """Train the RSSM to predict next-frame image, reward, and continuation.

    Uses MSE on pixels, MSE on reward, BCE on continuation. Fixed seeds make the
    run reproducible. Returns the trained model.
    """
    _set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for step in range(steps):
        batch = buffer.sample(batch_size, rng)
        obs = torch.tensor(batch["obs"].reshape(batch_size, -1), dtype=torch.float32)
        nobs = torch.tensor(batch["next_obs"].reshape(batch_size, -1), dtype=torch.float32)
        act = torch.tensor(batch["action"], dtype=torch.float32)
        rew = torch.tensor(batch["reward"], dtype=torch.float32)
        not_done = 1.0 - torch.tensor(batch["done"], dtype=torch.float32)

        # NOTE: the GRU recurrent state is reset to zeros EVERY training step
        # (batches are sampled independently, never unrolled as sequences),
        # so the recurrent carry-over is never trained. This stand-in is a
        # per-transition feedforward mapping; pilot conclusions do not extend
        # to a genuine recurrent world model (documented in README/leaderboard).
        h = model.init_state(batch_size)
        obs_mean, rew_pred, cont_logit, _h = model.step(obs, act, h)

        loss_obs = torch.mean((obs_mean - nobs) ** 2)
        loss_rew = torch.mean((rew_pred - rew) ** 2)
        loss_cont = bce(cont_logit, not_done)
        loss = loss_obs + 0.5 * loss_rew + 0.1 * loss_cont

        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0):
            print(f"    train step {step}: obs_mse={loss_obs.item():.4f} "
                  f"rew_mse={loss_rew.item():.4f}")
    model.eval()
    return model


def build_ensemble(obs_shape, action_dim: int, hidden: int, embed: int,
                   ensemble_size: int, buffer: ReplayBuffer,
                   steps: int, batch_size: int, lr: float,
                   base_seed: int = 0):
    """Train ``ensemble_size`` independent RSSMs with different init/data seeds.

    Returns a list of trained models. Ensemble disagreement feeds the
    uncertainty diagnostic.
    """
    members = []
    for e in range(ensemble_size):
        _set_seed(base_seed + 1000 + e)
        m = TinyRSSM(obs_shape, action_dim, hidden=hidden, embed=embed)
        train_wm(m, buffer, steps=steps, batch_size=batch_size, lr=lr,
                 seed=base_seed + 2000 + e)
        members.append(m)
    return members
