"""Dreamer-lite RSSM: a minimal, CPU-trainable recurrent state-space model.

This is *not* a full DreamerV3 -- it is a deliberately small stand-in so the
audit pipeline (train -> diagnostics -> close-loop -> correlate) completes in
CI in seconds. It encodes an observation image, advances a GRU recurrent state
conditioned on the action, and decodes a predicted next-frame image, reward,
and continuation. Real DreamerV3 / TD-MPC2 integrations plug in behind the same
``step`` / ``imagine`` interface.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TinyRSSM(nn.Module):
    def __init__(self, obs_shape, action_dim: int, hidden: int = 32, embed: int = 32):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = action_dim
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(self.obs_dim, embed), nn.ReLU())
        self.gru = nn.GRUCell(embed + action_dim, hidden)
        self.decoder = nn.Linear(hidden, self.obs_dim)
        self.reward_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, 1))
        self.cont_head = nn.Linear(hidden, 1)
        self.logvar = nn.Parameter(torch.zeros(1))

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.hidden)

    def step(self, obs_flat: torch.Tensor, action: torch.Tensor, h: torch.Tensor):
        """One RSSM step. Returns (next_obs_mean, reward_pred, cont_logit, next_h)."""
        e = self.encoder(obs_flat)
        x = torch.cat([e, action], dim=-1)
        nh = self.gru(x, h)
        next_obs_mean = torch.sigmoid(self.decoder(nh))
        reward_pred = self.reward_head(nh).squeeze(-1)
        cont_logit = self.cont_head(nh).squeeze(-1)
        return next_obs_mean, reward_pred, cont_logit, nh

    @torch.no_grad()
    def imagine(self, obs_flat: torch.Tensor, action_seq: torch.Tensor):
        """Roll out the model. action_seq: (B, H, action_dim). Returns summed
        predicted reward (B,) and latent trajectory (B, H, hidden)."""
        h = self.init_state(obs_flat.shape[0]).to(obs_flat.device)
        e = self.encoder(obs_flat)
        zero_act = torch.zeros(obs_flat.shape[0], self.action_dim, device=obs_flat.device)
        h = self.gru(torch.cat([e, zero_act], dim=-1), h)
        total_reward = torch.zeros(obs_flat.shape[0])
        latents = []
        cur = obs_flat
        for t in range(action_seq.shape[1]):
            mean, rew, _cont, h = self.step(cur, action_seq[:, t, :], h)
            total_reward = total_reward + rew
            latents.append(h)
            cur = mean
        return total_reward, torch.stack(latents, dim=1)
