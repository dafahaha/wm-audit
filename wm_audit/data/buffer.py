"""Fixed replay buffer and intervention-perturbation injection.

The buffer holds transitions collected under a *fixed* behavior policy so that
all diagnostics across (architecture, capacity, seed) are evaluated on the same
validation data -- otherwise differences in diagnostics would just reflect
differences in the data.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity,) + tuple(obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity,) + tuple(obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=bool)
        self.size = 0
        self.ptr = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done": self.dones[idx],
        }

    def all(self):
        sl = slice(0, self.size)
        return {
            "obs": self.obs[sl],
            "action": self.actions[sl],
            "reward": self.rewards[sl],
            "next_obs": self.next_obs[sl],
            "done": self.dones[sl],
        }


def inject_observation_perturbation(obs: np.ndarray, kind: str = "noise",
                                   strength: float = 0.1,
                                   rng: np.random.Generator = None) -> np.ndarray:
    """Apply an intervention perturbation to an image observation, staying in
    bounds [0, 1]. Used to probe diagnostic robustness under distribution shift.

    kind:
      * ``noise``  -- additive Gaussian noise, clipped to [0,1]
      * ``mask``   -- zero-out a centered block of side ~ strength
      * ``shift``  -- roll the image by a few pixels
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.array(obs, dtype=np.float32)
    if kind == "noise":
        out = out + rng.normal(0.0, strength, size=out.shape).astype(np.float32)
    elif kind == "mask":
        h, w = out.shape[0], out.shape[1]
        side = max(1, int(round(strength * min(h, w))))
        ch, cw = h // 2, w // 2
        out[max(0, ch - side // 2):ch + side // 2,
            max(0, cw - side // 2):cw + side // 2] = 0.0
    elif kind == "shift":
        d = max(1, int(round(strength * 5)))
        out = np.roll(out, shift=d, axis=1)
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)
