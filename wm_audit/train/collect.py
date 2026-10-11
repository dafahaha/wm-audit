"""Data collection under a fixed (random) behavior policy."""
from __future__ import annotations

import numpy as np

from ..data.buffer import ReplayBuffer


def collect_random_episodes(env, buffer: ReplayBuffer, n_steps: int,
                            seed: int = 0) -> int:
    """Fill ``buffer`` with rollouts from a uniform-random behavior policy.

    Returns the number of transitions actually added. Fixed seed -> identical
    dataset across model configurations, which keeps the comparison fair.
    """
    rng = np.random.default_rng(seed)
    added = 0
    obs = env.reset(seed=seed)
    for _ in range(n_steps):
        action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
        next_obs, reward, done, _info = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        added += 1
        obs = next_obs
        if done:
            obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return added
