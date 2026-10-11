"""Built-in deterministic pixel-navigation environment.

A 2D point (the agent) must move to a fixed goal under a bounded velocity. The
state transition is a deterministic linear dynamical system; the observation is
a rendered (N, N, 1) image of the agent blob plus the goal blob.

This env exists so the entire audit chain (train world model -> compute pixel /
control / uncertainty diagnostics -> close-loop control -> correlate diagnostics
with return) runs end-to-end on CPU in seconds inside CI, without pulling in
``dm_control``. It is a *real* environment driving *real* diagnostic and
statistic code -- not a mock of the diagnostics.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np

from .base import Env


def _to_pixel(v: float, n: int) -> int:
    # map [-1, 1] -> [0, n-1]
    return int(np.clip(round((v + 1.0) / 2.0 * (n - 1)), 0, n - 1))


class PixelNavEnv(Env):
    action_dim = 2

    def __init__(self, size: int = 16, max_steps: int = 50, render_goal: bool = True):
        self.size = size
        self.max_steps = max_steps
        self.render_goal = render_goal
        self.obs_shape = (size, size, 1)
        self._rng = np.random.default_rng(0)
        self._t = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.x = float(self._rng.uniform(-0.8, 0.8))
        self.y = float(self._rng.uniform(-0.8, 0.8))
        self.gx = float(self._rng.uniform(-0.7, 0.7))
        self.gy = float(self._rng.uniform(-0.7, 0.7))
        self.vx = 0.0
        self.vy = 0.0
        self._t = 0
        return self._render()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        ax, ay = float(action[0]), float(action[1])
        # deterministic damped dynamics
        self.vx = float(np.clip(0.9 * self.vx + 0.15 * ax, -1.0, 1.0))
        self.vy = float(np.clip(0.9 * self.vy + 0.15 * ay, -1.0, 1.0))
        self.x = float(np.clip(self.x + 0.08 * self.vx, -1.0, 1.0))
        self.y = float(np.clip(self.y + 0.08 * self.vy, -1.0, 1.0))
        dist = float(np.hypot(self.x - self.gx, self.y - self.gy))
        reward = -dist ** 2
        done = False
        if dist < 0.15:
            reward += 1.0
            done = True
        self._t += 1
        if self._t >= self.max_steps:
            done = True
        return self._render(), reward, done, {"dist": dist}

    def _render(self) -> np.ndarray:
        img = np.zeros((self.size, self.size, 1), dtype=np.float32)
        if self.render_goal:
            self._draw(img, self.gx, self.gy, value=0.35)
        self._draw(img, self.x, self.y, value=1.0)
        return img

    def _draw(self, img: np.ndarray, vx: float, vy: float, value: float) -> None:
        c = _to_pixel(vx, self.size)
        r = _to_pixel(vy, self.size)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    img[rr, cc, 0] = value
