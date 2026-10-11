"""Optional dm_control adapter (cartpole_swingup, 64x64 pixels).

This is an *extension path*, not part of the light CI. ``dm_control`` and its
MuJoCo dependency are heavy on Windows; install them as the ``dm_control``
optional extra (see pyproject) before using this. The smoke / CI path uses the
built-in :class:`PixelNavEnv` instead.
"""
from __future__ import annotations

import numpy as np

from .base import Env


class DMControlEnv(Env):
    action_dim = 1  # cartpole_swingup torque (scaled)

    def __init__(self, task: str = "cartpole_swingup", size: int = 64, action_repeat: int = 2):
        try:
            from dm_control import suite  # noqa: F401
        except Exception as e:  # pragma: no cover - exercised only when extra missing
            raise ImportError(
                "dm_control is not installed. Install the optional extra "
                "`pip install -e .[dm_control]` (requires MuJoCo) to use this "
                "env. CI uses the built-in PixelNavEnv instead."
            ) from e
        from dm_control import suite
        self._env = suite.load(domain_name="cartpole", task_name=task,
                               task_kwargs={"random": 0})
        self.size = size
        self.action_repeat = action_repeat
        self.obs_shape = (size, size, 3)

    def reset(self, seed: int | None = None) -> np.ndarray:  # pragma: no cover
        self._env.task._random = np.random.RandomState(seed)
        time_step = self._env.reset()
        return self._render_obs(time_step)

    def step(self, action):  # pragma: no cover
        reward = 0.0
        done = False
        for _ in range(self.action_repeat):
            time_step = self._env.step(np.asarray(action, dtype=np.float32))
            reward += float(time_step.reward or 0.0)
            done = bool(time_step.last())
            if done:
                break
        return self._render_obs(time_step), reward, done, {}

    def _render_obs(self, time_step):  # pragma: no cover
        img = self._env.physics.render(height=self.size, width=self.size,
                                       camera_id=0)
        return (img.astype(np.float32) / 255.0)
