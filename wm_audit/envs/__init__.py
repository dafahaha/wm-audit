"""Unified environment interface.

Every env exposes ``reset(seed) -> obs`` and ``step(action) -> (obs, reward,
done, info)``. Observations are float32 images in [0, 1] of shape (H, W, C).
The built-in :class:`PixelNavEnv` is fully deterministic and needs no heavy
dependency; dm_control / Atari adapters live behind optional extras.
"""
from .base import Env
from .pixel_nav import PixelNavEnv

__all__ = ["Env", "PixelNavEnv", "make_env"]


def make_env(name: str, **kwargs) -> Env:
    """Factory. Built-in envs are always available; heavy envs raise a clear
    ImportError pointing at the optional extra."""
    if name in ("pixel_nav", "PixelNav-v0"):
        return PixelNavEnv(**kwargs)
    if name in ("dm_control.cartpole_swingup", "cartpole"):
        from .dmc import DMControlEnv  # optional extra
        return DMControlEnv(**kwargs)
    raise ValueError(f"unknown env: {name}")
