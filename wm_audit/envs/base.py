"""Abstract env interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Env(ABC):
    """Minimal gym-like interface used across the audit."""

    action_dim: int
    obs_shape: Tuple[int, int, int]  # (H, W, C)

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset and return the initial observation (H, W, C) float32 in [0,1]."""

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Advance one step. Returns (obs, reward, done, info)."""

    def close(self) -> None:  # optional
        pass
