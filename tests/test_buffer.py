"""test_buffer: replay buffer shape/dtype and perturbation bounds."""
import numpy as np

from wm_audit.data import ReplayBuffer, inject_observation_perturbation


def test_buffer_shapes_and_types():
    buf = ReplayBuffer(capacity=100, obs_shape=(8, 8, 1), action_dim=2)
    obs = np.random.default_rng(0).random((8, 8, 1)).astype(np.float32)
    for i in range(10):
        buf.add(obs, np.array([i, i], dtype=np.float32), float(i),
                obs.copy(), done=(i % 3 == 0))
    assert buf.size == 10
    batch = buf.sample(4, np.random.default_rng(0))
    assert batch["obs"].shape == (4, 8, 8, 1)
    assert batch["action"].shape == (4, 2)
    assert batch["reward"].shape == (4,)
    assert batch["done"].dtype == bool
    assert batch["obs"].dtype == np.float32


def test_buffer_caps_at_capacity():
    buf = ReplayBuffer(capacity=5, obs_shape=(4, 4, 1), action_dim=1)
    obs = np.zeros((4, 4, 1), dtype=np.float32)
    for i in range(10):
        buf.add(obs, np.array([0.0], dtype=np.float32), 0.0, obs, done=False)
    assert buf.size == 5


def test_perturbation_stays_in_bounds():
    rng = np.random.default_rng(0)
    img = rng.random((16, 16, 1)).astype(np.float32)
    for kind in ("noise", "mask", "shift"):
        out = inject_observation_perturbation(img, kind=kind, strength=0.2, rng=rng)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.shape == img.shape
