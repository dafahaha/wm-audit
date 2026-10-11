"""test_env: env determinism and reset correctness."""
import numpy as np

from wm_audit.envs import make_env


def test_determinism_same_seed():
    a = make_env("pixel_nav")
    b = make_env("pixel_nav")
    oa = a.reset(seed=42)
    ob = b.reset(seed=42)
    assert np.allclose(oa, ob), "reset must be deterministic given same seed"
    rng = np.random.default_rng(0)
    for _ in range(10):
        act = rng.uniform(-1, 1, size=2).astype(np.float32)
        oa, ra, da, _ = a.step(act)
        ob, rb, db, _ = b.step(act.copy())
        assert np.allclose(oa, ob)
        assert ra == rb
        assert da == db


def test_reset_resets_state_and_shape():
    env = make_env("pixel_nav")
    o = env.reset(seed=1)
    assert o.shape == env.obs_shape
    assert o.dtype == np.float32
    assert o.min() >= 0.0 and o.max() <= 1.0
    # stepping then resetting must return to a consistent initial obs
    for _ in range(5):
        env.step(np.array([0.5, -0.5], dtype=np.float32))
    o2 = env.reset(seed=1)
    assert np.allclose(o, o2)


def test_step_signature():
    env = make_env("pixel_nav")
    env.reset(seed=0)
    obs, r, done, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert obs.shape == env.obs_shape
    assert isinstance(r, float)
    assert isinstance(done, bool)
