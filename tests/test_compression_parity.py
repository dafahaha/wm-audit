"""test_compression_parity: INT8 forward matches float within tolerance (smoke)."""
import numpy as np

from wm_audit.models import TinyRSSM
from wm_audit.deploy import quantize_dynamic_int8, forward_parity


def test_int8_parity_within_tolerance():
    import torch
    torch.manual_seed(0)
    model = TinyRSSM(obs_shape=(8, 8, 1), action_dim=2, hidden=16, embed=16)
    model.eval()
    q = quantize_dynamic_int8(model)
    res = forward_parity(model, q, obs_shape=(8, 8, 1), action_dim=2,
                         n=8, atol=0.25, seed=0)
    # P3-4: the guard must actually require parity, not just a non-absurd gap.
    # The reported within_tol flag is the authoritative gate (max_diff <= atol),
    # and for this tiny net the real gap is ~5e-4; pin a tight upper bound so a
    # large quantization drift fails the test instead of slipping under < 1.0.
    assert np.isfinite(res["max_abs_diff"])
    assert res["within_tol"], f"INT8 forward exceeded atol: {res}"
    assert res["max_abs_diff"] < 0.05, f"INT8 drift too large: {res}"


def test_int8_parity_reports_within_tol_flag():
    """P3-4: forward_parity must set within_tol from the actual gap; a wide
    atol yields True and a narrower-than-gap atol yields False."""
    import torch
    torch.manual_seed(0)
    model = TinyRSSM(obs_shape=(8, 8, 1), action_dim=2, hidden=16, embed=16)
    model.eval()
    q = quantize_dynamic_int8(model)
    loose = forward_parity(model, q, obs_shape=(8, 8, 1), action_dim=2,
                           n=8, atol=1.0, seed=0)
    tight = forward_parity(model, q, obs_shape=(8, 8, 1), action_dim=2,
                           n=8, atol=1e-6, seed=0)
    assert loose["within_tol"] is True
    assert tight["within_tol"] is False
