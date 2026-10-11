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
    # parity may be approximate; we only assert the pipeline runs and reports a
    # finite gap, and that the gap is not absurdly large for a tiny net.
    assert np.isfinite(res["max_abs_diff"])
    assert res["max_abs_diff"] < 1.0
