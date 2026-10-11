"""Compression / deployment axis (RQ3): INT8 quantization of a trained WM.

Thin wrapper around ``torch.quantization`` dynamic INT8 plus a parity check:
the compressed model's forward must match the float model within a tolerance.
Channel pruning / distillation / ONNX export are extension points.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import torch


def quantize_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    qmodel = copy.deepcopy(model)
    qmodel.eval()
    # torch.ao.quantization.quantize_dynamic is deprecated upstream (removed in
    # ~2.10, migration target: torchao). We keep it here because it is dependency
    # free and still fully functional on the supported CPU torch line; rather
    # than rebuild the whole quantization stack we suppress the single
    # DeprecationWarning and pin a note. Re-point this to torchao when the
    # installed torch drops this API.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # dynamic-int8 construction also emits a UserWarning that
        # torch.quint8/qint8 tensor ctors are deprecated (pytorch#184982); same
        # upstream-deprecation situation, same decision: keep the working path.
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*torch\\.(quint8|qint8|qint32).*")
        qmodel = torch.quantization.quantize_dynamic(
            qmodel, {torch.nn.Linear}, dtype=torch.qint8)
    return qmodel


def forward_parity(float_model: torch.nn.Module, q_model: torch.nn.Module,
                   obs_shape, action_dim: int, n: int = 8,
                   atol: float = 1e-1, seed: int = 0) -> dict:
    """Compare float vs INT8 forward on random (obs, action). Returns max abs
    output difference and whether it is within ``atol``."""
    rng = np.random.default_rng(seed)
    obs = torch.tensor(rng.random((n,) + tuple(obs_shape)), dtype=torch.float32)
    act = torch.tensor(rng.uniform(-1, 1, size=(n, action_dim)), dtype=torch.float32)
    h_f = float_model.init_state(n)
    h_q = q_model.init_state(n)
    flat = obs.reshape(n, -1)
    with torch.no_grad():
        out_f = float_model.step(flat, act, h_f)[0]
        out_q = q_model.step(flat, act, h_q)[0]
    max_diff = float(torch.max(torch.abs(out_f - out_q)))
    return {"max_abs_diff": max_diff, "within_tol": bool(max_diff <= atol), "atol": atol}
