"""The results artifact must be strict RFC 8259 JSON.

A NaN literal is invalid JSON: `jq`, node and every browser reject the whole
file, so one unmeasurable diagnostic would silently make the entire leaderboard
artifact unreadable. These tests pin the serialization rule that produces
`null` instead, and that the rule survives nested dicts/lists and numpy scalars.
"""
import json
import math
import os

import numpy as np
import pytest

from experiments.run_audit import _jsonable, _resolve_out_path


def test_nonfinite_floats_serialize_as_null():
    out = _jsonable({"rho": float("nan"), "ci_lo": -0.5, "ci_hi": float("inf")})
    assert out == {"rho": None, "ci_lo": -0.5, "ci_hi": None}
    # the whole point: this must not raise, because every value is JSON-legal
    assert json.loads(json.dumps(out, allow_nan=False))["rho"] is None


def test_nested_rows_and_numpy_scalars_survive():
    rows = [{"counterfactual_tau": np.float32(np.nan),
              "R_real": np.float64(22.4136),
              "seed": np.int64(3),
              "flag": np.bool_(True)}]
    out = _jsonable({"rows": rows})
    assert out["rows"][0]["counterfactual_tau"] is None
    assert out["rows"][0]["R_real"] == pytest.approx(22.4136)
    assert out["rows"][0]["seed"] == 3
    assert out["rows"][0]["flag"] is True
    json.dumps(out, allow_nan=False)   # no NaN token anywhere


def test_a_clean_payload_round_trips_bit_identically():
    payload = {"rows": [{"rho": 0.1297349946281605, "n": 15}], "H1": {"supports_H1": True}}
    assert _jsonable(payload) == payload
    assert json.loads(json.dumps(_jsonable(payload), allow_nan=False)) == payload


def test_default_out_path_is_gitignored_tmp():
    # A bare `python -m experiments.run_audit` (no --out) must default to a
    # scratch path instead of crashing on a None --out; this is the CI smoke
    # gate regression that left ci.yml red after the NaN-ranking refactor.
    p = _resolve_out_path("smoke", None)
    assert os.path.basename(os.path.dirname(p)) == ".tmp"
    assert os.path.basename(p) == "results_smoke.json"


def test_explicit_out_path_is_passed_through():
    assert _resolve_out_path("main", "leaderboard/results_main.json") == "leaderboard/results_main.json"
