"""Guard: the smoke leaderboard H3 row must render for BOTH branches.

`prereg_decision_h3` returns `assessable` only when compression rows matched
float baselines; the `not_assessable` branch omits
`pixel_composite_degradation` / `control_composite_degradation` /
`supports_H3` by design. `_smoke_verdicts` must not index those missing keys
(KeyError) -- it renders an em dash instead.
"""
import json
import os

from experiments.make_leaderboard import _smoke_verdicts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _assessable():
    return {
        "H1": {"observed_rho": -0.580, "supports_H1": False},
        "H2": {"observed_rho": -0.086, "supports_H2": False},
        "H3": {
            "status": "assessable",
            "n_pairs": 6,
            "supports_H3": False,
            "pixel_composite_degradation": 0.200,
            "control_composite_degradation": 0.111,
        },
    }


def _not_assessable():
    # No compression rows: verdict carries status/reason only, no degradation
    # keys and no supports_H3 (mirrors prereg_decision_h3's not_assessable).
    return {
        "H1": {"observed_rho": 0.100, "supports_H1": True},
        "H2": {"observed_rho": 0.900, "supports_H2": True},
        "H3": {
            "status": "not_assessable",
            "reason": "No compression configurations present in this run.",
        },
    }


def test_smoke_verdicts_assessable_shows_degradation():
    out = _smoke_verdicts(_assessable())
    assert "assessable" in out
    assert "pixel_deg=0.200" in out
    assert "control_deg=0.111" in out
    assert "supports_H3=False" in out


def test_smoke_verdicts_not_assessable_does_not_keyerror():
    # Must NOT raise KeyError on the absent degradation / supports_H3 keys.
    out = _smoke_verdicts(_not_assessable())
    assert "not_assessable" in out
    # Missing quantities render as an em dash, not a traced KeyError / blank.
    assert "pixel_deg=—" in out
    assert "control_deg=—" in out
    assert "supports_H3=—" in out


def test_committed_smoke_json_still_renders_assessable_block():
    # Reproducibility guard: the committed smoke placeholder stays assessable
    # and its rendered H3 row keeps the locked degradation numbers.
    path = os.path.join(ROOT, "leaderboard", "results_smoke.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = _smoke_verdicts(data)
    assert data["H3"]["status"] == "assessable"
    assert "pixel_deg=0.200" in out
    assert "control_deg=0.111" in out
