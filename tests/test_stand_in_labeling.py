"""Guard: pixel_lpips / rollout_fvd must never be silently over-read as real
learned perceptual distances.

The two metrics are fixed-filter stand-ins (``FixedFilterFeatures``) for the
pretrained VGG/AlexNet trunk behind real LPIPS and the feature distribution
behind real FVD. README's diagnostic table and the committed leaderboard must
say so explicitly; these string assertions go red if the labeling is dropped.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_readme_marks_lpips_fvd_as_fixed_filter_stand_in():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert "pixel_lpips" in readme and "rollout_fvd" in readme
    assert "stand-in" in readme.lower()
    # names the real learned trunk we are standing in for
    assert "VGG/AlexNet" in readme
    assert "not" in readme.lower() and "publication-grade" in readme.lower()


def test_leaderboard_marks_lpips_fvd_as_fixed_filter_stand_in():
    with open(os.path.join(ROOT, "leaderboard", "LEADERBOARD.md"),
              encoding="utf-8") as f:
        leaderboard = f.read()
    assert "pixel_lpips†" in leaderboard
    assert "rollout_fvd†" in leaderboard
    assert "stand-in" in leaderboard.lower()
