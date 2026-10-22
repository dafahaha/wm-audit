"""test_perceptual_metrics: MSE/SSIM/NLL ideal values on identity + known pairs."""
import numpy as np
import pytest

from wm_audit.diagnostics.perceptual import mse, ssim, gaussian_nll


def test_mse_identity_is_zero():
    img = np.random.default_rng(0).random((16, 16, 1))
    assert mse(img, img) == 0.0


def test_mse_known_value():
    a = np.zeros((2, 2))
    b = np.ones((2, 2)) * 2.0
    assert mse(a, b) == 4.0  # mean((0-2)^2) = 4


def test_ssim_identity_is_one():
    img = np.random.default_rng(1).random((20, 20, 1))
    assert abs(ssim(img, img) - 1.0) < 1e-6


def test_ssim_constant_offset_is_low():
    base = np.random.default_rng(2).random((24, 24, 1))
    # fully different constant image -> SSIM should be well below identity
    flat = np.full_like(base, 0.5)
    assert ssim(base, flat) < 0.5


def test_gaussian_nll_known_mode():
    # target == mean, logvar=0 -> NLL = 0.5*(0 + 0 + log(2pi))
    mean = np.zeros(3)
    logvar = np.zeros(3)
    target = np.zeros(3)
    expected = 0.5 * np.log(2 * np.pi)
    assert abs(gaussian_nll(mean, logvar, target) - expected) < 1e-5


def test_ssim_too_small_image_raises_instead_of_nan():
    # A 4x4 image has no valid 7x7 window: must raise ValueError rather than
    # return NaN from an empty window loop.
    small = np.random.default_rng(3).random((4, 4, 1))
    with pytest.raises(ValueError, match="at least 7x7"):
        ssim(small, small)


def test_ssim_mismatched_shape_raises():
    a = np.zeros((16, 16, 1))
    b = np.zeros((16, 12, 1))
    with pytest.raises(ValueError, match="share H,W"):
        ssim(a, b)
