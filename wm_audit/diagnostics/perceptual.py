"""Perceptual (pixel-space) fidelity diagnostics.

Implemented as pure numpy functions so they run on CPU in CI with no heavy
deep-learning deps. Identity input yields the ideal value (MSE = 0,
SSIM = 1). ``lpips_distance`` and ``frechet_distance`` accept a pluggable
feature extractor; the default feature extractors are deterministic and
dependency-free (documented lightweight stand-ins). When the real ``lpips``
package is installed (optional extra), pass its extractor to ``lpips_distance``.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Next-frame pixel error
# --------------------------------------------------------------------------- #
def mse(real, pred) -> float:
    """Mean squared error between two image tensors (any broadcast shape)."""
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean((real - pred) ** 2))


# --------------------------------------------------------------------------- #
# SSIM (windowed, Gaussian weights)
# --------------------------------------------------------------------------- #
def _gaussian_window(size: int = 7, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size) - size // 2
    g1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1d /= g1d.sum()
    return np.outer(g1d, g1d)


def ssim(img1, img2, data_range: float = 1.0, win_size: int = 7, sigma: float = 1.5) -> float:
    """Mean windowed SSIM. Inputs (H, W) or (H, W, C) in [0, data_range].

    Identity input -> 1.0. The window uses a 'valid' region; images smaller
    than ``win_size``×``win_size`` have no valid window and raise ValueError
    (instead of silently returning NaN from an empty window). This is the
    standard structural-similarity formula with C1/C2 constants (0.01 / 0.03 of
    the data range).
    """
    img1 = np.asarray(img1, dtype=float)
    img2 = np.asarray(img2, dtype=float)
    if img1.ndim == 2:
        img1 = img1[..., None]
        img2 = img2[..., None]
    H, W, C = img1.shape
    if H < win_size or W < win_size:
        raise ValueError(
            f"ssim needs images at least {win_size}x{win_size} for a valid "
            f"window; got {H}x{W}. Resize/pad before calling ssim."
        )
    if img2.shape[:2] != (H, W):
        raise ValueError(
            f"ssim inputs must share H,W: {img1.shape[:2]} vs {img2.shape[:2]}"
        )
    win = _gaussian_window(win_size, sigma)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    vals = []
    for c in range(C):
        a = img1[..., c]
        b = img2[..., c]
        for i in range(H - win_size + 1):
            for j in range(W - win_size + 1):
                aw = a[i:i + win_size, j:j + win_size]
                bw = b[i:i + win_size, j:j + win_size]
                mu1 = float(np.sum(aw * win))
                mu2 = float(np.sum(bw * win))
                var1 = float(np.sum(aw * aw * win)) - mu1 ** 2
                var2 = float(np.sum(bw * bw * win)) - mu2 ** 2
                cov = float(np.sum(aw * bw * win)) - mu1 * mu2
                num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
                den = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
                vals.append(num / den)
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# Gaussian negative log-likelihood (for a probabilistic world-model head)
# --------------------------------------------------------------------------- #
def gaussian_nll(mean, logvar, target) -> float:
    """Mean NLL under a diagonal Gaussian N(mean, exp(logvar))."""
    mean = np.asarray(mean, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    target = np.asarray(target, dtype=float)
    nll = 0.5 * (logvar + (target - mean) ** 2 / (np.exp(logvar) + 1e-8) + np.log(2.0 * np.pi))
    return float(np.mean(nll))


# --------------------------------------------------------------------------- #
# Lightweight LPIPS stand-in (deterministic fixed-filter features)
# --------------------------------------------------------------------------- #
class FixedFilterFeatures:
    """Deterministic, dependency-free feature extractor standing in for the
    pretrained VGG/AlexNet trunk used by real LPIPS. A small bank of fixed,
    seeded random convolutional filters responds to local texture; the output
    is L2-normalized per image. This is a *real* computable distance (not a
    mock) -- swap in the real ``lpips`` trunk via ``feature_extractor`` for
    publication-grade numbers.
    """

    def __init__(self, n_filters: int = 8, size: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.filters = rng.standard_normal((n_filters, size, size)) / np.sqrt(size * size)
        self.size = size

    def __call__(self, img) -> np.ndarray:
        # img: (..., H, W, C) or (H, W, C) or (H, W)
        img = np.asarray(img, dtype=float)
        single = img.ndim == 3 or img.ndim == 2
        if single:
            img = img[None, ...]
        if img.ndim == 3:
            img = img[..., None]
        B, H, W, C = img.shape
        pad = self.size // 2
        out = []
        for b in range(B):
            feats = []
            for f in self.filters:
                acc = np.zeros((H, W), dtype=float)
                for ch in range(C):
                    plane = img[b, :, :, ch]
                    padded = np.pad(plane, pad, mode="edge")
                    for di in range(self.size):
                        for dj in range(self.size):
                            acc += f[di, dj] * padded[di:di + H, dj:dj + W]
                feats.append(np.mean(acc ** 2))  # pool filter response energy
            v = np.asarray(feats, dtype=float)
            v = v / (np.linalg.norm(v) + 1e-8)
            out.append(v)
        return np.stack(out, axis=0)


def lpips_distance(real, pred, feature_extractor=None) -> float:
    """Mean L2 distance in a (pluggable) feature space. Identity input -> 0."""
    if feature_extractor is None:
        feature_extractor = FixedFilterFeatures()
    fr = feature_extractor(real)
    fp = feature_extractor(pred)
    return float(np.mean(np.sqrt(np.sum((fr - fp) ** 2, axis=-1) + 1e-12)))


# --------------------------------------------------------------------------- #
# Fréchet distance (rollout FVD in a feature/latent space)
# --------------------------------------------------------------------------- #
def _sym_matrix_sqrt(a: np.ndarray) -> np.ndarray:
    a = (a + a.T) / 2.0
    w, v = np.linalg.eigh(a)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.T


def frechet_distance(feats_real, feats_gen, eps: float = 1e-6) -> float:
    """Fréchet distance between two empirical Gaussians over feature vectors.

    ``feats_real`` / ``feats_gen``: (N, D) arrays. Used as rollout-FVD in the
    world model's latent feature space (the default feature extractor).
    """
    fr = np.asarray(feats_real, dtype=float)
    fg = np.asarray(feats_gen, dtype=float)
    if fr.ndim == 1:
        fr = fr[:, None]
    if fg.ndim == 1:
        fg = fg[:, None]
    mu1 = fr.mean(axis=0)
    mu2 = fg.mean(axis=0)
    if fr.shape[0] < 2:
        c1 = np.zeros((fr.shape[1], fr.shape[1]))
    else:
        c1 = np.cov(fr, rowvar=False)
    if fg.shape[0] < 2:
        c2 = np.zeros((fg.shape[1], fg.shape[1]))
    else:
        c2 = np.cov(fg, rowvar=False)
    if c1.ndim == 0:
        c1 = np.array([[float(c1)]])
        c2 = np.array([[float(c2)]])
    diff = mu1 - mu2
    covmean = _sym_matrix_sqrt(c1 @ c2)
    return float(diff @ diff + np.trace(c1 + c2 - 2.0 * covmean) + eps)
