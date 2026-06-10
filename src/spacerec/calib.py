"""Robust affine depth calibration: a * D_src + b ≈ D_ref on static pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthCalibration:
    a: float = 1.0
    b: float = 0.0
    inlier_frac: float = 0.0

    def apply(self, depth: np.ndarray) -> np.ndarray:
        return self.a * depth + self.b


def fit_affine_depth(src: np.ndarray, ref: np.ndarray,
                     valid: np.ndarray | None = None,
                     iters: int = 4, max_samples: int = 20000,
                     rng_seed: int = 0) -> DepthCalibration:
    """IRLS (Huber) fit of ref ≈ a*src + b over valid pixels."""
    s = src.ravel().astype(np.float64)
    r = ref.ravel().astype(np.float64)
    mask = np.isfinite(s) & np.isfinite(r) & (s > 1e-6) & (r > 1e-6)
    if valid is not None:
        mask &= valid.ravel()
    s, r = s[mask], r[mask]
    if len(s) < 100:
        return DepthCalibration()

    if len(s) > max_samples:
        idx = np.random.default_rng(rng_seed).choice(len(s), max_samples, replace=False)
        s, r = s[idx], r[idx]

    w = np.ones_like(s)
    a, b = 1.0, 0.0
    for _ in range(iters):
        sw = np.sqrt(w)
        A = np.column_stack([s * sw, sw])
        coef, *_ = np.linalg.lstsq(A, r * sw, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        resid = r - (a * s + b)
        scale = 1.4826 * np.median(np.abs(resid)) + 1e-9
        w = np.minimum(1.0, 1.345 * scale / (np.abs(resid) + 1e-12))

    resid = np.abs(r - (a * s + b))
    inlier = float((resid < 3 * (1.4826 * np.median(resid) + 1e-9)).mean())
    if a <= 0:  # depth 부호가 뒤집히는 보정은 신뢰 불가
        return DepthCalibration()
    return DepthCalibration(a=a, b=b, inlier_frac=inlier)
