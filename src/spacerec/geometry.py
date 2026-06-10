"""Sim(3) utilities: Umeyama alignment, application, composition, interpolation.

A Sim3 is represented as a tuple (s, R, t): p' = s * R @ p + t.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

Sim3 = tuple[float, np.ndarray, np.ndarray]

SIM3_IDENTITY: Sim3 = (1.0, np.eye(3), np.zeros(3))


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares similarity transform mapping src (N,3) onto dst (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float((D * np.diag(S)).sum() / var_s) if with_scale and var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def sim3_apply(T: Sim3, pts: np.ndarray) -> np.ndarray:
    s, R, t = T
    return s * (pts @ R.T) + t


def sim3_on_pose(T: Sim3, T_wc: np.ndarray) -> np.ndarray:
    """Map a camera-to-world SE3 pose into the Sim3's target frame.

    Orientation composes with R only; the camera center is mapped through the
    full similarity (scale affects position, not orientation).
    """
    s, R, t = T
    out = np.eye(4)
    out[:3, :3] = R @ T_wc[:3, :3]
    out[:3, 3] = s * R @ T_wc[:3, 3] + t
    return out


def sim3_compose(A: Sim3, B: Sim3) -> Sim3:
    """A ∘ B: apply B first, then A."""
    sa, Ra, ta = A
    sb, Rb, tb = B
    return sa * sb, Ra @ Rb, sa * Ra @ tb + ta


def sim3_inverse(T: Sim3) -> Sim3:
    s, R, t = T
    return 1.0 / s, R.T, -R.T @ t / s


def sim3_interp(A: Sim3, B: Sim3, alpha: float) -> Sim3:
    """Geodesic-ish interpolation from A (alpha=0) to B (alpha=1)."""
    sa, Ra, ta = A
    sb, Rb, tb = B
    s = float(np.exp((1 - alpha) * np.log(max(sa, 1e-12))
                     + alpha * np.log(max(sb, 1e-12))))
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))
    R = slerp(alpha).as_matrix()
    t = (1 - alpha) * ta + alpha * tb
    return s, R, t
