"""Floor-plane based gravity initialization for RDF camera coordinates."""

from __future__ import annotations

import numpy as np


_WORLD_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def estimate_floor(depth: np.ndarray, K: np.ndarray,
                   stride: int = 4, iters: int = 200,
                   rng_seed: int = 0) -> tuple[np.ndarray, float, float] | None:
    """Estimate the floor plane from a z-depth map with RANSAC.

    Samples the lower 60% of the image, backprojects valid depth pixels, fits
    planes from 3-point hypotheses, and returns (normal, d, inlier_frac) for
    dot(normal, point) = d. The normal is oriented toward RDF up (-Y).
    """
    points, z = _sample_points(depth, K, stride)
    if points is None or z is None or len(points) < 500:
        return None

    threshold = 0.02 * float(np.median(z))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return None

    rng = np.random.default_rng(rng_seed)
    min_up_dot = float(np.cos(np.radians(60.0)))
    best_mask: np.ndarray | None = None
    best_count = 0

    for _ in range(iters):
        sample_idx = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[sample_idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = float(normal @ p0)
        normal, d = _orient_up(normal, d)
        if float(normal @ _WORLD_UP) < min_up_dot:
            continue
        if -d <= 1e-6:
            continue

        mask = np.abs(points @ normal - d) < threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None:
        return None

    normal, d = _refit_plane(points[best_mask])
    normal, d = _orient_up(normal, d)
    if float(normal @ _WORLD_UP) < min_up_dot or -d <= 1e-6:
        return None

    inliers = np.abs(points @ normal - d) < threshold
    inlier_frac = float(inliers.mean())
    if inlier_frac < 0.25:
        return None

    return normal, d, inlier_frac


def gravity_align_rotation(normal_cam: np.ndarray) -> np.ndarray:
    """Return R such that R @ normal_cam == [0, -1, 0]."""
    normal = np.asarray(normal_cam, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("normal_cam must be non-zero")
    normal = normal / norm

    target = _WORLD_UP
    dot = float(np.clip(normal @ target, -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-12:
        axis = np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        axis = axis / np.linalg.norm(axis)
        return _axis_angle(axis, np.pi)

    axis_cross = np.cross(normal, target)
    skew = _skew(axis_cross)
    s2 = float(axis_cross @ axis_cross)
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / s2)


def _sample_points(depth: np.ndarray, K: np.ndarray,
                   stride: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    depth = np.asarray(depth)
    K = np.asarray(K, dtype=np.float64)
    if depth.ndim != 2 or K.shape != (3, 3) or stride <= 0:
        return None, None

    height, width = depth.shape
    row0 = int(height * 0.4)
    rows = np.arange(row0, height, stride)
    cols = np.arange(0, width, stride)
    if len(rows) == 0 or len(cols) == 0:
        return None, None

    vs, us = np.meshgrid(rows, cols, indexing="ij")
    z = depth[vs, us].astype(np.float64, copy=False)
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return None, None

    z_valid = z[valid]
    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z_valid
    y = (v - K[1, 2]) / K[1, 1] * z_valid
    return np.stack([x, y, z_valid], axis=1), z_valid


def _refit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return normal, float(normal @ centroid)


def _orient_up(normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    if float(normal @ _WORLD_UP) < 0.0:
        return -normal, -d
    return normal, d


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    skew = _skew(axis)
    return (np.eye(3, dtype=np.float64)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=np.float64)
