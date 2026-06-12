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

    min_up_dot = float(np.cos(np.radians(60.0)))
    result = _ransac_floor(points, threshold, iters, rng_seed, min_up_dot)
    if result is None:
        return None
    normal, d, inlier_frac = result
    if -d <= 1e-6:
        return None
    if inlier_frac < 0.25:
        return None

    return normal, d, inlier_frac


def estimate_floor_from_points(points: np.ndarray, *,
                               max_tilt_deg: float = 30.0,
                               iters: int = 200, rng_seed: int = 0
                               ) -> tuple[np.ndarray, float, float] | None:
    """Estimate a floor plane from global-frame points.

    Returns (normal, d, inlier_frac) for dot(normal, point) = d. The normal is
    oriented toward global up (-Y), and planes farther than max_tilt_deg from
    -Y are rejected.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if len(pts) < 2000:
        return None
    if not np.isfinite(max_tilt_deg) or max_tilt_deg <= 0.0:
        return None

    # 바닥 후보는 높이 하위 구간만 — 사무실 점군은 책상/벽/모니터가
    # 대부분이라 전체에서 바닥 비중이 수 %에 불과해 RANSAC이 못 찾는다
    # (office-loop 실측: 전체 inlier 3.4% → 하위 45%에서 22%). RDF는
    # Y가 아래이므로 y가 큰 쪽이 낮은 점이다.
    pts = pts[pts[:, 1] >= np.percentile(pts[:, 1], 55)]
    if len(pts) < 500:
        return None

    centroid = pts.mean(axis=0)
    spread = float(np.median(np.linalg.norm(pts - centroid, axis=1)))
    if not np.isfinite(spread) or spread <= 1e-9:
        spread = float(np.median(np.linalg.norm(pts, axis=1)))
    # 융합 점군의 바닥 표면 노이즈는 voxel·mono depth 오차가 합쳐져
    # 단일 프레임 depth보다 훨씬 크다 — 0.02×spread(≈0.009)는 바닥점을
    # 인라이어로 못 잡는다 (실측). 0.08×spread + 절대 바닥값 0.03 사용.
    threshold = max(0.08 * spread, 0.03)
    if not np.isfinite(threshold) or threshold <= 0.0:
        return None

    min_up_dot = float(np.cos(np.radians(max_tilt_deg)))
    result = _ransac_floor(pts, threshold, iters, rng_seed, min_up_dot)
    if result is None:
        return None
    normal, d, inlier_frac = result
    if inlier_frac < 0.15:
        return None
    return normal, d, inlier_frac


def floor_anchor_correction(normal_world: np.ndarray, cam_pos: np.ndarray,
                            height: float, y_floor_ref: float, *,
                            beta_rot: float = 0.5, beta_y: float = 0.5,
                            max_rot_rad: float = 0.035, max_dy: float = 0.08,
                            max_tilt_rad: float = 0.35,
                            max_y_err: float = 0.4) -> np.ndarray | None:
    """키프레임 바닥 측정으로 pose를 절대 기준에 재정렬하는 SE3 보정 (4x4).

    VO drift의 근본 대응: mono depth의 바닥 편향이 PnP 병진에 주입되어
    카메라가 수평 보행에서 계단식으로 가라앉는다 — 키프레임마다 그 프레임
    depth의 바닥(법선·높이)을 측정해 (a) 기울기를 중력(-Y)으로, (b) 바닥
    높이를 최초 기준으로 부분(β) 복원하면 drift가 키프레임 단위로 리셋된다.

    normal_world: 세계 좌표 바닥 법선(위쪽), cam_pos: 카메라 위치,
    height: 카메라-바닥 거리(депth 측정), y_floor_ref: 기준 바닥 y.
    게이트 미달(기울기 과대=오인 측정, 높이 차 과대=책상면)이면 None.
    """
    n = np.asarray(normal_world, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return None
    n = n / n_norm
    target = _WORLD_UP
    tilt = float(np.arccos(np.clip(n @ target, -1.0, 1.0)))
    if tilt > max_tilt_rad:
        return None

    y_floor_meas = float(cam_pos[1]) + float(height)
    dy_err = y_floor_meas - y_floor_ref
    if abs(dy_err) > max_y_err:
        return None

    R_c = np.eye(3)
    if tilt > 1e-6:
        axis = np.cross(n, target)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-12:
            step = min(beta_rot * tilt, max_rot_rad)
            R_c = _axis_angle(axis / axis_norm, step)
    dy = float(np.clip(beta_y * dy_err, -max_dy, max_dy))

    C = np.eye(4)
    C[:3, :3] = R_c
    # 회전은 카메라 위치를 고정점으로, 높이는 -dy만큼 복원
    C[:3, 3] = cam_pos - R_c @ cam_pos
    C[1, 3] -= dy
    return C


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


def _ransac_floor(points: np.ndarray, threshold: float, iters: int,
                  rng_seed: int, min_up_dot: float
                  ) -> tuple[np.ndarray, float, float] | None:
    if len(points) < 3 or iters <= 0:
        return None
    if not np.isfinite(threshold) or threshold <= 0.0:
        return None
    if not np.isfinite(min_up_dot):
        return None

    rng = np.random.default_rng(rng_seed)
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

        mask = np.abs(points @ normal - d) < threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < 3:
        return None

    normal, d = _refit_plane(points[best_mask])
    normal, d = _orient_up(normal, d)
    if float(normal @ _WORLD_UP) < min_up_dot:
        return None

    inliers = np.abs(points @ normal - d) < threshold
    return normal, d, float(inliers.mean())


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
