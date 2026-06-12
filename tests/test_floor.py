import numpy as np

from spacerec.floor import (estimate_floor, estimate_floor_from_points,
                            gravity_align_rotation)
from spacerec.vo import default_intrinsics


def _floor_depth(width=160, height=120, pitch_deg=15.0, camera_height=1.4):
    K = default_intrinsics(width, height)
    pitch = np.radians(pitch_deg)
    normal = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    d = -camera_height

    vs, us = np.mgrid[0:height, 0:width].astype(np.float64)
    rays = np.stack([(us - K[0, 2]) / K[0, 0],
                     (vs - K[1, 2]) / K[1, 1],
                     np.ones_like(us)], axis=-1)
    denom = rays @ normal
    depth = np.zeros((height, width), dtype=np.float32)
    visible = denom < -1e-6
    depth[visible] = (d / denom[visible]).astype(np.float32)
    return depth, K, normal


def _angle_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0)))


def _floor_points(normal, d=-1.0, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    z = rng.uniform(0.5, 4.0, n)
    y = (d - normal[0] * x - normal[2] * z) / normal[1]
    return np.stack([x, y, z], axis=1)


def test_estimate_floor_recovers_tilted_floor_normal():
    depth, K, expected_normal = _floor_depth()
    rng = np.random.default_rng(4)
    depth = depth.copy()
    valid = depth > 0
    depth[valid] += rng.normal(0.0, 0.002, valid.sum()).astype(np.float32)

    result = estimate_floor(depth, K, rng_seed=2)

    assert result is not None
    normal, _, inlier_frac = result
    assert _angle_deg(normal, expected_normal) < 1.0
    assert normal @ np.array([0.0, -1.0, 0.0]) > 0.0
    assert inlier_frac >= 0.5


def test_estimate_floor_from_points_recovers_floor_with_outliers():
    tilt = np.radians(12.0)
    normal = np.array([0.0, -np.cos(tilt), -np.sin(tilt)])
    points = _floor_points(normal, n=5000, seed=1)
    rng = np.random.default_rng(2)
    outliers = rng.uniform([-3.0, -1.0, 0.0], [3.0, 3.0, 5.0], (1000, 3))
    points = np.concatenate([points, outliers])

    result = estimate_floor_from_points(points, rng_seed=3)

    assert result is not None
    got_normal, _, inlier_frac = result
    assert _angle_deg(got_normal, normal) < 1.0
    assert inlier_frac > 0.7


def test_estimate_floor_from_points_rejects_tilt_past_gate():
    tilt = np.radians(45.0)
    normal = np.array([0.0, -np.cos(tilt), -np.sin(tilt)])
    points = _floor_points(normal, n=4000, seed=4)

    assert estimate_floor_from_points(points, rng_seed=5) is None


def test_estimate_floor_rejects_invalid_or_nonplanar_depth():
    K = default_intrinsics(160, 120)
    assert estimate_floor(np.zeros((120, 160), np.float32), K) is None

    rng = np.random.default_rng(5)
    noise_depth = rng.uniform(0.8, 3.0, (120, 160)).astype(np.float32)

    assert estimate_floor(noise_depth, K, rng_seed=6) is None


def test_gravity_align_rotation_maps_normal_to_world_up():
    normal = np.array([0.1, -0.85, -0.35], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    R = gravity_align_rotation(normal)

    np.testing.assert_allclose(R @ normal, [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) > 0.999999


def test_gravity_align_rotation_is_identity_for_already_aligned_normal():
    R = gravity_align_rotation(np.array([0.0, -1.0, 0.0]))

    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_floor_anchor_correction_restores_tilt_and_height():
    import pytest

    from spacerec.floor import floor_anchor_correction

    cam = np.array([1.0, 0.3, -2.0])   # 카메라가 0.3 가라앉음 (기준 바닥 y=1.4)
    tilt = np.radians(4.0)
    n_world = np.array([0.0, -np.cos(tilt), -np.sin(tilt)])
    # 측정 바닥 y = 0.3 + 1.4 = 1.7 (가라앉은 만큼 바닥도 내려간 계단식 drift)
    C = floor_anchor_correction(n_world, cam, height=1.4, y_floor_ref=1.4,
                                beta_rot=0.5, beta_y=0.5, max_dy=0.08)
    assert C is not None
    moved = C[:3, :3] @ cam + C[:3, 3]
    # 회전의 고정점은 카메라 위치 — x/z는 불변, y는 -dy만 이동
    np.testing.assert_allclose(moved[[0, 2]], cam[[0, 2]], atol=1e-9)
    # 높이 오차 0.3 → β=0.5면 0.15지만 max_dy=0.08 클램프 → 0.08 들림
    assert moved[1] == pytest.approx(cam[1] - 0.08)
    # 기울기는 부분 복원
    n_after = C[:3, :3] @ n_world
    tilt_after = np.arccos(np.clip(n_after @ [0, -1, 0.0], -1, 1))
    assert tilt_after < tilt


def test_floor_anchor_correction_gates():
    from spacerec.floor import floor_anchor_correction

    cam = np.zeros(3)
    flat = np.array([0.0, -1.0, 0.0])
    # 기준에서 0.4 초과 차이 = 책상면 등 → 기각
    assert floor_anchor_correction(flat, cam, height=1.0,
                                   y_floor_ref=1.6) is None
    # 기울기 과대(>20도) = 오인 측정 → 기각
    big = np.radians(30.0)
    n_big = np.array([0.0, -np.cos(big), -np.sin(big)])
    assert floor_anchor_correction(n_big, cam, height=1.0,
                                   y_floor_ref=1.0) is None


def test_vo_apply_keyframe_correction_transforms_state_consistently():
    import cv2

    from spacerec.config import VoCfg
    from spacerec.vo import VisualOdometry

    vo = VisualOdometry(default_intrinsics(128, 96), VoCfg())
    small = np.random.default_rng(0).integers(0, 255, (12, 16)).astype(np.uint8)
    gray = cv2.resize(small, (128, 96), interpolation=cv2.INTER_NEAREST)
    vo.process(gray, np.full((96, 128), 2.0, np.float32), 0.0, None)
    assert vo._pts3d is not None

    n = np.array([0.0, -0.99, -0.14])
    C = np.eye(4)
    C[:3, :3] = gravity_align_rotation(n / np.linalg.norm(n))
    C[:3, 3] = [0.1, -0.05, 0.0]
    pts_before = vo._pts3d.copy()
    T_before = vo.T_wc.copy()

    vo.apply_keyframe_correction(C)

    np.testing.assert_allclose(vo.T_wc, C @ T_before, atol=1e-12)
    np.testing.assert_allclose(vo.keyframe.T_wc, C @ T_before, atol=1e-12)
    np.testing.assert_allclose(vo._pts3d,
                               pts_before @ C[:3, :3].T + C[:3, 3], atol=1e-12)
