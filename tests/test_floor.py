import numpy as np

from spacerec.floor import estimate_floor, gravity_align_rotation
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
