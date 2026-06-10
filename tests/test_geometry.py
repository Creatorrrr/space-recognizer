import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spacerec.backend import robust_sim3
from spacerec.calib import fit_affine_depth
from spacerec.geometry import (sim3_apply, sim3_compose, sim3_interp,
                               sim3_inverse, sim3_on_pose, umeyama_sim3)


def _random_sim3(seed=0):
    rng = np.random.default_rng(seed)
    s = float(rng.uniform(0.5, 2.0))
    R = Rotation.random(random_state=int(rng.integers(1e6))).as_matrix()
    t = rng.uniform(-2, 2, 3)
    return s, R, t


def test_umeyama_recovers_known_sim3():
    rng = np.random.default_rng(1)
    src = rng.uniform(-3, 3, (40, 3))
    T = _random_sim3(2)
    dst = sim3_apply(T, src)
    est = umeyama_sim3(src, dst)
    assert est[0] == pytest.approx(T[0], rel=1e-9)
    assert np.allclose(est[1], T[1], atol=1e-9)
    assert np.allclose(est[2], T[2], atol=1e-9)


def test_sim3_compose_inverse():
    A, B = _random_sim3(3), _random_sim3(4)
    p = np.random.default_rng(5).uniform(-1, 1, (10, 3))
    assert np.allclose(sim3_apply(sim3_compose(A, B), p),
                       sim3_apply(A, sim3_apply(B, p)), atol=1e-9)
    assert np.allclose(sim3_apply(sim3_compose(A, sim3_inverse(A)), p), p, atol=1e-9)


def test_sim3_on_pose_maps_camera_center():
    T = _random_sim3(6)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.random(random_state=7).as_matrix()
    pose[:3, 3] = [1.0, -0.5, 2.0]
    mapped = sim3_on_pose(T, pose)
    assert np.allclose(mapped[:3, 3], sim3_apply(T, pose[:3, 3][None])[0], atol=1e-9)
    # 방향은 회전만 적용 (정규직교 유지)
    assert np.allclose(mapped[:3, :3] @ mapped[:3, :3].T, np.eye(3), atol=1e-9)


def test_sim3_interp_endpoints():
    A, B = _random_sim3(8), _random_sim3(9)
    for alpha, ref in ((0.0, A), (1.0, B)):
        out = sim3_interp(A, B, alpha)
        assert out[0] == pytest.approx(ref[0], rel=1e-9)
        assert np.allclose(out[1], ref[1], atol=1e-9)
        assert np.allclose(out[2], ref[2], atol=1e-9)


def test_robust_sim3_with_poses():
    T = _random_sim3(10)
    rng = np.random.default_rng(11)
    poses = []
    for _ in range(5):
        P = np.eye(4)
        P[:3, :3] = Rotation.random(random_state=int(rng.integers(1e6))).as_matrix()
        P[:3, 3] = rng.uniform(-2, 2, 3)
        poses.append(P)
    mapped = [sim3_on_pose(T, P) for P in poses]
    est = robust_sim3(poses, mapped)
    assert est[0] == pytest.approx(T[0], rel=1e-6)
    assert np.allclose(est[1], T[1], atol=1e-6)
    assert np.allclose(est[2], T[2], atol=1e-6)


def test_fit_affine_depth_with_outliers():
    rng = np.random.default_rng(12)
    src = rng.uniform(0.5, 5.0, (100, 200)).astype(np.float32)
    ref = 1.7 * src + 0.3
    # 15% 픽셀을 동적 물체처럼 오염
    n = src.size * 15 // 100
    idx = rng.choice(src.size, n, replace=False)
    ref.ravel()[idx] = rng.uniform(0.1, 8.0, n)
    cal = fit_affine_depth(src, ref)
    assert cal.a == pytest.approx(1.7, abs=0.02)
    assert cal.b == pytest.approx(0.3, abs=0.05)
    assert cal.inlier_frac > 0.7
