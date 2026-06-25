import cv2
import numpy as np
import pytest

from spacerec.config import VoCfg
from spacerec.vo import VisualOdometry, backproject, default_intrinsics


def test_default_intrinsics_fov():
    K = default_intrinsics(1280, 720, fov_deg=60.0)
    # 수평 FOV 60° -> fx = 640 / tan(30°)
    assert K[0, 0] == pytest.approx(640 / np.tan(np.radians(30)), rel=1e-6)
    assert K[0, 2] == 640 and K[1, 2] == 360


def test_backproject_roundtrip():
    K = default_intrinsics(640, 480)
    rng = np.random.default_rng(0)
    pts_cam = np.column_stack([rng.uniform(-1, 1, 50),
                               rng.uniform(-0.7, 0.7, 50),
                               rng.uniform(1.0, 5.0, 50)])
    uv = (K @ (pts_cam / pts_cam[:, 2:]).T).T[:, :2]
    inside = ((uv[:, 0] > 1) & (uv[:, 0] < 638) & (uv[:, 1] > 1) & (uv[:, 1] < 478))
    pts_cam, uv = pts_cam[inside], uv[inside]

    depth = np.zeros((480, 640), np.float32)
    depth[uv[:, 1].astype(int), uv[:, 0].astype(int)] = pts_cam[:, 2]
    recovered = backproject(uv, depth, K)
    # 픽셀 정수화 오차 범위 내에서 원점 복원
    assert np.allclose(recovered[:, 2], pts_cam[:, 2], atol=1e-6)
    assert np.allclose(recovered, pts_cam, atol=2.5e-2)


def _noise_image(shape, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, shape, dtype=np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


def test_reprojection_median_prefers_correct_candidate():
    W, H = 320, 240
    K = default_intrinsics(W, H)
    pts3d = np.array([
        [-0.2, -0.1, 2.0],
        [0.2, -0.1, 2.0],
        [-0.2, 0.1, 2.0],
        [0.2, 0.1, 2.0],
    ], dtype=np.float64)
    pts2d = (K @ (pts3d / pts3d[:, 2:]).T).T[:, :2]
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec_ok = np.zeros((3, 1), dtype=np.float64)
    tvec_bad = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    idx = np.arange(len(pts3d))

    from spacerec.vo import reprojection_median_px

    assert reprojection_median_px(pts3d, pts2d, rvec, tvec_ok, K, idx) < 1e-6
    assert reprojection_median_px(pts3d, pts2d, rvec, tvec_bad, K, idx) > 100.0


def test_vo_recovers_lateral_translation():
    """평면 장면(z=2)에서 카메라가 +x로 0.1 이동하면 영상은 -x로 균일하게 흐른다."""
    W, H, Z, TX = 640, 480, 2.0, 0.1
    K = default_intrinsics(W, H)
    shift = K[0, 0] * TX / Z  # pixels

    img0 = _noise_image((H, W))
    M = np.float32([[1, 0, -shift], [0, 1, 0]])
    img1 = cv2.warpAffine(img0, M, (W, H))
    depth = np.full((H, W), Z, np.float32)

    cfg = VoCfg(keyframe_interval_s=100.0, keyframe_min_flow_px=1e9,
                min_inlier_ratio=0.0)
    vo = VisualOdometry(K, cfg)
    r0 = vo.process(img0, depth, 0.0, None)
    assert r0.is_keyframe
    r1 = vo.process(img1, depth, 0.1, None)
    assert not r1.lost
    t = r1.T_wc[:3, 3]
    assert t[0] == pytest.approx(TX, abs=0.01)
    assert abs(t[1]) < 0.01 and abs(t[2]) < 0.02
    # 회전은 거의 없어야 함
    assert np.allclose(r1.T_wc[:3, :3], np.eye(3), atol=0.02)


def test_vo_rotation_prior_tracks_large_pure_rotation():
    W, H, Z = 320, 240, 2.0
    K = default_intrinsics(W, H)
    angle = np.radians(9.0)
    R_delta, _ = cv2.Rodrigues(np.array([0.0, angle, 0.0], dtype=np.float64))
    H_prev_to_cur = K @ R_delta @ np.linalg.inv(K)

    img0 = _noise_image((H, W), seed=4)
    img1 = cv2.warpPerspective(img0, H_prev_to_cur, (W, H),
                               flags=cv2.INTER_LINEAR)
    depth = np.full((H, W), Z, np.float32)
    cfg = VoCfg(keyframe_interval_s=100.0, keyframe_min_flow_px=1e9,
                min_inlier_ratio=0.0)

    plain = VisualOdometry(K, cfg)
    assert plain.process(img0, depth, 0.0, None).is_keyframe
    plain_result = plain.process(img1, depth, 0.1, None)

    aided = VisualOdometry(K, cfg)
    assert aided.process(img0, depth, 0.0, None).is_keyframe
    aided_result = aided.process(
        img1, depth, 0.1, None,
        R_delta_prev=R_delta,
        R_since_keyframe=R_delta,
    )

    assert plain_result.lost
    assert not aided_result.lost
    assert aided_result.n_tracked >= 8
    assert aided_result.inlier_ratio == pytest.approx(1.0)
    assert np.allclose(aided_result.T_wc[:3, :3], R_delta.T, atol=0.02)


def test_pnp_prior_remains_bounded_after_nonidentity_keyframe_pose():
    W, H, Z = 320, 240, 2.0
    K = default_intrinsics(W, H)
    cfg = VoCfg(keyframe_interval_s=0.2, keyframe_min_flow_px=1e9,
                min_inlier_ratio=0.0)
    img0 = _noise_image((H, W), seed=11)
    depth = np.full((H, W), Z, np.float32)

    vo = VisualOdometry(K, cfg)
    assert vo.process(img0, depth, 0.0, None).is_keyframe

    shift = K[0, 0] * 0.10 / Z
    img1 = cv2.warpAffine(img0, np.float32([[1, 0, -shift], [0, 1, 0]]), (W, H))
    r1 = vo.process(img1, depth, 0.1, None)
    assert not r1.lost
    prev_pos = r1.T_wc[:3, 3].copy()

    # Force a new keyframe at a non-identity pose.
    r2 = vo.process(img1, depth, 0.3, None)
    assert r2.is_keyframe

    angle = np.radians(5.0)
    R_delta, _ = cv2.Rodrigues(np.array([0.0, angle, 0.0], dtype=np.float64))
    img2 = cv2.warpPerspective(img1, K @ R_delta @ np.linalg.inv(K), (W, H))
    r3 = vo.process(
        img2,
        depth,
        0.4,
        None,
        R_delta_prev=R_delta,
        R_since_keyframe=R_delta,
    )

    assert not r3.lost
    assert np.linalg.norm(r3.T_wc[:3, 3] - prev_pos) < 0.5


def test_bad_pnp_prior_does_not_teleport_pose():
    W, H, Z = 320, 240, 2.0
    K = default_intrinsics(W, H)
    cfg = VoCfg(keyframe_interval_s=100.0, keyframe_min_flow_px=1e9,
                min_inlier_ratio=0.0,
                pnp_max_step_depth_frac=0.4,
                pnp_max_velocity_units_s=3.0,
                pnp_step_floor_units=0.10)
    img0 = _noise_image((H, W), seed=21)
    depth = np.full((H, W), Z, np.float32)
    vo = VisualOdometry(K, cfg)
    assert vo.process(img0, depth, 0.0, None).is_keyframe

    shift = K[0, 0] * 0.05 / Z
    img1 = cv2.warpAffine(img0, np.float32([[1, 0, -shift], [0, 1, 0]]), (W, H))
    bad_R, _ = cv2.Rodrigues(np.array([0.0, np.radians(25.0), 0.0], dtype=np.float64))
    result = vo.process(
        img1,
        depth,
        0.1,
        None,
        R_delta_prev=None,
        R_since_keyframe=bad_R,
    )

    assert not result.lost
    assert np.linalg.norm(result.T_wc[:3, 3]) < 0.5
