import cv2
import numpy as np
import pytest

from spacerec.config import VoCfg
from spacerec.vo import (Keyframe, PnpCandidate, PoseResult, VisualOdometry, backproject,
                         default_intrinsics, rotation_residual_deg)


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


def test_vo_defers_initial_keyframe_until_depth_available():
    W, H, Z = 320, 240, 2.0
    K = default_intrinsics(W, H)
    img = _noise_image((H, W), seed=11)
    depth = np.full((H, W), Z, np.float32)
    vo = VisualOdometry(K, VoCfg())

    r0 = vo.process(img, None, 0.0, None)
    r1 = vo.process(img, depth, 0.1, None)

    assert r0.lost
    assert not r0.is_keyframe
    assert r1.is_keyframe


def test_vo_defers_rekeyframe_when_depth_missing():
    W, H, Z = 320, 240, 2.0
    K = default_intrinsics(W, H)
    img0 = _noise_image((H, W), seed=12)
    img1 = _noise_image((H, W), seed=12)
    depth = np.full((H, W), Z, np.float32)
    cfg = VoCfg(keyframe_interval_s=0.0, keyframe_min_flow_px=1e9,
                min_inlier_ratio=0.0)
    vo = VisualOdometry(K, cfg)

    r0 = vo.process(img0, depth, 0.0, None)
    r1 = vo.process(img1, None, 1.0, None)

    assert r0.is_keyframe
    assert not r1.is_keyframe
    assert vo.keyframe is not None
    assert vo.keyframe.ts == 0.0


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


def _pnp_candidate(name, R, t, n):
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    cand = PnpCandidate(
        name=name,
        ok=True,
        rvec=rvec,
        tvec=np.asarray(t, dtype=np.float64).reshape(3, 1),
        inliers=np.arange(n, dtype=np.int32).reshape(-1, 1),
        n_total=n,
        reproj_median_px=0.2,
    )
    return cand


def _synthetic_pnp_scene():
    W, H = 320, 240
    K = default_intrinsics(W, H)
    xs = np.linspace(-0.6, 0.6, 5)
    ys = np.linspace(-0.4, 0.4, 4)
    pts3d = np.array([[x, y, 3.0 + 0.1 * x] for y in ys for x in xs],
                     dtype=np.float64)
    pts2d = (K @ (pts3d / pts3d[:, 2:]).T).T[:, :2]
    return K, pts3d, pts2d


def test_rotation_residual_deg_reports_so3_difference():
    R_visual, _ = cv2.Rodrigues(np.array([0.0, np.radians(10.0), 0.0]))
    R_imu, _ = cv2.Rodrigues(np.array([0.0, np.radians(3.0), 0.0]))

    assert rotation_residual_deg(R_visual, R_imu) == pytest.approx(7.0, abs=1e-3)


def test_imu_constrained_candidate_replaces_divergent_visual_when_reprojection_passes():
    K, pts3d, pts2d = _synthetic_pnp_scene()
    cfg = VoCfg(
        imu_rot_residual_warn_deg=3.0,
        imu_rot_residual_reject_deg=6.0,
        pnp_max_step_depth_frac=1.0,
    )
    vo = VisualOdometry(K, cfg)
    R_bad, _ = cv2.Rodrigues(np.array([0.0, np.radians(10.0), 0.0]))
    base = _pnp_candidate("base", R_bad, [0.0, 0.0, 0.0], len(pts3d))

    selection = vo._apply_imu_rotation_gate(
        base,
        pts3d,
        pts2d,
        np.eye(3),
        dt=0.1,
    )

    assert selection.candidate is not None
    assert selection.candidate.name == "imu_constrained"
    assert selection.low_confidence is False
    assert selection.fusion_skipped is False
    assert selection.rot_residual_deg == pytest.approx(10.0, abs=0.2)
    assert rotation_residual_deg(selection.candidate.rotation_matrix(), np.eye(3)) < 0.5


def test_bad_imu_does_not_override_good_visual_candidate():
    K, pts3d, pts2d = _synthetic_pnp_scene()
    cfg = VoCfg(
        imu_rot_residual_warn_deg=3.0,
        imu_rot_residual_reject_deg=6.0,
        pnp_max_step_depth_frac=1.0,
    )
    vo = VisualOdometry(K, cfg)
    base = _pnp_candidate("base", np.eye(3), [0.0, 0.0, 0.0], len(pts3d))
    R_bad_imu, _ = cv2.Rodrigues(np.array([0.0, np.radians(15.0), 0.0]))

    selection = vo._apply_imu_rotation_gate(
        base,
        pts3d,
        pts2d,
        R_bad_imu,
        dt=0.1,
    )

    assert selection.candidate is base
    assert selection.low_confidence is True
    assert selection.fusion_skipped is True
    assert selection.rotation_source == "visual_pnp"


def test_imu_rotation_gate_keeps_visual_for_warn_level_disagreement():
    K, pts3d, pts2d = _synthetic_pnp_scene()
    cfg = VoCfg(
        imu_rot_residual_warn_deg=3.0,
        imu_rot_residual_reject_deg=6.0,
        pnp_max_step_depth_frac=1.0,
    )
    vo = VisualOdometry(K, cfg)
    R_visual, _ = cv2.Rodrigues(np.array([0.0, np.radians(4.0), 0.0]))
    base = _pnp_candidate("base", R_visual, [0.0, 0.0, 0.0], len(pts3d))

    selection = vo._apply_imu_rotation_gate(
        base,
        pts3d,
        pts2d,
        np.eye(3),
        dt=0.1,
    )

    assert selection.candidate is base
    assert selection.rot_residual_deg == pytest.approx(4.0, abs=0.2)
    assert selection.low_confidence is False
    assert selection.fusion_skipped is False


def test_rejected_imu_visual_divergence_keeps_tracking_recoverable(monkeypatch):
    K, pts3d, pts2d = _synthetic_pnp_scene()
    cfg = VoCfg(
        imu_rot_residual_warn_deg=3.0,
        imu_rot_residual_reject_deg=6.0,
        pnp_max_step_depth_frac=1.0,
    )
    vo = VisualOdometry(K, cfg)
    gray = np.zeros((240, 320), dtype=np.uint8)
    vo.keyframe = Keyframe(
        ts=0.0,
        gray=gray,
        depth=np.ones_like(gray, dtype=np.float32),
        T_wc=np.eye(4),
        obj_masks=None,
    )
    vo._prev_gray = gray
    vo._prev_ts = 0.0
    vo._pts2d = pts2d.astype(np.float32)
    vo._pts3d = pts3d.copy()
    vo._pts3d_keyframe = pts3d.copy()
    vo._kf_pts2d = pts2d.astype(np.float32)
    previous_pose = np.eye(4)
    previous_pose[:3, 3] = [1.0, 0.0, 0.0]
    vo.T_wc = previous_pose.copy()

    monkeypatch.setattr(
        vo,
        "_lk_track",
        lambda *_args, **_kwargs: (
            pts2d.astype(np.float32).reshape(-1, 1, 2),
            np.ones((len(pts2d), 1), dtype=np.uint8),
            np.zeros(len(pts2d), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        vo,
        "_solve_pnp",
        lambda *_args, **_kwargs: _pnp_candidate(
            "base", np.eye(3), [0.0, 0.0, 0.0], len(pts3d)),
    )
    R_bad_imu, _ = cv2.Rodrigues(np.array([0.0, np.radians(15.0), 0.0]))

    result = vo._track(gray, ts=0.1, R_since_keyframe=R_bad_imu)

    expected_visual_pose = np.eye(4)
    assert result.low_confidence
    assert result.fusion_skipped
    assert result.rotation_source == "visual_pnp"
    assert result.pnp_candidate_source == "base"
    assert np.allclose(result.T_wc, expected_visual_pose)
    assert np.allclose(vo.T_wc, expected_visual_pose)
    assert not np.allclose(vo.T_wc, previous_pose)


def test_process_refreshes_tracking_keyframe_for_low_confidence_pose(monkeypatch):
    K = default_intrinsics(320, 240)
    cfg = VoCfg(keyframe_interval_s=0.1, keyframe_min_flow_px=1e9)
    vo = VisualOdometry(K, cfg)
    gray = np.zeros((240, 320), dtype=np.uint8)
    depth = np.ones_like(gray, dtype=np.float32)
    vo.keyframe = Keyframe(0.0, gray, depth, np.eye(4), None)
    vo._prev_ts = 0.0

    monkeypatch.setattr(
        vo,
        "_track",
        lambda *_args, **_kwargs: PoseResult(
            vo.T_wc.copy(),
            1.0,
            20,
            False,
            low_confidence=True,
            fusion_skipped=True,
        ),
    )
    keyframe_calls = []

    def fake_make_keyframe(_gray, _depth, ts, exclude_mask):
        keyframe_calls.append(ts)
        vo.keyframe = Keyframe(ts, _gray, _depth, vo.T_wc.copy(), exclude_mask)
        vo._prev_gray = _gray

    monkeypatch.setattr(vo, "_make_keyframe", fake_make_keyframe)

    result = vo.process(gray, depth, ts=0.2, exclude_mask=None)

    assert keyframe_calls == [0.2]
    assert result.is_keyframe
    assert result.low_confidence
    assert result.fusion_skipped
