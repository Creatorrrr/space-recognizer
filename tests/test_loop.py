"""루프 클로저 수학 코어: 3D-3D RANSAC Sim3, Sim3 pose graph 최적화."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spacerec.geometry import (SIM3_IDENTITY, sim3_apply, sim3_compose,
                               sim3_interp, sim3_inverse, sim3_on_pose)
from spacerec.loop import (LoopDetector, edge_residual, optimize_pose_graph,
                           sequential_edges, sim3_from_matches)


def _rand_sim3(rng, s=1.3, max_deg=25, max_t=0.5):
    R = Rotation.from_rotvec(rng.uniform(-1, 1, 3)
                             * np.radians(max_deg)).as_matrix()
    return (s, R, rng.uniform(-max_t, max_t, 3))


def test_sim3_from_matches_recovers_transform_with_outliers():
    rng = np.random.default_rng(7)
    T_true = _rand_sim3(rng)
    pts_b = rng.uniform(-2, 2, (80, 3))
    pts_a = sim3_apply(T_true, pts_b) + rng.normal(0, 0.005, (80, 3))
    # 25% outlier
    out = rng.choice(80, 20, replace=False)
    pts_a[out] += rng.uniform(0.5, 2.0, (20, 3))

    result = sim3_from_matches(pts_a, pts_b, inlier_dist=0.05)
    assert result is not None
    T, mask = result
    assert mask.sum() >= 55
    assert abs(T[0] - T_true[0]) < 0.02
    assert np.linalg.norm(T[2] - T_true[2]) < 0.05
    rot_err = Rotation.from_matrix(T[1] @ T_true[1].T).magnitude()
    assert rot_err < np.radians(2)


def test_sim3_from_matches_rejects_garbage():
    rng = np.random.default_rng(0)
    pts_a = rng.uniform(-2, 2, (40, 3))
    pts_b = rng.uniform(-2, 2, (40, 3))  # 무상관 — 기각되어야 함
    assert sim3_from_matches(pts_a, pts_b, inlier_dist=0.03,
                             min_inliers=15) is None


def _se3(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _global_floor_points(normal, d=-1.0, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    z = rng.uniform(0.5, 4.0, n)
    y = (d - normal[0] * x - normal[2] * z) / normal[1]
    return np.stack([x, y, z], axis=1)


def _angle_rad(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.arccos(np.clip(float(a @ b), -1.0, 1.0))


def test_edge_residual_is_zero_for_matching_relative_pose():
    P_i = _se3(np.eye(3), np.zeros(3))
    P_j = _se3(np.eye(3), np.array([1.0, 0.0, 0.0]))
    Z_ij = (1.0, np.eye(3), np.array([1.0, 0.0, 0.0]))

    assert edge_residual(P_i, P_j, Z_ij) < 1e-9


def test_edge_residual_grows_for_inconsistent_relative_pose():
    P_i = _se3(np.eye(3), np.zeros(3))
    P_j = _se3(np.eye(3), np.array([1.1, 0.0, 0.0]))
    Z_ij = (1.0, np.eye(3), np.array([1.0, 0.0, 0.0]))

    assert edge_residual(P_i, P_j, Z_ij) > 0.09


def test_pose_graph_corrects_square_loop_drift():
    """사각 궤적: 마지막 노드가 시작점으로 돌아오는데 drift로 어긋난 상황."""
    rng = np.random.default_rng(3)
    n = 21
    # 정답 궤적: 한 변 5노드짜리 사각형 (시작 = 끝 위치)
    gt = [np.eye(4)]
    legs = [(np.array([0.2, 0, 0]), 0), (np.array([0, 0, 0.2]), 0),
            (np.array([-0.2, 0, 0]), 0), (np.array([0, 0, -0.2]), 0)]
    for leg in range(4):
        step, _ = legs[leg]
        for _ in range(5):
            T = gt[-1].copy()
            T[:3, 3] = T[:3, 3] + step
            gt.append(T)
    gt = gt[:n]

    # drift 시뮬레이션: 실제 VO처럼 계통적 편향이 누적 (yaw 바이어스 +
    # 스케일 바이어스 + 약간의 랜덤 노이즈)
    yaw_bias = Rotation.from_rotvec([0, 0.012, 0]).as_matrix()
    drifted = [gt[0]]
    for k in range(1, n):
        rel = np.linalg.inv(gt[k - 1]) @ gt[k]
        noise_R = Rotation.from_rotvec(rng.normal(0, 0.004, 3)).as_matrix()
        rel = rel.copy()
        rel[:3, :3] = yaw_bias @ noise_R @ rel[:3, :3]
        rel[:3, 3] = rel[:3, 3] * 1.04 + rng.normal(0, 0.002, 3)
        drifted.append(drifted[-1] @ rel)

    end_err_before = np.linalg.norm(drifted[-1][:3, 3] - gt[-1][:3, 3])
    assert end_err_before > 0.05  # drift가 실제로 존재

    # 루프 엣지: 마지막 노드 ↔ 첫 노드, 정답 상대 Sim3 (검증된 측정으로 가정)
    A = (1.0, gt[0][:3, :3], gt[0][:3, 3])
    B = (1.0, gt[-1][:3, :3], gt[-1][:3, 3])
    Z_loop = sim3_compose(sim3_inverse(A), B)

    edges = sequential_edges(drifted) + [(0, n - 1, Z_loop, 1.0)]
    corrected = optimize_pose_graph(drifted, edges)

    end_err_after = np.linalg.norm(corrected[-1][2] - gt[-1][:3, 3])
    assert end_err_after < end_err_before * 0.2  # 5배 이상 개선
    # 게이지: 첫 노드는 고정
    np.testing.assert_allclose(corrected[0][2], gt[0][:3, 3], atol=1e-9)
    # 스케일 보정이 1 근처의 합리적 범위
    scales = [c[0] for c in corrected]
    assert all(0.7 < s < 1.3 for s in scales)


def test_pose_graph_partial_replay_converges_loop_residual():
    """Repeated partial loop corrections should keep reducing stored-edge error."""
    rng = np.random.default_rng(3)
    n = 21
    gt = [np.eye(4)]
    for step in (np.array([0.2, 0, 0]), np.array([0, 0, 0.2]),
                 np.array([-0.2, 0, 0]), np.array([0, 0, -0.2])):
        for _ in range(5):
            T = gt[-1].copy()
            T[:3, 3] = T[:3, 3] + step
            gt.append(T)
    gt = gt[:n]

    yaw_bias = Rotation.from_rotvec([0, 0.012, 0]).as_matrix()
    poses = [gt[0]]
    for k in range(1, n):
        rel = np.linalg.inv(gt[k - 1]) @ gt[k]
        noise_R = Rotation.from_rotvec(rng.normal(0, 0.004, 3)).as_matrix()
        rel = rel.copy()
        rel[:3, :3] = yaw_bias @ noise_R @ rel[:3, :3]
        rel[:3, 3] = rel[:3, 3] * 1.04 + rng.normal(0, 0.002, 3)
        poses.append(poses[-1] @ rel)

    A = (1.0, gt[0][:3, :3], gt[0][:3, 3])
    B = (1.0, gt[-1][:3, :3], gt[-1][:3, 3])
    Z_loop = sim3_compose(sim3_inverse(A), B)
    assert edge_residual(poses[0], poses[-1], Z_loop) > 0.05

    for _ in range(8):
        corrected = optimize_pose_graph(
            poses, sequential_edges(poses) + [(0, n - 1, Z_loop, 1.0)])
        node_corr = [
            sim3_compose(
                C, sim3_inverse((1.0, P[:3, :3], P[:3, 3])))
            for C, P in zip(corrected, poses)
        ]
        partial = [sim3_interp(SIM3_IDENTITY, C, 0.3) for C in node_corr]
        poses = [sim3_on_pose(C, P) for C, P in zip(partial, poses)]

    assert edge_residual(poses[0], poses[-1], Z_loop) < 0.05


def test_worker_replays_persisted_loop_edge_without_new_acceptance():
    from types import SimpleNamespace

    from spacerec.backend import _Worker

    worker = _Worker.__new__(_Worker)
    worker.detector = object()
    worker.loop_cfg = SimpleNamespace(persist_edges=True)
    Z_loop = (1.0, np.eye(3), np.zeros(3))
    worker._loop_edges = {(0, 7): (Z_loop, 1.0)}
    worker.kf_global_poses = {
        k: _se3(np.eye(3), np.array([0.1 * k, 0.0, 0.0]))
        for k in range(8)
    }
    worker._epoch_kfs = {0: list(range(8))}

    before = edge_residual(worker.kf_global_poses[0],
                           worker.kf_global_poses[7], Z_loop)
    corrections, log, corr_newest = worker._close_loops([])

    assert corrections is not None
    assert 0 in corrections
    assert np.linalg.norm(corr_newest[2]) > 0
    assert "persist r=" in log
    after = edge_residual(worker.kf_global_poses[0],
                          worker.kf_global_poses[7], Z_loop)
    assert after < before
    # 감쇠는 수렴(잔차 <= 0.05) 후에만 — 보정 진행 중에는 weight 유지
    expected_w = 0.9 if after <= 0.05 else 1.0
    assert worker._loop_edges[(0, 7)][1] == pytest.approx(expected_w)


def test_worker_snap_step_limit_is_three_times_normal_limit():
    from spacerec.backend import _Worker

    assert _Worker._SNAP_INLIERS == 150
    assert _Worker._step_limit(False) == (
        _Worker._MAX_STEP_T,
        _Worker._MAX_STEP_LOGS,
    )
    assert _Worker._step_limit(True) == (
        _Worker._MAX_STEP_T * 3.0,
        _Worker._MAX_STEP_LOGS * 3.0,
    )


def test_worker_snap_loop_uses_relaxed_step_and_logs_snap(monkeypatch):
    from types import SimpleNamespace

    import spacerec.loop as loop_mod
    from spacerec.backend import BackendKeyframe, _Worker

    worker = _Worker.__new__(_Worker)

    class Detector:
        def query(self, ts, emb):
            return 0, 0.99

        def add(self, kf_id, ts, emb):
            pass

    worker.detector = Detector()
    worker.loop_cfg = SimpleNamespace(
        persist_edges=True,
        inlier_dist=0.05,
        min_inliers=15,
        min_inlier_frac=0.45,
        max_kf_store=600,
    )
    worker._loop_edges = {}
    worker.kf_global_poses = {
        k: _se3(np.eye(3), np.zeros(3))
        for k in range(8)
    }
    worker._epoch_kfs = {0: list(range(8))}
    worker._kf_store = {
        0: (
            0.0,
            np.zeros((2, 2), np.uint8),
            np.ones((2, 2), np.float16),
            np.eye(3),
        )
    }

    def fake_match_3d3d(*args, **kwargs):
        pts = np.zeros((200, 3), np.float32)
        return pts, pts

    def fake_sim3_from_matches(*args, **kwargs):
        mask = np.zeros(200, dtype=bool)
        mask[:_Worker._SNAP_INLIERS] = True
        return (1.0, np.eye(3), np.zeros(3)), mask

    def fake_optimize_pose_graph(poses, edges):
        out = [(1.0, np.eye(3), np.zeros(3))]
        out.extend(
            (1.0, np.eye(3), np.array([2.0, 0.0, 0.0]))
            for _ in poses[1:]
        )
        return out

    monkeypatch.setattr(loop_mod, "match_3d3d", fake_match_3d3d)
    monkeypatch.setattr(loop_mod, "sim3_from_matches", fake_sim3_from_matches)
    monkeypatch.setattr(loop_mod, "optimize_pose_graph", fake_optimize_pose_graph)

    kf = BackendKeyframe(
        kf_id=7,
        ts=20.0,
        rgb=np.zeros((2, 2, 3), np.uint8),
        T_wc_live=np.eye(4),
        raw_depth=np.ones((2, 2), np.float32),
        dyn_mask=None,
        K=np.eye(3),
        emb=np.array([1.0], dtype=np.float32),
    )

    corrections, log, corr_newest = worker._close_loops([kf])

    assert corrections is not None
    assert corr_newest[2][0] == pytest.approx(_Worker._MAX_STEP_T * 3.0)
    assert "snap" in log


def test_worker_has_pending_loop_residual_checks_persisted_edges_only():
    from types import SimpleNamespace

    from spacerec.backend import _Worker

    worker = _Worker.__new__(_Worker)
    worker.detector = object()
    worker.loop_cfg = SimpleNamespace(persist_edges=True)
    Z_loop = (1.0, np.eye(3), np.zeros(3))
    worker._loop_edges = {(0, 7): (Z_loop, 1.0)}
    worker.kf_global_poses = {
        k: _se3(np.eye(3), np.array([0.1 * k, 0.0, 0.0]))
        for k in range(8)
    }

    assert worker._has_pending_loop_residual() is True

    worker.kf_global_poses[7] = _se3(np.eye(3), np.zeros(3))
    assert worker._has_pending_loop_residual() is False

    worker.kf_global_poses[7] = _se3(np.eye(3), np.array([0.7, 0.0, 0.0]))
    worker.loop_cfg = SimpleNamespace(persist_edges=False)
    assert worker._has_pending_loop_residual() is False

    worker.loop_cfg = SimpleNamespace(persist_edges=True)
    worker.detector = None
    assert worker._has_pending_loop_residual() is False


def test_worker_attitude_correction_clamps_rotation_and_keeps_anchor():
    from spacerec.backend import _ATT_STEP_RAD, _Worker

    worker = _Worker.__new__(_Worker)
    target = np.array([0.0, -1.0, 0.0])
    tilt = np.radians(5.0)
    normal = np.array([0.0, -np.cos(tilt), -np.sin(tilt)])
    points = _global_floor_points(normal, seed=8)
    anchor = np.array([0.4, 1.2, -0.3])

    C, dy = worker._attitude_correction(points, anchor)

    assert C is not None
    assert dy == 0.0  # 첫 측정은 높이 기준만 설정
    assert C[0] == pytest.approx(1.0)
    np.testing.assert_allclose(sim3_apply(C, anchor[None])[0], anchor, atol=1e-9)
    assert Rotation.from_matrix(C[1]).magnitude() == pytest.approx(_ATT_STEP_RAD)
    before = _angle_rad(normal, target)
    after = _angle_rad(C[1] @ normal, target)
    assert before - after == pytest.approx(_ATT_STEP_RAD, abs=1e-4)


def test_worker_attitude_correction_ignores_deadband_tilt():
    from spacerec.backend import _ATT_DEADBAND_RAD, _Worker

    worker = _Worker.__new__(_Worker)
    tilt = _ATT_DEADBAND_RAD * 0.5
    normal = np.array([0.0, -np.cos(tilt), -np.sin(tilt)])
    points = _global_floor_points(normal, seed=9)

    C, dy = worker._attitude_correction(points, np.zeros(3))
    assert C is None and dy == 0.0


def test_worker_height_servo_clamps_floor_offset():
    from spacerec.backend import _ATT_STEP_Y, _Worker

    worker = _Worker.__new__(_Worker)
    flat = np.array([0.0, -1.0, 0.0])
    # 첫 윈도: 바닥 y=1.0이 기준이 된다
    pts_ref = _global_floor_points(flat, d=-1.0, seed=10)
    C, dy = worker._attitude_correction(pts_ref, np.zeros(3))
    assert dy == 0.0
    # 다음 윈도: 바닥이 0.3 낮게(y=1.3) 들어옴 → dy는 +0.15로 클램프
    pts_low = _global_floor_points(flat, d=-1.3, seed=11)
    C, dy = worker._attitude_correction(pts_low, np.zeros(3))
    assert dy == pytest.approx(_ATT_STEP_Y)
    assert C is None  # 평평하므로 회전 보정은 없음


def test_pose_graph_noop_without_loop_edges():
    poses = [_se3(np.eye(3), np.array([0.1 * k, 0, 0])) for k in range(5)]
    corrected = optimize_pose_graph(poses, sequential_edges(poses))
    for c, p in zip(corrected, poses):
        assert abs(c[0] - 1.0) < 1e-6
        np.testing.assert_allclose(c[2], p[:3, 3], atol=1e-6)


def test_worldmap_apply_corrections_moves_only_target_epoch():
    from spacerec.config import BackendCfg
    from spacerec.worldmap import GlobalMap

    wm = GlobalMap(BackendCfg(voxel_size=0.02, max_points=100000))
    rng = np.random.default_rng(1)
    pts_e0 = rng.uniform(0, 1, (500, 3))
    pts_e1 = rng.uniform(2, 3, (500, 3))  # 공간적으로 분리된 다른 epoch
    cols = np.full((500, 3), 128.0)
    wm.fuse(pts_e0, cols, epoch=0)
    wm.fuse(pts_e1, cols, epoch=1)

    shift = np.array([0.5, 0.0, 0.0])
    wm.apply_corrections({1: (1.0, np.eye(3), shift)})

    # epoch 0 영역(0~1 박스)은 그대로, epoch 1 영역은 +0.5 이동
    in_e0 = wm.points[(wm.points[:, 0] < 1.5)]
    in_e1 = wm.points[(wm.points[:, 0] > 1.5)]
    assert len(in_e0) and len(in_e1)
    assert in_e0[:, 0].max() < 1.0 + 0.05
    assert in_e1[:, 0].min() > 2.5 - 0.05  # 2.0 → 2.5로 이동됨
    # 증거 가중치/카운트 보존 (총합)
    assert wm._cnt.sum() == 1000


def test_servo_gain_direction_and_clamp():
    from spacerec.backend import servo_gain

    # 기준과 같으면 보정 없음
    assert servo_gain(2.0, 2.0) == 1.0
    # mpu 상승(라이브 스케일 수축) → g > 1로 depth 확대
    g = servo_gain(2.4, 2.0)
    assert 1.0 < g <= 1.05
    # mpu 하락 → g < 1
    g = servo_gain(1.6, 2.0)
    assert 0.95 <= g < 1.0
    # 큰 drift도 윈도당 ±5%로 클램프 (점진 흡수)
    assert servo_gain(6.0, 2.0) == 1.05
    assert servo_gain(0.5, 2.0) == 0.95
    # 비정상 입력은 항등
    assert servo_gain(0.0, 2.0) == 1.0
    assert servo_gain(2.0, 0.0) == 1.0


def test_rescale_live_keeps_global_positions():
    from spacerec.config import BackendCfg
    from spacerec.worldmap import GlobalMap

    wm = GlobalMap(BackendCfg())
    R = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    wm.set_correction_target((1.4, R, np.array([0.5, -0.2, 1.0])))
    for _ in range(100):
        wm.step_correction()
    p_live = np.array([[0.3, 0.1, 2.0], [1.0, -0.5, 0.7]])
    before = wm.to_global_points(p_live)

    g = 1.05
    wm.rescale_live(g)
    after = wm.to_global_points(p_live * g)  # live 길이가 g배가 된 같은 점
    np.testing.assert_allclose(after, before, atol=1e-9)


def test_vo_rescale_consistency():
    from spacerec.config import VoCfg
    from spacerec.vo import VisualOdometry, default_intrinsics

    import cv2

    vo = VisualOdometry(default_intrinsics(128, 96), VoCfg())
    # 코너가 풍부한 패턴 (블록 노이즈를 NEAREST 확대 → 격자 모서리)
    small = np.random.default_rng(0).integers(0, 255, (12, 16)).astype(np.uint8)
    gray = cv2.resize(small, (128, 96), interpolation=cv2.INTER_NEAREST)
    depth = np.full((96, 128), 2.0, np.float32)
    vo.process(gray, depth, 0.0, None)  # 키프레임 생성
    assert vo._pts3d is not None
    pts_before = vo._pts3d.copy()

    g = 1.05
    vo.rescale(g)
    np.testing.assert_allclose(vo._pts3d, pts_before * g, atol=1e-12)
    np.testing.assert_allclose(vo.keyframe.depth, depth * g, atol=1e-6)


def test_loop_detector_gap_and_threshold():
    det = LoopDetector(sim_thresh=0.6, min_gap_s=10.0)
    e1 = np.zeros(8)
    e1[0] = 1.0
    e2 = np.zeros(8)
    e2[1] = 1.0
    det.add(0, ts=0.0, emb=e1)
    det.add(1, ts=1.0, emb=e2)
    # 같은 외형이지만 시간 갭 부족 → 후보 없음
    assert det.query(ts=5.0, emb=e1) is None
    # 갭 충족 + 유사도 통과 → kf 0 매칭
    hit = det.query(ts=20.0, emb=e1)
    assert hit is not None and hit[0] == 0
    # 유사도 미달
    e3 = np.ones(8) / np.sqrt(8)
    assert det.query(ts=20.0, emb=e3) is None
