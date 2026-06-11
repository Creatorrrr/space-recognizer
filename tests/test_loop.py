"""루프 클로저 수학 코어: 3D-3D RANSAC Sim3, Sim3 pose graph 최적화."""

import numpy as np
from scipy.spatial.transform import Rotation

from spacerec.geometry import sim3_apply, sim3_compose, sim3_inverse
from spacerec.loop import (LoopDetector, optimize_pose_graph,
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
