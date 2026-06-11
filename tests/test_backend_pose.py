"""build_pose_inputs: VO pose → DA3 pose-conditioned 입력 변환과 퇴화 게이트."""

import numpy as np
import pytest

from spacerec.backend import BackendKeyframe, build_pose_inputs


def _kf(kf_id: int, pos, K=None, rot=None) -> BackendKeyframe:
    T = np.eye(4)
    if rot is not None:
        T[:3, :3] = rot
    T[:3, 3] = pos
    return BackendKeyframe(kf_id=kf_id, ts=float(kf_id), rgb=None,
                           T_wc_live=T, raw_depth=None, dyn_mask=None, K=K)


def _rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


K = np.array([[400.0, 0, 252], [0, 400.0, 140], [0, 0, 1]])


def test_w2c_inverts_scaled_c2w_pose():
    window = [_kf(i, [0.1 * i, 0.02 * i, 0.0], K=K, rot=_rot_y(5 * i))
              for i in range(4)]
    out = build_pose_inputs(window)
    assert out is not None
    exts, ixts = out
    assert exts.shape == (4, 4, 4)
    assert ixts.shape == (4, 3, 3)
    np.testing.assert_allclose(ixts[0], K)

    centers = np.array([kf.T_wc_live[:3, 3] for kf in window])
    med = np.median(np.linalg.norm(centers - centers[0], axis=1))
    for w2c, kf in zip(exts, window):
        # 회전은 그대로, 병진은 median 거리=1로 사전 스케일된 c2w의 역
        c2w_scaled = np.eye(4)
        c2w_scaled[:3, :3] = kf.T_wc_live[:3, :3]
        c2w_scaled[:3, 3] = kf.T_wc_live[:3, 3] / med
        np.testing.assert_allclose(w2c @ c2w_scaled, np.eye(4), atol=1e-10)

    # 사전 스케일 결과: 첫 뷰 기준 median 카메라 거리 == 1 (DA3 클램프 무력화)
    c_scaled = np.array([np.linalg.inv(e)[:3, 3] for e in exts])
    d = np.linalg.norm(c_scaled - c_scaled[0], axis=1)
    np.testing.assert_allclose(np.median(d), 1.0, atol=1e-10)


def test_degenerate_baseline_falls_back():
    # 카메라가 사실상 제자리 (스프레드 < min_spread) → 무조건화 폴백
    window = [_kf(i, [1e-5 * i, 0, 0], K=K) for i in range(4)]
    assert build_pose_inputs(window) is None


def test_missing_intrinsics_falls_back():
    window = [_kf(i, [0.1 * i, 0, 0], K=K) for i in range(4)]
    window[2].K = None
    assert build_pose_inputs(window) is None


def test_too_few_views_falls_back():
    window = [_kf(i, [0.5 * i, 0, 0], K=K) for i in range(2)]
    assert build_pose_inputs(window) is None


@pytest.mark.parametrize("spread_scale, expected", [(1.0, True), (1e-4, False)])
def test_spread_gate(spread_scale, expected):
    window = [_kf(i, [0.05 * i * spread_scale, 0, 0], K=K) for i in range(6)]
    assert (build_pose_inputs(window) is not None) is expected
