"""증거 기반 갱신 테스트: 지도 carving과 객체 부재 처리."""

import numpy as np

from spacerec.config import BackendCfg, ObjectsCfg
from spacerec.detect import Detection
from spacerec.objects import Observation, ObjectRegistry
from spacerec.vo import default_intrinsics
from spacerec.worldmap import GlobalMap


def _wall(z: float, n: int = 20, extent: float = 1.0):
    """원점에서 보이는 z 평면 위의 점 격자."""
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent * 0.6, extent * 0.6, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)


def test_carving_erases_corrected_surface():
    """잘못 재구성된 벽(z=2)이, 그 너머(z=3)가 보이는 재촬영으로 지워진다.

    재촬영 depth는 실제처럼 조밀하게(광선 간격 ≈ voxel), 시야는 잘못된 벽
    전체를 덮도록 넓게 구성한다.
    """
    wm = GlobalMap(BackendCfg(voxel_size=0.05))
    origin = np.zeros((1, 3))
    bad_wall = _wall(2.0, n=20, extent=1.0)
    colors = np.full((len(bad_wall), 3), 128)
    wm.fuse(bad_wall, colors)  # 시선 정보 없는 융합 (초기 오류)
    n_bad0 = (wm.points[:, 2] < 2.5).sum()
    assert n_bad0 > 100

    # 재촬영: 같은 방향 시선이 이제 z=3 벽에 닿음 (z=2 자리를 관통)
    true_wall = _wall(3.0, n=60, extent=1.6)
    for _ in range(6):
        wm.fuse(true_wall, np.full((len(true_wall), 3), 200),
                origins=origin, view_idx=np.zeros(len(true_wall), np.uint8))

    near = (wm.points[:, 2] < 2.5).sum()
    far = (wm.points[:, 2] > 2.5).sum()
    assert far > 500              # 올바른 벽은 융합되어 있고
    assert near < 0.15 * n_bad0   # 잘못된 벽은 대부분 지워짐


def test_carving_resists_single_bad_pass():
    """반대 방향: 좋은 지도가 단발성 불량 관측으로는 지워지지 않는다."""
    wm = GlobalMap(BackendCfg(voxel_size=0.05))
    origin = np.zeros((1, 3))
    good_wall = _wall(2.0)
    colors = np.full((len(good_wall), 3), 128)
    # 좋은 벽을 여러 윈도에 걸쳐 관측 (가중치 적립)
    for _ in range(5):
        wm.fuse(good_wall, colors, origins=origin,
                view_idx=np.zeros(len(good_wall), np.uint8))
    before = (np.abs(wm.points[:, 2] - 2.0) < 0.1).sum()

    # 불량 관측 1회: depth가 잘못 커서 시선이 벽을 관통
    bad = _wall(3.5)
    wm.fuse(bad, np.full((len(bad), 3), 50), origins=origin,
            view_idx=np.zeros(len(bad), np.uint8))
    after = (np.abs(wm.points[:, 2] - 2.0) < 0.1).sum()
    assert after > 0.5 * before  # 한 번의 오류로는 대부분 살아남음


def _obs(track_id, pos, cls_name="chair"):
    det = Detection(track_id=track_id, cls_name=cls_name, conf=0.9,
                    box=np.array([0, 0, 10, 10], np.float32), mask=None)
    return Observation(det=det, position=np.asarray(pos, float), size=0.3)


def test_absent_object_removed_when_visible_area_empty():
    """시야 중앙·비가림인데 계속 미검출 → 노드 제거."""
    reg = ObjectRegistry(ObjectsCfg())
    K = default_intrinsics(640, 480)
    pos = np.array([0.0, 0.0, 1.0])  # 화면 정중앙, 1.0 거리
    reg.update([_obs(1, pos)], ts=0.0)
    for _ in range(2):  # n_obs를 3으로 (프루닝 대상에서 제외)
        reg.update([_obs(1, pos)], ts=0.1)
    assert len(reg.objects) == 1
    oid = next(iter(reg.objects))
    positions_live = {oid: pos}

    depth_far = np.full((480, 640), 3.0, np.float32)  # 그 자리가 비어 보임
    for k in range(5):
        reg.decay_absent(set(), [], positions_live, np.eye(4), K,
                         depth_far, limit=5)
    assert len(reg.objects) == 0


def test_absent_object_kept_when_occluded():
    """노드 위치가 더 가까운 표면에 가려져 있으면 부재로 세지 않는다."""
    reg = ObjectRegistry(ObjectsCfg())
    K = default_intrinsics(640, 480)
    pos = np.array([0.0, 0.0, 1.0])
    reg.update([_obs(1, pos)], ts=0.0)
    oid = next(iter(reg.objects))
    positions_live = {oid: pos}

    depth_near = np.full((480, 640), 0.5, np.float32)  # 앞에 가림막
    for _ in range(20):
        reg.decay_absent(set(), [], positions_live, np.eye(4), K,
                         depth_near, limit=5)
    assert len(reg.objects) == 1
    assert next(iter(reg.objects.values())).miss_count == 0


def test_absent_skips_when_same_class_detected_nearby():
    """근처에 같은 클래스 검출이 있으면 (연관 실패 가능) 부재로 세지 않는다."""
    reg = ObjectRegistry(ObjectsCfg())
    K = default_intrinsics(640, 480)
    pos = np.array([0.0, 0.0, 1.0])
    reg.update([_obs(1, pos)], ts=0.0)
    oid = next(iter(reg.objects))
    depth_far = np.full((480, 640), 3.0, np.float32)
    nearby = _obs(9, pos + 0.1)
    for _ in range(20):
        reg.decay_absent(set(), [nearby], {oid: pos}, np.eye(4), K,
                         depth_far, limit=5)
    assert len(reg.objects) == 1
