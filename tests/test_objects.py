import numpy as np

from spacerec.config import GraphCfg, ObjectsCfg
from spacerec.detect import Detection
from spacerec.graph import build_graph
from spacerec.objects import ObjectRegistry, WorldObject


def _det(track_id, cls_name="chair"):
    return Detection(track_id=track_id, cls_name=cls_name, conf=0.9,
                     box=np.array([0, 0, 10, 10], dtype=np.float32), mask=None)


def test_registry_persistence_and_ema():
    reg = ObjectRegistry(ObjectsCfg(ema_alpha=0.5))
    p0 = np.array([1.0, 0.0, 2.0])
    visible = reg.update([(_det(1), p0)], ts=0.0)
    assert len(reg.objects) == 1 and len(visible) == 1
    obj = next(iter(reg.objects.values()))

    # 같은 track id로 약간 이동 -> EMA 갱신
    reg.update([(_det(1), p0 + 0.1)], ts=0.5)
    assert np.allclose(obj.position, p0 + 0.05)

    # 화면에서 사라져도 객체와 위치는 유지 (영속성)
    visible = reg.update([], ts=1.0)
    assert visible == set()
    assert len(reg.objects) == 1
    assert np.allclose(obj.position, p0 + 0.05)


def test_registry_reidentify_same_class_nearby():
    reg = ObjectRegistry(ObjectsCfg(merge_radius=0.5))
    pos = np.array([0.0, 0.0, 1.0])
    reg.update([(_det(1), pos)], ts=0.0)
    # 트랙이 끊긴 뒤 새 track id(7)로 같은 자리 근처에 재등장 -> 병합
    reg.update([(_det(7), pos + 0.2)], ts=2.0)
    assert len(reg.objects) == 1
    # 같은 클래스라도 멀리 떨어져 있으면 새 객체
    reg.update([(_det(9), pos + np.array([2.0, 0, 0]))], ts=3.0)
    assert len(reg.objects) == 2
    # 다른 클래스는 같은 자리여도 새 객체
    reg.update([(_det(11, "cup"), pos)], ts=4.0)
    assert len(reg.objects) == 3


def test_registry_dynamic_flag():
    reg = ObjectRegistry(ObjectsCfg(dynamic_var_thresh=0.15))
    for k in range(8):  # 한 방향으로 계속 움직이는 물체
        reg.update([(_det(1, "person"), np.array([0.3 * k, 0.0, 1.0]))], ts=float(k))
    obj = next(iter(reg.objects.values()))
    assert obj.is_dynamic
    assert len(obj.trajectory) >= 2


def _obj(oid, cls_name, x, y, z):
    return WorldObject(oid, cls_name, np.array([x, y, z], float), 0.0)


def test_graph_relations():
    cfg = GraphCfg(near_dist=1.5, vertical_ratio=1.5)
    cup = _obj(0, "cup", 0.0, -0.5, 1.0)      # y는 아래가 +, 컵이 책상 위
    desk = _obj(1, "desk", 0.05, 0.0, 1.0)
    chair = _obj(2, "chair", 0.8, 0.0, 1.0)
    far = _obj(3, "sofa", 5.0, 0.0, 5.0)
    edges = build_graph([cup, desk, chair, far], cfg)

    rel = {tuple(sorted((e.a.obj_id, e.b.obj_id))): e for e in edges}
    assert (0, 1) in rel and rel[(0, 1)].relation == "above"
    assert rel[(0, 1)].a.obj_id == 0  # above 엣지는 (위, 아래) 순서로 정규화됨
    assert (1, 2) in rel and rel[(1, 2)].relation == "beside"
    assert not any(3 in pair for pair in rel)  # 멀리 있는 객체는 엣지 없음
