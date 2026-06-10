import numpy as np

from spacerec.config import GraphCfg, ObjectsCfg
from spacerec.detect import Detection
from spacerec.graph import build_graph
from spacerec.objects import Observation, ObjectRegistry, WorldObject


def _det(track_id, cls_name="chair"):
    return Detection(track_id=track_id, cls_name=cls_name, conf=0.9,
                     box=np.array([0, 0, 10, 10], dtype=np.float32), mask=None)


def _obs(track_id, pos, cls_name="chair", size=0.3, emb=None):
    return Observation(det=_det(track_id, cls_name), position=np.asarray(pos, float),
                       size=size, emb=None if emb is None else np.asarray(emb, float))


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=384)
    return v / np.linalg.norm(v)


def test_registry_persistence_and_ema():
    reg = ObjectRegistry(ObjectsCfg(ema_alpha=0.5))
    p0 = np.array([1.0, 0.0, 2.0])
    visible = reg.update([_obs(1, p0)], ts=0.0)
    assert len(reg.objects) == 1 and len(visible) == 1
    obj = next(iter(reg.objects.values()))

    # 같은 track id로 약간 이동 -> EMA 갱신
    reg.update([_obs(1, p0 + 0.1)], ts=0.5)
    assert np.allclose(obj.position, p0 + 0.05)

    # 화면에서 사라져도 객체와 위치는 유지 (영속성)
    visible = reg.update([], ts=1.0)
    assert visible == set()
    assert len(reg.objects) == 1
    assert np.allclose(obj.position, p0 + 0.05)


def test_registry_reidentify_same_class_nearby():
    reg = ObjectRegistry(ObjectsCfg(merge_radius=0.5))
    pos = np.array([0.0, 0.0, 1.0])
    reg.update([_obs(1, pos, size=0.5)], ts=0.0)
    # 트랙이 끊긴 뒤 새 track id(7)로 같은 자리 근처에 재등장 -> 병합
    reg.update([_obs(7, pos + 0.2, size=0.5)], ts=2.0)
    assert len(reg.objects) == 1
    # 같은 클래스라도 멀리 떨어져 있으면 새 객체
    reg.update([_obs(9, pos + np.array([2.0, 0, 0]), size=0.5)], ts=3.0)
    assert len(reg.objects) == 2
    # 다른 클래스는 같은 자리여도 새 객체
    reg.update([_obs(11, pos, cls_name="cup")], ts=4.0)
    assert len(reg.objects) == 3


def test_registry_dynamic_flag():
    reg = ObjectRegistry(ObjectsCfg(dynamic_var_thresh=0.15))
    for k in range(8):  # 한 방향으로 계속 움직이는 물체
        reg.update([_obs(1, [0.3 * k, 0.0, 1.0], cls_name="person")], ts=float(k))
    obj = next(iter(reg.objects.values()))
    assert obj.is_dynamic
    assert len(obj.trajectory) >= 2


def test_appearance_separates_nearby_same_class():
    """위치 게이트가 겹치는 같은 클래스 두 물체도 외형이 다르면 분리 유지."""
    cfg = ObjectsCfg(merge_radius=0.6, app_gate=0.4)
    reg = ObjectRegistry(cfg)
    bed_a, bed_b = _emb(1), _emb(2)  # 서로 직교에 가까움 (cos ≈ 0)
    reg.update([_obs(1, [0.0, 0, 1.0], "bed", size=0.6, emb=bed_a)], ts=0.0)
    # 추적이 끊긴 뒤(track id 새로 발급) 0.3 떨어진 곳에 '다른' 침대 등장
    reg.update([_obs(5, [0.3, 0, 1.0], "bed", size=0.6, emb=bed_b)], ts=2.0)
    assert len(reg.objects) == 2  # 외형이 다르므로 병합 금지


def test_appearance_reidentifies_after_gap():
    """화면 밖으로 나갔다 돌아온 물체는 외형으로 같은 노드에 복원."""
    cfg = ObjectsCfg(merge_radius=0.6, app_gate=0.4)
    reg = ObjectRegistry(cfg)
    bed = _emb(3)
    noise = 0.05 * _emb(4)
    reg.update([_obs(1, [0.0, 0, 1.0], "bed", size=0.6, emb=bed)], ts=0.0)
    first_id = next(iter(reg.objects.keys()))
    reg.update([], ts=3.0)  # 공백
    # 위치가 약간 드리프트한 채 재등장 (track id 새로 발급, 외형은 유사)
    emb2 = bed + noise
    emb2 /= np.linalg.norm(emb2)
    reg.update([_obs(9, [0.25, 0, 1.0], "bed", size=0.6, emb=emb2)], ts=5.0)
    assert len(reg.objects) == 1
    assert first_id in reg.objects
    assert reg.objects[first_id].n_obs == 2


def test_prune_removes_one_shot_ghosts():
    reg = ObjectRegistry(ObjectsCfg())
    reg.update([_obs(1, [0, 0, 1.0])], ts=0.0)
    reg.update([], ts=10.0)  # 8초 넘게 재확인 안 됨 -> 제거
    assert len(reg.objects) == 0


def _wobj(oid, cls_name, x, y, z):
    return WorldObject(oid, cls_name, np.array([x, y, z], float), 0.0)


def test_graph_relations():
    cfg = GraphCfg(near_dist=1.5, vertical_ratio=1.5)
    cup = _wobj(0, "cup", 0.0, -0.5, 1.0)      # y는 아래가 +, 컵이 책상 위
    desk = _wobj(1, "desk", 0.05, 0.0, 1.0)
    chair = _wobj(2, "chair", 0.8, 0.0, 1.0)
    far = _wobj(3, "sofa", 5.0, 0.0, 5.0)
    edges = build_graph([cup, desk, chair, far], cfg)

    rel = {tuple(sorted((e.a.obj_id, e.b.obj_id))): e for e in edges}
    assert (0, 1) in rel and rel[(0, 1)].relation == "above"
    assert rel[(0, 1)].a.obj_id == 0  # above 엣지는 (위, 아래) 순서로 정규화됨
    assert (1, 2) in rel and rel[(1, 2)].relation == "beside"
    assert not any(3 in pair for pair in rel)  # 멀리 있는 객체는 엣지 없음
