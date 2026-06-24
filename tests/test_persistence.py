import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spacerec.config import BackendCfg, MeshCfg, ObjectsCfg
from spacerec.geometry import sim3_apply
from spacerec.mesh import MeshMap
from spacerec.objects import ObjectRegistry, WorldObject
from spacerec.persistence import (SavedState, load_mesh_state, load_state,
                                  merge_into_session, merge_mesh_into_session,
                                  relocalize, save_mesh_state, save_state)
from spacerec.worldmap import GlobalMap


def _emb(seed):
    v = np.random.default_rng(seed).normal(size=384)
    return v / np.linalg.norm(v)


def _registry_with(objs):
    reg = ObjectRegistry(ObjectsCfg())
    for i, (cls, pos, emb) in enumerate(objs):
        o = WorldObject(i, cls, np.asarray(pos, float), last_seen=0.0,
                        n_obs=5, embedding=emb)
        reg.objects[i] = o
    reg._next_id = len(objs)
    return reg


def test_save_load_roundtrip(tmp_path):
    wm = GlobalMap(BackendCfg(voxel_size=0.05))
    rng = np.random.default_rng(0)
    wm.fuse(rng.uniform(-1, 1, (500, 3)), rng.integers(0, 255, (500, 3)))
    reg = _registry_with([("bed", [1, 0, 2], _emb(1)), ("chair", [0, 0, 1], _emb(2))])
    path = tmp_path / "state.npz"
    n = save_state(path, wm, reg, meters_per_unit=3.5)
    assert n == 2
    s = load_state(path)
    assert s is not None
    assert len(s.points) == len(wm.points)
    assert s.obj_classes == ["bed", "chair"]
    assert s.meters_per_unit == pytest.approx(3.5)


def test_relocalize_recovers_transform_and_merges():
    # 이전 세션: 객체 4개 + 지도
    embs = [_emb(i) for i in range(4)]
    old_pos = np.array([[0, 0, 1.0], [1, 0, 1.5], [0.5, -0.4, 2.0], [-0.6, 0.1, 1.2]])
    rng = np.random.default_rng(7)
    old_points = rng.uniform(-1, 1, (800, 3))
    saved = SavedState(
        points=old_points, colors=np.zeros((800, 3), np.uint8),
        obj_classes=["bed", "chair", "lamp", "rug"],
        obj_positions=old_pos, obj_sizes=np.full(4, 0.4),
        obj_n_obs=np.array([10, 8, 6, 7]),
        obj_embs=np.stack(embs).astype(np.float32),
        obj_dynamic=np.zeros(4, bool), meters_per_unit=3.0)

    # 새 세션 좌표계 = 이전을 Sim3 (s=1.3, 회전+이동) 변환한 것
    s, R = 1.3, Rotation.from_euler("y", 25, degrees=True).as_matrix()
    t = np.array([0.4, -0.1, 0.7])
    T_true = (s, R, t)
    # 새 세션에서 그중 3개를 관측 (임베딩 약간 노이즈)
    reg = _registry_with([
        ("bed", sim3_apply(T_true, old_pos[0][None])[0], embs[0]),
        ("chair", sim3_apply(T_true, old_pos[1][None])[0], embs[1]),
        ("lamp", sim3_apply(T_true, old_pos[2][None])[0], embs[2]),
    ])

    result = relocalize(saved, reg)
    assert result is not None
    T, matches, rms = result
    assert len(matches) == 3 and rms < 1e-6
    assert T[0] == pytest.approx(1.3, rel=1e-6)

    wm = GlobalMap(BackendCfg(voxel_size=0.02))
    merge_into_session(saved, T, matches, wm, reg, now=10.0)
    # 매칭 안 된 rug가 새 노드로 들어오고, 위치는 변환되어 있어야 함
    rugs = [o for o in reg.objects.values() if o.cls_name == "rug"]
    assert len(rugs) == 1
    assert np.allclose(rugs[0].position, sim3_apply(T_true, old_pos[3][None])[0],
                       atol=1e-6)
    # 매칭된 bed는 이전 관측 횟수를 승계
    assert reg.objects[0].n_obs == 5 + 10
    # 지도 포인트 융합됨
    assert len(wm.points) > 0


def test_relocalize_rejects_wrong_scene():
    saved = SavedState(
        points=np.zeros((10, 3)), colors=np.zeros((10, 3), np.uint8),
        obj_classes=["bed", "chair", "lamp"],
        obj_positions=np.array([[0, 0, 1.0], [1, 0, 1.0], [0, 1, 1.0]]),
        obj_sizes=np.full(3, 0.4), obj_n_obs=np.array([5, 5, 5]),
        obj_embs=np.stack([_emb(1), _emb(2), _emb(3)]).astype(np.float32),
        obj_dynamic=np.zeros(3, bool),
        meters_per_unit=3.0)
    # 다른 장면: 클래스는 겹치지만 임베딩이 전혀 다름
    reg = _registry_with([
        ("bed", [0, 0, 2.0], _emb(11)),
        ("chair", [1, 0, 2.0], _emb(12)),
        ("lamp", [0, 1, 2.0], _emb(13)),
    ])
    assert relocalize(saved, reg) is None


def test_save_load_mesh_state_roundtrip(tmp_path):
    mm = MeshMap(MeshCfg(voxel_size=0.05, trunc_margin=0.15))
    K = np.array([[60.0, 0.0, 16.0], [0.0, 60.0, 12.0], [0.0, 0.0, 1.0]])
    depth = np.full((24, 32), 1.0, np.float32)
    color = np.full((24, 32, 3), 180, np.uint8)
    pose = np.eye(4)
    mm.integrate_views(
        np.stack([depth, depth]),
        np.stack([color, color]),
        np.stack([depth > 0, depth > 0]),
        np.stack([pose, pose]),
        np.stack([K, K]),
        window_ids=[0, 1],
    )

    path = tmp_path / "mesh_state.npz"
    assert save_mesh_state(path, mm) == 1
    loaded = load_mesh_state(path, MeshCfg(voxel_size=0.05, trunc_margin=0.15))

    assert len(loaded.submaps) == 1
    mesh = next(iter(loaded.submaps.values())).mesh
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_merge_mesh_into_session_applies_sim3_to_saved_anchors():
    saved = MeshMap(MeshCfg(voxel_size=0.05, trunc_margin=0.15))
    K = np.array([[60.0, 0.0, 16.0], [0.0, 60.0, 12.0], [0.0, 0.0, 1.0]])
    depth = np.full((24, 32), 1.0, np.float32)
    color = np.full((24, 32, 3), 180, np.uint8)
    pose = np.eye(4)
    saved.integrate_views(
        np.stack([depth, depth]),
        np.stack([color, color]),
        np.stack([depth > 0, depth > 0]),
        np.stack([pose, pose]),
        np.stack([K, K]),
        window_ids=[0, 1],
    )
    original = next(iter(saved.submaps.values()))
    original_local = original.mesh.vertices.copy()
    original_global = original.global_vertices().copy()
    T = (1.1, np.eye(3), np.array([0.2, 0.3, -0.1]))
    current = MeshMap(MeshCfg(voxel_size=0.05, trunc_margin=0.15))

    assert merge_mesh_into_session(saved, T, current) == 1
    merged = next(iter(current.submaps.values()))

    assert np.allclose(merged.mesh.vertices, original_local)
    assert np.allclose(merged.global_vertices(), sim3_apply(T, original_global), atol=1e-6)
