"""Cross-session persistence: save/load the world state and relocalize.

새 세션의 좌표계는 매번 새로 시작되므로, 저장된 지도를 그대로 쓸 수 없다.
저장해 둔 객체들의 외형 임베딩(DINOv2)과 현재 세션에서 관측된 객체들을
매칭해 Sim(3)을 풀고(umeyama), 포인트클라우드 ICP로 정밀화한 뒤 이전
지도·객체를 현재 좌표계로 변환해 융합한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import Sim3, sim3_apply, umeyama_sim3
from .objects import ObjectRegistry, WorldObject
from .worldmap import GlobalMap

_INF = 1e9


@dataclass
class SavedState:
    points: np.ndarray            # (N,3) 지도 (저장 세션 좌표계)
    colors: np.ndarray            # (N,3) uint8
    obj_classes: list[str]
    obj_positions: np.ndarray     # (M,3)
    obj_sizes: np.ndarray         # (M,)
    obj_n_obs: np.ndarray         # (M,)
    obj_embs: np.ndarray          # (M,D) — 행이 0이면 임베딩 없음
    obj_dynamic: np.ndarray       # (M,) bool
    meters_per_unit: float


def save_state(path: str | Path, worldmap: GlobalMap, registry: ObjectRegistry,
               meters_per_unit: float | None) -> int:
    """안정 객체(n_obs>=3)와 지도를 저장. 반환: 저장된 객체 수.

    dynamic 플래그가 있는 객체도 저장한다 — '마지막으로 본 위치를 기억'이
    이 시스템의 계약이고, 대형 가구가 dynamic으로 오판되는 경우도 있다.
    """
    objs = [o for o in registry.objects.values() if o.n_obs >= 3]
    dim = next((len(o.embedding) for o in objs if o.embedding is not None), 384)
    embs = np.zeros((len(objs), dim), np.float32)
    for i, o in enumerate(objs):
        if o.embedding is not None:
            embs[i] = o.embedding
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=worldmap.points, colors=worldmap.colors,
        obj_classes=np.array([o.cls_name for o in objs]),
        obj_positions=np.array([o.position for o in objs]).reshape(-1, 3),
        obj_sizes=np.array([o.size for o in objs]),
        obj_n_obs=np.array([o.n_obs for o in objs]),
        obj_embs=embs,
        obj_dynamic=np.array([o.is_dynamic for o in objs], bool),
        meters_per_unit=np.float64(meters_per_unit or 0.0),
    )
    return len(objs)


def load_state(path: str | Path) -> SavedState | None:
    if not Path(path).exists():
        return None
    d = np.load(path, allow_pickle=False)
    return SavedState(
        points=d["points"], colors=d["colors"],
        obj_classes=[str(c) for c in d["obj_classes"]],
        obj_positions=d["obj_positions"], obj_sizes=d["obj_sizes"],
        obj_n_obs=d["obj_n_obs"], obj_embs=d["obj_embs"],
        obj_dynamic=d["obj_dynamic"],
        meters_per_unit=float(d["meters_per_unit"]),
    )


def relocalize(saved: SavedState, registry: ObjectRegistry,
               min_matches: int = 3, min_cos: float = 0.5
               ) -> tuple[Sim3, list[tuple[int, int]], float] | None:
    """저장 객체 ↔ 현재 객체 임베딩 매칭으로 Sim3(old→new)를 추정.

    반환: (T, [(cur_obj_id, saved_idx)], rms) 또는 실패 시 None.
    """
    cur = [o for o in registry.objects.values()
           if o.n_obs >= 3 and not o.is_dynamic and o.embedding is not None]
    if len(cur) < min_matches:
        return None

    cost = np.full((len(cur), len(saved.obj_classes)), _INF)
    for i, o in enumerate(cur):
        for j, cls in enumerate(saved.obj_classes):
            # dynamic으로 저장된 객체는 위치 신뢰도가 낮아 정렬 기준에서 제외
            if (cls != o.cls_name or not saved.obj_embs[j].any()
                    or saved.obj_dynamic[j]):
                continue
            cos = float(o.embedding @ saved.obj_embs[j])
            if cos >= min_cos:
                cost[i, j] = 1.0 - cos
    rows, cols = linear_sum_assignment(np.minimum(cost, _INF))
    pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < _INF]
    if len(pairs) < min_matches:
        return None

    src = saved.obj_positions[[c for _, c in pairs]]
    dst = np.array([cur[r].position for r, _ in pairs])
    spread = float(np.linalg.norm(dst - dst.mean(0), axis=1).mean())
    if spread < 0.1:  # 매칭점들이 한곳에 몰려 있으면 자세가 부정
        return None
    T = umeyama_sim3(src, dst)
    rms = float(np.sqrt(np.mean(np.sum((sim3_apply(T, src) - dst) ** 2, axis=1))))
    if rms > 0.5 * spread:  # 잔차가 배치 크기에 비해 크면 오매칭
        return None
    return T, [(cur[r].obj_id, c) for r, c in pairs], rms


def icp_refine(T0: Sim3, old_points: np.ndarray, new_points: np.ndarray,
               voxel: float) -> Sim3:
    """객체 매칭으로 얻은 초기 Sim3를 포인트클라우드 ICP(SE3)로 정밀화."""
    import open3d as o3d

    if len(old_points) < 100 or len(new_points) < 100:
        return T0
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(sim3_apply(T0, old_points))
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(new_points)
    result = o3d.pipelines.registration.registration_icp(
        src, dst, max_correspondence_distance=4 * voxel,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    if result.fitness < 0.2:
        return T0
    M = np.asarray(result.transformation)
    s0, R0, t0 = T0
    return s0, M[:3, :3] @ R0, M[:3, :3] @ t0 + M[:3, 3]


def merge_into_session(saved: SavedState, T: Sim3,
                       matches: list[tuple[int, int]],
                       worldmap: GlobalMap, registry: ObjectRegistry,
                       now: float) -> None:
    """정렬된 이전 지도·객체를 현재 세션에 흡수한다."""
    worldmap.fuse(sim3_apply(T, saved.points), saved.colors)

    matched_saved = {c for _, c in matches}
    matched_cur = {r: c for r, c in matches}
    # 매칭된 노드: 현재 노드에 이전 관측 횟수를 승계
    for cur_id, saved_idx in matched_cur.items():
        obj = registry.objects.get(cur_id)
        if obj is not None:
            obj.n_obs += int(saved.obj_n_obs[saved_idx])
    # 매칭 안 된 이전 객체: '기억된 위치'로 새 노드 등록 (이번 세션에서
    # 아직 안 보였을 뿐일 수 있음 — 보이면 외형 re-ID로 다시 잡힌다)
    for j, cls in enumerate(saved.obj_classes):
        if j in matched_saved:
            continue
        pos = sim3_apply(T, saved.obj_positions[j][None])[0]
        emb = saved.obj_embs[j]
        obj = WorldObject(
            registry._next_id, cls, pos, last_seen=0.0,
            size=float(saved.obj_sizes[j] * T[0]),
            n_obs=int(saved.obj_n_obs[j]),
            embedding=emb.copy() if emb.any() else None)
        registry.objects[obj.obj_id] = obj
        registry._next_id += 1
