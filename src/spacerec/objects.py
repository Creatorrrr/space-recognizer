"""Object 3D localization and the persistent world-object registry.

The registry remembers every recognized object's position in the *global*
frame. Objects that leave the view or get occluded keep their last known
position; when an object reappears it is matched back to its node by
**position + appearance** (Hungarian assignment with per-object gates).

위치 단서만 쓰면 같은 클래스의 이웃 물체(한 방의 침대 두 대)를 구분할 수
없고, 잘못 병합된 노드를 EMA가 끌고 가며 오류가 전파된다 — 실측으로 확인된
실패 모드라서 외형 임베딩과 전역 매칭을 함께 사용한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import ObjectsCfg
from .detect import Detection

_INF = 1e9


@dataclass
class Observation:
    det: Detection
    position: np.ndarray          # global frame
    size: float                   # 대략적 3D 지름 (장면 단위)
    emb: np.ndarray | None = None  # 외형 임베딩 (L2 정규화, AppearanceEmbedder)


def _robust_depth_sample(depth: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                         conf: np.ndarray | None = None) -> float | None:
    vals = depth[ys, xs]
    valid = np.isfinite(vals) & (vals > 1e-6)
    if conf is not None:
        valid &= conf[ys, xs] > 0
    vals = vals[valid]
    if len(vals) == 0:
        return None
    if len(vals) >= 20:
        lo, hi = np.percentile(vals, [10, 90])
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        if len(trimmed):
            vals = trimmed
    return float(np.median(vals))


def localize_objects(detections: list[Detection], depth: np.ndarray | None,
                     K: np.ndarray, T_wc: np.ndarray,
                     conf: np.ndarray | None = None) -> list[Observation]:
    """Mask-interior robust depth -> camera-frame 3D -> world-frame position."""
    if depth is None:
        return []
    results = []
    for det in detections:
        if det.mask is not None and det.mask.any():
            mask = det.mask
            if mask.sum() >= 25:
                eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                   iterations=1).astype(bool)
                if eroded.any():
                    mask = eroded
            ys, xs = np.nonzero(mask)
            z = _robust_depth_sample(depth, ys, xs, conf)
            if z is None:
                continue
            u, v = float(np.median(xs)), float(np.median(ys))
        else:
            x0, y0, x1, y1 = det.box
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            ui = int(np.clip(u, 0, depth.shape[1] - 1))
            vi = int(np.clip(v, 0, depth.shape[0] - 1))
            y0p, y1p = max(0, vi - 2), min(depth.shape[0], vi + 3)
            x0p, x1p = max(0, ui - 2), min(depth.shape[1], ui + 3)
            yy, xx = np.mgrid[y0p:y1p, x0p:x1p]
            z = _robust_depth_sample(depth, yy.ravel(), xx.ravel(), conf)
            if z is None:
                continue
        if z <= 1e-6:
            continue
        cam = np.array([(u - K[0, 2]) / K[0, 0] * z,
                        (v - K[1, 2]) / K[1, 1] * z,
                        z])
        world = T_wc[:3, :3] @ cam + T_wc[:3, 3]
        x0, y0, x1, y1 = det.box
        size = float(np.hypot(x1 - x0, y1 - y0) * z / K[0, 0])
        results.append(Observation(det=det, position=world, size=size))
    return results


@dataclass
class WorldObject:
    obj_id: int
    cls_name: str
    position: np.ndarray            # global frame (EMA)
    last_seen: float
    size: float = 0.3               # 3D 지름 (EMA)
    n_obs: int = 1
    is_dynamic: bool = False
    embedding: np.ndarray | None = None
    miss_count: int = 0  # '보여야 하는데 안 보임' 연속 증거 수
    history: deque = field(default_factory=lambda: deque(maxlen=12))
    trajectory: list = field(default_factory=list)  # (ts, pos) — dynamic 객체용

    @property
    def label(self) -> str:
        flag = "~" if self.is_dynamic else ""
        return f"{flag}{self.cls_name}#{self.obj_id}"


class ObjectRegistry:
    def __init__(self, cfg: ObjectsCfg):
        self.cfg = cfg
        self.objects: dict[int, WorldObject] = {}
        self._track_to_obj: dict[int, int] = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    def update(self, observations: list[Observation], ts: float) -> set[int]:
        """Associate this frame's localized detections; returns visible obj ids."""
        observations = self._dedup(observations)
        visible: set[int] = set()

        # 1) 트래커 id 지름길: 직전 프레임부터 이어지는 추적은 신뢰하되,
        #    위치 게이트로 트래커의 id 스위치 사고는 걸러낸다.
        pending: list[Observation] = []
        for obs in observations:
            obj = None
            if obs.det.track_id >= 0 and obs.det.track_id in self._track_to_obj:
                cand = self.objects.get(self._track_to_obj[obs.det.track_id])
                if cand is not None and cand.cls_name == obs.det.cls_name:
                    # 끊김 없이 이어지는 추적은 거리와 무관하게 신뢰한다
                    # (움직이는 물체는 EMA 위치가 뒤처져 게이트를 벗어난다).
                    # 추적이 한동안 끊겼다 같은 id가 오면 게이트로 재검증.
                    gap = ts - cand.last_seen
                    if (gap < 1.5
                            or np.linalg.norm(cand.position - obs.position)
                            < 1.5 * self._gate(cand)):
                        obj = cand
            if obj is not None and obj.obj_id not in visible:
                self._apply(obj, obs, ts)
                visible.add(obj.obj_id)
            else:
                pending.append(obs)

        # 2) 나머지는 헝가리안 전역 매칭 (위치 + 외형 비용)
        candidates = [o for o in self.objects.values() if o.obj_id not in visible]
        if pending and candidates:
            cost = np.full((len(pending), len(candidates)), _INF)
            for i, obs in enumerate(pending):
                for j, obj in enumerate(candidates):
                    cost[i, j] = self._match_cost(obs, obj)
            rows, cols = linear_sum_assignment(np.minimum(cost, _INF))
            matched_idx = set()
            for r, c in zip(rows, cols):
                if cost[r, c] >= _INF:
                    continue
                obj = candidates[c]
                gap = ts - obj.last_seen
                if gap > 1.0:
                    print(f"[reid] {obj.label} 재획득 (공백 {gap:.1f}s, "
                          f"cost={cost[r, c]:.2f})")
                self._apply(obj, pending[r], ts)
                visible.add(obj.obj_id)
                if pending[r].det.track_id >= 0:
                    self._track_to_obj[pending[r].det.track_id] = obj.obj_id
                matched_idx.add(r)
            pending = [obs for i, obs in enumerate(pending) if i not in matched_idx]

        # 3) 매칭되지 않은 검출 -> 새 객체
        for obs in pending:
            obj = WorldObject(self._next_id, obs.det.cls_name,
                              obs.position.copy(), ts, size=obs.size,
                              embedding=None if obs.emb is None else obs.emb.copy())
            self.objects[obj.obj_id] = obj
            self._next_id += 1
            if obs.det.track_id >= 0:
                self._track_to_obj[obs.det.track_id] = obj.obj_id
            obj.history.append(obs.position.copy())
            visible.add(obj.obj_id)
            print(f"[obj] new {obj.label} size={obj.size:.2f}")

        self._prune(ts)
        return visible

    # ------------------------------------------------------------------
    def _gate(self, obj: WorldObject) -> float:
        """물체 크기에 비례하는 연관 반경 (큰 가구일수록 위치 추정이 출렁임)."""
        return float(np.clip(0.8 * obj.size, 0.15, self.cfg.merge_radius))

    def _match_cost(self, obs: Observation, obj: WorldObject) -> float:
        if obj.cls_name != obs.det.cls_name:
            return _INF
        gate = self._gate(obj)
        dist = float(np.linalg.norm(obj.position - obs.position))
        if dist > gate:
            return _INF
        cost = dist / gate
        if obs.emb is not None and obj.embedding is not None:
            cos = float(obs.emb @ obj.embedding)
            if cos < self.cfg.app_gate:
                return _INF  # 외형이 다르면 위치가 겹쳐도 다른 물체
            cost += self.cfg.app_weight * (1.0 - cos)
        return cost

    def _apply(self, obj: WorldObject, obs: Observation, ts: float) -> None:
        # 동적 물체는 EMA 지연을 줄여 실제 위치를 빠르게 따라가게 한다
        a = 0.6 if obj.is_dynamic else self.cfg.ema_alpha
        obj.position = (1 - a) * obj.position + a * obs.position
        obj.size = (1 - a) * obj.size + a * obs.size
        if obs.emb is not None:
            if obj.embedding is None:
                obj.embedding = obs.emb.copy()
            else:
                obj.embedding = 0.8 * obj.embedding + 0.2 * obs.emb
                obj.embedding /= np.linalg.norm(obj.embedding) + 1e-9
        obj.last_seen = ts
        obj.n_obs += 1
        obj.miss_count = 0
        obj.history.append(obs.position.copy())
        self._update_dynamic(obj, ts)

    def _dedup(self, observations: list[Observation]) -> list[Observation]:
        """동일 클래스가 거의 같은 3D 위치에 중복 검출되면 conf 높은 쪽만 유지."""
        keep: list[Observation] = []
        for obs in sorted(observations, key=lambda o: -o.det.conf):
            radius = max(0.15, 0.5 * obs.size)
            if any(obs.det.cls_name == k.det.cls_name
                   and np.linalg.norm(obs.position - k.position) < radius
                   for k in keep):
                continue
            keep.append(obs)
        return keep

    def _prune(self, ts: float) -> None:
        """한두 번 관측되고 다시는 확인되지 않는 노드(불량 검출)는 제거."""
        stale = [oid for oid, o in self.objects.items()
                 if o.n_obs <= 2 and ts - o.last_seen > 8.0]
        for oid in stale:
            del self.objects[oid]
            self._track_to_obj = {t: i for t, i in self._track_to_obj.items()
                                  if i != oid}

    def decay_absent(self, visible: set[int],
                     observations: list[Observation],
                     positions_live: dict[int, np.ndarray],
                     T_wc_live: np.ndarray, K: np.ndarray,
                     depth: np.ndarray, limit: int) -> None:
        """부재 증거 처리: 노드가 카메라 시야 중앙에 있고 가려지지도 않았는데
        해당 클래스가 검출되지 않는 상황이 누적되면 제거한다.

        잘못된 클래스로 박제된 유령 노드와 실제로 치워진 물체가 모두 이
        경로로 정리된다. 동적 물체는 '마지막 위치 기억' 계약을 지키기 위해
        제외한다.
        """
        h, w = depth.shape
        R, t = T_wc_live[:3, :3], T_wc_live[:3, 3]
        for obj in list(self.objects.values()):
            if obj.obj_id in visible or obj.is_dynamic:
                continue
            # 같은 클래스 검출이 근처에 있으면 연관 실패일 수 있으므로 패스
            if any(o.det.cls_name == obj.cls_name
                   and np.linalg.norm(o.position - obj.position)
                   < 2 * self._gate(obj) for o in observations):
                continue
            p_live = positions_live.get(obj.obj_id)
            if p_live is None:
                continue
            cam = R.T @ (p_live - t)
            if cam[2] < 0.3:
                continue
            u = K[0, 0] * cam[0] / cam[2] + K[0, 2]
            v = K[1, 1] * cam[1] / cam[2] + K[1, 2]
            mx, my = 0.1 * w, 0.1 * h  # 화면 가장자리는 판정 제외
            if not (mx < u < w - mx and my < v < h - my):
                continue
            ui, vi = int(u), int(v)
            patch = depth[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
            if patch.size == 0:
                continue
            z_meas = float(np.median(patch))
            # 측정 depth가 노드 위치보다 멀다 = 그 자리가 비어 있는 게 보인다
            if z_meas > cam[2] - max(0.1, 0.3 * obj.size):
                obj.miss_count += 1
                if obj.miss_count >= limit:
                    print(f"[obj] {obj.label} 제거 — 부재 증거 {limit}회 "
                          f"(시야 안·비가림인데 미검출)")
                    del self.objects[obj.obj_id]
                    self._track_to_obj = {
                        tid: oid for tid, oid in self._track_to_obj.items()
                        if oid != obj.obj_id}

    def stable_objects(self, now: float) -> list[WorldObject]:
        """시각화/그래프용: 1회성 오검출을 걸러낸 객체 목록."""
        return [o for o in self.objects.values()
                if o.n_obs >= 3 or now - o.last_seen < 1.0]

    def _update_dynamic(self, obj: WorldObject, ts: float) -> None:
        # 측정 노이즈(분산)가 아니라 방향성 있는 순변위로 움직임을 판정한다.
        # 최근 관측 6개만 보는 이유: 카메라가 궤도를 돌면 정적 물체도 mask 중심이
        # 시점에 따라 서서히 밀리는데, 긴 구간 누적 변위는 이를 움직임으로 오판한다.
        if len(obj.history) >= 6:
            h = np.array(obj.history)[-6:]
            displacement = float(np.linalg.norm(h[-2:].mean(0) - h[:2].mean(0)))
            if displacement > self.cfg.dynamic_var_thresh:
                obj.is_dynamic = True
        if obj.is_dynamic:
            obj.trajectory.append((ts, obj.position.copy()))
