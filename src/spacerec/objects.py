"""Object 3D localization and the persistent world-object registry.

The registry remembers every recognized object's position in the *global*
frame. Objects that leave the view or get occluded keep their last known
position; when a same-class object reappears near a lost object's position
it is merged back into the same node (simple re-identification).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import ObjectsCfg
from .detect import Detection


def localize_objects(detections: list[Detection], depth: np.ndarray,
                     K: np.ndarray, T_wc: np.ndarray
                     ) -> list[tuple[Detection, np.ndarray]]:
    """Mask-interior depth median -> camera-frame 3D -> world-frame position."""
    results = []
    for det in detections:
        if det.mask is not None and det.mask.any():
            ys, xs = np.nonzero(det.mask)
            z = float(np.median(depth[ys, xs]))
            u, v = float(np.median(xs)), float(np.median(ys))
        else:
            x0, y0, x1, y1 = det.box
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            z = float(depth[int(np.clip(v, 0, depth.shape[0] - 1)),
                            int(np.clip(u, 0, depth.shape[1] - 1))])
        if z <= 1e-6:
            continue
        cam = np.array([(u - K[0, 2]) / K[0, 0] * z,
                        (v - K[1, 2]) / K[1, 1] * z,
                        z])
        world = T_wc[:3, :3] @ cam + T_wc[:3, 3]
        results.append((det, world))
    return results


@dataclass
class WorldObject:
    obj_id: int
    cls_name: str
    position: np.ndarray            # global frame (EMA)
    last_seen: float
    n_obs: int = 1
    is_dynamic: bool = False
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

    def update(self, located: list[tuple[Detection, np.ndarray]], ts: float
               ) -> set[int]:
        """Associate this frame's localized detections; returns visible obj ids."""
        located = self._dedup(located)
        visible: set[int] = set()
        for det, pos in located:
            obj = None
            if det.track_id >= 0 and det.track_id in self._track_to_obj:
                obj = self.objects.get(self._track_to_obj[det.track_id])
                if obj is not None and obj.cls_name != det.cls_name:
                    obj = None  # 트래커 id 재사용으로 클래스가 바뀐 경우
            if obj is None:
                obj = self._reidentify(det, pos, visible)
            if obj is None:
                obj = WorldObject(self._next_id, det.cls_name, pos.copy(), ts)
                self.objects[obj.obj_id] = obj
                self._next_id += 1
            else:
                a = self.cfg.ema_alpha
                obj.position = (1 - a) * obj.position + a * pos
                obj.last_seen = ts
                obj.n_obs += 1
            if det.track_id >= 0:
                self._track_to_obj[det.track_id] = obj.obj_id
            obj.history.append(pos.copy())
            self._update_dynamic(obj, ts)
            visible.add(obj.obj_id)
        return visible

    def _dedup(self, located: list[tuple[Detection, np.ndarray]]
               ) -> list[tuple[Detection, np.ndarray]]:
        """동일 클래스가 거의 같은 3D 위치에 중복 검출되면 conf 높은 쪽만 유지."""
        keep: list[tuple[Detection, np.ndarray]] = []
        for det, pos in sorted(located, key=lambda x: -x[0].conf):
            radius = self.cfg.merge_radius * 0.4
            if any(det.cls_name == k.cls_name
                   and np.linalg.norm(pos - kp) < radius for k, kp in keep):
                continue
            keep.append((det, pos))
        return keep

    def stable_objects(self, now: float) -> list[WorldObject]:
        """시각화/그래프용: 1회성 오검출을 걸러낸 객체 목록."""
        return [o for o in self.objects.values()
                if o.n_obs >= 3 or now - o.last_seen < 1.0]

    def _reidentify(self, det: Detection, pos: np.ndarray,
                    claimed: set[int]) -> WorldObject | None:
        """Same class + within merge radius -> same world object."""
        best, best_d = None, self.cfg.merge_radius
        for obj in self.objects.values():
            if obj.obj_id in claimed or obj.cls_name != det.cls_name:
                continue
            d = float(np.linalg.norm(obj.position - pos))
            if d < best_d:
                best, best_d = obj, d
        return best

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
