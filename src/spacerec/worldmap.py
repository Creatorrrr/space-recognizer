"""Global static map: evidence-weighted voxel hash + live→global Sim3 correction.

단순 누적(append-only)이 아니라 증거 기반으로 갱신된다:
- 관측된 표면은 voxel 가중치를 올리고 (상한 있음),
- 새 관측의 시선이 기존 voxel을 '관통'하면(그 너머가 보이면) 빈 공간 증거로
  가중치를 깎아, 0이 되면 제거한다 (free-space carving).

이로써 잘못 재구성된 옛 표면은 다시 촬영하면 지워지고, 반대로 단발성 불량
관측이 좋은 지도를 바꾸려면 반복 증거가 필요해 양방향으로 견고하다.
"""

from __future__ import annotations

import numpy as np

from .config import BackendCfg
from .geometry import SIM3_IDENTITY, Sim3, sim3_apply, sim3_interp, sim3_on_pose

_MAX_W = 6.0        # voxel당 증거 가중치 상한 (오류가 과도하게 굳지 않게)
_CARVE = 1.0        # 시선 관통 1회당 감점
_CARVE_CAP = 3      # 한 윈도에서 같은 voxel에 줄 수 있는 최대 감점 횟수
_RAY_STRIDE = 4     # carving에 쓸 광선 서브샘플링
_MAX_STEPS = 64     # 광선당 샘플 수 상한
_OFF = 1 << 20      # voxel 좌표 패킹 오프셋


class GlobalMap:
    def __init__(self, cfg: BackendCfg):
        self.cfg = cfg
        self._keys = np.empty(0, np.int64)       # 정렬 유지
        self._weight = np.empty(0, np.float32)
        self._psum = np.empty((0, 3), np.float64)
        self._csum = np.empty((0, 3), np.float64)
        self._cnt = np.empty(0, np.int64)
        self.points = np.empty((0, 3), np.float32)
        self.colors = np.empty((0, 3), np.uint8)
        self._carve_pass = 0
        # live VO frame -> global map frame correction
        self._T_gl_current: Sim3 = SIM3_IDENTITY
        self._T_gl_target: Sim3 = SIM3_IDENTITY

    # ---- voxel hashing -------------------------------------------------
    def _quantize(self, pts: np.ndarray) -> np.ndarray:
        q = np.floor(pts / self.cfg.voxel_size).astype(np.int64)
        np.clip(q, -_OFF + 1, _OFF - 1, out=q)
        return ((q[:, 0] + _OFF) << 42) | ((q[:, 1] + _OFF) << 21) | (q[:, 2] + _OFF)

    # ---- point fusion --------------------------------------------------
    def fuse(self, points: np.ndarray, colors: np.ndarray,
             origins: np.ndarray | None = None,
             view_idx: np.ndarray | None = None,
             weight: float = 1.0) -> None:
        """관측 포인트를 융합하고, 시선 정보가 있으면 free-space carving 수행."""
        points = np.asarray(points, np.float64).reshape(-1, 3)
        if len(points):
            colors = np.asarray(colors, np.float64).reshape(-1, 3)
            new_keys = self._quantize(points)
            uk, inv, counts = np.unique(new_keys, return_inverse=True,
                                        return_counts=True)
            psum = np.zeros((len(uk), 3))
            csum = np.zeros((len(uk), 3))
            np.add.at(psum, inv, points)
            np.add.at(csum, inv, colors)
            wnew = np.minimum(counts * weight, _MAX_W).astype(np.float32)

            merged, minv = np.unique(np.concatenate([self._keys, uk]),
                                     return_inverse=True)
            n_old = len(self._keys)
            w = np.zeros(len(merged), np.float32)
            ps = np.zeros((len(merged), 3))
            cs = np.zeros((len(merged), 3))
            ct = np.zeros(len(merged), np.int64)
            w[minv[:n_old]] = self._weight
            ps[minv[:n_old]] = self._psum
            cs[minv[:n_old]] = self._csum
            ct[minv[:n_old]] = self._cnt
            idx_new = minv[n_old:]
            w[idx_new] = np.minimum(w[idx_new] + wnew, _MAX_W)
            ps[idx_new] += psum
            cs[idx_new] += csum
            ct[idx_new] += counts
            self._keys, self._weight = merged, w
            self._psum, self._csum, self._cnt = ps, cs, ct

        if origins is not None and view_idx is not None and len(points):
            self._carve(points, view_idx, np.asarray(origins, np.float64))

        self._enforce_cap()
        self._materialize()

    def _carve(self, points: np.ndarray, view_idx: np.ndarray,
               origins: np.ndarray) -> None:
        """각 시선(origin→측정점)의 중간 구간을 빈 공간 증거로 사용.

        샘플 간격을 voxel 크기에 맞추고, 호출마다 광선 선택과 샘플 위상을
        바꿔(지터) 여러 패스에 걸쳐 커버리지가 누적되게 한다 — 간격이 성기면
        잘못된 표면 voxel이 광선 사이로 빠져나가 영영 안 지워진다.
        """
        self._carve_pass += 1
        rng = np.random.default_rng(self._carve_pass)
        samples = []
        for v in range(len(origins)):
            offset = (self._carve_pass + v) % _RAY_STRIDE
            pts = points[view_idx == v][offset::_RAY_STRIDE]
            if len(pts) == 0:
                continue
            med_len = float(np.median(np.linalg.norm(pts - origins[v], axis=1)))
            steps = int(np.clip(0.75 * med_len / self.cfg.voxel_size,
                                8, _MAX_STEPS))
            fr = np.linspace(0.1, 0.85, steps) + rng.uniform(0, 0.75 / steps)
            fr = fr[fr < 0.88]
            seg = origins[v] + (pts - origins[v])[:, None, :] * fr[None, :, None]
            samples.append(seg.reshape(-1, 3))
        if not samples:
            return
        keys = self._quantize(np.concatenate(samples))
        uk, counts = np.unique(keys, return_counts=True)
        pos = np.searchsorted(self._keys, uk)
        pos_c = np.minimum(pos, len(self._keys) - 1)
        hit = (len(self._keys) > 0) & (self._keys[pos_c] == uk)
        idx = pos_c[hit]
        self._weight[idx] -= _CARVE * np.minimum(counts[hit], _CARVE_CAP)
        keep = self._weight > 0
        if not keep.all():
            self._filter(keep)

    def _filter(self, keep: np.ndarray) -> None:
        self._keys = self._keys[keep]
        self._weight = self._weight[keep]
        self._psum = self._psum[keep]
        self._csum = self._csum[keep]
        self._cnt = self._cnt[keep]

    def _enforce_cap(self) -> None:
        if len(self._keys) > self.cfg.max_points:
            # 증거가 약한 voxel부터 버린다
            order = np.argpartition(self._weight, len(self._weight)
                                    - self.cfg.max_points)
            keep = np.zeros(len(self._keys), bool)
            keep[order[-self.cfg.max_points:]] = True
            self._filter(keep)

    def _materialize(self) -> None:
        cnt = np.maximum(self._cnt, 1)[:, None]
        self.points = (self._psum / cnt).astype(np.float32)
        self.colors = np.clip(self._csum / cnt, 0, 255).astype(np.uint8)

    # ---- live -> global correction ------------------------------------
    def set_correction_target(self, T: Sim3) -> None:
        self._T_gl_target = T

    def step_correction(self, alpha: float = 0.2) -> None:
        """Called once per live frame: ease toward the target so object/camera
        positions never teleport when the backend re-anchors the map."""
        self._T_gl_current = sim3_interp(self._T_gl_current, self._T_gl_target, alpha)

    @property
    def T_global_live(self) -> Sim3:
        return self._T_gl_current

    def to_global_points(self, pts_live: np.ndarray) -> np.ndarray:
        return sim3_apply(self._T_gl_current, pts_live)

    def to_global_pose(self, T_wc_live: np.ndarray) -> np.ndarray:
        return sim3_on_pose(self._T_gl_current, T_wc_live)
