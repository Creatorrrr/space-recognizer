"""루프 클로저: place recognition + 상대 Sim3 검증 + pose graph 최적화.

(docs/upgrade-plan.md Tier 4) 장시간 사용 시 VO drift가 누적되는 문제를,
같은 장소 재방문을 감지해 보정한다.

설계 메모 — 루프 검증은 DA3 멀티뷰가 아니라 **ORB 매칭 + 양쪽 키프레임
depth의 3D-3D RANSAC Umeyama**를 쓴다: DA3 pose 헤드는 병진을 과소추정해
신뢰할 수 없고(benchmarks.md Phase 3 / Tier 2), 3D-3D 방식은 monocular
drift의 스케일 성분까지 Sim3로 직접 측정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .geometry import Sim3, sim3_compose, sim3_inverse, umeyama_sim3

_ORB = None  # 프로세스당 1회 생성


@dataclass
class LoopEdge:
    i: int                # 과거 키프레임 노드 인덱스
    j: int                # 현재 키프레임 노드 인덱스
    T_ij: Sim3            # 측정: j-카메라 좌표 → i-카메라 좌표 (relative Sim3)
    inliers: int = 0


# ---------------------------------------------------------------------------
# 1) place recognition
# ---------------------------------------------------------------------------

class LoopDetector:
    """키프레임 전역 임베딩 저장 + 시간 갭을 둔 최근접 후보 탐색."""

    def __init__(self, sim_thresh: float = 0.62, min_gap_s: float = 10.0,
                 max_store: int = 600):
        self.sim_thresh = sim_thresh
        self.min_gap_s = min_gap_s
        self.max_store = max_store
        self._ids: list[int] = []
        self._ts: list[float] = []
        self._embs: list[np.ndarray] = []

    def add(self, kf_id: int, ts: float, emb: np.ndarray) -> None:
        self._ids.append(kf_id)
        self._ts.append(ts)
        self._embs.append(emb)
        if len(self._ids) > self.max_store:
            self._ids.pop(0)
            self._ts.pop(0)
            self._embs.pop(0)

    def query(self, ts: float, emb: np.ndarray) -> tuple[int, float] | None:
        """시간 갭 조건을 만족하는 저장 키프레임 중 최고 유사도 후보."""
        best_id, best_sim = -1, self.sim_thresh
        for kid, kts, kemb in zip(self._ids, self._ts, self._embs):
            if ts - kts < self.min_gap_s:
                continue
            sim = float(emb @ kemb)
            if sim > best_sim:
                best_id, best_sim = kid, sim
        return (best_id, best_sim) if best_id >= 0 else None


# ---------------------------------------------------------------------------
# 2) 상대 Sim3 검증
# ---------------------------------------------------------------------------

def sim3_from_matches(pts_a: np.ndarray, pts_b: np.ndarray,
                      inlier_dist: float = 0.05, iters: int = 200,
                      min_inliers: int = 12, rng_seed: int = 0,
                      diag: dict | None = None
                      ) -> tuple[Sim3, np.ndarray] | None:
    """대응점 (N,3)x2에서 RANSAC Umeyama로 b→a Sim3 추정.

    반환: (T_ab: pts_b를 pts_a 좌표로 보내는 Sim3, inlier mask) 또는 None.
    diag를 주면 기각 사유 진단 정보를 채운다: n(매칭 수),
    med_scale(샘플 스케일 중앙값 — 게이트 밖이면 스케일 drift가 원인),
    best_inl(최대 인라이어 수 — 적으면 기하 불일치가 원인).
    """
    n = len(pts_a)
    if diag is not None:
        diag.update(n=n, med_scale=float("nan"), best_inl=0)
    if n < max(4, min_inliers):
        return None
    # 인라이어 임계값은 depth 비례 — mono depth의 3D 오차는 z에 비례해
    # 커지므로 고정 임계값은 원거리 매칭을 부당하게 떨어뜨린다 (desk/office
    # 실측: 인라이어가 문턱 직전 탈락). 근거리는 inlier_dist 바닥 유지.
    thr = inlier_dist * np.maximum(1.0, np.abs(pts_a[:, 2]))
    rng = np.random.default_rng(rng_seed)
    best_mask = None
    scales = []
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            T = umeyama_sim3(pts_b[idx], pts_a[idx])
        except np.linalg.LinAlgError:
            continue
        scales.append(T[0])
        if not (0.2 < T[0] < 5.0):   # 비상식적 스케일은 기각
            continue
        d = np.linalg.norm(pts_a - (T[0] * pts_b @ T[1].T + T[2]), axis=1)
        mask = d < thr
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask
    if diag is not None and scales:
        diag["med_scale"] = float(np.median(scales))
        diag["best_inl"] = int(best_mask.sum()) if best_mask is not None else 0
    if best_mask is None or best_mask.sum() < min_inliers:
        return None
    T = umeyama_sim3(pts_b[best_mask], pts_a[best_mask])
    d = np.linalg.norm(pts_a - (T[0] * pts_b @ T[1].T + T[2]), axis=1)
    mask = d < thr
    if mask.sum() < min_inliers or not (0.2 < T[0] < 5.0):
        return None
    return T, mask


def match_3d3d(rgb_a: np.ndarray, depth_a: np.ndarray, K_a: np.ndarray,
               rgb_b: np.ndarray, depth_b: np.ndarray, K_b: np.ndarray,
               max_features: int = 2500, ratio: float = 0.75
               ) -> tuple[np.ndarray, np.ndarray]:
    """ORB 매칭 → 각 프레임의 depth로 카메라 좌표 3D 대응점 추출."""
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(max_features)
    gray_a = rgb_a if rgb_a.ndim == 2 else cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY)
    gray_b = rgb_b if rgb_b.ndim == 2 else cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
    kp_a, des_a = _ORB.detectAndCompute(gray_a, None)
    kp_b, des_b = _ORB.detectAndCompute(gray_b, None)
    empty = (np.empty((0, 3)), np.empty((0, 3)))
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return empty
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_a, des_b, k=2)
    pts_a, pts_b = [], []
    for m_pair in pairs:
        if len(m_pair) < 2 or m_pair[0].distance > ratio * m_pair[1].distance:
            continue
        m = m_pair[0]
        ua, va = kp_a[m.queryIdx].pt
        ub, vb = kp_b[m.trainIdx].pt
        za = depth_a[int(va), int(ua)]
        zb = depth_b[int(vb), int(ub)]
        if za <= 1e-6 or zb <= 1e-6:
            continue
        pts_a.append([(ua - K_a[0, 2]) / K_a[0, 0] * za,
                      (va - K_a[1, 2]) / K_a[1, 1] * za, za])
        pts_b.append([(ub - K_b[0, 2]) / K_b[0, 0] * zb,
                      (vb - K_b[1, 2]) / K_b[1, 1] * zb, zb])
    if not pts_a:
        return empty
    return np.array(pts_a), np.array(pts_b)


# ---------------------------------------------------------------------------
# 3) Sim3 pose graph 최적화
# ---------------------------------------------------------------------------

def _node_to_sim3(x: np.ndarray) -> Sim3:
    return (float(np.exp(x[6])), Rotation.from_rotvec(x[:3]).as_matrix(),
            x[3:6].copy())


def _sim3_log(T: Sim3) -> np.ndarray:
    s, R, t = T
    return np.concatenate([Rotation.from_matrix(R).as_rotvec(), t,
                           [np.log(max(s, 1e-12))]])


def optimize_pose_graph(poses: list[np.ndarray],
                        edges: list[tuple[int, int, Sim3, object]],
                        max_nfev: int = 60) -> list[Sim3]:
    """Sim3 pose graph LM 최적화.

    poses: 노드별 초기 cam-to-world SE3 (4x4). 노드 0은 고정 (게이지).
    edges: (i, j, Z_ij, weight) — Z_ij는 j-카메라 좌표를 i-카메라 좌표로
           보내는 측정 Sim3 (순차 엣지 = 초기 pose에서 유도·scale 1,
           루프 엣지 = 3D-3D 검증 결과). weight는 스칼라 또는 (7,) 벡터
           [rot×3, trans×3, log_scale] — 성분별 신뢰도를 따로 줄 수 있다.
           스케일 측정은 mono depth 비율이라 루프 엣지에서 노이즈가 크므로,
           순차 엣지(스케일=1)는 뻣뻣하게, 루프 엣지는 보통으로 주는 것이
           기본 설계다 (sequential_edges/_close_loops 참고).
    반환: 노드별 보정된 cam-to-world Sim3 (s, R, t). scale은 그 노드 주변
          지도의 스케일 보정량으로 해석한다.
    """
    n = len(poses)
    if n < 2 or not edges:
        return [(1.0, P[:3, :3].copy(), P[:3, 3].copy()) for P in poses]

    x0 = np.zeros(7 * (n - 1))
    for k in range(1, n):
        P = poses[k]
        x0[7 * (k - 1):7 * k] = _sim3_log((1.0, P[:3, :3], P[:3, 3]))
    node0: Sim3 = (1.0, poses[0][:3, :3].copy(), poses[0][:3, 3].copy())
    weights = [np.asarray(ew, np.float64) * np.ones(7)
               for _, _, _, ew in edges]

    def decode(x: np.ndarray, k: int) -> Sim3:
        if k == 0:
            return node0
        return _node_to_sim3(x[7 * (k - 1):7 * k])

    def fun(x: np.ndarray) -> np.ndarray:
        res = np.empty(7 * len(edges))
        for e, (i, j, Z, _) in enumerate(edges):
            pred = sim3_compose(sim3_inverse(decode(x, i)), decode(x, j))
            err = sim3_compose(sim3_inverse(Z), pred)
            res[7 * e:7 * e + 7] = _sim3_log(err) * weights[e]
        return res

    sparsity = lil_matrix((7 * len(edges), 7 * (n - 1)), dtype=np.int8)
    for e, (i, j, _, _) in enumerate(edges):
        for k in (i, j):
            if k > 0:
                sparsity[7 * e:7 * e + 7, 7 * (k - 1):7 * k] = 1

    sol = least_squares(fun, x0, jac_sparsity=sparsity, method="trf",
                        max_nfev=max_nfev, verbose=0)
    return [decode(sol.x, k) for k in range(n)]


_SEQ_W = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0])  # 체인 스케일 강성


def sequential_edges(poses: list[np.ndarray]
                     ) -> list[tuple[int, int, Sim3, object]]:
    """인접 노드 간 측정 엣지 — 초기(VO) 상대 pose, scale 1 (스케일 뻣뻣)."""
    edges = []
    for k in range(len(poses) - 1):
        A: Sim3 = (1.0, poses[k][:3, :3], poses[k][:3, 3])
        B: Sim3 = (1.0, poses[k + 1][:3, :3], poses[k + 1][:3, 3])
        edges.append((k, k + 1, sim3_compose(sim3_inverse(A), B), _SEQ_W))
    return edges
