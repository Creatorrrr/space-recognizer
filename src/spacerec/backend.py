"""5-second-cadence multi-view reconstruction backend (DA3 any-view).

Runs in its own *process*: PyTorch MPS is not thread-safe — concurrent use
of one Metal command queue from two threads crashes with an
IOGPUMetalCommandBuffer assertion. A separate process gets its own Metal
context, so live inference and backend inference can overlap safely.

Every period the worker takes a window of keyframes — a few
already-reconstructed ones for overlap plus the newest ones — runs DA3
multi-view inference (poses + depth + intrinsics), aligns the window into
the global map frame with a Sim(3), fuses static points into the map, and
re-estimates (a) the live→global correction and (b) the live mono-depth
affine calibration.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import floor as floor_mod
from .calib import DepthCalibration, fit_affine_depth
from .config import BackendCfg
from .geometry import (SIM3_IDENTITY, Sim3, sim3_apply, sim3_compose,
                       sim3_interp, sim3_inverse, sim3_on_pose, umeyama_sim3)


_ATT_STEP_RAD = float(np.radians(2.0))
_ATT_DEADBAND_RAD = float(np.radians(1.0))


@dataclass
class BackendKeyframe:
    kf_id: int
    ts: float
    rgb: np.ndarray                 # processing-res RGB (backend input)
    T_wc_live: np.ndarray           # VO pose (live frame)
    raw_depth: np.ndarray | None    # uncalibrated live mono depth (small)
    dyn_mask: np.ndarray | None     # dynamic-object mask (small, bool)
    calib_ab: tuple[float, float] = (1.0, 0.0)  # 생성 시점의 mono 보정 계수
    K: np.ndarray | None = None     # VO 고정 intrinsics (rgb 해상도 기준, 3x3)
    emb: np.ndarray | None = None   # DINOv2 전역 임베딩 (루프 클로저용)


@dataclass
class BackendResult:
    points: np.ndarray              # new global-frame points (N,3)
    colors: np.ndarray              # (N,3) uint8
    T_global_live: Sim3
    calib: DepthCalibration
    kf_global_poses: dict[int, np.ndarray]
    kf_ts: dict[int, float] | None = None
    intrinsics: np.ndarray | None = None  # DA3 추정 K (process 해상도 기준)
    depth_size: tuple[int, int] = (0, 0)  # (w, h) of the K's reference image
    meters_per_unit: float | None = None  # metric 앵커 환산 계수 (표시용)
    view_origins: np.ndarray | None = None  # (V,3) 각 뷰의 카메라 원점 (전역)
    point_view_idx: np.ndarray | None = None  # (N,) 각 포인트가 나온 뷰 번호
    window_ids: list[int] = field(default_factory=list)
    runtime_s: float = 0.0
    pose_conditioned: bool = False  # 이번 윈도가 pose 조건화로 추론됐는지
    win_alpha: float = 1.0          # 멀티뷰→라이브 스케일 정합 계수 (모니터링용:
    win_beta: float = 0.0           #  pose 조건화 시 항등에 수렴해야 정상)
    epoch: int = 0                  # 이번 윈도의 지도 epoch (voxel 출처 추적)
    # 스케일 서보 이득 — main이 calib·VO 상태·T_global_live에 일관 적용해야
    # 한다 (calib만 키우면 frame_scale 피드백이 1/g로 즉시 상쇄함, 실측)
    servo_gain_g: float = 1.0
    # 루프 클로저가 수락된 경우: epoch별 지도 보정 Sim3 + 로그 문자열
    loop_corrections: dict[int, Sim3] | None = None
    loop_log: str = ""
    loop_converging: bool = False


def robust_sim3(src_poses: list[np.ndarray], dst_poses: list[np.ndarray]) -> Sim3:
    """Sim3 mapping src camera poses onto dst camera poses.

    Uses Umeyama on camera centers when they span enough space; otherwise
    falls back to anchoring on the last pose (scale from pairwise distances
    when at least two poses exist).
    """
    src_c = np.array([T[:3, 3] for T in src_poses])
    dst_c = np.array([T[:3, 3] for T in dst_poses])
    spread = float(np.linalg.norm(src_c - src_c.mean(0), axis=1).mean())
    if len(src_poses) >= 3 and spread > 1e-4:
        return umeyama_sim3(src_c, dst_c)

    s = 1.0
    if len(src_poses) >= 2:
        d_src = np.linalg.norm(src_c[-1] - src_c[0])
        d_dst = np.linalg.norm(dst_c[-1] - dst_c[0])
        # 베이스라인이 노이즈 수준이면 두 미세 거리의 비율이 그대로 스케일이
        # 되어 폭주한다 (실측 9.8x — 카메라 정지 구간). 충분한 거리에서만
        # 추정하고 비율도 상식 범위로 클램프.
        if d_src > 1e-3 and d_dst > 1e-3:
            s = float(np.clip(d_dst / d_src, 0.5, 2.0))
    R = dst_poses[-1][:3, :3] @ src_poses[-1][:3, :3].T
    t = dst_c[-1] - s * R @ src_c[-1]
    return s, R, t


def servo_gain(mpu: float, mpu_ref: float, k: float = 0.25,
               step: float = 0.05) -> float:
    """스케일 서보 이득: 이번 윈도에서 calib(a,b)에 곱할 보정 계수.

    mpu가 기준보다 크다 = 라이브 단위가 미터로 더 비싸졌다 = 라이브 스케일이
    수축했다 → g>1로 depth를 키워 복귀. 윈도당 ±step 비율로 클램프해 VO의
    keyframe 체인이 점진 흡수할 수 있게 한다 (급격한 스케일 변화는 기존
    지도/객체와 불연속을 만든다).
    """
    if mpu_ref <= 0 or mpu <= 0:
        return 1.0
    return float(np.clip((mpu / mpu_ref) ** k, 1.0 - step, 1.0 + step))


def build_pose_inputs(window: list[BackendKeyframe],
                      min_spread: float = 1e-3
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """윈도 키프레임의 VO pose/K를 DA3 입력 형식으로 변환.

    반환: (extrinsics (V,4,4) w2c, intrinsics (V,3,3)) — rgb 해상도 기준.
    K가 없거나 카메라 중심의 스프레드가 퇴화하면 None (무조건화 폴백):
    베이스라인이 노이즈 수준이면 입력 pose와 예측 pose의 Umeyama 스케일
    정합(align_to_input_ext_scale)이 불안정해 depth가 오염될 수 있다.

    병진은 첫 뷰 기준 median 거리가 1이 되도록 미리 스케일한다. DA3의
    _normalize_extrinsics가 median 거리로 나누되 min 0.1로 클램프하는데,
    라이브 단위의 윈도 베이스라인(~0.02)은 클램프에 걸려 조건 신호가
    5~10x 축소돼 기하가 왜곡된다 (실측: extent 0.64x 수축, 커버리지 붕괴).
    미리 1로 맞추면 클램프가 무력화되고, 스케일 차이는 어차피 윈도 α,β
    정합이 흡수한다.
    """
    if any(kf.K is None for kf in window):
        return None
    centers = np.array([kf.T_wc_live[:3, 3] for kf in window])
    spread = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean())
    if len(window) < 3 or spread < min_spread:
        return None
    med = float(np.median(np.linalg.norm(centers - centers[0], axis=1)))
    if med < min_spread:
        return None
    exts = []
    for kf in window:
        R = kf.T_wc_live[:3, :3]
        c = kf.T_wc_live[:3, 3] / med
        w2c = np.eye(4)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ c
        exts.append(w2c)
    return np.stack(exts), np.stack([kf.K for kf in window])


class _Worker:
    """Backend state + window processing. Lives entirely in the child process."""

    def __init__(self, cfg: BackendCfg, model_name: str, device: str,
                 process_res: int, metric_model: str | None = None,
                 loop_cfg=None):
        from depth_anything_3.api import DepthAnything3

        self.cfg = cfg
        self.device = device
        self.process_res = process_res
        self.model = DepthAnything3.from_pretrained(model_name).to(device).eval()
        self.metric_model = None
        if cfg.metric_anchor and metric_model:
            self.metric_model = (DepthAnything3.from_pretrained(metric_model)
                                 .to(device).eval())
        self._meters_per_unit: float | None = None
        self._mpu_ref: float | None = None   # 스케일 서보 기준 (최초 유효 mpu)
        self.kf_global_poses: dict[int, np.ndarray] = {}
        self._kf_ts: dict[int, float] = {}
        self._pending: list[BackendKeyframe] = []
        self._reconstructed: list[BackendKeyframe] = []
        # ---- 루프 클로저 상태 (loop_cfg가 있을 때만) ----
        self.loop_cfg = loop_cfg
        self.detector = None
        if loop_cfg is not None and loop_cfg.enabled:
            from .loop import LoopDetector
            self.detector = LoopDetector(loop_cfg.sim_thresh,
                                         loop_cfg.min_gap_s,
                                         loop_cfg.max_kf_store)
        self._epoch = 0
        self._epoch_kfs: dict[int, list[int]] = {}   # epoch -> 그 윈도의 kf ids
        # 루프 검증용 키프레임 저장: id -> (ts, gray u8, depth f16, K)
        self._kf_store: dict[int, tuple] = {}
        self._loop_edges: dict[tuple[int, int], tuple[Sim3, float]] = {}

    def run(self, in_q: mp.Queue, out_q: mp.Queue) -> None:
        last_run = time.monotonic()
        while True:
            try:
                item = in_q.get(timeout=0.1)
                if item is None:
                    return
                self._pending.append(item)
            except queue.Empty:
                pass
            # 첫 윈도는 전역 좌표계의 스케일을 정의하므로 충분한 시차를 요구
            min_kf = 5 if not self._reconstructed else 2
            if (time.monotonic() - last_run < self.cfg.period_s
                    or len(self._pending) < min_kf):
                continue
            last_run = time.monotonic()
            try:
                print(f"[backend-worker] window start ({len(self._pending)} pending)",
                      flush=True)
                result = self._run_window()
                if result is not None:
                    out_q.put(result)
                    print(f"[backend-worker] window done in {result.runtime_s:.1f}s",
                          flush=True)
                if self.device == "cuda":
                    # 윈도 사이(수 초)에 예약 VRAM을 반환 — 라이브/GS 프로세스와
                    # 16GB를 나눠 쓰므로 캐시를 쥐고 있으면 WDDM 스왑을 유발
                    import torch
                    torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()

    def _run_window(self) -> BackendResult | None:
        t0 = time.monotonic()
        n_new = self.cfg.window_size - self.cfg.overlap
        new_kfs = self._pending[-n_new:]
        self._pending = []
        old_kfs = self._reconstructed[-self.cfg.overlap:]
        window = old_kfs + new_kfs
        ids = [kf.kf_id for kf in window]
        self._epoch_kfs[self._epoch] = ids
        for kf in window:
            self._kf_ts[kf.kf_id] = kf.ts

        # ---- pose-conditioned 추론 (옵션): VO pose/K를 입력 조건으로 주면
        # 출력 depth가 입력 pose 스케일로 정합되어 나온다 (패키지가 내부에서
        # 예측 pose를 입력 pose에 Umeyama 정합 후 depth /= scale). 아래의
        # α,β 정합은 안전망으로 유지하며, 조건화 시 항등에 수렴해야 정상.
        pose_inputs = (build_pose_inputs(window)
                       if self.cfg.pose_conditioned else None)
        if pose_inputs is not None:
            exts, ixts = pose_inputs
            pred = self.model.inference([kf.rgb for kf in window],
                                        extrinsics=exts, intrinsics=ixts,
                                        align_to_input_ext_scale=True,
                                        process_res=self.process_res)
        else:
            pred = self.model.inference([kf.rgb for kf in window],
                                        process_res=self.process_res)
        V, dh, dw = pred.depth.shape

        # 윈도 pose는 라이브 VO pose를 그대로 사용한다. DA3-small의 pose 헤드는
        # 자기 depth 대비 병진을 4-8x 과소추정하고, 띄엄띄엄한 키프레임 간
        # LK/PnP 재추정은 베이스라인이 노이즈 바닥 아래라 붕괴한다
        # (docs/benchmarks.md). 연속 추적되는 라이브 VO가 유일하게 신뢰 가능한
        # 병진 소스이므로, 백엔드는 멀티뷰 depth 품질과 보정에 집중한다.
        T_wc_win = [kf.T_wc_live for kf in window]

        # ---- align window into the global map frame (살아있는 메커니즘으로
        # 유지하지만, 윈도 pose가 라이브 pose이므로 사실상 항등에 가깝다) ----
        shared = [(i, kf) for i, kf in enumerate(window)
                  if kf.kf_id in self.kf_global_poses]
        if shared:
            S = robust_sim3([T_wc_win[i] for i, _ in shared],
                            [self.kf_global_poses[kf.kf_id] for _, kf in shared])
        else:
            S = SIM3_IDENTITY  # 첫 윈도가 전역 좌표계를 정의한다

        for i, kf in enumerate(window):
            self.kf_global_poses[kf.kf_id] = sim3_on_pose(S, T_wc_win[i])

        conf_thresh = (np.percentile(pred.conf, self.cfg.conf_percentile)
                       if pred.conf is not None else None)

        def static_valid(i: int, kf: BackendKeyframe) -> np.ndarray:
            keep = pred.depth[i] > 1e-6
            if conf_thresh is not None:
                keep &= pred.conf[i] >= conf_thresh
            if kf.dyn_mask is not None:
                keep &= ~cv2.resize(kf.dyn_mask.astype(np.uint8), (dw, dh),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            return keep

        # ---- 멀티뷰 depth를 라이브 스케일로 정합 (α·d_win + β ≈ d_live) ----
        # 멀티뷰 depth는 윈도마다 정규화 스케일이 달라질 수 있으므로, pose가
        # 살고 있는 라이브 스케일로 변환한 뒤에 융합해야 지도가 일관된다.
        alpha, beta = 1.0, 0.0
        newest = window[-1]
        if newest.raw_depth is not None:
            a0, b0 = newest.calib_ab
            d_live = a0 * cv2.resize(newest.raw_depth, (dw, dh),
                                     interpolation=cv2.INTER_LINEAR) + b0
            win_cal = fit_affine_depth(pred.depth[-1], d_live,
                                       static_valid(V - 1, newest))
            if win_cal.inlier_frac > 0.3:
                alpha, beta = win_cal.a, win_cal.b

        # ---- fuse static points (라이브 스케일, 전역 frame) ----
        # 각 포인트의 출처 뷰와 카메라 원점도 함께 보낸다 — 지도 쪽에서
        # 시선 관통(free-space carving)으로 잘못된 옛 표면을 지우는 데 사용.
        pts_list, col_list, vidx_list = [], [], []
        view_origins = np.array(
            [sim3_apply(S, T[:3, 3][None])[0] for T in T_wc_win])
        for i, kf in enumerate(window):
            keep = static_valid(i, kf)
            vs, us = np.nonzero(keep)
            if len(vs) == 0:
                continue
            K = pred.intrinsics[i]
            z = alpha * pred.depth[i][vs, us] + beta
            cam = np.stack([(us - K[0, 2]) / K[0, 0] * z,
                            (vs - K[1, 2]) / K[1, 1] * z, z], axis=1)
            T = T_wc_win[i]
            world = sim3_apply(S, cam @ T[:3, :3].T + T[:3, 3])
            rgb_small = cv2.resize(kf.rgb, (dw, dh), interpolation=cv2.INTER_AREA)
            pts_list.append(world)
            col_list.append(rgb_small[vs, us])
            vidx_list.append(np.full(len(world), i, np.uint8))

        points = np.concatenate(pts_list) if pts_list else np.empty((0, 3))
        colors = np.concatenate(col_list) if col_list else np.empty((0, 3), np.uint8)
        point_view_idx = (np.concatenate(vidx_list) if vidx_list
                          else np.empty(0, np.uint8))

        # ---- live frame -> global correction ----
        T_gl = robust_sim3([kf.T_wc_live for kf in window],
                           [self.kf_global_poses[kf.kf_id] for kf in window])

        # ---- live mono depth calibration: raw mono -> 라이브 스케일의
        # 멀티뷰 depth (mono의 프레임별 스케일 요동을 멀티뷰 기준으로 고정) ----
        calib = DepthCalibration()
        if newest.raw_depth is not None:
            raw = cv2.resize(newest.raw_depth, (dw, dh),
                             interpolation=cv2.INTER_LINEAR)
            ref = alpha * pred.depth[-1] + beta
            calib = fit_affine_depth(raw, ref, static_valid(V - 1, newest))

        # ---- (선택) metric 앵커: 라이브 스케일 1단위 = 몇 미터인지 추정 ----
        # 지도/pose의 스케일은 건드리지 않고 표시용 환산 계수만 갱신한다.
        servo_g = 1.0
        if self.metric_model is not None:
            metric = self.metric_model.inference(
                [newest.rgb], process_res=self.process_res).depth[0]
            d_live = alpha * pred.depth[-1] + beta
            valid = static_valid(V - 1, newest) & (d_live > 1e-6) & (metric > 1e-6)
            if valid.sum() > 500:
                mpu = float(np.median(metric[valid] / d_live[valid]))
                self._meters_per_unit = (mpu if self._meters_per_unit is None
                                         else 0.7 * self._meters_per_unit + 0.3 * mpu)
                # ---- 스케일 서보: 라이브 스케일의 장기 drift를 metric 앵커로
                # 묶는다 (calib a가 1~2분에 걸쳐 0.8→0.1로 붕괴하던 문제).
                # 여기서는 이득 g만 계산해 결과로 보낸다 — 적용은 main이
                # calib·VO 상태·T_global_live에 일관되게 수행 (분리 적용 시
                # frame_scale 피드백이 상쇄함).
                if self.cfg.scale_servo and calib.inlier_frac > 0.3:
                    if self._mpu_ref is None:
                        self._mpu_ref = self._meters_per_unit
                    else:
                        g = servo_gain(self._meters_per_unit, self._mpu_ref)
                        if abs(g - 1.0) > 1e-3:
                            servo_g = g
                            # 라이브 depth가 g배 되면 다음 측정 mpu는 1/g배 —
                            # EMA 추정치도 같이 당겨 서보 지연을 줄인다
                            self._meters_per_unit /= g

        self._reconstructed.extend(new_kfs)
        # 윈도 중첩에 쓰일 최근 키프레임만 이미지를 유지 (메모리 해제)
        self._reconstructed = self._reconstructed[-self.cfg.overlap:]

        # ---- 루프 클로저: 재방문 감지 → pose graph → epoch별 지도 보정 ----
        corrections, loop_log, corr_newest = self._close_loops(new_kfs)
        if corrections is None and getattr(self.cfg, "attitude_servo", False):
            anchor_pose = self.kf_global_poses.get(newest.kf_id)
            if anchor_pose is not None:
                C = self._attitude_correction(points, anchor_pose[:3, 3])
                if C is not None:
                    corrections = {e: C for e in self._epoch_kfs}
                    for k, pose in list(self.kf_global_poses.items()):
                        self.kf_global_poses[k] = sim3_on_pose(C, pose)
                    corr_newest = C
                    loop_log = (
                        f"attitude theta={self._last_attitude_deg:.1f}deg "
                        f"inl={self._last_attitude_inlier_frac:.2f}")
        if corrections is not None:
            # live→global 갱신은 재피팅이 아니라 *합성*으로: pose graph가
            # 최신 노드를 얼마나 움직였는지(corr_newest)가 곧 현재 시점의
            # 전역 보정량이다. pose 집합에 Umeyama를 다시 맞추면 보정이
            # 비균일하게 변형시킨 카메라 배치 탓에 스케일이 폭주한다
            # (실측: 윈도 전체 1.4~1.5, 신규만 써도 정지 구간에서 2.6~9.8).
            T_gl = sim3_compose(corr_newest, T_gl)

        epoch = self._epoch
        self._epoch += 1

        return BackendResult(
            points=points, colors=colors, T_global_live=T_gl, calib=calib,
            kf_global_poses=dict(self.kf_global_poses),
            kf_ts=dict(self._kf_ts),
            intrinsics=np.median(pred.intrinsics, axis=0),
            depth_size=(dw, dh), meters_per_unit=self._meters_per_unit,
            view_origins=view_origins, point_view_idx=point_view_idx,
            window_ids=ids, runtime_s=time.monotonic() - t0,
            pose_conditioned=pose_inputs is not None,
            win_alpha=alpha, win_beta=beta,
            epoch=epoch, loop_corrections=corrections, loop_log=loop_log,
            servo_gain_g=servo_g,
            loop_converging=self._has_pending_loop_residual())

    def _attitude_correction(self, points: np.ndarray,
                             anchor: np.ndarray) -> Sim3 | None:
        """Return a small rotation correction that moves the floor normal to -Y."""
        self._last_attitude_deg = 0.0
        self._last_attitude_inlier_frac = 0.0
        result = floor_mod.estimate_floor_from_points(points)
        if result is None:
            return None

        normal, _, inlier_frac = result
        target = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            return None
        normal = normal / normal_norm

        dot = float(np.clip(normal @ target, -1.0, 1.0))
        angle = float(np.arccos(dot))
        self._last_attitude_deg = float(np.degrees(angle))
        self._last_attitude_inlier_frac = float(inlier_frac)
        if angle <= _ATT_DEADBAND_RAD:
            return None

        axis = np.cross(normal, target)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return None
        R_c = floor_mod._axis_angle(axis / axis_norm,
                                    min(angle, _ATT_STEP_RAD))
        anchor = np.asarray(anchor, dtype=np.float64)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            return None
        return 1.0, R_c, anchor - R_c @ anchor

    # ------------------------------------------------------------------
    # 루프 클로저 (docs/upgrade-plan.md Tier 4)
    # ------------------------------------------------------------------
    _MAX_GRAPH_NODES = 400
    _MAX_LOOP_EDGES = 50
    _RESIDUAL_THRESH = 0.05
    _SNAP_INLIERS = 150
    # 윈도당 보정 상한 — 초과분은 부분 적용(λ<1)하고 다음 윈도의 루프
    # 재검출로 반복 수렴시킨다. drift가 극단으로 누적된 뒤 한 번에 당기면
    # (실측 |t|=4.7, scale 3.1) 지도·객체가 출렁이고 병합이 과격해진다.
    _MAX_STEP_T = 0.5            # 병진 (live 단위)
    _MAX_STEP_LOGS = 0.405       # 스케일 (=log 1.5)

    @classmethod
    def _step_limit(cls, snap: bool) -> tuple[float, float]:
        factor = 3.0 if snap else 1.0
        return cls._MAX_STEP_T * factor, cls._MAX_STEP_LOGS * factor

    def _remember_loop_edge(self, old_id: int, new_id: int, T_ab: Sim3,
                            weight: float) -> None:
        key = (old_id, new_id)
        prev = self._loop_edges.get(key)
        if prev is None or weight >= prev[1]:
            if prev is not None:
                self._loop_edges.pop(key)
            self._loop_edges[key] = (T_ab, weight)
        while len(self._loop_edges) > self._MAX_LOOP_EDGES:
            # 증거가 가장 약한(감쇠된) 엣지부터 버린다
            weakest = min(self._loop_edges, key=lambda k: self._loop_edges[k][1])
            self._loop_edges.pop(weakest)

    def _decay_loop_edges(self, keys: list[tuple[int, int]]) -> None:
        for key in keys:
            if key not in self._loop_edges:
                continue
            T_ab, weight = self._loop_edges[key]
            self._loop_edges[key] = (T_ab, max(0.3, weight * 0.9))

    def _has_pending_loop_residual(self) -> bool:
        if self.detector is None:
            return False
        if not bool(getattr(self.loop_cfg, "persist_edges", True)):
            return False
        if not self._loop_edges:
            return False

        from .loop import edge_residual

        for (old_id, new_id), (T_ab, _) in self._loop_edges.items():
            if old_id not in self.kf_global_poses:
                continue
            if new_id not in self.kf_global_poses:
                continue
            residual = edge_residual(self.kf_global_poses[old_id],
                                     self.kf_global_poses[new_id], T_ab)
            if residual > self._RESIDUAL_THRESH:
                return True
        return False

    def _close_loops(self, new_kfs: list[BackendKeyframe]
                     ) -> tuple[dict[int, Sim3] | None, str, Sim3]:
        """새 키프레임에서 재방문을 찾아 수락되면 pose graph로 drift 보정.

        반환: (epoch별 지도 보정 Sim3 dict | None, 로그 문자열,
               최신 키프레임 노드의 보정 Sim3 — T_global_live 합성용).
        kf_global_poses는 in-place로 보정된다.
        """
        if self.detector is None:
            return None, "", SIM3_IDENTITY
        from .loop import (edge_residual, match_3d3d, optimize_pose_graph,
                           sequential_edges, sim3_from_matches)

        lcfg = self.loop_cfg
        persist_edges = bool(getattr(lcfg, "persist_edges", True))
        accepted = []
        for kf in new_kfs:
            if kf.emb is None or kf.K is None or kf.raw_depth is None:
                continue
            a0, b0 = kf.calib_ab
            gray = cv2.cvtColor(kf.rgb, cv2.COLOR_RGB2GRAY)
            depth = (a0 * kf.raw_depth + b0).astype(np.float16)
            hit = self.detector.query(kf.ts, kf.emb)
            if (hit is not None and hit[0] in self._kf_store
                    and hit[0] in self.kf_global_poses):
                old_id, sim = hit
                _, ogray, odepth, oK = self._kf_store[old_id]
                pts_a, pts_b = match_3d3d(
                    ogray, odepth.astype(np.float32), oK,
                    gray, depth.astype(np.float32), kf.K)
                dg: dict = {}
                res = sim3_from_matches(pts_a, pts_b, lcfg.inlier_dist,
                                        min_inliers=lcfg.min_inliers, diag=dg)
                if res is not None and (res[1].mean()
                                        >= lcfg.min_inlier_frac):
                    T_ab, mask = res
                    inl = int(mask.sum())
                    weight = min(1.5, inl / 50.0)
                    accepted.append((old_id, kf.kf_id, T_ab, inl, sim,
                                     weight))
                    if persist_edges:
                        self._remember_loop_edge(old_id, kf.kf_id, T_ab,
                                                 weight)
                else:
                    print(f"[loop] 후보 기각 kf{old_id}->kf{kf.kf_id} "
                          f"(sim={sim:.2f}, matches={dg.get('n', 0)}, "
                          f"best_inl={dg.get('best_inl', 0)}, "
                          f"med_s={dg.get('med_scale', float('nan')):.2f})",
                          flush=True)
            self.detector.add(kf.kf_id, kf.ts, kf.emb)
            self._kf_store[kf.kf_id] = (kf.ts, gray, depth, kf.K)
            while len(self._kf_store) > lcfg.max_kf_store:
                self._kf_store.pop(next(iter(self._kf_store)))

        if persist_edges:
            active_loop_edges = [
                (old_id, new_id, T_ab, weight)
                for (old_id, new_id), (T_ab, weight) in self._loop_edges.items()
                if old_id in self.kf_global_poses
                and new_id in self.kf_global_poses
            ]
        else:
            active_loop_edges = [
                (old_id, new_id, T_ab, weight)
                for old_id, new_id, T_ab, _, _, weight in accepted
            ]

        residuals = [
            edge_residual(self.kf_global_poses[old_id],
                          self.kf_global_poses[new_id], T_ab)
            for old_id, new_id, T_ab, _ in active_loop_edges
        ]
        max_residual = max(residuals, default=0.0)
        replay_persisted = (
            persist_edges
            and bool(active_loop_edges)
            and max_residual > self._RESIDUAL_THRESH
        )
        snap = any(inl >= self._SNAP_INLIERS
                   for _, _, _, inl, _, _ in accepted)

        if ((not accepted and not replay_persisted)
                or len(self.kf_global_poses) < 8):
            return None, "", SIM3_IDENTITY

        # ---- pose graph 노드 선정 (장시간 세션은 솎아냄) ----
        all_ids = sorted(self.kf_global_poses)
        must = {i for edge in active_loop_edges for i in edge[:2]}
        if len(all_ids) > self._MAX_GRAPH_NODES:
            stride = int(np.ceil(len(all_ids) / self._MAX_GRAPH_NODES))
            node_ids = sorted(set(all_ids[::stride]) | must | {all_ids[-1]})
        else:
            node_ids = all_ids
        idx = {k: i for i, k in enumerate(node_ids)}
        poses = [self.kf_global_poses[k] for k in node_ids]

        edges = sequential_edges(poses)
        used_loop_keys: list[tuple[int, int]] = []
        for old_id, new_id, T_ab, weight in active_loop_edges:
            if old_id in idx and new_id in idx:
                edges.append((idx[old_id], idx[new_id], T_ab, weight))
                used_loop_keys.append((old_id, new_id))
        corrected = optimize_pose_graph(poses, edges)

        # 노드별 보정 Sim3 (corrected ∘ old⁻¹); 비노드 kf는 최근접 노드 보정 적용
        node_corr: list[Sim3] = []
        for k, old in zip(node_ids, poses):
            node_corr.append(sim3_compose(
                corrected[idx[k]],
                sim3_inverse((1.0, old[:3, :3], old[:3, 3]))))

        # ---- 대형 보정은 부분 적용 (점진 수렴) ----
        mag_t = max(np.linalg.norm(c[2]) for c in node_corr)
        mag_s = max(abs(np.log(max(c[0], 1e-12))) for c in node_corr)
        max_step_t, max_step_logs = self._step_limit(snap)
        lam = min(1.0, max_step_t / max(mag_t, 1e-9),
                  max_step_logs / max(mag_s, 1e-9))
        partial = ""
        if lam < 1.0:
            node_corr = [sim3_interp(SIM3_IDENTITY, C, lam) for C in node_corr]
            partial = f"부분 적용 λ={lam:.2f}"
        node_arr = np.array(node_ids)
        for k in all_ids:
            j = int(np.argmin(np.abs(node_arr - k)))
            self.kf_global_poses[k] = sim3_on_pose(node_corr[j],
                                                   self.kf_global_poses[k])

        if persist_edges and used_loop_keys:
            self._decay_loop_edges(used_loop_keys)

        # epoch별 지도 보정: 각 윈도의 최신 kf에 대응하는 노드 보정 사용
        corrections: dict[int, Sim3] = {}
        for e, kf_ids in self._epoch_kfs.items():
            j = int(np.argmin(np.abs(node_arr - kf_ids[-1])))
            corrections[e] = node_corr[j]

        mag = max(np.linalg.norm(c[2]) for c in node_corr)
        log_parts = []
        if accepted:
            log_parts.append("; ".join(
                f"kf{o}↔kf{nn} inl={i} sim={s_:.2f}"
                for o, nn, _, i, s_, _ in accepted))
        if persist_edges and active_loop_edges:
            log_parts.append(f"persist r={max_residual:.3f}")
        if snap:
            log_parts.append(f"snap inl>={self._SNAP_INLIERS}")
        log_parts.append(f"max|t|={mag:.3f}")
        if partial:
            log_parts.append(partial)
        newest_id = new_kfs[-1].kf_id if new_kfs else all_ids[-1]
        j_newest = int(np.argmin(np.abs(node_arr - newest_id)))
        return (corrections, " | ".join(log_parts),
                node_corr[j_newest])


def _worker_main(cfg: BackendCfg, model_name: str, device: str,
                 process_res: int, metric_model: str | None, loop_cfg,
                 in_q: mp.Queue, out_q: mp.Queue) -> None:
    import spacerec  # noqa: F401  (env vars in the child process)

    worker = _Worker(cfg, model_name, device, process_res, metric_model,
                     loop_cfg=loop_cfg)
    out_q.put("ready")
    worker.run(in_q, out_q)


class ReconstructionBackend:
    """Main-process handle: feeds keyframes to / reads results from the child."""

    def __init__(self, cfg: BackendCfg, model_name: str, device: str,
                 process_res: int = 504, metric_model: str | None = None,
                 loop_cfg=None):
        ctx = mp.get_context("spawn")
        self._in_q: mp.Queue = ctx.Queue()
        self.results: mp.Queue = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(cfg, model_name, device, process_res, metric_model,
                  loop_cfg, self._in_q, self.results),
            daemon=True)

    def start(self) -> None:
        self._proc.start()

    def wait_ready(self, timeout: float = 900.0) -> None:
        """모델 가중치 최초 다운로드(수백 MB~GB)까지 포함해 기다린다."""
        msg = self.results.get(timeout=timeout)
        assert msg == "ready", f"unexpected backend message: {msg!r}"

    def add_keyframe(self, kf: BackendKeyframe) -> None:
        self._in_q.put(kf)

    def stop(self) -> None:
        if self._proc.is_alive():
            self._in_q.put(None)
            self._proc.join(timeout=30)
            if self._proc.is_alive():
                self._proc.terminate()
        _drain_and_close(self._in_q, self.results)


def _drain_and_close(*queues: mp.Queue) -> None:
    """종료 행 방지: feeder 스레드 join을 포기시키고 큐를 닫는다.

    큐에 남은 대형 항목이 exit 시 feeder join을 영원히 막아 좀비 프로세스가
    CUDA 컨텍스트(VRAM)를 계속 쥐는 사고가 실측됨. 주의 — 잔여 항목을
    get_nowait()로 비우려 하면 안 된다: Windows에서 죽은 자식이 쓰다 만
    찢어진 메시지가 파이프에 있으면 poll()은 True인데 _recv_bytes()가
    영원히 블록한다 (py-spy 스택으로 실측, 또 다른 행 경로)."""
    for q in queues:
        q.cancel_join_thread()
        q.close()
