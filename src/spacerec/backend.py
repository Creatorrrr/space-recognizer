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

from .calib import DepthCalibration, fit_affine_depth
from .config import BackendCfg
from .geometry import SIM3_IDENTITY, Sim3, sim3_apply, sim3_on_pose, umeyama_sim3


@dataclass
class BackendKeyframe:
    kf_id: int
    ts: float
    rgb: np.ndarray                 # processing-res RGB (backend input)
    T_wc_live: np.ndarray           # VO pose (live frame)
    raw_depth: np.ndarray | None    # uncalibrated live mono depth (small)
    dyn_mask: np.ndarray | None     # dynamic-object mask (small, bool)
    calib_ab: tuple[float, float] = (1.0, 0.0)  # 생성 시점의 mono 보정 계수


@dataclass
class BackendResult:
    points: np.ndarray              # new global-frame points (N,3)
    colors: np.ndarray              # (N,3) uint8
    T_global_live: Sim3
    calib: DepthCalibration
    kf_global_poses: dict[int, np.ndarray]
    intrinsics: np.ndarray | None = None  # DA3 추정 K (process 해상도 기준)
    depth_size: tuple[int, int] = (0, 0)  # (w, h) of the K's reference image
    meters_per_unit: float | None = None  # metric 앵커 환산 계수 (표시용)
    window_ids: list[int] = field(default_factory=list)
    runtime_s: float = 0.0


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
        if d_src > 1e-6 and d_dst > 1e-6:
            s = float(d_dst / d_src)
    R = dst_poses[-1][:3, :3] @ src_poses[-1][:3, :3].T
    t = dst_c[-1] - s * R @ src_c[-1]
    return s, R, t


class _Worker:
    """Backend state + window processing. Lives entirely in the child process."""

    def __init__(self, cfg: BackendCfg, model_name: str, device: str,
                 process_res: int, metric_model: str | None = None):
        from depth_anything_3.api import DepthAnything3

        self.cfg = cfg
        self.process_res = process_res
        self.model = DepthAnything3.from_pretrained(model_name).to(device).eval()
        self.metric_model = None
        if cfg.metric_anchor and metric_model:
            self.metric_model = (DepthAnything3.from_pretrained(metric_model)
                                 .to(device).eval())
        self._meters_per_unit: float | None = None
        self.kf_global_poses: dict[int, np.ndarray] = {}
        self._pending: list[BackendKeyframe] = []
        self._reconstructed: list[BackendKeyframe] = []

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

        conf_thresh = (np.percentile(pred.conf, 30.0)
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
        pts_list, col_list = [], []
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

        points = np.concatenate(pts_list) if pts_list else np.empty((0, 3))
        colors = np.concatenate(col_list) if col_list else np.empty((0, 3), np.uint8)

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
        if self.metric_model is not None:
            metric = self.metric_model.inference(
                [newest.rgb], process_res=self.process_res).depth[0]
            d_live = alpha * pred.depth[-1] + beta
            valid = static_valid(V - 1, newest) & (d_live > 1e-6) & (metric > 1e-6)
            if valid.sum() > 500:
                mpu = float(np.median(metric[valid] / d_live[valid]))
                self._meters_per_unit = (mpu if self._meters_per_unit is None
                                         else 0.7 * self._meters_per_unit + 0.3 * mpu)

        self._reconstructed.extend(new_kfs)
        # 윈도 중첩에 쓰일 최근 키프레임만 이미지를 유지 (메모리 해제)
        self._reconstructed = self._reconstructed[-self.cfg.overlap:]

        return BackendResult(
            points=points, colors=colors, T_global_live=T_gl, calib=calib,
            kf_global_poses=dict(self.kf_global_poses),
            intrinsics=np.median(pred.intrinsics, axis=0),
            depth_size=(dw, dh), meters_per_unit=self._meters_per_unit,
            window_ids=ids, runtime_s=time.monotonic() - t0)


def _worker_main(cfg: BackendCfg, model_name: str, device: str,
                 process_res: int, metric_model: str | None,
                 in_q: mp.Queue, out_q: mp.Queue) -> None:
    import spacerec  # noqa: F401  (env vars in the child process)

    worker = _Worker(cfg, model_name, device, process_res, metric_model)
    out_q.put("ready")
    worker.run(in_q, out_q)


class ReconstructionBackend:
    """Main-process handle: feeds keyframes to / reads results from the child."""

    def __init__(self, cfg: BackendCfg, model_name: str, device: str,
                 process_res: int = 504, metric_model: str | None = None):
        ctx = mp.get_context("spawn")
        self._in_q: mp.Queue = ctx.Queue()
        self.results: mp.Queue = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(cfg, model_name, device, process_res, metric_model,
                  self._in_q, self.results),
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
