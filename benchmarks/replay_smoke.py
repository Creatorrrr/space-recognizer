"""Smoke-check recorded OAK sessions without requiring a live device.

Usage:
  .venv/bin/python benchmarks/replay_smoke.py sources/session_... --frames 120
  .venv/bin/python benchmarks/replay_smoke.py sources/session_* --frames 60 --full-models
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

import spacerec  # noqa: F401
from spacerec.backend import BackendKeyframe, _Worker
from spacerec.config import Config
from spacerec.device import configure_torch_runtime, select_torch_device
from spacerec.directfusion import DirectFusionBackend, DirectFusionKeyframe
from spacerec.imu import estimate_camera_rotation, should_accept_backend_keyframe
from spacerec.objects import ObjectRegistry, localize_objects
from spacerec.replay import RecordedOakSource, _nearest
from spacerec.vo import VisualOdometry
from spacerec.worldmap import GlobalMap


class MetricTimer:
    def __init__(self, timings: dict, name: str):
        self.timings = timings
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        entry = self.timings.setdefault(self.name, {"count": 0, "total_ms": 0.0})
        entry["count"] += 1
        entry["total_ms"] += (time.perf_counter() - self.t0) * 1000.0
        return False


def _summarize_timings(timings: dict) -> dict:
    summary = {}
    for name, entry in sorted(timings.items()):
        count = int(entry.get("count", 0))
        total_ms = float(entry.get("total_ms", 0.0))
        summary[f"{name}_count"] = count
        summary[f"{name}_total_ms"] = total_ms
        summary[f"{name}_avg_ms"] = total_ms / count if count else 0.0
    return summary


def _cuda_memory_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "max_reserved": int(torch.cuda.max_memory_reserved()),
    }


def _apply_runtime_overrides(cfg: Config, args) -> None:
    if getattr(args, "precision", None):
        cfg.compute.precision = args.precision
    if getattr(args, "no_tf32", False):
        cfg.compute.tf32 = False
    if getattr(args, "metric_anchor_every_n_windows", None) is not None:
        cfg.backend.metric_anchor_every_n_windows = int(args.metric_anchor_every_n_windows)
    if getattr(args, "metric_anchor_process_res", None) is not None:
        cfg.backend.metric_anchor_process_res = int(args.metric_anchor_process_res)


def _dynamic_mask(detections, shape, dynamic_classes):
    mask = None
    for det in detections:
        if det.cls_name in dynamic_classes and det.mask is not None:
            mask = det.mask if mask is None else (mask | det.mask)
    return mask


def _pairing_stats(src: RecordedOakSource) -> tuple[int, float, float]:
    deltas = []
    for rgb in src.rgb_events:
        depth = _nearest(src.depth_events, rgb.ts_device_ns)
        if depth is None:
            continue
        deltas.append(abs(depth.ts_device_ns - rgb.ts_device_ns) / 1e6)
    if not deltas:
        return 0, 0.0, 0.0
    return len(deltas), float(np.median(deltas)), float(np.max(deltas))


def _payload_missing(src: RecordedOakSource) -> int:
    missing = 0
    for events in src.events.values():
        for event in events:
            if event.payload_path is not None and not event.payload_path.is_file():
                missing += 1
    return missing


def _run_backend_window(
    cfg: Config,
    keyframes: list[BackendKeyframe],
    *,
    metric_anchor: bool = False,
) -> dict:
    if not keyframes:
        return {
            "backend_points": 0,
            "backend_runtime_s": 0.0,
            "backend_timings_ms": {},
            "backend_cuda_memory": _cuda_memory_snapshot(),
        }
    bcfg = replace(
        cfg.backend,
        window_size=max(1, len(keyframes)),
        overlap=0,
        metric_anchor=bool(metric_anchor),
    )
    metric_model = cfg.depth.metric_model if metric_anchor else None
    worker = _Worker(bcfg, cfg.depth.model, select_torch_device(None),
                     min(cfg.depth.process_res, 252), metric_model=metric_model,
                     precision=cfg.compute.precision)
    worker._pending.extend(keyframes)
    result = worker._run_window()
    if result is None:
        return {
            "backend_points": 0,
            "backend_runtime_s": 0.0,
            "backend_timings_ms": {},
            "backend_cuda_memory": _cuda_memory_snapshot(),
        }
    return {
        "backend_points": int(len(result.points)),
        "backend_runtime_s": float(result.runtime_s),
        "backend_timings_ms": dict(result.timings_ms),
        "backend_cuda_memory": dict(result.cuda_memory),
    }


def run_session(path: Path, cfg: Config, frames: int, full_models: bool,
                backend: bool, imu_enabled: bool | None = None,
                backend_metric_anchor: bool = False,
                direct_fusion: bool = False) -> dict:
    src = RecordedOakSource(
        path,
        proc_width=cfg.proc_width,
        realtime=False,
        depth_mode=cfg.capture.replay_depth_mode,
        max_pair_dt_ms=cfg.capture.replay_pair_tolerance_ms,
    )
    detector = None
    registry = None
    if full_models:
        from spacerec.detect import ObjectDetector

        detector = ObjectDetector(cfg.detect.model, conf=cfg.detect.conf,
                                  vocabulary=cfg.detect.vocabulary)
        registry = ObjectRegistry(cfg.objects)

    vo = VisualOdometry(src.K, cfg.vo)
    use_imu = cfg.imu.enabled if imu_enabled is None else bool(imu_enabled)
    R_cam_imu = src.R_cam_imu
    imu_since_keyframe = []
    imu_max_angle_rad = np.radians(cfg.imu.max_rotation_deg)
    prev_frame_ts_for_imu: float | None = None
    imu_keyframe_ts: float | None = None
    last_backend_keyframe_ts: float | None = None
    stats = Counter()
    timings = {}
    valid_fracs = []
    inliers = []
    tracked = []
    classes = Counter()
    keyframes: list[BackendKeyframe] = []
    direct_keyframes = 0
    direct_backend = DirectFusionBackend(cfg.fusion, cfg.capture) if direct_fusion else None
    direct_map = GlobalMap(cfg.backend) if direct_fusion else None
    payload_missing = _payload_missing(src)
    pair_count, pair_median_ms, pair_max_ms = _pairing_stats(src)

    def drain_direct() -> None:
        if direct_backend is None or direct_map is None:
            return
        while True:
            try:
                res = direct_backend.results.get_nowait()
            except Exception:
                return
            direct_map.fuse(res.points, res.colors,
                            origins=res.view_origins, view_idx=res.point_view_idx)

    try:
        for frame in src.frames():
            if stats["frames"] >= frames:
                break
            stats["frames"] += 1
            if frame.depth_m is not None:
                valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
                valid_fracs.append(float(valid.mean()))
                stats["depth_frames"] += 1
            gray = frame.gray_track
            if gray is None:
                gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
            detections = []
            if detector is not None:
                with MetricTimer(timings, "detect_track"):
                    detections = detector.track(frame.bgr)
            for det in detections:
                classes[det.cls_name] += 1
            imu_samples = list(frame.imu_samples or [])
            imu_candidate = imu_since_keyframe + imu_samples
            imu_time_aligned = True
            if frame.metadata is not None:
                imu_time_aligned = bool(frame.metadata.get("imu_timestamp_aligned", True))
            delta_t0 = prev_frame_ts_for_imu if imu_time_aligned else None
            delta_t1 = frame.ts if imu_time_aligned else None
            keyframe_t0 = imu_keyframe_ts if imu_time_aligned else None
            keyframe_t1 = frame.ts if imu_time_aligned else None
            delta_prior = None
            since_prior = None
            if use_imu:
                if R_cam_imu is None:
                    stats["imu_no_extrinsics"] += 1
                elif not imu_samples:
                    stats["imu_no_samples"] += 1
                else:
                    if cfg.imu.use_lk_prior:
                        delta_prior = estimate_camera_rotation(
                            imu_samples,
                            R_cam_imu,
                            min_samples=cfg.imu.min_rotation_samples,
                            max_angle_rad=imu_max_angle_rad,
                            t0=delta_t0,
                            t1=delta_t1,
                        )
                    if cfg.imu.use_pnp_prior:
                        since_prior = estimate_camera_rotation(
                            imu_candidate,
                            R_cam_imu,
                            min_samples=cfg.imu.min_rotation_samples,
                            max_angle_rad=imu_max_angle_rad,
                            t0=keyframe_t0,
                            t1=keyframe_t1,
                        )
                    stats["imu_lk_priors"] += int(delta_prior is not None)
                    stats["imu_pnp_priors"] += int(since_prior is not None)
                    stats["imu_prior_frames"] += int(
                        delta_prior is not None or since_prior is not None)
            omega_norm = None
            if delta_prior is not None:
                omega_norm = delta_prior.omega_norm
            elif since_prior is not None:
                omega_norm = since_prior.omega_norm
            with MetricTimer(timings, "vo_process"):
                pose = vo.process(
                    gray,
                    frame.depth_m if frame.depth_m is not None else np.ones(gray.shape, np.float32),
                    frame.ts,
                    _dynamic_mask(detections, gray.shape, set(cfg.detect.dynamic_classes)),
                    R_delta_prev=None if delta_prior is None else delta_prior.R,
                    R_since_keyframe=None if since_prior is None else since_prior.R,
                    omega_norm=omega_norm,
                )
            if pose.is_keyframe:
                imu_since_keyframe = []
                imu_keyframe_ts = frame.ts
            else:
                imu_since_keyframe = imu_candidate
            prev_frame_ts_for_imu = frame.ts
            accept_backend_keyframe = should_accept_backend_keyframe(
                frame_ts=frame.ts,
                last_backend_keyframe_ts=last_backend_keyframe_ts,
                omega_norm=omega_norm,
                blur_omega_rad_s=cfg.imu.keyframe_blur_omega_rad_s,
                max_delay_s=cfg.imu.keyframe_max_delay_s,
            )
            if pose.is_keyframe and not accept_backend_keyframe:
                stats["imu_blur_skipped_kf"] += 1
            elif (pose.is_keyframe and omega_norm is not None
                  and omega_norm > cfg.imu.keyframe_blur_omega_rad_s):
                stats["imu_blur_forced_kf"] += 1
            if (backend and pose.is_keyframe and accept_backend_keyframe
                    and frame.depth_m is not None and len(keyframes) < 6):
                bw = min(cfg.depth.process_res, 252)
                bh = int(frame.bgr.shape[0] * bw / frame.bgr.shape[1])
                dyn = _dynamic_mask(detections, gray.shape, set(cfg.detect.dynamic_classes))
                keyframes.append(BackendKeyframe(
                    kf_id=len(keyframes),
                    ts=frame.ts,
                    rgb=cv2.cvtColor(
                        cv2.resize(frame.bgr, (bw, bh), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB,
                    ),
                    T_wc_live=pose.T_wc.copy(),
                    raw_depth=cv2.resize(frame.depth_m, (bw, bh),
                                         interpolation=cv2.INTER_NEAREST),
                    dyn_mask=None if dyn is None else cv2.resize(
                        dyn.astype(np.uint8), (bw, bh),
                        interpolation=cv2.INTER_NEAREST).astype(bool),
                    calib_ab=(1.0, 0.0),
                ))
            if (direct_backend is not None and pose.is_keyframe
                    and accept_backend_keyframe and frame.depth_m is not None):
                dyn = _dynamic_mask(detections, gray.shape, set(cfg.detect.dynamic_classes))
                direct_backend.add_keyframe(DirectFusionKeyframe(
                    kf_id=direct_keyframes,
                    ts=frame.ts,
                    bgr=frame.bgr.copy(),
                    depth_m=frame.depth_m.copy(),
                    K=vo.K.copy(),
                    T_wc=pose.T_wc.copy(),
                    dyn_mask=None if dyn is None else dyn.copy(),
                    depth_conf=None if frame.depth_conf is None else frame.depth_conf.copy(),
                ))
                direct_keyframes += 1
                drain_direct()
            if pose.is_keyframe and accept_backend_keyframe:
                last_backend_keyframe_ts = frame.ts
            stats["lost"] += int(pose.lost)
            stats["keyframes"] += int(pose.is_keyframe)
            if pose.n_tracked:
                tracked.append(float(pose.n_tracked))
                inliers.append(float(pose.inlier_ratio))
            if detections and registry is not None and frame.depth_m is not None:
                with MetricTimer(timings, "object_localize"):
                    obs = localize_objects(detections, frame.depth_m, vo.K, pose.T_wc,
                                           frame.depth_conf)
                stats["detections"] += len(detections)
                stats["object_observations"] += len(obs)
                registry.update(obs, frame.ts)
    finally:
        drain_direct()
        src.release()
    backend_metrics = (_run_backend_window(
        cfg, keyframes, metric_anchor=backend_metric_anchor) if backend else {
        "backend_points": 0,
        "backend_runtime_s": 0.0,
        "backend_timings_ms": {},
        "backend_cuda_memory": _cuda_memory_snapshot(),
    })

    return {
        "session": path.name,
        "imu": "on" if use_imu else "off",
        "payload_missing": payload_missing,
        "pair_count": pair_count,
        "pair_median_ms": pair_median_ms,
        "pair_max_ms": pair_max_ms,
        "frames": stats["frames"],
        "depth_frames": stats["depth_frames"],
        "depth_valid_mean": float(np.mean(valid_fracs)) if valid_fracs else 0.0,
        "lost": stats["lost"],
        "keyframes": stats["keyframes"],
        "avg_tracked": float(np.mean(tracked)) if tracked else 0.0,
        "avg_inlier": float(np.mean(inliers)) if inliers else 0.0,
        "detections": stats["detections"],
        "object_observations": stats["object_observations"],
        "objects": 0 if registry is None else len(registry.objects),
        "imu_prior_frames": stats["imu_prior_frames"],
        "imu_lk_priors": stats["imu_lk_priors"],
        "imu_pnp_priors": stats["imu_pnp_priors"],
        "imu_no_extrinsics": stats["imu_no_extrinsics"],
        "imu_no_samples": stats["imu_no_samples"],
        "imu_blur_skipped_kf": stats["imu_blur_skipped_kf"],
        "imu_blur_forced_kf": stats["imu_blur_forced_kf"],
        "backend_keyframes": len(keyframes),
        "backend_points": backend_metrics["backend_points"],
        "backend_runtime_s": backend_metrics["backend_runtime_s"],
        "backend_timings_ms": backend_metrics["backend_timings_ms"],
        "backend_cuda_memory": backend_metrics["backend_cuda_memory"],
        "timings": _summarize_timings(timings),
        "cuda_memory": _cuda_memory_snapshot(),
        "direct_keyframes": direct_keyframes,
        "direct_points": 0 if direct_map is None else len(direct_map.points),
        "top_classes": classes.most_common(6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--full-models", action="store_true",
                    help="load YOLOE and report detections/object observations")
    ap.add_argument("--backend", action="store_true",
                    help="run one direct DA3 backend window and report point count")
    ap.add_argument("--backend-metric-anchor", action="store_true",
                    help="include the optional DA3METRIC anchor in the direct backend window")
    ap.add_argument("--direct-fusion", action="store_true",
                    help="run OAK metric-depth direct fusion and report map point count")
    ap.add_argument("--imu", action="store_true",
                    help="enable gyro-derived VO rotation priors for this run")
    ap.add_argument("--compare-imu", action="store_true",
                    help="run each session twice: visual-only, then IMU-assisted")
    ap.add_argument("--metrics-out",
                    help="write replay metrics JSON for before/after CUDA comparisons")
    ap.add_argument("--precision", choices=["fp32", "bf16"],
                    help="override compute.precision for this run")
    ap.add_argument("--metric-anchor-every-n-windows", type=int,
                    help="override backend.metric_anchor_every_n_windows for this run")
    ap.add_argument("--metric-anchor-process-res", type=int,
                    help="override backend.metric_anchor_process_res for this run")
    ap.add_argument("--no-tf32", action="store_true",
                    help="disable CUDA TF32 runtime switches for this run")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    _apply_runtime_overrides(cfg, args)
    configure_torch_runtime(
        None,
        tf32=cfg.compute.tf32,
        cudnn_benchmark=cfg.compute.cudnn_benchmark,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = []
    for session in args.sessions:
        modes = (False, True) if args.compare_imu else (bool(args.imu or cfg.imu.enabled),)
        for imu_enabled in modes:
            result = run_session(Path(session), cfg, args.frames, args.full_models,
                                 args.backend, imu_enabled=imu_enabled,
                                 backend_metric_anchor=args.backend_metric_anchor,
                                 direct_fusion=args.direct_fusion)
            results.append(result)
            print(
                "REPLAY_SMOKE "
                f"session={result['session']} imu={result['imu']} "
                f"payload_missing={result['payload_missing']} "
                f"pairs={result['pair_count']} pair_median_ms={result['pair_median_ms']:.1f} "
                f"pair_max_ms={result['pair_max_ms']:.1f} frames={result['frames']} "
                f"depth_frames={result['depth_frames']} "
                f"depth_valid_mean={result['depth_valid_mean']:.3f} "
                f"lost={result['lost']} keyframes={result['keyframes']} "
                f"avg_tracked={result['avg_tracked']:.1f} "
                f"avg_inlier={result['avg_inlier']:.2f} "
                f"imu_prior_frames={result['imu_prior_frames']} "
                f"imu_lk_priors={result['imu_lk_priors']} "
                f"imu_pnp_priors={result['imu_pnp_priors']} "
                f"imu_no_extrinsics={result['imu_no_extrinsics']} "
                f"imu_no_samples={result['imu_no_samples']} "
                f"imu_blur_skipped_kf={result['imu_blur_skipped_kf']} "
                f"imu_blur_forced_kf={result['imu_blur_forced_kf']} "
                f"detections={result['detections']} "
                f"object_observations={result['object_observations']} "
                f"objects={result['objects']} "
                f"backend_keyframes={result['backend_keyframes']} "
                f"backend_points={result['backend_points']} "
                f"backend_runtime_s={result['backend_runtime_s']:.2f} "
                f"direct_keyframes={result['direct_keyframes']} "
                f"direct_points={result['direct_points']} "
                f"top_classes={result['top_classes']}"
            )
    if args.metrics_out:
        path = Path(args.metrics_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "config": {
                "compute": cfg.compute.__dict__,
                "backend": cfg.backend.__dict__,
                "depth": cfg.depth.__dict__,
            },
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
