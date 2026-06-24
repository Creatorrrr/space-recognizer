"""Smoke-check recorded OAK sessions without requiring a live device.

Usage:
  .venv/bin/python benchmarks/replay_smoke.py sources/session_... --frames 120
  .venv/bin/python benchmarks/replay_smoke.py sources/session_* --frames 60 --full-models
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import spacerec  # noqa: F401
from spacerec.backend import BackendKeyframe, _Worker
from spacerec.config import Config
from spacerec.device import select_torch_device
from spacerec.objects import ObjectRegistry, localize_objects
from spacerec.replay import RecordedOakSource, _nearest
from spacerec.vo import VisualOdometry


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


def _run_backend_window(cfg: Config, keyframes: list[BackendKeyframe]) -> int:
    if not keyframes:
        return 0
    bcfg = replace(
        cfg.backend,
        window_size=max(1, len(keyframes)),
        overlap=0,
        metric_anchor=False,
    )
    worker = _Worker(bcfg, cfg.depth.model, select_torch_device(None),
                     min(cfg.depth.process_res, 252), metric_model=None)
    worker._pending.extend(keyframes)
    result = worker._run_window()
    return 0 if result is None else int(len(result.points))


def run_session(path: Path, cfg: Config, frames: int, full_models: bool,
                backend: bool) -> dict:
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
    stats = Counter()
    valid_fracs = []
    inliers = []
    tracked = []
    classes = Counter()
    keyframes: list[BackendKeyframe] = []
    payload_missing = _payload_missing(src)
    pair_count, pair_median_ms, pair_max_ms = _pairing_stats(src)
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
            detections = detector.track(frame.bgr) if detector is not None else []
            for det in detections:
                classes[det.cls_name] += 1
            pose = vo.process(
                gray,
                frame.depth_m if frame.depth_m is not None else np.ones(gray.shape, np.float32),
                frame.ts,
                _dynamic_mask(detections, gray.shape, set(cfg.detect.dynamic_classes)),
            )
            if backend and pose.is_keyframe and frame.depth_m is not None and len(keyframes) < 6:
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
            stats["lost"] += int(pose.lost)
            stats["keyframes"] += int(pose.is_keyframe)
            if pose.n_tracked:
                tracked.append(float(pose.n_tracked))
                inliers.append(float(pose.inlier_ratio))
            if detections and registry is not None and frame.depth_m is not None:
                obs = localize_objects(detections, frame.depth_m, vo.K, pose.T_wc,
                                       frame.depth_conf)
                stats["detections"] += len(detections)
                stats["object_observations"] += len(obs)
                registry.update(obs, frame.ts)
    finally:
        src.release()
    backend_points = _run_backend_window(cfg, keyframes) if backend else 0

    return {
        "session": path.name,
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
        "backend_keyframes": len(keyframes),
        "backend_points": backend_points,
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
    args = ap.parse_args()

    cfg = Config.load(args.config)
    for session in args.sessions:
        result = run_session(Path(session), cfg, args.frames, args.full_models,
                             args.backend)
        print(
            "REPLAY_SMOKE "
            f"session={result['session']} payload_missing={result['payload_missing']} "
            f"pairs={result['pair_count']} pair_median_ms={result['pair_median_ms']:.1f} "
            f"pair_max_ms={result['pair_max_ms']:.1f} frames={result['frames']} "
            f"depth_frames={result['depth_frames']} "
            f"depth_valid_mean={result['depth_valid_mean']:.3f} "
            f"lost={result['lost']} keyframes={result['keyframes']} "
            f"avg_tracked={result['avg_tracked']:.1f} "
            f"avg_inlier={result['avg_inlier']:.2f} "
            f"detections={result['detections']} "
            f"object_observations={result['object_observations']} "
            f"objects={result['objects']} "
            f"backend_keyframes={result['backend_keyframes']} "
            f"backend_points={result['backend_points']} "
            f"top_classes={result['top_classes']}"
        )


if __name__ == "__main__":
    main()
