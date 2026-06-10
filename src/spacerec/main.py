"""Orchestrator: live depth + detection + visual odometry + 3D visualization."""

from __future__ import annotations

import argparse
import logging
import time

import spacerec  # noqa: F401  (env vars must be set before heavy imports)

logging.disable(logging.INFO)  # DA3 래퍼가 추론마다 INFO 3줄을 출력해 콘솔이 가려짐

import cv2
import numpy as np

from .capture import VideoSource
from .config import Config
from .depth import DepthEstimator
from .detect import Detection, ObjectDetector
from .objects import localize_objects
from .viz import Visualizer
from .vo import VisualOdometry, default_intrinsics


def dynamic_mask(detections: list[Detection], shape: tuple[int, int],
                 dynamic_classes: set[str]) -> np.ndarray | None:
    """Union mask of objects that are likely to move (excluded from VO/map)."""
    mask = None
    for det in detections:
        if det.cls_name in dynamic_classes and det.mask is not None:
            mask = det.mask if mask is None else (mask | det.mask)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="spacerec")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None,
                        help="video file path or webcam index (overrides config)")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--no-realtime", action="store_true",
                        help="process every frame instead of wall-clock pacing")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.source is not None:
        cfg.source = int(args.source) if args.source.isdigit() else args.source
    if args.no_realtime:
        cfg.realtime = False

    print("loading models...")
    detector = ObjectDetector(cfg.detect.model, conf=cfg.detect.conf)
    depth_est = DepthEstimator(cfg.depth.model, process_res=cfg.depth.process_res)
    viz = Visualizer(memory_limit=cfg.viz.memory_limit)
    source = VideoSource(cfg.source, proc_width=cfg.proc_width, realtime=cfg.realtime)
    W, H = source.proc_width, source.proc_height
    K = default_intrinsics(W, H)
    vo = VisualOdometry(K, cfg.vo)
    dyn_classes = set(cfg.detect.dynamic_classes)
    sub = cfg.viz.point_subsample

    print(f"running on {cfg.source!r} ({W}x{H})")
    frame_count, t_start = 0, time.monotonic()
    try:
        for frame in source.frames():
            if args.max_seconds and frame.ts > args.max_seconds:
                break
            viz.set_time(frame.ts)
            detections = detector.track(frame.bgr)
            depth = depth_est.infer(frame.bgr)
            gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
            excl = dynamic_mask(detections, (H, W), dyn_classes)
            pose = vo.process(gray, depth, frame.ts, excl)

            viz.log_frame(frame.bgr, depth, detections)
            viz.log_camera(pose.T_wc, vo.K, W, H)
            if pose.is_keyframe:
                # 키프레임마다 현재 프레임 포인트클라우드 미리보기 (카메라 좌표계)
                d = depth[::sub, ::sub]
                vs, us = np.mgrid[0:H:sub, 0:W:sub].astype(np.float32)
                pts = np.stack([(us - K[0, 2]) / K[0, 0] * d,
                                (vs - K[1, 2]) / K[1, 1] * d, d], axis=-1)
                cols = frame.bgr[::sub, ::sub, ::-1]
                viz.log_live_points(pts.reshape(-1, 3), cols.reshape(-1, 3))
            located = localize_objects(detections, depth, vo.K, pose.T_wc)
            viz.log_live_objects(
                [f"{d.cls_name}#{d.track_id}" for d, _ in located],
                np.array([p for _, p in located]).reshape(-1, 3))
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.monotonic() - t_start)
                p = pose.T_wc[:3, 3]
                print(f"t={frame.ts:5.1f}s processed={frame_count} avg {fps:.1f} FPS | "
                      f"pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"inliers={pose.inlier_ratio:.2f} n={pose.n_tracked}"
                      f"{' LOST' if pose.lost else ''}")
    finally:
        source.release()
        elapsed = time.monotonic() - t_start
        if frame_count:
            print(f"done: {frame_count} frames in {elapsed:.1f}s "
                  f"({frame_count / elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
