"""Orchestrator. Phase 1: live 2D recognition (depth + detection + tracking)."""

from __future__ import annotations

import argparse
import logging
import time

import spacerec  # noqa: F401  (env vars must be set before heavy imports)

logging.disable(logging.INFO)  # DA3 래퍼가 추론마다 INFO 3줄을 출력해 콘솔이 가려짐

from .capture import VideoSource
from .config import Config
from .depth import DepthEstimator
from .detect import ObjectDetector
from .viz import Visualizer


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

    print(f"running on {cfg.source!r} ({source.proc_width}x{source.proc_height})")
    frame_count, t_start = 0, time.monotonic()
    try:
        for frame in source.frames():
            if args.max_seconds and frame.ts > args.max_seconds:
                break
            viz.set_time(frame.ts)
            detections = detector.track(frame.bgr)
            depth = depth_est.infer(frame.bgr)
            viz.log_frame(frame.bgr, depth, detections)
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.monotonic() - t_start)
                print(f"t={frame.ts:5.1f}s processed={frame_count} avg {fps:.1f} FPS")
    finally:
        source.release()
        elapsed = time.monotonic() - t_start
        if frame_count:
            print(f"done: {frame_count} frames in {elapsed:.1f}s "
                  f"({frame_count / elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
