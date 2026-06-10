"""Orchestrator: live depth + detection + visual odometry + 3D visualization."""

from __future__ import annotations

import argparse
import logging
import time

import spacerec  # noqa: F401  (env vars must be set before heavy imports)

logging.disable(logging.INFO)  # DA3 래퍼가 추론마다 INFO 3줄을 출력해 콘솔이 가려짐

import cv2
import numpy as np

from .backend import BackendKeyframe, ReconstructionBackend
from .calib import DepthCalibration
from .capture import VideoSource
from .config import Config
from .depth import DepthEstimator
from .detect import Detection, ObjectDetector
from .graph import build_graph
from .objects import ObjectRegistry, localize_objects
from .viz import Visualizer
from .vo import VisualOdometry, default_intrinsics
from .worldmap import GlobalMap


def _drain_backend_results(backend: ReconstructionBackend, worldmap: GlobalMap,
                           viz: Visualizer, calib: DepthCalibration,
                           vo: VisualOdometry, frame_wh: tuple[int, int]
                           ) -> DepthCalibration:
    while True:
        try:
            res = backend.results.get_nowait()
        except Exception:
            return calib
        worldmap.fuse(res.points, res.colors)
        worldmap.set_correction_target(res.T_global_live)
        if res.calib.inlier_frac > 0.3:
            calib = res.calib
        if res.intrinsics is not None and res.depth_size[0] > 0:
            # DA3가 추정한 intrinsics를 라이브 해상도로 환산해 VO에 반영
            K = res.intrinsics.copy()
            K[0] *= frame_wh[0] / res.depth_size[0]
            K[1] *= frame_wh[1] / res.depth_size[1]
            vo.set_intrinsics(K)
        viz.log_global_map(worldmap.points, worldmap.colors)
        print(f"[backend] window={len(res.window_ids)}kf "
              f"{res.runtime_s:.1f}s map={len(worldmap.points)}pts "
              f"calib a={calib.a:.3f} b={calib.b:.3f} "
              f"scale={res.T_global_live[0]:.3f}")


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
    worldmap = GlobalMap(cfg.backend)
    backend = ReconstructionBackend(cfg.backend, cfg.depth.model,
                                    depth_est.device, cfg.depth.process_res)
    backend.start()
    print("waiting for backend process...")
    backend.wait_ready()
    calib = DepthCalibration()
    registry = ObjectRegistry(cfg.objects)
    dyn_classes = set(cfg.detect.dynamic_classes)
    sub = cfg.viz.point_subsample
    bw = cfg.depth.process_res  # 백엔드 입력 가로 해상도
    bh = int(H * bw / W)
    kf_counter = 0

    print(f"running on {cfg.source!r} ({W}x{H})")
    frame_count, t_start = 0, time.monotonic()
    try:
        for frame in source.frames():
            if args.max_seconds and frame.ts > args.max_seconds:
                break
            viz.set_time(frame.ts)
            detections = detector.track(frame.bgr)
            raw_depth = depth_est.infer(frame.bgr)
            depth = calib.apply(raw_depth)
            gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
            excl = dynamic_mask(detections, (H, W), dyn_classes)
            pose = vo.process(gray, depth, frame.ts, excl)

            if pose.is_keyframe:
                small = cv2.resize(frame.bgr, (bw, bh), interpolation=cv2.INTER_AREA)
                backend.add_keyframe(BackendKeyframe(
                    kf_id=kf_counter, ts=frame.ts,
                    rgb=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                    T_wc_live=pose.T_wc.copy(),
                    raw_depth=cv2.resize(raw_depth, (bw, bh)),
                    dyn_mask=None if excl is None else
                             cv2.resize(excl.astype(np.uint8), (bw, bh),
                                        interpolation=cv2.INTER_NEAREST).astype(bool),
                ))
                kf_counter += 1

            # 백엔드 결과 반영 (논블로킹)
            calib = _drain_backend_results(backend, worldmap, viz, calib, vo, (W, H))
            worldmap.step_correction()

            T_wc_global = worldmap.to_global_pose(pose.T_wc)
            viz.log_frame(frame.bgr, depth, detections)
            viz.log_camera(T_wc_global, vo.K, W, H)
            if pose.is_keyframe:
                # 키프레임마다 현재 프레임 포인트클라우드 미리보기 (카메라 좌표계,
                # world/camera 트리의 전역 pose 변환이 적용됨)
                scale = worldmap.T_global_live[0]
                d = depth[::sub, ::sub] * scale
                Kv = vo.K
                vs, us = np.mgrid[0:H:sub, 0:W:sub].astype(np.float32)
                pts = np.stack([(us - Kv[0, 2]) / Kv[0, 0] * d,
                                (vs - Kv[1, 2]) / Kv[1, 1] * d, d], axis=-1)
                cols = frame.bgr[::sub, ::sub, ::-1]
                viz.log_live_points(pts.reshape(-1, 3), cols.reshape(-1, 3))
            located = localize_objects(detections, depth, vo.K, pose.T_wc)
            located_global = [(det, worldmap.to_global_points(p[None])[0])
                              for det, p in located]
            visible = registry.update(located_global, frame.ts)
            objects = registry.stable_objects(frame.ts)
            viz.log_objects(objects, build_graph(objects, cfg.graph), visible)
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.monotonic() - t_start)
                p = pose.T_wc[:3, 3]
                print(f"t={frame.ts:5.1f}s processed={frame_count} avg {fps:.1f} FPS | "
                      f"pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"inliers={pose.inlier_ratio:.2f} n={pose.n_tracked}"
                      f"{' LOST' if pose.lost else ''}")
        # 영상 종료 후 진행 중인 백엔드 윈도 결과를 기다려 지도에 반영
        print("video ended; draining backend...")
        idle_since = time.monotonic()
        while time.monotonic() - idle_since < 8.0:
            before = len(worldmap.points)
            calib = _drain_backend_results(backend, worldmap, viz, calib, vo, (W, H))
            for _ in range(10):
                worldmap.step_correction()
            if len(worldmap.points) != before:
                idle_since = time.monotonic()
            time.sleep(0.3)
    finally:
        backend.stop()
        source.release()
        elapsed = time.monotonic() - t_start
        if frame_count:
            print(f"done: {frame_count} frames in {elapsed:.1f}s "
                  f"({frame_count / elapsed:.1f} FPS)")
        if registry.objects:
            print(f"world objects ({len(registry.objects)}):")
            for o in registry.objects.values():
                print(f"  {o.label:>16} pos=({o.position[0]:+.2f},"
                      f"{o.position[1]:+.2f},{o.position[2]:+.2f}) "
                      f"obs={o.n_obs} last_seen={o.last_seen:.1f}s")


if __name__ == "__main__":
    main()
