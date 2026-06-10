"""Headless full-pipeline run: saves the fused map + trajectory to /tmp/map.npz.

Usage: .venv/bin/python benchmarks/headless_run.py [--stride 3]
"""

import argparse
import time

import cv2
import numpy as np

import spacerec  # noqa: F401
from spacerec.backend import BackendKeyframe, ReconstructionBackend
from spacerec.calib import DepthCalibration
from spacerec.capture import VideoSource
from spacerec.config import Config
from spacerec.depth import DepthEstimator
from spacerec.detect import ObjectDetector
from spacerec.main import dynamic_mask
from spacerec.vo import VisualOdometry, default_intrinsics
from spacerec.worldmap import GlobalMap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="/tmp/map.npz")
    args = ap.parse_args()

    cfg = Config.load()
    det = ObjectDetector(cfg.detect.model, conf=cfg.detect.conf)
    dep = DepthEstimator(cfg.depth.model)
    src = VideoSource(cfg.source, proc_width=cfg.proc_width, realtime=False)
    W, H = src.proc_width, src.proc_height
    vo = VisualOdometry(default_intrinsics(W, H), cfg.vo)
    wm = GlobalMap(cfg.backend)
    be = ReconstructionBackend(cfg.backend, cfg.depth.model, dep.device,
                               cfg.depth.process_res)
    be.start()
    be.wait_ready()
    calib = DepthCalibration()
    bw = cfg.depth.process_res
    bh = int(H * bw / W)
    kf_id = 0
    dyn = set(cfg.detect.dynamic_classes)
    traj = []

    def drain() -> None:
        nonlocal calib
        while True:
            try:
                res = be.results.get_nowait()
            except Exception:
                return
            wm.fuse(res.points, res.colors)
            wm.set_correction_target(res.T_global_live)
            if res.calib.inlier_frac > 0.3:
                calib = res.calib
            if res.intrinsics is not None:
                K = res.intrinsics.copy()
                K[0] *= W / res.depth_size[0]
                K[1] *= H / res.depth_size[1]
                vo.set_intrinsics(K)
            print(f"[res] map={len(wm.points)} scale={res.T_global_live[0]:.3f} "
                  f"a={calib.a:.3f} fx={vo.K[0, 0]:.0f}")

    for i, frame in enumerate(src.frames()):
        if i % args.stride:
            continue
        raw = dep.infer(frame.bgr)
        d = calib.apply(raw)
        dets = det.track(frame.bgr)
        gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
        excl = dynamic_mask(dets, (H, W), dyn)
        r = vo.process(gray, d, frame.ts, excl)
        traj.append(wm.to_global_pose(r.T_wc)[:3, 3])
        if r.is_keyframe:
            small = cv2.resize(frame.bgr, (bw, bh))
            be.add_keyframe(BackendKeyframe(
                kf_id, frame.ts, cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                r.T_wc.copy(), cv2.resize(raw, (bw, bh)),
                None if excl is None else
                cv2.resize(excl.astype(np.uint8), (bw, bh)).astype(bool),
                (calib.a, calib.b)))
            kf_id += 1
        drain()
        wm.step_correction()
    src.release()

    t_end = time.monotonic() + 12
    while time.monotonic() < t_end:
        drain()
        wm.step_correction()
        time.sleep(0.3)
    be.stop()
    np.savez(args.out, pts=wm.points, cols=wm.colors, traj=np.array(traj))
    print("saved:", args.out, len(wm.points), "pts,", len(traj), "poses")


if __name__ == "__main__":
    main()
