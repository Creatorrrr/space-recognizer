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
    det = ObjectDetector(cfg.detect.model, conf=cfg.detect.conf,
                         vocabulary=cfg.detect.vocabulary)
    dep = DepthEstimator(cfg.depth.model, precision=cfg.compute.precision)
    src = VideoSource(cfg.source, proc_width=cfg.proc_width, realtime=False)
    W, H = src.proc_width, src.proc_height
    vo = VisualOdometry(default_intrinsics(W, H), cfg.vo)
    wm = GlobalMap(cfg.backend)
    be = ReconstructionBackend(cfg.backend, cfg.depth.model, dep.device,
                               cfg.depth.process_res,
                               precision=cfg.compute.precision)
    be.start()
    be.wait_ready()
    calib = DepthCalibration()
    bw = cfg.depth.process_res
    bh = int(H * bw / W)
    kf_id = 0
    dyn = set(cfg.detect.dynamic_classes)
    traj = []

    frame_scale = 1.0
    K_samples: list[np.ndarray] = []

    def drain() -> None:
        nonlocal calib, frame_scale
        while True:
            try:
                res = be.results.get_nowait()
            except Exception:
                return
            wm.fuse(res.points, res.colors)
            wm.set_correction_target(res.T_global_live)
            if res.calib.inlier_frac > 0.3:
                calib = res.calib
                frame_scale = 1.0  # main.py와 동일: 이중 적용 방지
            # K는 첫 프레임에서 고정 — 도중에 바꾸면 지도 스케일이 갈라진다
            print(f"[res] map={len(wm.points)} scale={res.T_global_live[0]:.3f} "
                  f"a={calib.a:.3f} fx={vo.K[0, 0]:.0f}")

    for i, frame in enumerate(src.frames()):
        if i % args.stride:
            continue
        raw = dep.infer(frame.bgr)
        if len(K_samples) < 10:  # main.py와 동일한 K 워밍업
            if dep.last_K is not None:
                K_samples.append(dep.last_K)
            if len(K_samples) == 10:
                vo.set_intrinsics(np.median(np.stack(K_samples), axis=0))
            else:
                continue
        d = calib.apply(raw) * frame_scale
        dets = det.track(frame.bgr)
        gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
        excl = dynamic_mask(dets, (H, W), dyn)
        r = vo.process(gray, d, frame.ts, excl)
        if r.feat_uv is not None and len(r.feat_uv) >= 20:
            u = r.feat_uv[:, 0].astype(int).clip(0, W - 1)
            v = r.feat_uv[:, 1].astype(int).clip(0, H - 1)
            z_meas = d[v, u]
            ok = (z_meas > 1e-6) & (r.feat_z > 1e-6)
            if ok.sum() >= 20:
                ratio = float(np.clip(np.median(r.feat_z[ok] / z_meas[ok]),
                                      0.8, 1.25))
                frame_scale = float(np.clip(frame_scale * ratio ** 0.3, 0.5, 2.0))
        traj.append(wm.to_global_pose(r.T_wc)[:3, 3])
        if r.is_keyframe:
            small = cv2.resize(frame.bgr, (bw, bh))
            be.add_keyframe(BackendKeyframe(
                kf_id, frame.ts, cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                r.T_wc.copy(), cv2.resize(raw, (bw, bh)),
                None if excl is None else
                cv2.resize(excl.astype(np.uint8), (bw, bh)).astype(bool),
                (calib.a * frame_scale, calib.b * frame_scale)))
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
