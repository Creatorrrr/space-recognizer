"""Smoke-check a connected OAK device and print usable calibration metadata.

Usage:
  .venv/bin/python benchmarks/oak_smoke.py
  .venv/bin/python benchmarks/oak_smoke.py --frames 30
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import spacerec  # noqa: F401
from spacerec.config import Config
from spacerec.oak import OakSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.capture.source_kind = "oak"
    src = OakSource(cfg.capture, proc_width=cfg.proc_width)
    try:
        print("metadata:", src.metadata)
        if src.K is None:
            print("intrinsics: unavailable")
        else:
            print("intrinsics:")
            print(np.array2string(src.K, precision=2, suppress_small=True))

        t0 = time.monotonic()
        n_depth = 0
        n_imu = 0
        valid_fracs = []
        for i, frame in zip(range(args.frames), src.frames()):
            if frame.depth_m is not None:
                valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
                n_depth += 1
                valid_fracs.append(float(valid.mean()))
            imu_text = "no"
            if frame.imu is not None:
                n_imu += 1
                accel = frame.imu.get("accel")
                gyro = frame.imu.get("gyro")
                parts = []
                if accel is not None:
                    parts.append("accel=" + np.array2string(accel, precision=2))
                if gyro is not None:
                    parts.append("gyro=" + np.array2string(gyro, precision=2))
                imu_text = " ".join(parts) if parts else "yes"
            print(
                f"frame={i} rgb={frame.bgr.shape[1]}x{frame.bgr.shape[0]} "
                f"depth={'yes' if frame.depth_m is not None else 'no'} "
                f"imu={imu_text}"
            )
        dt = max(time.monotonic() - t0, 1e-6)
        print(f"fps={args.frames / dt:.1f} depth_frames={n_depth} imu_frames={n_imu}")
        if valid_fracs:
            print(f"depth_valid_mean={np.mean(valid_fracs):.3f}")
    finally:
        src.release()


if __name__ == "__main__":
    main()
