"""Phase 0 spike: measure model speeds on this machine (M1 Max, MPS).

Usage: KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python benchmarks/bench_models.py <video>

Measures:
  1. DA3-Small monocular depth, single frame (live-layer budget)
  2. DA3-Small multi-view inference, 8 views (backend budget)
  3. YOLO seg + ByteTrack, single frame (live-layer budget)
"""

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "sources/10135156-uhd_3840_2160_30fps.mp4"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def grab_frames(path: str, n: int, width: int = 1280) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        h = int(frame.shape[0] * width / frame.shape[1])
        frames.append(cv2.resize(frame, (width, h)))
    cap.release()
    return frames


def timeit(fn, warmup: int = 2, runs: int = 10) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs * 1000


def bench_da3(frames: list[np.ndarray]) -> None:
    from depth_anything_3.api import DepthAnything3

    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    model = DepthAnything3(model_name="da3-small", device=DEVICE)
    model.eval()

    ms = timeit(lambda: model.inference([rgb[0]]), warmup=3, runs=10)
    print(f"DA3-small mono 1-view @1280px input: {ms:.0f} ms/frame ({1000 / ms:.1f} FPS)")

    t0 = time.perf_counter()
    pred = model.inference(rgb[:8])
    dt = time.perf_counter() - t0
    print(f"DA3-small multi-view 8 views: {dt:.1f} s "
          f"(depth {pred.depth.shape}, ext {None if pred.extrinsics is None else pred.extrinsics.shape}, "
          f"ixt {None if pred.intrinsics is None else pred.intrinsics.shape})")

    t0 = time.perf_counter()
    model.inference(rgb[:16])
    print(f"DA3-small multi-view 16 views: {time.perf_counter() - t0:.1f} s")


def bench_yolo(frames: list[np.ndarray]) -> None:
    from ultralytics import YOLO

    for name in ("yolo26n-seg.pt", "yolo11n-seg.pt"):
        try:
            model = YOLO(name)
            ms = timeit(lambda: model.predict(frames[0], device=DEVICE, verbose=False),
                        warmup=3, runs=10)
            print(f"{name} predict @1280px: {ms:.0f} ms/frame ({1000 / ms:.1f} FPS)")
            ms = timeit(lambda: model.track(frames[0], device=DEVICE, persist=True,
                                            tracker="bytetrack.yaml", verbose=False),
                        warmup=2, runs=10)
            print(f"{name} track   @1280px: {ms:.0f} ms/frame ({1000 / ms:.1f} FPS)")
            break
        except Exception as e:
            print(f"{name} failed: {type(e).__name__}: {e}")


def main() -> None:
    print(f"device={DEVICE}, torch={torch.__version__}")
    frames = grab_frames(VIDEO, 16)
    print(f"frames: {len(frames)} @ {frames[0].shape}")
    bench_yolo(frames)
    bench_da3(frames)


if __name__ == "__main__":
    main()
