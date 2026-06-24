"""DA3-Small monocular depth wrapper (MPS, fp32)."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .calib import DepthCalibration, fit_affine_depth
from .device import select_torch_device


class DepthEstimator:
    def __init__(self, model_name: str = "depth-anything/DA3-SMALL",
                 process_res: int = 504, device: str | None = None):
        from depth_anything_3.api import DepthAnything3

        self.device = select_torch_device(device)
        self.process_res = process_res
        self.model = DepthAnything3.from_pretrained(model_name).to(self.device).eval()
        self.last_K: np.ndarray | None = None  # DA3 추정 intrinsics (frame 해상도)

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Relative depth map at full frame resolution (float32, HxW)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = self.model.inference([rgb], process_res=self.process_res)
        depth = pred.depth[0].astype(np.float32)
        if pred.intrinsics is not None:
            K = pred.intrinsics[0].astype(np.float64).copy()
            dh, dw = depth.shape
            K[0] *= bgr.shape[1] / dw
            K[1] *= bgr.shape[0] / dh
            self.last_K = K
        return cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_LINEAR)


def fuse_metric_depth(primary_m: np.ndarray,
                      fallback_relative: np.ndarray | None = None,
                      valid_mask: np.ndarray | None = None,
                      min_depth_m: float = 0.3,
                      max_depth_m: float = 8.0,
                      min_valid: int = 500
                      ) -> tuple[np.ndarray, DepthCalibration, np.ndarray]:
    """Use metric stereo depth first, then optionally metric-fit DA3 holes.

    `primary_m` is expected in meters, already aligned to the RGB frame.
    `fallback_relative` can be any positive relative depth map at the same
    resolution (or resizable to it). It is affine-fitted to the reliable
    stereo pixels and only used where stereo has no valid measurement.
    """
    primary = np.asarray(primary_m, dtype=np.float32)
    valid = np.isfinite(primary) & (primary >= min_depth_m) & (primary <= max_depth_m)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    depth = np.where(valid, primary, 0.0).astype(np.float32)
    calib = DepthCalibration(inlier_frac=1.0 if int(valid.sum()) else 0.0)

    if fallback_relative is None or int(valid.sum()) < min_valid:
        return depth, calib, valid

    fallback = np.asarray(fallback_relative, dtype=np.float32)
    if fallback.shape != primary.shape:
        fallback = cv2.resize(fallback, (primary.shape[1], primary.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

    fit = fit_affine_depth(fallback, primary, valid)
    if fit.inlier_frac <= 0.3:
        return depth, fit, valid

    filled = fit.apply(fallback).astype(np.float32)
    fill = (~valid
            & np.isfinite(filled)
            & (filled >= min_depth_m)
            & (filled <= max_depth_m))
    depth[fill] = filled[fill]
    return depth, fit, valid
