"""DA3-Small monocular depth wrapper (MPS, fp32)."""

from __future__ import annotations

import cv2
import numpy as np
import torch

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
