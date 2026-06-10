"""DA3-Small monocular depth wrapper (MPS, fp32)."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, model_name: str = "depth-anything/DA3-SMALL",
                 process_res: int = 504, device: str | None = None):
        from depth_anything_3.api import DepthAnything3

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.process_res = process_res
        self.model = DepthAnything3.from_pretrained(model_name).to(self.device).eval()

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Relative depth map at full frame resolution (float32, HxW)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = self.model.inference([rgb], process_res=self.process_res)
        depth = pred.depth[0].astype(np.float32)
        return cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
