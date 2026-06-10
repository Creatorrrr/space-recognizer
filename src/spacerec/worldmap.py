"""Global static map: voxel-fused point cloud + live→global Sim3 correction."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from .config import BackendCfg
from .geometry import SIM3_IDENTITY, Sim3, sim3_apply, sim3_interp, sim3_on_pose


class GlobalMap:
    def __init__(self, cfg: BackendCfg):
        self.cfg = cfg
        self.points = np.empty((0, 3), np.float32)
        self.colors = np.empty((0, 3), np.uint8)
        # live VO frame -> global map frame correction
        self._T_gl_current: Sim3 = SIM3_IDENTITY
        self._T_gl_target: Sim3 = SIM3_IDENTITY

    # ---- point fusion -------------------------------------------------
    def fuse(self, points: np.ndarray, colors: np.ndarray) -> None:
        self.points = np.concatenate([self.points, points.astype(np.float32)])
        self.colors = np.concatenate([self.colors, colors.astype(np.uint8)])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(self.colors.astype(np.float64) / 255.0)
        pc = pc.voxel_down_sample(self.cfg.voxel_size)
        self.points = np.asarray(pc.points, dtype=np.float32)
        self.colors = (np.asarray(pc.colors) * 255).astype(np.uint8)
        if len(self.points) > self.cfg.max_points:
            idx = np.random.default_rng(0).choice(
                len(self.points), self.cfg.max_points, replace=False)
            self.points, self.colors = self.points[idx], self.colors[idx]

    # ---- live -> global correction ------------------------------------
    def set_correction_target(self, T: Sim3) -> None:
        self._T_gl_target = T

    def step_correction(self, alpha: float = 0.2) -> None:
        """Called once per live frame: ease toward the target so object/camera
        positions never teleport when the backend re-anchors the map."""
        self._T_gl_current = sim3_interp(self._T_gl_current, self._T_gl_target, alpha)

    @property
    def T_global_live(self) -> Sim3:
        return self._T_gl_current

    def to_global_points(self, pts_live: np.ndarray) -> np.ndarray:
        return sim3_apply(self._T_gl_current, pts_live)

    def to_global_pose(self, T_wc_live: np.ndarray) -> np.ndarray:
        return sim3_on_pose(self._T_gl_current, T_wc_live)
