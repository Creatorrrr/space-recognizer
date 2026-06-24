"""Direct RGB-D fusion for metric-depth sources such as OAK-D-Lite.

This path intentionally does not run DA3. It converts metric RGB-D keyframes
into the existing BackendResult shape so GlobalMap, MeshMap, persistence, and
visualization keep sharing the same downstream code.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .backend import BackendResult
from .calib import DepthCalibration
from .config import CaptureCfg, FusionCfg
from .geometry import SIM3_IDENTITY


@dataclass
class DirectFusionKeyframe:
    kf_id: int
    ts: float
    bgr: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray
    dyn_mask: np.ndarray | None = None
    depth_conf: np.ndarray | None = None


class DirectFusionBackend:
    """Backend-compatible direct fusion handle.

    The implementation is synchronous on add_keyframe for now, but results are
    still queued and drained through the same main-loop path as DA3 backend
    results. Heavy TSDF work remains in MeshMap integration downstream.
    """

    def __init__(self, cfg: FusionCfg, capture: CaptureCfg):
        self.cfg = cfg
        self.capture = capture
        self.results: queue.Queue = queue.Queue()
        self._mesh_views: list[dict[str, np.ndarray | int]] = []

    def start(self) -> None:
        pass

    def wait_ready(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def add_keyframe(self, keyframe: DirectFusionKeyframe) -> None:
        self.results.put(self._build_result(keyframe))

    def _build_result(self, keyframe: DirectFusionKeyframe) -> BackendResult:
        t0 = time.monotonic()
        depth = np.asarray(keyframe.depth_m, dtype=np.float32)
        bgr = _resize_color_to_depth(np.asarray(keyframe.bgr), depth.shape)
        K = np.asarray(keyframe.K, dtype=np.float64)
        T_wc = np.asarray(keyframe.T_wc, dtype=np.float64)
        valid = _valid_depth_mask(depth, keyframe.dyn_mask, keyframe.depth_conf,
                                  self.capture, self.cfg)
        points, colors = _backproject_points(depth, bgr, valid, K, T_wc,
                                             self.cfg.direct_point_subsample)
        mesh_payload = self._buffer_mesh_view(keyframe, bgr, depth, valid, K, T_wc)
        origin = T_wc[:3, 3].astype(np.float64)[None]
        point_view_idx = np.zeros(len(points), dtype=np.uint8)
        window_ids = (mesh_payload["window_ids"] if mesh_payload is not None
                      else [int(keyframe.kf_id)])

        return BackendResult(
            points=points,
            colors=colors,
            T_global_live=SIM3_IDENTITY,
            calib=DepthCalibration(a=1.0, b=0.0, inlier_frac=1.0),
            kf_global_poses={int(keyframe.kf_id): T_wc.copy()},
            intrinsics=K.copy(),
            depth_size=(int(depth.shape[1]), int(depth.shape[0])),
            meters_per_unit=1.0,
            view_origins=origin,
            point_view_idx=point_view_idx,
            view_depths=None if mesh_payload is None else mesh_payload["depths"],
            view_valid=None if mesh_payload is None else mesh_payload["valids"],
            view_colors=None if mesh_payload is None else mesh_payload["colors"],
            view_poses=None if mesh_payload is None else mesh_payload["poses"],
            view_intrinsics=None if mesh_payload is None else mesh_payload["Ks"],
            anchor_kf_id=(None if mesh_payload is None
                          else int(mesh_payload["window_ids"][0])),
            window_ids=window_ids,
            runtime_s=time.monotonic() - t0,
        )

    def _buffer_mesh_view(
        self,
        keyframe: DirectFusionKeyframe,
        bgr: np.ndarray,
        depth: np.ndarray,
        valid: np.ndarray,
        K: np.ndarray,
        T_wc: np.ndarray,
    ) -> dict[str, np.ndarray | list[int]] | None:
        depth_v = np.where(valid, depth, 0.0).astype(np.float32)
        rgb = bgr[..., ::-1].astype(np.uint8)
        depth_v, rgb, valid_v, K_v = _downsample_view(
            depth_v, rgb, valid, K, self.cfg.direct_mesh_downsample)
        self._mesh_views.append({
            "kf_id": int(keyframe.kf_id),
            "depth": depth_v,
            "color": rgb,
            "valid": valid_v,
            "pose": T_wc.copy(),
            "K": K_v,
        })

        window_size = max(2, int(self.cfg.direct_mesh_window_size))
        if len(self._mesh_views) < window_size:
            return None

        window = list(self._mesh_views)
        overlap = int(np.clip(self.cfg.direct_mesh_overlap, 0, window_size - 1))
        self._mesh_views = window[-overlap:] if overlap else []
        return {
            "depths": np.stack([v["depth"] for v in window]).astype(np.float32),
            "colors": np.stack([v["color"] for v in window]).astype(np.uint8),
            "valids": np.stack([v["valid"] for v in window]).astype(bool),
            "poses": np.stack([v["pose"] for v in window]).astype(np.float64),
            "Ks": np.stack([v["K"] for v in window]).astype(np.float64),
            "window_ids": [int(v["kf_id"]) for v in window],
        }


def _resize_color_to_depth(bgr: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    if bgr.shape[:2] == depth_shape:
        return bgr.astype(np.uint8, copy=False)
    return cv2.resize(bgr, (depth_shape[1], depth_shape[0]),
                      interpolation=cv2.INTER_AREA).astype(np.uint8)


def _resize_mask(mask: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.shape[:2] != depth_shape:
        mask_arr = cv2.resize(mask_arr.astype(np.uint8), (depth_shape[1], depth_shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    return mask_arr.astype(bool)


def _valid_depth_mask(
    depth: np.ndarray,
    dyn_mask: np.ndarray | None,
    depth_conf: np.ndarray | None,
    capture: CaptureCfg,
    fusion: FusionCfg,
) -> np.ndarray:
    valid = (np.isfinite(depth)
             & (depth >= float(capture.oak_depth_min_m))
             & (depth <= float(capture.oak_depth_max_m)))
    if depth_conf is not None:
        valid &= _resize_mask(depth_conf, depth.shape)
    if dyn_mask is not None:
        mask = _resize_mask(dyn_mask, depth.shape)
        radius = max(0, int(fusion.direct_mask_dilate_px))
        if radius > 0 and np.any(mask):
            kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        valid &= ~mask
    if bool(fusion.direct_edge_filter):
        valid &= _edge_consistent_mask(depth, float(fusion.direct_edge_rel_thresh))
    return valid


def _edge_consistent_mask(depth: np.ndarray, rel_thresh: float) -> np.ndarray:
    if rel_thresh <= 0:
        return np.ones(depth.shape, dtype=bool)
    z = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(z) & (z > 1e-6)
    edge = np.zeros(z.shape, dtype=bool)
    if z.shape[1] > 1:
        denom = np.maximum(np.minimum(z[:, 1:], z[:, :-1]), 1e-6)
        bad = finite[:, 1:] & finite[:, :-1] & (np.abs(z[:, 1:] - z[:, :-1]) / denom > rel_thresh)
        edge[:, 1:] |= bad
        edge[:, :-1] |= bad
    if z.shape[0] > 1:
        denom = np.maximum(np.minimum(z[1:, :], z[:-1, :]), 1e-6)
        bad = finite[1:, :] & finite[:-1, :] & (np.abs(z[1:, :] - z[:-1, :]) / denom > rel_thresh)
        edge[1:, :] |= bad
        edge[:-1, :] |= bad
    return ~edge


def _backproject_points(
    depth: np.ndarray,
    bgr: np.ndarray,
    valid: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(stride))
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride].astype(np.float32)
    z_grid = depth[::stride, ::stride]
    keep = valid[::stride, ::stride] & (z_grid > 0)
    if not np.any(keep):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    z = z_grid[keep].astype(np.float64)
    u = us[keep].astype(np.float64)
    v = vs[keep].astype(np.float64)
    cam = np.stack([
        (u - K[0, 2]) / K[0, 0] * z,
        (v - K[1, 2]) / K[1, 1] * z,
        z,
    ], axis=1)
    world = cam @ T_wc[:3, :3].T + T_wc[:3, 3]
    colors = bgr[::stride, ::stride, ::-1][keep].astype(np.uint8)
    return world.astype(np.float32), colors


def _downsample_view(
    depth: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
    K: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    factor = max(1, int(factor))
    if factor == 1:
        return depth.astype(np.float32), rgb.astype(np.uint8), valid.astype(bool), K.copy()
    h, w = depth.shape
    new_w = max(1, int(round(w / factor)))
    new_h = max(1, int(round(h / factor)))
    depth_small = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    valid_small = cv2.resize(valid.astype(np.uint8), (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    K_small = K.copy()
    K_small[0] *= new_w / w
    K_small[1] *= new_h / h
    return (depth_small.astype(np.float32), rgb_small.astype(np.uint8),
            valid_small, K_small)
