"""Lightweight visual odometry: LK optical flow + PnP RANSAC against keyframes.

Replaces the real-time pose role of a learned SLAM frontend. Keyframe features
are back-projected to 3D with the live (DA3) depth; subsequent frames track
those features with pyramidal Lucas-Kanade and solve PnP for the camera pose
in the *live* world frame (the first camera's frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import VoCfg


def default_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Horizontal-FOV pinhole guess until the backend provides real intrinsics."""
    fx = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    return np.array([[fx, 0, width / 2],
                     [0, fx, height / 2],
                     [0, 0, 1]], dtype=np.float64)


def backproject(pts2d: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel coords (N,2) + z-depth map -> camera-frame 3D points (N,3)."""
    u = pts2d[:, 0]
    v = pts2d[:, 1]
    z = depth[np.clip(v.astype(int), 0, depth.shape[0] - 1),
              np.clip(u.astype(int), 0, depth.shape[1] - 1)]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


@dataclass
class Keyframe:
    ts: float
    gray: np.ndarray
    depth: np.ndarray           # live-calibrated z-depth at frame resolution
    T_wc: np.ndarray            # camera-to-world (live frame), 4x4
    obj_masks: np.ndarray | None  # bool HxW: pixels belonging to dynamic objects
    bgr: np.ndarray | None = None  # kept only for backend keyframes


@dataclass
class PoseResult:
    T_wc: np.ndarray
    inlier_ratio: float
    n_tracked: int
    is_keyframe: bool
    lost: bool = False
    # PnP inlier 특징점들: 현재 프레임 픽셀 좌표와, 키프레임 3D로부터 예측한
    # 카메라 z-depth. depth 맵의 프레임별 스케일 검증/보정에 사용한다.
    feat_uv: np.ndarray | None = None
    feat_z: np.ndarray | None = None


_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class VisualOdometry:
    def __init__(self, K: np.ndarray, cfg: VoCfg):
        self.K = K.copy()
        self.cfg = cfg
        self.T_wc = np.eye(4)
        self.keyframe: Keyframe | None = None
        self._prev_gray: np.ndarray | None = None
        self._pts2d: np.ndarray | None = None   # tracked keyframe features, current pos
        self._pts3d: np.ndarray | None = None   # their world-frame 3D (fixed per keyframe)
        self._kf_pts2d: np.ndarray | None = None  # positions at keyframe creation

    def set_intrinsics(self, K: np.ndarray) -> None:
        self.K = K.copy()

    def process(self, gray: np.ndarray, depth: np.ndarray, ts: float,
                exclude_mask: np.ndarray | None,
                R_delta_prev: np.ndarray | None = None,
                R_since_keyframe: np.ndarray | None = None,
                omega_norm: float | None = None) -> PoseResult:
        if self.keyframe is None:
            self._make_keyframe(gray, depth, ts, exclude_mask)
            return PoseResult(self.T_wc.copy(), 1.0, 0, True)

        result = self._track(gray, R_delta_prev, R_since_keyframe)
        need_kf = (
            result.lost
            or ts - self.keyframe.ts >= self.cfg.keyframe_interval_s
            or result.inlier_ratio < self.cfg.min_inlier_ratio
            or self._median_flow() > self.cfg.keyframe_min_flow_px
        )
        if need_kf:
            self._make_keyframe(gray, depth, ts, exclude_mask)
            result.is_keyframe = True
        else:
            self._prev_gray = gray
        return result

    def _median_flow(self) -> float:
        if self._pts2d is None or self._kf_pts2d is None or len(self._pts2d) == 0:
            return np.inf
        return float(np.median(np.linalg.norm(self._pts2d - self._kf_pts2d, axis=1)))

    def _warp_points_with_rotation(self, pts: np.ndarray,
                                   R_delta: np.ndarray) -> np.ndarray:
        H = self.K @ np.asarray(R_delta, dtype=np.float64).reshape(3, 3) @ np.linalg.inv(self.K)
        homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
        warped = (H @ homog.T).T
        z = warped[:, 2:3]
        valid = np.abs(z) > 1e-12
        out = pts.astype(np.float32).copy()
        out[valid[:, 0]] = (warped[valid[:, 0], :2] / z[valid[:, 0]]).astype(np.float32)
        return out

    def _lk_track(self, gray: np.ndarray, p0: np.ndarray,
                  initial: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = dict(_LK_PARAMS)
        flags = int(params.pop("flags", 0))
        if initial is None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, p0, None, flags=flags, **params)
        else:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, p0, initial.copy(),
                flags=flags | cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
        if p1 is None or st is None:
            empty = np.zeros((len(p0), 1, 2), dtype=np.float32)
            return empty, np.zeros((len(p0), 1), dtype=np.uint8), np.full(len(p0), np.inf)
        p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, p1, None, flags=flags, **params)
        if p0r is None or st_b is None:
            return p1, st, np.full(len(p0), np.inf)
        fb_err = np.linalg.norm(p0 - p0r, axis=2).ravel()
        good_back = st_b.ravel() == 1
        fb_err[~good_back] = np.inf
        return p1, st, fb_err

    def _choose_lk_result(self, base: tuple[np.ndarray, np.ndarray, np.ndarray],
                          aided: tuple[np.ndarray, np.ndarray, np.ndarray] | None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if aided is None:
            return base

        def score(item: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, float]:
            _, st, fb = item
            good = (st.ravel() == 1) & (fb < 1.5)
            median = float(np.median(fb[good])) if np.any(good) else np.inf
            return int(good.sum()), median

        base_good, base_med = score(base)
        aided_good, aided_med = score(aided)
        if aided_good > base_good or (aided_good == base_good and aided_med <= base_med):
            return aided
        return base

    def _solve_pnp(self, pts3d: np.ndarray, pts2d: np.ndarray,
                   R_guess: np.ndarray | None = None):
        kwargs = dict(reprojectionError=4.0, iterationsCount=100,
                      flags=cv2.SOLVEPNP_ITERATIVE)
        if R_guess is not None:
            rvec0, _ = cv2.Rodrigues(np.asarray(R_guess, dtype=np.float64).reshape(3, 3))
            kwargs.update(rvec=rvec0, tvec=np.zeros((3, 1), dtype=np.float64),
                          useExtrinsicGuess=True)
        return cv2.solvePnPRansac(
            pts3d.astype(np.float64), pts2d.astype(np.float64), self.K, None,
            **kwargs)

    def _track(self, gray: np.ndarray, R_delta_prev: np.ndarray | None = None,
               R_since_keyframe: np.ndarray | None = None) -> PoseResult:
        if self._pts2d is None or len(self._pts2d) < 8:
            return PoseResult(self.T_wc.copy(), 0.0, 0, False, lost=True)

        p0 = self._pts2d.astype(np.float32).reshape(-1, 1, 2)
        base_lk = self._lk_track(gray, p0)
        aided_lk = None
        if R_delta_prev is not None:
            initial = self._warp_points_with_rotation(p0.reshape(-1, 2), R_delta_prev)
            aided_lk = self._lk_track(gray, p0, initial.reshape(-1, 1, 2))
        p1, st, fb_err = self._choose_lk_result(base_lk, aided_lk)
        good = (st.ravel() == 1) & (fb_err < 1.5)

        self._pts2d = p1.reshape(-1, 2)[good]
        self._pts3d = self._pts3d[good]
        self._kf_pts2d = self._kf_pts2d[good]
        n = len(self._pts2d)
        if n < 8:
            return PoseResult(self.T_wc.copy(), 0.0, n, False, lost=True)

        ok, rvec, tvec, inliers = self._solve_pnp(
            self._pts3d, self._pts2d, R_since_keyframe)
        if (not ok or inliers is None or len(inliers) < 6) and R_since_keyframe is not None:
            ok, rvec, tvec, inliers = self._solve_pnp(self._pts3d, self._pts2d)
        if not ok or inliers is None or len(inliers) < 6:
            return PoseResult(self.T_wc.copy(), 0.0, n, False, lost=True)

        R, _ = cv2.Rodrigues(rvec)
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = tvec.ravel()
        self.T_wc = np.linalg.inv(T_cw)

        idx = inliers.ravel()
        z_pred = (self._pts3d[idx] @ R.T + tvec.ravel())[:, 2]
        return PoseResult(self.T_wc.copy(), len(inliers) / n, n, False,
                          feat_uv=self._pts2d[idx].copy(), feat_z=z_pred)

    def _make_keyframe(self, gray: np.ndarray, depth: np.ndarray, ts: float,
                       exclude_mask: np.ndarray | None) -> None:
        feat_mask = np.full(gray.shape, 255, np.uint8)
        if exclude_mask is not None:
            feat_mask[exclude_mask] = 0
        # 화면 가장자리는 왜곡/롤링셔터 영향이 커서 제외
        m = int(min(gray.shape) * 0.02)
        feat_mask[:m] = feat_mask[-m:] = 0
        feat_mask[:, :m] = feat_mask[:, -m:] = 0

        corners = cv2.goodFeaturesToTrack(gray, maxCorners=self.cfg.max_corners,
                                          qualityLevel=0.01, minDistance=12,
                                          mask=feat_mask, blockSize=7)
        self.keyframe = Keyframe(ts=ts, gray=gray, depth=depth,
                                 T_wc=self.T_wc.copy(), obj_masks=exclude_mask)
        self._prev_gray = gray
        if corners is None or len(corners) < 8:
            self._pts2d = self._pts3d = self._kf_pts2d = None
            return

        pts2d = corners.reshape(-1, 2)
        cam_pts = backproject(pts2d, depth, self.K)
        valid = cam_pts[:, 2] > 1e-6
        pts2d, cam_pts = pts2d[valid], cam_pts[valid]
        world = (self.T_wc[:3, :3] @ cam_pts.T).T + self.T_wc[:3, 3]
        self._pts2d = pts2d.copy()
        self._kf_pts2d = pts2d.copy()
        self._pts3d = world
