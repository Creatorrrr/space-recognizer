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


@dataclass
class PnpCandidate:
    name: str
    ok: bool
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    inliers: np.ndarray | None
    n_total: int
    reproj_median_px: float = np.inf

    @property
    def n_inliers(self) -> int:
        return 0 if self.inliers is None else int(len(self.inliers))

    @property
    def inlier_ratio(self) -> float:
        return 0.0 if self.n_total <= 0 else self.n_inliers / self.n_total


def reprojection_median_px(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    idx: np.ndarray,
) -> float:
    if len(idx) == 0:
        return np.inf
    proj, _ = cv2.projectPoints(
        pts3d[idx].astype(np.float64),
        rvec.astype(np.float64),
        tvec.astype(np.float64),
        K.astype(np.float64),
        None,
    )
    err = np.linalg.norm(proj.reshape(-1, 2) - pts2d[idx], axis=1)
    return float(np.median(err)) if len(err) else np.inf


class VisualOdometry:
    def __init__(self, K: np.ndarray, cfg: VoCfg):
        self.K = K.copy()
        self.cfg = cfg
        self.T_wc = np.eye(4)
        self.keyframe: Keyframe | None = None
        self._prev_gray: np.ndarray | None = None
        self._pts2d: np.ndarray | None = None   # tracked keyframe features, current pos
        self._pts3d: np.ndarray | None = None   # their world-frame 3D (fixed per keyframe)
        self._pts3d_keyframe: np.ndarray | None = None
        self._kf_pts2d: np.ndarray | None = None  # positions at keyframe creation
        self._prev_ts: float | None = None

    def set_intrinsics(self, K: np.ndarray) -> None:
        self.K = K.copy()

    def process(self, gray: np.ndarray, depth: np.ndarray, ts: float,
                exclude_mask: np.ndarray | None,
                R_delta_prev: np.ndarray | None = None,
                R_since_keyframe: np.ndarray | None = None,
                omega_norm: float | None = None) -> PoseResult:
        if self.keyframe is None:
            self._make_keyframe(gray, depth, ts, exclude_mask)
            self._prev_ts = ts
            return PoseResult(self.T_wc.copy(), 1.0, 0, True)

        result = self._track(gray, ts, R_delta_prev, R_since_keyframe)
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
        self._prev_ts = ts
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

    def _relative_guess_from_previous(self, R_since_keyframe: np.ndarray | None):
        if self.keyframe is None:
            return None, None
        T_cw_prev = np.linalg.inv(self.T_wc)
        T_prev_keyframe = T_cw_prev @ self.keyframe.T_wc
        R_guess = T_prev_keyframe[:3, :3]
        if R_since_keyframe is not None:
            R_guess = np.asarray(R_since_keyframe, dtype=np.float64).reshape(3, 3)
        return R_guess, T_prev_keyframe[:3, 3].reshape(3, 1)

    def _candidate_to_world_pose(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        if self.keyframe is None:
            return self.T_wc.copy()
        R_ck, _ = cv2.Rodrigues(rvec)
        T_ck = np.eye(4)
        T_ck[:3, :3] = R_ck
        T_ck[:3, 3] = tvec.ravel()
        return self.keyframe.T_wc @ np.linalg.inv(T_ck)

    def _pose_step_limit(self, dt: float, median_z: float) -> float:
        velocity_cap = self.cfg.pnp_max_velocity_units_s * max(dt, 0.0)
        depth_cap = self.cfg.pnp_max_step_depth_frac * max(float(median_z), 0.0)
        return max(self.cfg.pnp_step_floor_units, velocity_cap, depth_cap)

    def _candidate_pose_step(self, cand: PnpCandidate) -> float:
        if not cand.ok or cand.rvec is None or cand.tvec is None:
            return np.inf
        T_wc = self._candidate_to_world_pose(cand.rvec, cand.tvec)
        return float(np.linalg.norm(T_wc[:3, 3] - self.T_wc[:3, 3]))

    def _candidate_median_z(self, cand: PnpCandidate, pts3d: np.ndarray) -> float:
        if not cand.ok or cand.rvec is None or cand.tvec is None or cand.inliers is None:
            return 0.0
        R, _ = cv2.Rodrigues(cand.rvec)
        idx = cand.inliers.ravel()
        z = (pts3d[idx] @ R.T + cand.tvec.ravel())[:, 2]
        z = z[np.isfinite(z) & (z > 1e-6)]
        return float(np.median(z)) if len(z) else 0.0

    def _passes_motion_gate(self, cand: PnpCandidate, pts3d: np.ndarray, dt: float) -> bool:
        median_z = self._candidate_median_z(cand, pts3d)
        return self._candidate_pose_step(cand) <= self._pose_step_limit(dt, median_z)

    def _choose_pnp_result(
        self,
        base: PnpCandidate,
        aided: PnpCandidate | None,
        pts3d: np.ndarray,
        dt: float,
    ) -> PnpCandidate | None:
        valid_base = base.ok and base.inliers is not None and base.n_inliers >= 6
        valid_aided = (
            aided is not None and aided.ok and aided.inliers is not None
            and aided.n_inliers >= 6
        )
        if valid_aided and not self._passes_motion_gate(aided, pts3d, dt):
            valid_aided = False
        if valid_base and not self._passes_motion_gate(base, pts3d, dt):
            valid_base = False
        if not valid_base and not valid_aided:
            return None
        if valid_base and not valid_aided:
            return base
        if valid_aided and not valid_base:
            return aided

        assert aided is not None
        inlier_ok = aided.n_inliers >= base.n_inliers + self.cfg.pnp_aided_min_inlier_delta
        reproj_ok = aided.reproj_median_px <= base.reproj_median_px * self.cfg.pnp_aided_reproj_tol
        base_step = max(self._candidate_pose_step(base), self.cfg.pnp_step_floor_units)
        aided_step = self._candidate_pose_step(aided)
        divergence_ok = aided_step <= base_step * self.cfg.pnp_divergence_step_factor
        return aided if inlier_ok and reproj_ok and divergence_ok else base

    def _solve_pnp(self, pts3d: np.ndarray, pts2d: np.ndarray,
                   R_guess: np.ndarray | None = None,
                   tvec_guess: np.ndarray | None = None,
                   name: str = "base") -> PnpCandidate:
        kwargs = dict(reprojectionError=4.0, iterationsCount=100,
                      flags=cv2.SOLVEPNP_ITERATIVE)
        if R_guess is not None and tvec_guess is not None:
            rvec0, _ = cv2.Rodrigues(np.asarray(R_guess, dtype=np.float64).reshape(3, 3))
            kwargs.update(rvec=rvec0,
                          tvec=np.asarray(tvec_guess, dtype=np.float64).reshape(3, 1),
                          useExtrinsicGuess=True)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64), pts2d.astype(np.float64), self.K, None,
            **kwargs)
        cand = PnpCandidate(name, bool(ok), rvec, tvec, inliers, len(pts2d))
        if cand.ok and cand.inliers is not None and cand.rvec is not None and cand.tvec is not None:
            idx = cand.inliers.ravel()
            cand.reproj_median_px = reprojection_median_px(
                pts3d, pts2d, cand.rvec, cand.tvec, self.K, idx)
        return cand

    def _track(self, gray: np.ndarray, ts: float,
               R_delta_prev: np.ndarray | None = None,
               R_since_keyframe: np.ndarray | None = None) -> PoseResult:
        if (self._pts2d is None or self._pts3d is None
                or self._pts3d_keyframe is None or self._kf_pts2d is None
                or len(self._pts2d) < 8):
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
        self._pts3d_keyframe = self._pts3d_keyframe[good]
        self._kf_pts2d = self._kf_pts2d[good]
        n = len(self._pts2d)
        if n < 8:
            return PoseResult(self.T_wc.copy(), 0.0, n, False, lost=True)

        pts3d_for_pnp = self._pts3d_keyframe
        if pts3d_for_pnp is None or len(pts3d_for_pnp) < 8:
            return PoseResult(self.T_wc.copy(), 0.0, n, False, lost=True)

        base = self._solve_pnp(pts3d_for_pnp, self._pts2d, name="base")
        aided = None
        if R_since_keyframe is not None:
            R_guess, t_guess = self._relative_guess_from_previous(R_since_keyframe)
            aided = self._solve_pnp(
                pts3d_for_pnp, self._pts2d,
                R_guess=R_guess, tvec_guess=t_guess, name="imu")
        dt = max(0.0, ts - self._prev_ts) if self._prev_ts is not None else 0.0
        cand = self._choose_pnp_result(base, aided, pts3d_for_pnp, dt)
        if cand is None or cand.rvec is None or cand.tvec is None or cand.inliers is None:
            return PoseResult(self.T_wc.copy(), 0.0, n, False, lost=True)

        self.T_wc = self._candidate_to_world_pose(cand.rvec, cand.tvec)
        idx = cand.inliers.ravel()
        R, _ = cv2.Rodrigues(cand.rvec)
        z_pred = (pts3d_for_pnp[idx] @ R.T + cand.tvec.ravel())[:, 2]
        return PoseResult(self.T_wc.copy(), cand.inlier_ratio, n, False,
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
            self._pts2d = self._pts3d = self._pts3d_keyframe = self._kf_pts2d = None
            return

        pts2d = corners.reshape(-1, 2)
        cam_pts = backproject(pts2d, depth, self.K)
        valid = cam_pts[:, 2] > 1e-6
        pts2d, cam_pts = pts2d[valid], cam_pts[valid]
        world = (self.T_wc[:3, :3] @ cam_pts.T).T + self.T_wc[:3, 3]
        self._pts2d = pts2d.copy()
        self._kf_pts2d = pts2d.copy()
        self._pts3d_keyframe = cam_pts.copy()
        self._pts3d = world
