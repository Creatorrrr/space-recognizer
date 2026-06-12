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

    def apply_keyframe_correction(self, C: np.ndarray) -> None:
        """방금 만든 키프레임의 pose에 SE3 보정 C를 좌측 합성한다.

        바닥 anchoring용 — 키프레임 생성 직후에만 호출해야 한다 (특징점
        3D는 키프레임 pose로 막 역투영된 상태라 같은 변환으로 일관 이동).
        """
        self.T_wc = C @ self.T_wc
        if self.keyframe is not None:
            self.keyframe.T_wc = C @ self.keyframe.T_wc
        if self._pts3d is not None:
            self._pts3d = self._pts3d @ C[:3, :3].T + C[:3, 3]

    def rescale(self, g: float) -> None:
        """live 좌표계 전체의 길이 단위를 g배로 — 스케일 서보용.

        키프레임 3D 특징점·pose 병진·키프레임 depth를 함께 키워야 새 depth
        스케일(calib×g)과 일관되고, frame_scale 피드백이 보정을 상쇄하지
        않는다. 호출자는 T_global_live 쪽도 1/g로 보정해야 한다.
        """
        self.T_wc[:3, 3] *= g
        if self._pts3d is not None:
            self._pts3d *= g
        if self.keyframe is not None:
            self.keyframe.depth = self.keyframe.depth * g
            self.keyframe.T_wc[:3, 3] *= g

    def process(self, gray: np.ndarray, depth: np.ndarray, ts: float,
                exclude_mask: np.ndarray | None) -> PoseResult:
        if self.keyframe is None:
            self._make_keyframe(gray, depth, ts, exclude_mask)
            return PoseResult(self.T_wc.copy(), 1.0, 0, True)

        result = self._track(gray)
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

    def _track(self, gray: np.ndarray) -> PoseResult:
        if self._pts2d is None or len(self._pts2d) < 8:
            return PoseResult(self.T_wc.copy(), 0.0, 0, False, lost=True)

        p0 = self._pts2d.astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **_LK_PARAMS)
        p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, **_LK_PARAMS)
        fb_err = np.linalg.norm(p0 - p0r, axis=2).ravel()
        good = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < 1.5)

        self._pts2d = p1.reshape(-1, 2)[good]
        self._pts3d = self._pts3d[good]
        self._kf_pts2d = self._kf_pts2d[good]
        n = len(self._pts2d)
        if n < 8:
            return PoseResult(self.T_wc.copy(), 0.0, n, False, lost=True)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            self._pts3d.astype(np.float64), self._pts2d.astype(np.float64),
            self.K, None, reprojectionError=4.0, iterationsCount=100,
            flags=cv2.SOLVEPNP_ITERATIVE)
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
