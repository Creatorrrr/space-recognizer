"""Small IMU helpers for gyro-aided visual odometry.

The reconstruction pipeline still treats visual VO and metric depth as the
source of translation. IMU support starts with short-window rotation estimates
and stationary diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class ImuSample:
    """One timestamped IMU sample in a single stream time domain."""

    t: float
    gyro: np.ndarray
    accel: np.ndarray


@dataclass(frozen=True)
class RotationEstimate:
    """Camera-frame short-window rotation derived from IMU gyro samples."""

    R: np.ndarray
    omega_norm: float
    dt: float
    sample_count: int


def _as_vec3(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    return arr


def _sorted_samples(samples: list[ImuSample]) -> list[ImuSample]:
    return sorted(samples, key=lambda sample: sample.t)


def _interp_gyro(samples: list[ImuSample], t: float) -> np.ndarray:
    if not samples:
        return np.zeros(3, dtype=np.float64)
    if t <= samples[0].t:
        return _as_vec3(samples[0].gyro)
    if t >= samples[-1].t:
        return _as_vec3(samples[-1].gyro)
    for left, right in zip(samples, samples[1:]):
        if left.t <= t <= right.t:
            dt = right.t - left.t
            if dt <= 0:
                return _as_vec3(right.gyro)
            alpha = (t - left.t) / dt
            return (1.0 - alpha) * _as_vec3(left.gyro) + alpha * _as_vec3(right.gyro)
    return _as_vec3(samples[-1].gyro)


def integrate_gyro(samples: list[ImuSample], t0: float, t1: float,
                   bias: np.ndarray | None = None) -> np.ndarray:
    """Integrate angular velocity over ``(t0, t1]`` into a relative rotation.

    Gyro units are radians per second. The implementation uses trapezoidal
    angular velocity over the provided sample timeline and composes Rodrigues
    increments. Empty or degenerate windows return identity.
    """

    if t1 <= t0 or not samples:
        return np.eye(3, dtype=np.float64)
    ordered = _sorted_samples(samples)
    bias_vec = _as_vec3(bias)
    times = [float(t0)]
    times.extend(sample.t for sample in ordered if t0 < sample.t < t1)
    times.append(float(t1))
    times = sorted(set(times))

    R = np.eye(3, dtype=np.float64)
    for a, b in zip(times, times[1:]):
        dt = b - a
        if dt <= 0:
            continue
        omega = 0.5 * (_interp_gyro(ordered, a) + _interp_gyro(ordered, b)) - bias_vec
        dR, _ = cv2.Rodrigues((omega * dt).astype(np.float64))
        R = dR @ R
    return R


def estimate_gyro_bias(samples: list[ImuSample],
                       max_omega: float = 0.05) -> np.ndarray | None:
    """Estimate gyro bias from a stationary low-angular-rate window."""

    if not samples:
        return None
    gyros = np.asarray([_as_vec3(sample.gyro) for sample in samples], dtype=np.float64)
    keep = np.linalg.norm(gyros, axis=1) <= float(max_omega)
    if not np.any(keep):
        return None
    return gyros[keep].mean(axis=0)


def gravity_direction(samples: list[ImuSample], max_omega: float = 0.05,
                      accel_tol: float = 0.3) -> np.ndarray | None:
    """Return a unit gravity direction in IMU coordinates for stationary samples."""

    if not samples:
        return None
    gyros = np.asarray([_as_vec3(sample.gyro) for sample in samples], dtype=np.float64)
    accels = np.asarray([_as_vec3(sample.accel) for sample in samples], dtype=np.float64)
    accel_norms = np.linalg.norm(accels, axis=1)
    keep = ((np.linalg.norm(gyros, axis=1) <= float(max_omega))
            & (np.abs(accel_norms - GRAVITY_MPS2) <= float(accel_tol)))
    if not np.any(keep):
        return None
    direction = accels[keep].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return direction / norm


def rotate_to_camera(R_imu: np.ndarray, R_cam_imu: np.ndarray) -> np.ndarray:
    """Convert an IMU-frame relative rotation into the camera frame."""

    R_i = np.asarray(R_imu, dtype=np.float64).reshape(3, 3)
    R_ci = np.asarray(R_cam_imu, dtype=np.float64).reshape(3, 3)
    return R_ci @ R_i @ R_ci.T


def _rotation_angle(R: np.ndarray) -> float:
    value = float((np.trace(R) - 1.0) * 0.5)
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def estimate_camera_rotation(samples: list[ImuSample],
                             R_cam_imu: np.ndarray | None,
                             bias: np.ndarray | None = None,
                             min_samples: int = 2,
                             max_angle_rad: float | None = None,
                             t0: float | None = None,
                             t1: float | None = None
                             ) -> RotationEstimate | None:
    """Estimate a camera-frame rotation prior from a gyro sample window.

    Missing extrinsics, too few samples, non-positive duration, non-finite
    values, or an implausibly large integrated angle return None so callers can
    fall back to visual-only odometry without guessing.
    """

    if R_cam_imu is None or len(samples) < int(min_samples):
        return None
    ordered = _sorted_samples(samples)
    start = float(ordered[0].t) if t0 is None else float(t0)
    end = float(ordered[-1].t) if t1 is None else float(t1)
    dt = end - start
    if dt <= 0:
        return None
    gyros = np.asarray([_as_vec3(sample.gyro) for sample in ordered],
                       dtype=np.float64)
    bias_vec = _as_vec3(bias)
    if not np.all(np.isfinite(gyros)) or not np.all(np.isfinite(bias_vec)):
        return None
    omega_norm = float(np.max(np.linalg.norm(gyros - bias_vec, axis=1)))
    R_imu = integrate_gyro(ordered, start, end, bias=bias_vec)
    R_cam = rotate_to_camera(R_imu, R_cam_imu)
    if not np.all(np.isfinite(R_cam)):
        return None
    if max_angle_rad is not None and _rotation_angle(R_cam) > float(max_angle_rad):
        return None
    return RotationEstimate(R=R_cam, omega_norm=omega_norm, dt=dt,
                            sample_count=len(ordered))


def should_accept_backend_keyframe(frame_ts: float,
                                   last_backend_keyframe_ts: float | None,
                                   omega_norm: float | None,
                                   blur_omega_rad_s: float,
                                   max_delay_s: float) -> bool:
    """Gate backend keyframes during high angular-rate intervals.

    Returning False delays only backend promotion; VO keyframe maintenance still
    happens normally. Starvation protection accepts a high-rate frame when no
    backend keyframe has been accepted recently.
    """

    if omega_norm is None or blur_omega_rad_s <= 0:
        return True
    if float(omega_norm) <= float(blur_omega_rad_s):
        return True
    if last_backend_keyframe_ts is None:
        return True
    return float(frame_ts) - float(last_backend_keyframe_ts) >= float(max_delay_s)
