import numpy as np
import pytest

from spacerec.imu import (ImuSample, estimate_camera_rotation,
                          estimate_gyro_bias, gravity_direction, integrate_gyro,
                          rotate_to_camera, should_accept_backend_keyframe)


def _sample(t, gyro, accel=(0.0, 9.80665, 0.0)):
    return ImuSample(t=t,
                     gyro=np.asarray(gyro, dtype=np.float64),
                     accel=np.asarray(accel, dtype=np.float64))


def _angle_axis(R):
    rvec, _ = __import__("cv2").Rodrigues(R)
    return rvec.ravel()


def test_integrate_gyro_constant_rate_uses_sample_interval():
    samples = [
        _sample(0.0, (0.0, 0.0, 0.2)),
        _sample(0.5, (0.0, 0.0, 0.2)),
        _sample(1.0, (0.0, 0.0, 0.2)),
    ]

    R = integrate_gyro(samples, 0.0, 1.0)

    rvec = _angle_axis(R)
    assert rvec[:2] == pytest.approx([0.0, 0.0], abs=1e-7)
    assert rvec[2] == pytest.approx(0.2, abs=1e-7)


def test_integrate_gyro_handles_nonuniform_samples_and_bias():
    samples = [
        _sample(0.0, (0.01, -0.02, 0.1)),
        _sample(0.2, (0.01, -0.02, 0.3)),
        _sample(1.0, (0.01, -0.02, 1.1)),
    ]

    R = integrate_gyro(samples, 0.0, 1.0, bias=np.array([0.01, -0.02, 0.0]))

    rvec = _angle_axis(R)
    assert rvec[:2] == pytest.approx([0.0, 0.0], abs=1e-7)
    # Trapezoid: 0.2 * (0.1 + 0.3) / 2 + 0.8 * (0.3 + 1.1) / 2 = 0.6
    assert rvec[2] == pytest.approx(0.6, abs=1e-7)


def test_estimate_gyro_bias_returns_stationary_mean_only():
    stationary = [
        _sample(0.0, (0.01, -0.02, 0.005)),
        _sample(0.1, (0.012, -0.018, 0.006)),
        _sample(0.2, (0.008, -0.021, 0.004)),
    ]
    moving = [_sample(0.0, (0.2, 0.0, 0.0)), _sample(0.1, (0.3, 0.0, 0.0))]

    assert estimate_gyro_bias(stationary, max_omega=0.05) == pytest.approx(
        np.array([0.01, -0.0196666667, 0.005]))
    assert estimate_gyro_bias(moving, max_omega=0.05) is None


def test_gravity_direction_requires_stationary_gravity_magnitude():
    samples = [
        _sample(0.0, (0.01, 0.0, 0.0), (0.0, 9.80, 0.1)),
        _sample(0.1, (0.01, 0.0, 0.0), (0.0, 9.82, -0.1)),
    ]

    direction = gravity_direction(samples, max_omega=0.05, accel_tol=0.2)

    assert direction == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=2e-3)
    assert gravity_direction(
        [_sample(0.0, (0.2, 0.0, 0.0), (0.0, 9.8, 0.0))],
        max_omega=0.05,
        accel_tol=0.2,
    ) is None
    assert gravity_direction(
        [_sample(0.0, (0.0, 0.0, 0.0), (0.0, 5.0, 0.0))],
        max_omega=0.05,
        accel_tol=0.2,
    ) is None


def test_rotate_to_camera_conjugates_imu_rotation():
    R_imu = integrate_gyro([
        _sample(0.0, (0.0, 0.0, 1.0)),
        _sample(1.0, (0.0, 0.0, 1.0)),
    ], 0.0, 1.0)
    R_cam_imu, _ = __import__("cv2").Rodrigues(np.array([np.pi / 2, 0.0, 0.0]))

    R_cam = rotate_to_camera(R_imu, R_cam_imu)

    assert R_cam == pytest.approx(R_cam_imu @ R_imu @ R_cam_imu.T)


def test_estimate_camera_rotation_requires_samples_and_extrinsics():
    samples = [
        _sample(0.0, (0.0, 0.0, 1.0)),
        _sample(0.1, (0.0, 0.0, 1.0)),
    ]

    assert estimate_camera_rotation(samples, None) is None
    assert estimate_camera_rotation(samples[:1], np.eye(3), min_samples=2) is None


def test_estimate_camera_rotation_converts_window_to_camera_frame():
    samples = [
        _sample(0.0, (0.0, 0.0, 1.0)),
        _sample(0.1, (0.0, 0.0, 1.0)),
        _sample(0.2, (0.0, 0.0, 1.0)),
    ]
    R_cam_imu, _ = __import__("cv2").Rodrigues(np.array([np.pi / 2, 0.0, 0.0]))

    estimate = estimate_camera_rotation(
        samples,
        R_cam_imu,
        min_samples=2,
        max_angle_rad=1.0,
    )

    assert estimate is not None
    assert estimate.sample_count == 3
    assert estimate.dt == pytest.approx(0.2)
    assert estimate.omega_norm == pytest.approx(1.0)
    assert estimate.R == pytest.approx(
        rotate_to_camera(integrate_gyro(samples, 0.0, 0.2), R_cam_imu))


def test_estimate_camera_rotation_can_use_frame_boundaries():
    samples = [
        _sample(0.01, (0.0, 0.0, 1.0)),
        _sample(0.05, (0.0, 0.0, 1.0)),
        _sample(0.09, (0.0, 0.0, 1.0)),
    ]

    estimate = estimate_camera_rotation(
        samples,
        np.eye(3),
        min_samples=2,
        t0=0.0,
        t1=0.1,
    )

    assert estimate is not None
    assert estimate.dt == pytest.approx(0.1)
    assert _angle_axis(estimate.R)[2] == pytest.approx(0.1)


def test_should_accept_backend_keyframe_gates_high_angular_rate_with_starvation():
    assert should_accept_backend_keyframe(
        frame_ts=1.0,
        last_backend_keyframe_ts=0.5,
        omega_norm=None,
        blur_omega_rad_s=2.0,
        max_delay_s=1.0,
    )
    assert should_accept_backend_keyframe(
        frame_ts=1.0,
        last_backend_keyframe_ts=0.5,
        omega_norm=1.5,
        blur_omega_rad_s=2.0,
        max_delay_s=1.0,
    )
    assert not should_accept_backend_keyframe(
        frame_ts=1.0,
        last_backend_keyframe_ts=0.5,
        omega_norm=3.0,
        blur_omega_rad_s=2.0,
        max_delay_s=1.0,
    )
    assert should_accept_backend_keyframe(
        frame_ts=1.6,
        last_backend_keyframe_ts=0.5,
        omega_norm=3.0,
        blur_omega_rad_s=2.0,
        max_delay_s=1.0,
    )
