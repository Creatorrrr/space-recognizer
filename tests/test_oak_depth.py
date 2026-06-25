import numpy as np
import pytest

from spacerec.depth import fuse_metric_depth
from spacerec.imu import ImuSample
from spacerec.oak import OakSource


def test_oak_source_advertises_metric_depth_capability():
    assert OakSource.has_metric_depth is True


def test_fuse_metric_depth_keeps_stereo_and_fills_holes():
    yy, xx = np.mgrid[0:20, 0:20]
    stereo = (1.0 + 0.02 * yy + 0.01 * xx).astype(np.float32)
    stereo[:, -3:] = 0.0
    # Fallback is relative but affine-compatible with the stereo valid pixels:
    # metric = 2 * relative.
    fallback = (stereo.copy() / 2.0).astype(np.float32)
    fallback[:, -3:] = (1.0 + 0.02 * yy[:, -3:] + 0.01 * xx[:, -3:]) / 2.0

    fused, cal, valid = fuse_metric_depth(
        stereo, fallback, min_depth_m=0.3, max_depth_m=3.0, min_valid=100)

    assert valid.sum() == 340
    assert cal.inlier_frac > 0.5
    assert np.allclose(fused[valid], stereo[valid])
    assert fused[0, -1] == pytest.approx(1.19)
    assert fused[10, -1] == pytest.approx(1.39)
    assert fused[19, -1] == pytest.approx(1.57)


def test_fuse_metric_depth_rejects_out_of_range_primary_depth():
    stereo = np.array([[0.1, 1.0, 9.0]], dtype=np.float32)

    fused, _, valid = fuse_metric_depth(stereo, min_depth_m=0.3, max_depth_m=8.0)

    assert valid.tolist() == [[False, True, False]]
    assert fused.tolist() == [[0.0, 1.0, 0.0]]


class _FakeDuration:
    def __init__(self, seconds):
        self.seconds = seconds

    def total_seconds(self):
        return self.seconds


class _FakeMsg:
    def __init__(self, seconds, sequence=None, cv_frame=None, frame=None):
        self.seconds = seconds
        self.sequence = sequence
        self.cv_frame = cv_frame
        self.frame = frame

    def getTimestampDevice(self):
        return _FakeDuration(self.seconds)

    def getSequenceNum(self):
        return self.sequence

    def getCvFrame(self):
        return self.cv_frame

    def getFrame(self):
        return self.frame


class _FakeQueue:
    def __init__(self, get_items=None, try_items=None):
        self.get_items = list(get_items or [])
        self.try_items = list(try_items or [])

    def get(self):
        return self.get_items.pop(0)

    def tryGet(self):
        if not self.try_items:
            return None
        return self.try_items.pop(0)


def _imu_sample(t):
    return ImuSample(
        t=t,
        gyro=np.zeros(3, dtype=np.float64),
        accel=np.zeros(3, dtype=np.float64),
    )


def test_oak_timestamp_helpers_prefer_device_timestamp():
    src = object.__new__(OakSource)

    assert src._message_timestamp_s(_FakeMsg(12.5)) == pytest.approx(12.5)


def test_oak_imu_window_uses_previous_to_current_rgb_interval():
    src = object.__new__(OakSource)
    src._prev_rgb_imu_ts_s = 0.0
    src._imu_pending_samples = [_imu_sample(0.01), _imu_sample(0.05),
                                _imu_sample(0.11)]

    window = src._pop_imu_window(0.10)

    assert [sample.t for sample in window] == pytest.approx([0.01, 0.05])
    assert [sample.t for sample in src._imu_pending_samples] == pytest.approx([0.11])

    src._prev_rgb_imu_ts_s = 0.10
    src._imu_pending_samples.append(_imu_sample(0.15))
    window = src._pop_imu_window(0.20)

    assert [sample.t for sample in window] == pytest.approx([0.11, 0.15])
    assert src._imu_pending_samples == []


def test_oak_live_frames_do_not_carry_forward_stale_depth():
    src = object.__new__(OakSource)
    src._closed = False
    src.proc_width = 2
    src.proc_height = 2
    src.K = np.eye(3)
    src.metadata = {}
    src._imu_t0_s = None
    src._prev_rgb_imu_ts_s = None
    src._imu_pending_samples = []

    bgr0 = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr1 = np.full((2, 2, 3), 10, dtype=np.uint8)
    depth_mm = np.full((2, 2), 1234, dtype=np.uint16)
    src._queues = {
        "rgb": _FakeQueue(get_items=[
            _FakeMsg(10.0, sequence=1, cv_frame=bgr0),
            _FakeMsg(10.1, sequence=2, cv_frame=bgr1),
        ]),
        "depth": _FakeQueue(try_items=[
            _FakeMsg(10.005, sequence=7, frame=depth_mm),
        ]),
        "left": _FakeQueue(),
    }

    frames = src.frames()
    first = next(frames)
    second = next(frames)

    assert first.depth_m is not None
    assert first.depth_m[0, 0] == pytest.approx(1.234)
    assert first.metadata["depth_matched"] is True
    assert first.metadata["depth_seq"] == 7
    assert first.metadata["depth_age_ms"] == pytest.approx(5.0)
    assert first.metadata["depth_pairing_policy"] == "fresh_depth_packet_only"

    assert second.depth_m is None
    assert second.metadata["depth_matched"] is False
    assert second.metadata["depth_seq"] is None
    assert second.metadata["depth_age_ms"] is None
