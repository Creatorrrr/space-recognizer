import json
from pathlib import Path

import numpy as np
import pytest

from spacerec.replay import (
    RecordedOakSource,
    ReplayFormatError,
    crop_scaled_intrinsics,
    is_recorded_oak_session,
    reproject_depth_to_rgb,
)


def _metadata():
    return {
        "schema_version": 1,
        "viewer_config": {"fps": 10.0, "rgb_width": 4, "rgb_height": 2},
        "calibration": {
            "cameras": {
                "CAM_A": {
                    "default_intrinsics": [
                        [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]],
                        8,
                        6,
                    ],
                    "distortion_coefficients": [],
                },
                "CAM_B": {
                    "default_intrinsics": [
                        [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                        4,
                        4,
                    ],
                    "distortion_coefficients": [],
                },
            },
            "eeprom": {"cameraData": []},
        },
    }


def _write_event(root: Path, event: dict) -> None:
    with (root / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _make_session(tmp_path: Path) -> Path:
    root = tmp_path / "session_20260624_test"
    (root / "streams" / "rgb").mkdir(parents=True)
    (root / "streams" / "depth").mkdir(parents=True)
    (root / "streams" / "left").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    rgb0 = np.zeros((2, 4, 3), dtype=np.uint8)
    rgb1 = np.full((2, 4, 3), 20, dtype=np.uint8)
    depth0 = np.array([[1000, 0], [1500, 2000]], dtype=np.uint16)
    depth1 = np.full((2, 2), 2500, dtype=np.uint16)
    np.save(root / "streams" / "rgb" / "00000001_seq1.npy", rgb0)
    np.save(root / "streams" / "rgb" / "00000002_seq2.npy", rgb1)
    np.save(root / "streams" / "depth" / "00000001_seq1.npy", depth0)
    np.save(root / "streams" / "depth" / "00000002_seq2.npy", depth1)
    for idx, ts in enumerate([1_000_000_000, 1_100_000_000], start=1):
        _write_event(root, {
            "type": "frame", "stream": "rgb", "seq": idx,
            "ts_device_ns": ts, "ts_host_ns": ts,
            "payload_path": f"streams/rgb/0000000{idx}_seq{idx}.npy",
            "shape": [2, 4, 3], "dtype": "uint8",
            "metadata": {"mode": "bgr"},
        })
        _write_event(root, {
            "type": "frame", "stream": "depth", "seq": idx,
            "ts_device_ns": ts + 1_000_000, "ts_host_ns": ts + 1_000_000,
            "payload_path": f"streams/depth/0000000{idx}_seq{idx}.npy",
            "shape": [2, 2], "dtype": "uint16",
            "metadata": {"mode": "uint16_mm"},
        })
        _write_event(root, {
            "type": "imu", "stream": "imu", "seq": idx,
            "ts_device_ns": ts, "ts_host_ns": ts,
            "records": [{
                "accel": {"x": 1.0, "y": 2.0, "z": 3.0},
                "gyro": {"x": 0.1, "y": 0.2, "z": 0.3},
                "ts_device_ns": ts,
            }],
        })
    return root


def test_crop_scaled_intrinsics_accounts_for_center_crop():
    meta = _metadata()

    K = crop_scaled_intrinsics(meta, "CAM_A", 4, 2)

    assert K[0, 0] == pytest.approx(5.0)
    assert K[1, 1] == pytest.approx(5.0)
    assert K[0, 2] == pytest.approx(2.0)
    # 8x6 sensor cropped vertically to 8x4 before scaling to 4x2.
    assert K[1, 2] == pytest.approx(1.0)


def test_recorded_oak_source_yields_metric_frames(tmp_path):
    root = _make_session(tmp_path)

    src = RecordedOakSource(root, depth_mode="resize")
    frames = list(src.frames())

    assert is_recorded_oak_session(root)
    assert len(frames) == 2
    assert frames[0].bgr.shape == (2, 4, 3)
    assert frames[0].depth_m.shape == (2, 4)
    assert frames[0].depth_conf.shape == (2, 4)
    assert frames[0].K.shape == (3, 3)
    assert frames[0].imu["accel"].tolist() == [1.0, 2.0, 3.0]
    assert frames[1].ts == pytest.approx(0.1)


def test_recorded_oak_source_rejects_payload_path_escape(tmp_path):
    root = _make_session(tmp_path)
    lines = (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["payload_path"] = "../outside.npy"
    lines[0] = json.dumps(first)
    (root / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ReplayFormatError, match="escapes"):
        RecordedOakSource(root)


def test_reproject_depth_to_rgb_identity_transform():
    depth = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    aligned = reproject_depth_to_rgb(depth, K, K, (np.eye(3), np.zeros(3)), (2, 2))

    assert aligned.tolist() == [[1.0, 2.0], [0.0, 3.0]]
