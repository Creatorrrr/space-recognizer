from spacerec.config import Config
from spacerec.device import select_torch_device


def test_load_reads_utf8_config_comments(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_bytes(
        b"# \xed\x95\x9c\xea\xb8\x80 \xec\xa3\xbc\xec\x84\x9d\n"
        b"source: sources/sample_720p.mp4\n"
        b"realtime: false\n"
    )

    cfg = Config.load(path)

    assert cfg.source == "sources/sample_720p.mp4"
    assert cfg.realtime is False


def test_load_reads_oak_capture_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
capture:
  source_kind: oak
  oak_rgb_width: 960
  oak_fps: 12
  oak_enable_imu: true
  oak_imu_rate_hz: 200
  replay_depth_mode: resize
  replay_pair_tolerance_ms: 12
depth:
  oak_fill_missing: false
mesh:
  enabled: true
  voxel_size: 0.04
  trunc_margin: 0.16
  min_surface_observations: 3
  max_active_submaps: 8
""",
        encoding="utf-8",
    )

    cfg = Config.load(path)

    assert cfg.capture.source_kind == "oak"
    assert cfg.capture.oak_rgb_width == 960
    assert cfg.capture.oak_fps == 12
    assert cfg.capture.oak_enable_imu is True
    assert cfg.capture.oak_imu_rate_hz == 200
    assert cfg.capture.replay_depth_mode == "resize"
    assert cfg.capture.replay_pair_tolerance_ms == 12
    assert cfg.depth.oak_fill_missing is False
    assert cfg.mesh.enabled is True
    assert cfg.mesh.voxel_size == 0.04
    assert cfg.mesh.trunc_margin == 0.16
    assert cfg.mesh.min_surface_observations == 3
    assert cfg.mesh.max_active_submaps == 8


def test_select_torch_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert select_torch_device() == "cuda"


def test_select_torch_device_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert select_torch_device() == "mps"


def test_select_torch_device_honors_explicit_device(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert select_torch_device("cpu") == "cpu"
