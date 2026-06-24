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
  render_mode: both
  canonical_cell_size: 0.18
  canonical_distance_m: 0.09
  canonical_normal_cos: 0.80
  canonical_min_support: 2
  canonical_support_weight: 1.5
  canonical_residual_weight: 0.4
  canonical_recency_weight: 0.2
imu:
  enabled: true
  use_lk_prior: false
  use_pnp_prior: true
  min_rotation_samples: 3
  max_rotation_deg: 25.0
  keyframe_blur_omega_rad_s: 1.8
  keyframe_max_delay_s: 0.7
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
    assert cfg.mesh.render_mode == "both"
    assert cfg.mesh.canonical_cell_size == 0.18
    assert cfg.mesh.canonical_distance_m == 0.09
    assert cfg.mesh.canonical_normal_cos == 0.80
    assert cfg.mesh.canonical_min_support == 2
    assert cfg.mesh.canonical_support_weight == 1.5
    assert cfg.mesh.canonical_residual_weight == 0.4
    assert cfg.mesh.canonical_recency_weight == 0.2
    assert cfg.imu.enabled is True
    assert cfg.imu.use_lk_prior is False
    assert cfg.imu.use_pnp_prior is True
    assert cfg.imu.min_rotation_samples == 3
    assert cfg.imu.max_rotation_deg == 25.0
    assert cfg.imu.keyframe_blur_omega_rad_s == 1.8
    assert cfg.imu.keyframe_max_delay_s == 0.7


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
