import torch

from spacerec.config import Config, apply_runtime_profile
from spacerec.device import autocast_context, configure_torch_runtime, select_torch_device


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


def test_repository_default_disables_oak_depth_hole_fill():
    cfg = Config.load("config.yaml")

    assert cfg.depth.oak_fill_missing is False


def test_dataclass_default_disables_oak_depth_hole_fill():
    assert Config().depth.oak_fill_missing is False


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
detect:
  every_n_frames: 2
backend:
  enabled: false
  live_apply: false
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
objects:
  appearance: true
  appearance_keyframes_only: true
  appearance_every_n_frames: 3
viz:
  point_subsample: 6
  frame_every: 2
  depth_every: 3
  objects_every: 4
  trajectory_every: 5
  trajectory_max_points: 200
  global_map_every: 2
  global_map_max_points: 10000
  jpeg_quality: 55
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
    assert cfg.detect.every_n_frames == 2
    assert cfg.backend.enabled is False
    assert cfg.backend.live_apply is False
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
    assert cfg.objects.appearance is True
    assert cfg.objects.appearance_keyframes_only is True
    assert cfg.objects.appearance_every_n_frames == 3
    assert cfg.viz.point_subsample == 6
    assert cfg.viz.frame_every == 2
    assert cfg.viz.depth_every == 3
    assert cfg.viz.objects_every == 4
    assert cfg.viz.trajectory_every == 5
    assert cfg.viz.trajectory_max_points == 200
    assert cfg.viz.global_map_every == 2
    assert cfg.viz.global_map_max_points == 10000
    assert cfg.viz.jpeg_quality == 55


def test_load_reads_fusion_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
fusion:
  mode: direct
  direct_point_subsample: 3
  direct_mesh_window_size: 5
  direct_mesh_overlap: 2
  direct_mesh_downsample: 2
  direct_edge_filter: false
  direct_edge_rel_thresh: 0.08
  direct_mask_dilate_px: 1
  require_aligned_depth: false
""",
        encoding="utf-8",
    )

    cfg = Config.load(path)

    assert cfg.fusion.mode == "direct"
    assert cfg.fusion.direct_point_subsample == 3
    assert cfg.fusion.direct_mesh_window_size == 5
    assert cfg.fusion.direct_mesh_overlap == 2
    assert cfg.fusion.direct_mesh_downsample == 2
    assert cfg.fusion.direct_edge_filter is False
    assert cfg.fusion.direct_edge_rel_thresh == 0.08
    assert cfg.fusion.direct_mask_dilate_px == 1
    assert cfg.fusion.require_aligned_depth is False


def test_fusion_config_defaults_preserve_backend_mode():
    cfg = Config()

    assert cfg.fusion.mode == "backend"
    assert cfg.fusion.direct_point_subsample >= 1
    assert cfg.fusion.direct_mesh_window_size >= 2
    assert cfg.fusion.direct_mesh_overlap < cfg.fusion.direct_mesh_window_size
    assert cfg.fusion.require_aligned_depth is True


def test_realtime_runtime_profile_applies_tail_latency_overrides():
    cfg = Config()
    cfg.depth.oak_fill_missing = True
    cfg.backend.metric_anchor = True
    cfg.mesh.enabled = True
    cfg.objects.appearance = True

    apply_runtime_profile(cfg, "realtime")

    assert cfg.depth.oak_fill_missing is False
    assert cfg.backend.enabled is False
    assert cfg.backend.metric_anchor is False
    assert cfg.backend.live_apply is False
    assert cfg.detect.every_n_frames >= 5
    assert cfg.backend.period_s >= 10.0
    assert cfg.backend.window_size <= 8
    assert cfg.backend.overlap <= cfg.backend.window_size // 2
    assert cfg.mesh.enabled is False
    assert cfg.objects.appearance is False
    assert cfg.viz.point_subsample >= 8
    assert cfg.viz.frame_every >= 2
    assert cfg.viz.depth_every >= 2
    assert cfg.viz.global_map_max_points <= 150000


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


def test_load_reads_compute_and_metric_anchor_knobs(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
compute:
  precision: bf16
  tf32: true
  cudnn_benchmark: false
backend:
  metric_anchor: true
  metric_anchor_every_n_windows: 3
  metric_anchor_process_res: 196
""",
        encoding="utf-8",
    )

    cfg = Config.load(path)

    assert cfg.compute.precision == "bf16"
    assert cfg.compute.tf32 is True
    assert cfg.compute.cudnn_benchmark is False
    assert cfg.backend.metric_anchor is True
    assert cfg.backend.metric_anchor_every_n_windows == 3
    assert cfg.backend.metric_anchor_process_res == 196


def test_configure_torch_runtime_enables_cuda_tf32(monkeypatch):
    monkeypatch.setattr("spacerec.device.select_torch_device", lambda device=None: "cuda")

    configure_torch_runtime(tf32=True, cudnn_benchmark=True)

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cudnn.benchmark is True


def test_autocast_context_uses_bfloat16_on_cuda(monkeypatch):
    seen = {}

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_autocast(device_type, dtype):
        seen["device_type"] = device_type
        seen["dtype"] = dtype
        return _Ctx()

    monkeypatch.setattr(torch, "autocast", fake_autocast)

    with autocast_context("cuda", "bf16"):
        pass

    assert seen == {"device_type": "cuda", "dtype": torch.bfloat16}
