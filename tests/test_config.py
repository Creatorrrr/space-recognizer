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
loop_closure:
  enabled: true
  check_every_frames: 8
  min_matches: 4
  min_observations: 5
  min_distinct_classes: 3
  min_spread: 0.7
  max_match_distance: 1.8
  match_size_factor: 2.5
  min_cos: 0.55
  require_appearance: false
  app_weight: 0.4
  allow_scale: false
  max_rms: 0.25
  max_rms_frac: 0.2
  max_yaw_delta_deg: 12.0
  max_translation_delta: 1.4
  max_scale_delta: 0.08
  min_accept_interval_s: 2.0
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
vo:
  pnp_aided_reproj_tol: 1.4
  pnp_aided_min_inlier_delta: -2
  pnp_max_step_depth_frac: 0.5
  pnp_max_velocity_units_s: 3.0
  pnp_step_floor_units: 0.10
  pnp_divergence_step_factor: 3.0
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
    assert cfg.loop_closure.enabled is True
    assert cfg.loop_closure.check_every_frames == 8
    assert cfg.loop_closure.min_matches == 4
    assert cfg.loop_closure.min_observations == 5
    assert cfg.loop_closure.min_distinct_classes == 3
    assert cfg.loop_closure.min_spread == 0.7
    assert cfg.loop_closure.max_match_distance == 1.8
    assert cfg.loop_closure.match_size_factor == 2.5
    assert cfg.loop_closure.min_cos == 0.55
    assert cfg.loop_closure.require_appearance is False
    assert cfg.loop_closure.app_weight == 0.4
    assert cfg.loop_closure.allow_scale is False
    assert cfg.loop_closure.max_rms == 0.25
    assert cfg.loop_closure.max_rms_frac == 0.2
    assert cfg.loop_closure.max_yaw_delta_deg == 12.0
    assert cfg.loop_closure.max_translation_delta == 1.4
    assert cfg.loop_closure.max_scale_delta == 0.08
    assert cfg.loop_closure.min_accept_interval_s == 2.0
    assert cfg.viz.point_subsample == 6
    assert cfg.viz.frame_every == 2
    assert cfg.viz.depth_every == 3
    assert cfg.viz.objects_every == 4
    assert cfg.viz.trajectory_every == 5
    assert cfg.viz.trajectory_max_points == 200
    assert cfg.viz.global_map_every == 2
    assert cfg.viz.global_map_max_points == 10000
    assert cfg.viz.jpeg_quality == 55
    assert cfg.vo.pnp_aided_reproj_tol == 1.4
    assert cfg.vo.pnp_aided_min_inlier_delta == -2
    assert cfg.vo.pnp_max_step_depth_frac == 0.5
    assert cfg.vo.pnp_max_velocity_units_s == 3.0
    assert cfg.vo.pnp_step_floor_units == 0.10
    assert cfg.vo.pnp_divergence_step_factor == 3.0


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


def test_loop_closure_defaults_do_not_estimate_scale():
    cfg = Config()

    assert cfg.loop_closure.enabled is True
    assert cfg.loop_closure.allow_scale is False


def test_realtime_runtime_profile_applies_tail_latency_overrides():
    cfg = Config()
    cfg.depth.oak_fill_missing = True
    cfg.backend.metric_anchor = True
    cfg.mesh.enabled = True
    cfg.objects.appearance = True
    cfg.loop_closure.max_match_distance = 2.0
    cfg.loop_closure.max_yaw_delta_deg = 20.0

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
    assert cfg.loop_closure.allow_scale is False
    assert cfg.loop_closure.max_match_distance <= 0.9
    assert cfg.loop_closure.max_yaw_delta_deg <= 12.0
    assert cfg.loop_closure.max_translation_delta <= 1.0
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
