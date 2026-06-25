"""YAML-backed dataclass configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DepthCfg:
    model: str = "depth-anything/DA3-SMALL"
    process_res: int = 504
    metric_model: str = "depth-anything/DA3METRIC-LARGE"
    oak_fill_missing: bool = False
    oak_fill_min_valid: int = 500


@dataclass
class ComputeCfg:
    precision: str = "fp32"  # "fp32" or CUDA-only "bf16".
    tf32: bool = True
    cudnn_benchmark: bool = True


@dataclass
class CaptureCfg:
    source_kind: str = "video"  # "video" or "oak"
    oak_rgb_width: int = 1280
    oak_rgb_height: int = 720
    oak_depth_resolution: str = "400p"
    oak_fps: float = 15.0
    oak_queue_size: int = 4
    oak_align_depth_to_rgb: bool = True
    oak_lr_check: bool = True
    oak_subpixel: bool = True
    oak_extended_disparity: bool = False
    oak_median_filter: str = "7x7"
    oak_depth_min_m: float = 0.3
    oak_depth_max_m: float = 8.0
    oak_enable_imu: bool = True
    oak_imu_rate_hz: int = 100
    replay_depth_mode: str = "calibrated"  # "calibrated" or "resize"
    replay_pair_tolerance_ms: float = 20.0


@dataclass
class DetectCfg:
    model: str = "models/yoloe-11s-seg.pt"
    conf: float = 0.35
    every_n_frames: int = 1
    dynamic_classes: list[str] = field(default_factory=lambda: ["person"])
    # 비어 있지 않으면 YOLOE 오픈 보캐뷸러리 모드로 동작
    vocabulary: list[str] = field(default_factory=list)


@dataclass
class VoCfg:
    max_corners: int = 600
    keyframe_interval_s: float = 0.5
    keyframe_min_flow_px: float = 40.0
    min_inlier_ratio: float = 0.5
    pnp_aided_reproj_tol: float = 1.4
    pnp_aided_min_inlier_delta: int = -2
    pnp_max_step_depth_frac: float = 0.5
    pnp_max_velocity_units_s: float = 3.0
    pnp_step_floor_units: float = 0.10
    pnp_divergence_step_factor: float = 3.0
    imu_rot_residual_warn_deg: float = 3.0
    imu_rot_residual_reject_deg: float = 6.0
    imu_constrained_reproj_error_px: float = 4.0
    imu_constrained_min_inliers: int = 12


@dataclass
class ImuCfg:
    enabled: bool = False
    use_lk_prior: bool = True
    use_pnp_prior: bool = True
    min_rotation_samples: int = 2
    max_rotation_deg: float = 35.0
    keyframe_blur_omega_rad_s: float = 2.5
    keyframe_max_delay_s: float = 1.0


@dataclass
class BackendCfg:
    enabled: bool = True
    live_apply: bool = True
    period_s: float = 5.0
    window_size: int = 12
    overlap: int = 6
    voxel_size: float = 0.03
    max_points: int = 800_000
    metric_anchor: bool = False  # DA3METRIC로 미터 단위 추정 (느림, 선택)
    metric_anchor_every_n_windows: int = 1
    metric_anchor_process_res: int | None = None


@dataclass
class FusionCfg:
    mode: str = "backend"  # "backend", "direct", "none", or "auto"
    direct_point_subsample: int = 4
    direct_mesh_window_size: int = 6
    direct_mesh_overlap: int = 2
    direct_mesh_downsample: int = 1
    direct_edge_filter: bool = True
    direct_edge_rel_thresh: float = 0.06
    direct_mask_dilate_px: int = 2
    require_aligned_depth: bool = True


@dataclass
class MeshCfg:
    enabled: bool = True
    voxel_size: float = 0.05
    trunc_margin: float = 0.15
    depth_trunc_m: float = 8.0
    min_surface_observations: int = 2
    max_active_submaps: int = 32
    render_mode: str = "canonical"  # "canonical", "raw", or "both"
    canonical_cell_size: float = 0.10
    canonical_distance_m: float = 0.10
    canonical_normal_cos: float = 0.85
    canonical_min_support: int = 1
    canonical_support_weight: float = 1.0
    canonical_residual_weight: float = 0.25
    canonical_recency_weight: float = 0.10
    persist_evidence: bool = False
    export_on_exit: bool = False


@dataclass
class ObjectsCfg:
    ema_alpha: float = 0.3
    merge_radius: float = 0.5      # 연관 게이트의 상한 (크기 비례 게이트가 기본)
    dynamic_var_thresh: float = 0.3
    appearance: bool = True        # DINOv2 외형 임베딩 re-ID 사용
    appearance_keyframes_only: bool = False
    appearance_every_n_frames: int = 1
    app_weight: float = 0.6        # 매칭 비용에서 외형 항의 가중치
    app_gate: float = 0.4          # 이보다 낮은 cos 유사도면 같은 물체로 안 봄
    absence_limit: int = 12        # '보여야 하는데 안 보임' 누적 시 노드 제거


@dataclass
class LoopClosureCfg:
    enabled: bool = True
    check_every_frames: int = 10
    min_matches: int = 3
    min_observations: int = 3
    min_distinct_classes: int = 2
    min_spread: float = 0.5
    max_match_distance: float = 1.0
    match_size_factor: float = 2.0
    min_cos: float = 0.45
    require_appearance: bool = False
    app_weight: float = 0.5
    allow_scale: bool = False
    max_rms: float = 0.25
    max_rms_frac: float = 0.30
    max_yaw_delta_deg: float = 15.0
    max_translation_delta: float = 1.0
    max_scale_delta: float = 0.12
    min_accept_interval_s: float = 1.0


@dataclass
class GraphCfg:
    near_dist: float = 1.2
    vertical_ratio: float = 1.5


@dataclass
class VizCfg:
    memory_limit: str = "4GB"
    point_subsample: int = 4
    frame_every: int = 1
    depth_every: int = 1
    objects_every: int = 1
    trajectory_every: int = 1
    trajectory_max_points: int = 0
    global_map_every: int = 1
    global_map_max_points: int = 0
    jpeg_quality: int = 75


@dataclass
class Config:
    source: str | int = 0
    realtime: bool = True
    runtime_profile: str | None = None
    proc_width: int = 1280
    compute: ComputeCfg = field(default_factory=ComputeCfg)
    capture: CaptureCfg = field(default_factory=CaptureCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    detect: DetectCfg = field(default_factory=DetectCfg)
    vo: VoCfg = field(default_factory=VoCfg)
    imu: ImuCfg = field(default_factory=ImuCfg)
    backend: BackendCfg = field(default_factory=BackendCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    mesh: MeshCfg = field(default_factory=MeshCfg)
    objects: ObjectsCfg = field(default_factory=ObjectsCfg)
    loop_closure: LoopClosureCfg = field(default_factory=LoopClosureCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    viz: VizCfg = field(default_factory=VizCfg)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sections = {
            "compute": ComputeCfg, "capture": CaptureCfg, "depth": DepthCfg, "detect": DetectCfg,
            "vo": VoCfg, "imu": ImuCfg, "backend": BackendCfg,
            "fusion": FusionCfg, "mesh": MeshCfg,
            "objects": ObjectsCfg, "loop_closure": LoopClosureCfg,
            "graph": GraphCfg, "viz": VizCfg,
        }
        kwargs = {}
        for key, value in raw.items():
            if key in sections:
                kwargs[key] = sections[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def apply_runtime_profile(cfg: Config, profile: str | None) -> None:
    """Apply runtime-oriented config overrides in-place.

    Profiles are intentionally conservative overlays: explicit CLI/config values
    still load first, then the profile picks safer live defaults for tail latency.
    """
    if not profile or profile == "quality":
        return
    if profile != "realtime":
        raise ValueError(f"unknown runtime profile: {profile}")

    cfg.depth.oak_fill_missing = False
    cfg.backend.enabled = False
    cfg.backend.metric_anchor = False
    cfg.backend.live_apply = False
    cfg.backend.period_s = max(float(cfg.backend.period_s), 10.0)
    cfg.backend.window_size = max(2, min(int(cfg.backend.window_size), 8))
    cfg.backend.overlap = min(
        int(cfg.backend.overlap),
        max(1, cfg.backend.window_size // 2),
        cfg.backend.window_size - 1,
    )
    cfg.mesh.enabled = False
    cfg.detect.every_n_frames = max(int(cfg.detect.every_n_frames), 5)
    cfg.objects.appearance = False
    cfg.objects.appearance_keyframes_only = True
    cfg.objects.appearance_every_n_frames = max(int(cfg.objects.appearance_every_n_frames), 1)
    cfg.loop_closure.allow_scale = False
    cfg.loop_closure.max_match_distance = min(float(cfg.loop_closure.max_match_distance), 0.9)
    cfg.loop_closure.match_size_factor = min(float(cfg.loop_closure.match_size_factor), 2.0)
    cfg.loop_closure.max_rms = min(float(cfg.loop_closure.max_rms), 0.25)
    cfg.loop_closure.max_yaw_delta_deg = min(float(cfg.loop_closure.max_yaw_delta_deg), 12.0)
    cfg.loop_closure.max_translation_delta = min(
        float(cfg.loop_closure.max_translation_delta), 1.0)
    cfg.loop_closure.min_accept_interval_s = max(
        float(cfg.loop_closure.min_accept_interval_s), 1.0)
    cfg.viz.point_subsample = max(int(cfg.viz.point_subsample), 8)
    cfg.viz.frame_every = max(int(cfg.viz.frame_every), 2)
    cfg.viz.depth_every = max(int(cfg.viz.depth_every), 2)
    cfg.viz.objects_every = max(int(cfg.viz.objects_every), 2)
    cfg.viz.trajectory_every = max(int(cfg.viz.trajectory_every), 5)
    cfg.viz.trajectory_max_points = (
        500 if int(cfg.viz.trajectory_max_points) <= 0
        else min(int(cfg.viz.trajectory_max_points), 500)
    )
    cfg.viz.global_map_every = max(int(cfg.viz.global_map_every), 2)
    cfg.viz.global_map_max_points = (
        150_000 if int(cfg.viz.global_map_max_points) <= 0
        else min(int(cfg.viz.global_map_max_points), 150_000)
    )
    cfg.viz.jpeg_quality = min(int(cfg.viz.jpeg_quality), 60)
