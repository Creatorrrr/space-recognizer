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
    dynamic_classes: list[str] = field(default_factory=lambda: ["person"])
    # 비어 있지 않으면 YOLOE 오픈 보캐뷸러리 모드로 동작
    vocabulary: list[str] = field(default_factory=list)


@dataclass
class VoCfg:
    max_corners: int = 600
    keyframe_interval_s: float = 0.5
    keyframe_min_flow_px: float = 40.0
    min_inlier_ratio: float = 0.5


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
    period_s: float = 5.0
    window_size: int = 12
    overlap: int = 6
    voxel_size: float = 0.03
    max_points: int = 800_000
    metric_anchor: bool = False  # DA3METRIC로 미터 단위 추정 (느림, 선택)


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
    app_weight: float = 0.6        # 매칭 비용에서 외형 항의 가중치
    app_gate: float = 0.4          # 이보다 낮은 cos 유사도면 같은 물체로 안 봄
    absence_limit: int = 12        # '보여야 하는데 안 보임' 누적 시 노드 제거


@dataclass
class GraphCfg:
    near_dist: float = 1.2
    vertical_ratio: float = 1.5


@dataclass
class VizCfg:
    memory_limit: str = "4GB"
    point_subsample: int = 4


@dataclass
class Config:
    source: str | int = 0
    realtime: bool = True
    proc_width: int = 1280
    capture: CaptureCfg = field(default_factory=CaptureCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    detect: DetectCfg = field(default_factory=DetectCfg)
    vo: VoCfg = field(default_factory=VoCfg)
    imu: ImuCfg = field(default_factory=ImuCfg)
    backend: BackendCfg = field(default_factory=BackendCfg)
    mesh: MeshCfg = field(default_factory=MeshCfg)
    objects: ObjectsCfg = field(default_factory=ObjectsCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    viz: VizCfg = field(default_factory=VizCfg)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sections = {
            "capture": CaptureCfg, "depth": DepthCfg, "detect": DetectCfg,
            "vo": VoCfg, "imu": ImuCfg, "backend": BackendCfg, "mesh": MeshCfg,
            "objects": ObjectsCfg, "graph": GraphCfg, "viz": VizCfg,
        }
        kwargs = {}
        for key, value in raw.items():
            if key in sections:
                kwargs[key] = sections[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
