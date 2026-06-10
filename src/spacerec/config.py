"""YAML-backed dataclass configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DepthCfg:
    model: str = "da3-small"
    process_res: int = 504


@dataclass
class DetectCfg:
    model: str = "models/yolo26n-seg.pt"
    conf: float = 0.35
    dynamic_classes: list[str] = field(default_factory=lambda: ["person"])


@dataclass
class VoCfg:
    max_corners: int = 600
    keyframe_interval_s: float = 0.5
    keyframe_min_flow_px: float = 40.0
    min_inlier_ratio: float = 0.5


@dataclass
class BackendCfg:
    period_s: float = 5.0
    window_size: int = 12
    overlap: int = 6
    voxel_size: float = 0.03
    max_points: int = 800_000


@dataclass
class ObjectsCfg:
    ema_alpha: float = 0.3
    merge_radius: float = 0.5
    dynamic_var_thresh: float = 0.3


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
    depth: DepthCfg = field(default_factory=DepthCfg)
    detect: DetectCfg = field(default_factory=DetectCfg)
    vo: VoCfg = field(default_factory=VoCfg)
    backend: BackendCfg = field(default_factory=BackendCfg)
    objects: ObjectsCfg = field(default_factory=ObjectsCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    viz: VizCfg = field(default_factory=VizCfg)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        sections = {
            "depth": DepthCfg, "detect": DetectCfg, "vo": VoCfg,
            "backend": BackendCfg, "objects": ObjectsCfg,
            "graph": GraphCfg, "viz": VizCfg,
        }
        kwargs = {}
        for key, value in raw.items():
            if key in sections:
                kwargs[key] = sections[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
