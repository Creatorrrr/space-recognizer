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
    # 백엔드(5초 주기 멀티뷰)는 지연이 덜 중요하므로 더 큰 모델/해상도 사용.
    # 비우면 라이브와 동일한 model/process_res를 쓴다 (CUDA 이전 동작).
    backend_model: str = ""
    backend_process_res: int = 0

    @property
    def backend_model_resolved(self) -> str:
        return self.backend_model or self.model

    @property
    def backend_process_res_resolved(self) -> int:
        return self.backend_process_res or self.process_res


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
class BackendCfg:
    period_s: float = 5.0
    window_size: int = 12
    overlap: int = 6
    voxel_size: float = 0.03
    max_points: int = 800_000
    metric_anchor: bool = False  # DA3METRIC로 미터 단위 추정 (느림, 선택)
    # 멀티뷰 conf 하위 퍼센타일 컷 (모델 변형마다 conf 분포가 달라 튜닝 대상)
    conf_percentile: float = 30.0


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
    depth: DepthCfg = field(default_factory=DepthCfg)
    detect: DetectCfg = field(default_factory=DetectCfg)
    vo: VoCfg = field(default_factory=VoCfg)
    backend: BackendCfg = field(default_factory=BackendCfg)
    objects: ObjectsCfg = field(default_factory=ObjectsCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    viz: VizCfg = field(default_factory=VizCfg)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
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
