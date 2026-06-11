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
    # VO pose를 DA3 멀티뷰의 입력 조건으로 전달 (pose-conditioned 추론).
    # 출력 depth가 처음부터 라이브 pose 스케일로 정합되어 윈도 간 일관성이
    # 좋아진다. 베이스라인이 퇴화한 윈도는 자동으로 무조건화 폴백.
    pose_conditioned: bool = False


@dataclass
class LoopCfg:
    """루프 클로저 (docs/upgrade-plan.md Tier 4).

    DINOv2 키프레임 임베딩으로 재방문을 감지하고, ORB+depth 3D-3D RANSAC으로
    상대 Sim3를 검증한 뒤 pose graph로 누적 drift를 보정한다.
    objects.appearance가 켜져 있어야 동작한다 (임베더 공유).
    """
    enabled: bool = False
    sim_thresh: float = 0.62    # 임베딩 cos 유사도 후보 임계값
    min_gap_s: float = 10.0     # 이보다 가까운 시점끼리는 루프로 안 봄
    min_inliers: int = 25       # 3D-3D RANSAC inlier 하한 (기각 게이트)
    inlier_dist: float = 0.05   # inlier 거리 임계값 (장면 단위, ≈12cm@2.5m/unit)
    max_kf_store: int = 600     # 루프 탐색용 키프레임 저장 상한


@dataclass
class GaussianCfg:
    """Gaussian Splatting 품질 레이어 (CUDA 전용, docs/upgrade-plan.md Tier 3).

    voxel 지도(기하·증거 레이어)와 별개의 시각 품질 레이어 — 꺼도 기존
    파이프라인은 동일하게 동작한다.
    """
    enabled: bool = False
    period_s: float = 15.0      # 최적화 주기 (느슨해도 됨 — anytime 설계)
    max_gaussians: int = 400_000
    opt_steps: int = 150        # 주기당 최적화 스텝
    spawn_stride: int = 4       # spawn 픽셀 서브샘플링
    depth_loss_w: float = 0.3   # depth L1 가중치 (적은 뷰에서 기하 고정)
    holdout_every: int = 8      # N번째 키프레임은 학습 제외 (PSNR 검증용)


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
    gaussian: GaussianCfg = field(default_factory=GaussianCfg)
    loop: LoopCfg = field(default_factory=LoopCfg)
    objects: ObjectsCfg = field(default_factory=ObjectsCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    viz: VizCfg = field(default_factory=VizCfg)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sections = {
            "depth": DepthCfg, "detect": DetectCfg, "vo": VoCfg,
            "backend": BackendCfg, "gaussian": GaussianCfg, "loop": LoopCfg,
            "objects": ObjectsCfg, "graph": GraphCfg, "viz": VizCfg,
        }
        kwargs = {}
        for key, value in raw.items():
            if key in sections:
                kwargs[key] = sections[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
