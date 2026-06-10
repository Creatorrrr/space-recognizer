"""Rerun visualization: 2D camera panels now, 3D world view in later phases.

Entity tree:
  world/                      3D world frame (right-handed, Y down = camera convention)
    points                    global static point cloud
    camera/                   live camera pose (Transform3D)
      image                   pinhole + RGB / depth / detections
    objects/                  object nodes + graph edges
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .detect import Detection


def _color_for(name: str) -> list[int]:
    digest = hashlib.md5(name.encode()).digest()
    return [96 + digest[0] % 160, 96 + digest[1] % 160, 96 + digest[2] % 160, 255]


class Visualizer:
    def __init__(self, app_id: str = "spacerec", memory_limit: str = "4GB"):
        rr.init(app_id)
        # venv를 활성화하지 않고 실행해도 뷰어 바이너리를 찾도록 명시 경로 사용
        viewer = shutil.which("rerun") or str(Path(sys.executable).parent / "rerun")
        rr.spawn(memory_limit=memory_limit, executable_path=viewer)
        rr.send_blueprint(self._blueprint())
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    @staticmethod
    def _blueprint() -> rrb.Blueprint:
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(origin="world/camera/image", name="Live RGB",
                                      contents=["world/camera/image/rgb",
                                                "world/camera/image/detections"]),
                    rrb.Spatial2DView(origin="world/camera/image/depth", name="Depth"),
                ),
                rrb.Spatial3DView(origin="world", name="3D World"),
                column_shares=[1, 2],
            ),
        )

    def set_time(self, ts: float) -> None:
        rr.set_time("video", duration=ts)

    def log_frame(self, bgr: np.ndarray, depth: np.ndarray | None,
                  detections: list[Detection]) -> None:
        rr.log("world/camera/image/rgb",
               rr.Image(bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=75))
        if depth is not None:
            # 시각화 부담을 줄이기 위해 절반 해상도로 로깅
            rr.log("world/camera/image/depth",
                   rr.DepthImage(depth[::2, ::2], colormap=rr.components.Colormap.Viridis))
        if detections:
            labels = [f"{d.cls_name}#{d.track_id}" for d in detections]
            rr.log("world/camera/image/detections", rr.Boxes2D(
                array=np.array([d.box for d in detections], dtype=np.float32),
                array_format=rr.Box2DFormat.XYXY,
                labels=labels,
                colors=[_color_for(label) for label in labels],
            ))
        else:
            rr.log("world/camera/image/detections", rr.Clear(recursive=False))
