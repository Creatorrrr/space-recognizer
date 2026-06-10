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
        self._trajectory: list[np.ndarray] = []
        self.meters_per_unit: float | None = None  # metric 앵커가 있으면 거리 라벨에 사용
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
                    rrb.Spatial2DView(origin="world/camera/depth", name="Depth"),
                    rrb.TimeSeriesView(origin="calib", name="Depth Calibration"),
                    row_shares=[2, 2, 1],
                ),
                rrb.Spatial3DView(origin="world", name="3D World"),
                column_shares=[1, 2],
            ),
        )

    def set_time(self, ts: float) -> None:
        rr.set_time("video", duration=ts)

    def log_camera(self, T_wc: np.ndarray, K: np.ndarray,
                   width: int, height: int) -> None:
        rr.log("world/camera", rr.Transform3D(translation=T_wc[:3, 3],
                                              mat3x3=T_wc[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(
            image_from_camera=K.astype(np.float32),
            resolution=[width, height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.3,
        ))
        self._trajectory.append(T_wc[:3, 3].copy())
        if len(self._trajectory) >= 2:
            rr.log("world/trajectory", rr.LineStrips3D(
                [np.array(self._trajectory, dtype=np.float32)],
                colors=[[120, 180, 255, 255]], radii=0.004))

    def log_live_points(self, points_world: np.ndarray, colors: np.ndarray) -> None:
        """Latest-keyframe preview cloud, already in *global* coordinates.

        카메라 엔티티 아래(카메라 좌표)에 로깅하면 다음 키프레임까지 이전 점들이
        움직이는 카메라에 붙어 따라다니며 지도와 어긋나 보인다. 전역 좌표로 변환해
        독립 엔티티에 로깅해야 제자리에 고정된다.
        """
        rr.log("world/live_preview", rr.Points3D(points_world, colors=colors,
                                                 radii=0.008))

    def log_calibration(self, a: float, b: float, frame_scale: float) -> None:
        """depth 캘리브레이션이 실제로 동작 중인지 시계열로 보여준다."""
        rr.log("calib/a", rr.Scalars([a]))
        rr.log("calib/b", rr.Scalars([b]))
        rr.log("calib/frame_scale", rr.Scalars([frame_scale]))

    def log_global_map(self, points: np.ndarray, colors: np.ndarray) -> None:
        rr.log("world/points", rr.Points3D(points, colors=colors, radii=0.006))

    def log_objects(self, objects: list, edges: list, visible: set[int]) -> None:
        """월드 오브젝트 노드(영속) + 관계 엣지 그래프.

        현재 보이는 노드는 불투명, 화면 밖/가려진 노드는 반투명으로 그려
        '기억된 위치'임을 시각적으로 구분한다.
        """
        if not objects:
            rr.log("world/objects", rr.Clear(recursive=True))
            return
        positions = np.array([o.position for o in objects], dtype=np.float32)
        colors = []
        for o in objects:
            c = _color_for(o.cls_name)
            colors.append(c if o.obj_id in visible else c[:3] + [80])
        rr.log("world/objects/nodes", rr.Points3D(
            positions, labels=[o.label for o in objects],
            colors=colors, radii=0.035))

        if edges:
            strips = np.array([[e.a.position, e.b.position] for e in edges],
                              dtype=np.float32)
            edge_colors = [[255, 170, 60, 200] if e.relation == "above"
                           else [160, 160, 170, 140] for e in edges]
            mpu = self.meters_per_unit
            labels = [(f"{'↑' if e.relation == 'above' else '—'} "
                       + (f"{e.dist * mpu:.2f}m" if mpu else f"{e.dist:.2f}"))
                      for e in edges]
            rr.log("world/objects/edges", rr.LineStrips3D(
                strips, colors=edge_colors, radii=0.0035,
                labels=labels, show_labels=False))
        else:
            rr.log("world/objects/edges", rr.Clear(recursive=False))

        # 동적 객체의 시간 궤적 (후순위 기능, 있으면 그려줌)
        dyn = [o for o in objects if o.is_dynamic and len(o.trajectory) >= 2]
        if dyn:
            rr.log("world/objects/dyn_traj", rr.LineStrips3D(
                [np.array([p for _, p in o.trajectory], dtype=np.float32)
                 for o in dyn],
                colors=[_color_for(o.cls_name) for o in dyn], radii=0.002))

    def log_frame(self, bgr: np.ndarray, depth: np.ndarray | None,
                  detections: list[Detection], K: np.ndarray | None = None) -> None:
        rr.log("world/camera/image/rgb",
               rr.Image(bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=75))
        if depth is not None:
            # 절반 해상도로 로깅하되, 반드시 그에 맞는 절반 스케일 Pinhole을 함께
            # 단다. (전체 해상도 Pinhole 밑에 절반 해상도 depth를 넣으면 3D 뷰의
            # depth 역투영이 2배 어긋난다 — 실제로 겪었던 버그)
            half = depth[::2, ::2]
            if K is not None:
                K_half = K.copy()
                K_half[:2] *= 0.5
                rr.log("world/camera/depth", rr.Pinhole(
                    image_from_camera=K_half.astype(np.float32),
                    resolution=[half.shape[1], half.shape[0]],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.01,
                ))
            rr.log("world/camera/depth",
                   rr.DepthImage(half, meter=1.0,
                                 colormap=rr.components.Colormap.Viridis))
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
