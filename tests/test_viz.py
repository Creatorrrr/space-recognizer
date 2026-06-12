import numpy as np

from spacerec.config import VizCfg
from spacerec.viz import Visualizer


def _bare_visualizer(cfg: VizCfg) -> Visualizer:
    viz = Visualizer.__new__(Visualizer)
    viz._cfg = cfg
    return viz


def test_point_cloud_visibility_flags_skip_rerun_logs(monkeypatch):
    calls = []
    monkeypatch.setattr("spacerec.viz.rr.log", lambda *args, **kwargs: calls.append(args))
    viz = _bare_visualizer(VizCfg(
        show_map_points=False,
        show_live_preview=False,
        show_gaussians=False,
    ))
    points = np.zeros((1, 3), dtype=np.float32)
    colors = np.zeros((1, 3), dtype=np.uint8)

    viz.log_global_map(points, colors)
    viz.log_live_points(points, colors)
    viz.log_gaussians(points, colors, render=np.zeros((2, 2, 3), dtype=np.uint8))

    assert calls == []


def test_point_cloud_visibility_flags_allow_rerun_logs(monkeypatch):
    calls = []
    monkeypatch.setattr("spacerec.viz.rr.log", lambda *args, **kwargs: calls.append(args))
    viz = _bare_visualizer(VizCfg())
    points = np.zeros((1, 3), dtype=np.float32)
    colors = np.zeros((1, 3), dtype=np.uint8)

    viz.log_global_map(points, colors)
    viz.log_live_points(points, colors)
    viz.log_gaussians(points, colors)

    assert [call[0] for call in calls] == [
        "world/points",
        "world/live_preview",
        "world/gaussians",
    ]
