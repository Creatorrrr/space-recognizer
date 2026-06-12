import numpy as np

from spacerec.config import VizCfg
from spacerec.viz import Visualizer


def _bare_visualizer(cfg: VizCfg) -> Visualizer:
    viz = Visualizer.__new__(Visualizer)
    viz._cfg = cfg
    viz._trajectory = []
    return viz


def _pose_at(position) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array(position, dtype=float)
    return T


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


def test_trajectory_correction_replaces_path_in_timestamp_order(monkeypatch):
    calls = []
    monkeypatch.setattr("spacerec.viz.rr.log", lambda *args, **kwargs: calls.append(args))
    viz = _bare_visualizer(VizCfg())
    viz._trajectory = [np.array([99.0, 99.0, 99.0])]
    kf_poses = {
        30: _pose_at([3.0, 0.0, 0.0]),
        10: _pose_at([1.0, 0.0, 0.0]),
        20: _pose_at([2.0, 0.0, 0.0]),
    }
    kf_ts = {20: 2.0, 10: 1.0, 30: 3.0}

    viz.log_trajectory_correction(kf_poses, kf_ts)

    assert any(call[0] == "world/trajectory" for call in calls)
    np.testing.assert_allclose(
        np.array(viz._trajectory),
        np.array([[1.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0],
                  [3.0, 0.0, 0.0]]),
    )
