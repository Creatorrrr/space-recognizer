import queue

import numpy as np

from spacerec.config import CaptureCfg, FusionCfg
from spacerec.directfusion import DirectFusionBackend, DirectFusionKeyframe


def _keyframe(
    kf_id: int,
    depth: np.ndarray,
    bgr: np.ndarray | None = None,
    dyn_mask: np.ndarray | None = None,
    depth_conf: np.ndarray | None = None,
    T_wc: np.ndarray | None = None,
) -> DirectFusionKeyframe:
    if bgr is None:
        bgr = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if T_wc is None:
        T_wc = np.eye(4, dtype=np.float64)
    K = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return DirectFusionKeyframe(
        kf_id=kf_id,
        ts=float(kf_id),
        bgr=bgr,
        depth_m=depth.astype(np.float32),
        K=K,
        T_wc=T_wc,
        dyn_mask=dyn_mask,
        depth_conf=depth_conf,
    )


def test_direct_fusion_backprojects_subsampled_metric_depth_with_rgb_colors():
    depth = np.ones((4, 4), dtype=np.float32)
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[0, 0] = [0, 0, 255]
    bgr[0, 2] = [0, 255, 0]
    bgr[2, 0] = [255, 0, 0]
    bgr[2, 2] = [10, 20, 30]
    fusion = FusionCfg(direct_point_subsample=2, direct_mesh_window_size=2)
    backend = DirectFusionBackend(fusion, CaptureCfg())

    backend.add_keyframe(_keyframe(7, depth, bgr))
    result = backend.results.get_nowait()

    assert result.points.tolist() == [
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
    ]
    assert result.colors.tolist() == [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [30, 20, 10],
    ]
    assert result.view_origins.tolist() == [[0.0, 0.0, 0.0]]
    assert result.point_view_idx.tolist() == [0, 0, 0, 0]
    assert result.meters_per_unit == 1.0
    assert result.view_depths is None


def test_direct_fusion_filters_invalid_depth_confidence_and_dynamic_masks():
    depth = np.array([
        [1.0, 0.0, 1.0],
        [9.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)
    depth_conf = np.ones((3, 3), dtype=np.uint8)
    depth_conf[0, 2] = 0
    dyn_mask = np.zeros((3, 3), dtype=bool)
    dyn_mask[1, 1] = True
    capture = CaptureCfg(oak_depth_min_m=0.3, oak_depth_max_m=8.0)
    fusion = FusionCfg(
        direct_point_subsample=1,
        direct_mesh_window_size=2,
        direct_mask_dilate_px=0,
        direct_edge_filter=False,
    )
    backend = DirectFusionBackend(fusion, capture)

    backend.add_keyframe(_keyframe(0, depth, dyn_mask=dyn_mask, depth_conf=depth_conf))
    result = backend.results.get_nowait()

    assert result.points.tolist() == [
        [0.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
        [0.0, 2.0, 1.0],
        [1.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
    ]


def test_direct_fusion_flushes_rgbd_mesh_window_as_backend_result_views():
    depth = np.ones((3, 3), dtype=np.float32)
    bgr = np.zeros((3, 3, 3), dtype=np.uint8)
    bgr[..., 2] = 255
    fusion = FusionCfg(
        direct_point_subsample=2,
        direct_mesh_window_size=2,
        direct_mesh_overlap=1,
    )
    backend = DirectFusionBackend(fusion, CaptureCfg())

    backend.add_keyframe(_keyframe(0, depth, bgr))
    first = backend.results.get_nowait()
    backend.add_keyframe(_keyframe(1, depth * 2.0, bgr))
    second = backend.results.get_nowait()

    assert first.view_depths is None
    assert second.view_depths.shape == (2, 3, 3)
    assert second.view_valid.shape == (2, 3, 3)
    assert second.view_colors.shape == (2, 3, 3, 3)
    assert second.view_poses.shape == (2, 4, 4)
    assert second.view_intrinsics.shape == (2, 3, 3)
    assert second.window_ids == [0, 1]
    assert second.anchor_kf_id == 0
    assert second.view_colors[0, 0, 0].tolist() == [255, 0, 0]

    backend.add_keyframe(_keyframe(2, depth * 3.0, bgr))
    third = backend.results.get_nowait()
    assert third.window_ids == [1, 2]


def test_direct_fusion_start_stop_are_backend_compatible_noops():
    backend = DirectFusionBackend(FusionCfg(), CaptureCfg())

    backend.start()
    backend.wait_ready()
    backend.stop()

    try:
        backend.results.get_nowait()
    except queue.Empty:
        pass
    else:
        raise AssertionError("no readiness sentinel should be queued")
