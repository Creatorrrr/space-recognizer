import numpy as np
from scipy.spatial.transform import Rotation

from spacerec.backend import BackendResult
from spacerec.calib import DepthCalibration
from spacerec.config import BackendCfg
from spacerec.geometry import SIM3_IDENTITY, sim3_apply
from spacerec.main import _apply_backend_result
from spacerec.viz import NullVisualizer
from spacerec.worldmap import GlobalMap


def test_direct_backend_result_keeps_existing_correction_target():
    worldmap = GlobalMap(BackendCfg(voxel_size=0.01))
    T_global_live = (
        1.0,
        Rotation.from_euler("y", 10.0, degrees=True).as_matrix(),
        np.array([0.2, 0.0, 0.3]),
    )
    worldmap.set_correction_target(T_global_live)
    worldmap.step_correction(alpha=1.0)
    points_live = np.array([[1.0, 0.0, 2.0]])
    result = BackendResult(
        points=points_live,
        colors=np.array([[10, 20, 30]], dtype=np.uint8),
        T_global_live=SIM3_IDENTITY,
        calib=DepthCalibration(a=1.0, b=0.0, inlier_frac=1.0),
        kf_global_poses={0: np.eye(4)},
        view_origins=np.array([[0.0, 0.0, 0.0]]),
        point_view_idx=np.array([0], dtype=np.uint8),
        window_ids=[0],
    )

    _apply_backend_result(
        result,
        worldmap,
        NullVisualizer(),
        DepthCalibration(),
        apply_correction_target=False,
    )

    assert np.allclose(worldmap.points[0], sim3_apply(T_global_live, points_live)[0])
    assert np.allclose(worldmap.T_global_live[1], T_global_live[1])
    assert np.allclose(worldmap.T_global_live[2], T_global_live[2])
