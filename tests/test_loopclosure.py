import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spacerec.config import LoopClosureCfg, ObjectsCfg
from spacerec.detect import Detection
from spacerec.geometry import SIM3_IDENTITY, sim3_apply, sim3_inverse
from spacerec.loopclosure import LoopClosureEstimator
from spacerec.objects import Observation, ObjectRegistry, WorldObject


def _det(cls_name, track_id=1):
    return Detection(track_id=track_id, cls_name=cls_name, conf=0.9,
                     box=np.array([0, 0, 10, 10], dtype=np.float32),
                     mask=None)


def _obs(cls_name, pos, track_id=1, size=0.5, emb=None):
    return Observation(det=_det(cls_name, track_id),
                       position=np.asarray(pos, dtype=np.float64),
                       size=size,
                       emb=None if emb is None else np.asarray(emb, dtype=np.float64))


def _registry_with(classes, positions):
    reg = ObjectRegistry(ObjectsCfg())
    for idx, (cls_name, pos) in enumerate(zip(classes, positions)):
        reg.objects[idx] = WorldObject(
            idx,
            cls_name,
            np.asarray(pos, dtype=np.float64),
            last_seen=0.0,
            size=0.5,
            n_obs=6,
            is_dynamic=False,
        )
    reg._next_id = len(classes)
    return reg


def test_loop_correction_recovers_known_yaw_and_translation():
    classes = ["bed", "chair", "lamp", "rug"]
    global_pts = np.array([
        [0.0, 0.0, 1.0],
        [1.3, 0.1, 1.7],
        [-0.7, -0.2, 2.4],
        [0.4, 0.0, 3.2],
    ])
    T_true = (
        1.0,
        Rotation.from_euler("y", 8.0, degrees=True).as_matrix(),
        np.array([0.25, -0.05, 0.35]),
    )
    live_pts = sim3_apply(sim3_inverse(T_true), global_pts)
    observations = [
        _obs(cls_name, pos, track_id=i)
        for i, (cls_name, pos) in enumerate(zip(classes, live_pts))
    ]
    reg = _registry_with(classes, global_pts)
    estimator = LoopClosureEstimator(LoopClosureCfg(
        check_every_frames=1,
        min_spread=0.1,
        max_yaw_delta_deg=15.0,
        max_translation_delta=1.0,
    ))

    result = estimator.estimate(0, 10.0, observations, reg, SIM3_IDENTITY)

    assert result.accepted
    assert result.reason == "accepted"
    assert result.match_count == 4
    assert result.rms < 1e-6
    assert result.yaw_delta_deg == pytest.approx(8.0, abs=1e-6)
    assert result.T_global_live is not None
    assert np.allclose(sim3_apply(result.T_global_live, live_pts), global_pts,
                       atol=1e-6)


def test_loop_correction_rejects_insufficient_matches():
    classes = ["bed", "chair"]
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    reg = _registry_with(classes, pts)
    observations = [_obs(cls_name, pos, track_id=i)
                    for i, (cls_name, pos) in enumerate(zip(classes, pts))]
    estimator = LoopClosureEstimator(LoopClosureCfg(check_every_frames=1))

    result = estimator.estimate(0, 1.0, observations, reg, SIM3_IDENTITY)

    assert result.attempted
    assert not result.accepted
    assert result.reason == "insufficient_inputs"


def test_loop_correction_rejects_low_spread():
    classes = ["bed", "chair", "lamp"]
    pts = np.array([
        [0.00, 0.0, 1.00],
        [0.03, 0.0, 1.02],
        [0.06, 0.0, 1.01],
    ])
    reg = _registry_with(classes, pts)
    observations = [_obs(cls_name, pos, track_id=i)
                    for i, (cls_name, pos) in enumerate(zip(classes, pts))]
    estimator = LoopClosureEstimator(LoopClosureCfg(
        check_every_frames=1,
        min_spread=0.5,
    ))

    result = estimator.estimate(0, 1.0, observations, reg, SIM3_IDENTITY)

    assert result.attempted
    assert not result.accepted
    assert result.reason == "low_spread"


def test_loop_correction_rejects_large_yaw_jump():
    classes = ["bed", "chair", "lamp"]
    global_pts = np.array([
        [0.0, 0.0, 1.0],
        [1.2, 0.0, 1.7],
        [-0.6, 0.0, 2.5],
    ])
    T_large = (
        1.0,
        Rotation.from_euler("y", 35.0, degrees=True).as_matrix(),
        np.zeros(3),
    )
    live_pts = sim3_apply(sim3_inverse(T_large), global_pts)
    reg = _registry_with(classes, global_pts)
    observations = [_obs(cls_name, pos, track_id=i)
                    for i, (cls_name, pos) in enumerate(zip(classes, live_pts))]
    estimator = LoopClosureEstimator(LoopClosureCfg(
        check_every_frames=1,
        min_spread=0.1,
        max_yaw_delta_deg=15.0,
        max_match_distance=3.0,
        match_size_factor=6.0,
    ))

    result = estimator.estimate(0, 1.0, observations, reg, SIM3_IDENTITY)

    assert result.attempted
    assert not result.accepted
    assert result.reason == "yaw_delta"


def test_loop_correction_caps_size_scaled_match_gate():
    classes = ["bed", "chair", "lamp"]
    global_pts = np.array([
        [0.0, 0.0, 1.0],
        [1.2, 0.0, 1.7],
        [-0.6, 0.0, 2.5],
    ])
    reg = _registry_with(classes, global_pts)
    for obj in reg.objects.values():
        obj.size = 3.0
    observations = [
        _obs(cls_name, pos + np.array([2.0, 0.0, 0.0]), track_id=i, size=3.0)
        for i, (cls_name, pos) in enumerate(zip(classes, global_pts))
    ]
    estimator = LoopClosureEstimator(LoopClosureCfg(
        check_every_frames=1,
        min_spread=0.1,
        max_match_distance=0.75,
        match_size_factor=10.0,
    ))

    result = estimator.estimate(0, 1.0, observations, reg, SIM3_IDENTITY)

    assert result.attempted
    assert not result.accepted
    assert result.reason == "insufficient_matches"


def test_loop_correction_rejects_tilt_only_alignment():
    classes = ["bed", "chair", "lamp", "desk"]
    global_pts = np.array([
        [0.0, -0.5, 1.0],
        [1.2, 0.2, 1.7],
        [-0.6, 0.4, 2.5],
        [0.5, -0.1, 3.2],
    ])
    T_tilt = (
        1.0,
        Rotation.from_euler("x", 12.0, degrees=True).as_matrix(),
        np.zeros(3),
    )
    live_pts = sim3_apply(sim3_inverse(T_tilt), global_pts)
    reg = _registry_with(classes, global_pts)
    observations = [
        _obs(cls_name, pos, track_id=i)
        for i, (cls_name, pos) in enumerate(zip(classes, live_pts))
    ]
    estimator = LoopClosureEstimator(LoopClosureCfg(
        check_every_frames=1,
        min_spread=0.1,
        max_match_distance=3.0,
        match_size_factor=6.0,
        max_rms=0.05,
    ))

    result = estimator.estimate(0, 1.0, observations, reg, SIM3_IDENTITY)

    assert result.attempted
    assert not result.accepted
    assert result.reason == "rms_abs"
