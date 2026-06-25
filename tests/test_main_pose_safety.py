import numpy as np

from spacerec.main import (_pose_trusted_for_world_updates,
                           _should_send_reconstruction_keyframe)
from spacerec.vo import PoseResult


def _pose(**kwargs):
    data = dict(
        T_wc=np.eye(4),
        inlier_ratio=1.0,
        n_tracked=100,
        is_keyframe=True,
        lost=False,
    )
    data.update(kwargs)
    return PoseResult(**data)


def test_reconstruction_keyframe_skips_low_confidence_pose():
    pose = _pose(low_confidence=True, fusion_skipped=True)

    assert not _should_send_reconstruction_keyframe(
        pose,
        accept_backend_keyframe=True,
        fusion_mode="direct",
    )


def test_reconstruction_keyframe_skips_warn_level_low_confidence_pose():
    pose = _pose(low_confidence=True, fusion_skipped=False)

    assert not _should_send_reconstruction_keyframe(
        pose,
        accept_backend_keyframe=True,
        fusion_mode="direct",
    )


def test_reconstruction_keyframe_accepts_normal_pose():
    pose = _pose()

    assert _should_send_reconstruction_keyframe(
        pose,
        accept_backend_keyframe=True,
        fusion_mode="direct",
    )


def test_reconstruction_keyframe_skips_lost_pose():
    pose = _pose(lost=True)

    assert not _should_send_reconstruction_keyframe(
        pose,
        accept_backend_keyframe=True,
        fusion_mode="backend",
    )


def test_world_updates_skip_low_confidence_pose():
    assert not _pose_trusted_for_world_updates(_pose(low_confidence=True))
    assert not _pose_trusted_for_world_updates(_pose(fusion_skipped=True))
    assert not _pose_trusted_for_world_updates(_pose(lost=True))
    assert _pose_trusted_for_world_updates(_pose())
