import numpy as np
import pytest

from spacerec.depth import fuse_metric_depth
from spacerec.oak import OakSource


def test_oak_source_advertises_metric_depth_capability():
    assert OakSource.has_metric_depth is True


def test_fuse_metric_depth_keeps_stereo_and_fills_holes():
    yy, xx = np.mgrid[0:20, 0:20]
    stereo = (1.0 + 0.02 * yy + 0.01 * xx).astype(np.float32)
    stereo[:, -3:] = 0.0
    # Fallback is relative but affine-compatible with the stereo valid pixels:
    # metric = 2 * relative.
    fallback = (stereo.copy() / 2.0).astype(np.float32)
    fallback[:, -3:] = (1.0 + 0.02 * yy[:, -3:] + 0.01 * xx[:, -3:]) / 2.0

    fused, cal, valid = fuse_metric_depth(
        stereo, fallback, min_depth_m=0.3, max_depth_m=3.0, min_valid=100)

    assert valid.sum() == 340
    assert cal.inlier_frac > 0.5
    assert np.allclose(fused[valid], stereo[valid])
    assert fused[0, -1] == pytest.approx(1.19)
    assert fused[10, -1] == pytest.approx(1.39)
    assert fused[19, -1] == pytest.approx(1.57)


def test_fuse_metric_depth_rejects_out_of_range_primary_depth():
    stereo = np.array([[0.1, 1.0, 9.0]], dtype=np.float32)

    fused, _, valid = fuse_metric_depth(stereo, min_depth_m=0.3, max_depth_m=8.0)

    assert valid.tolist() == [[False, True, False]]
    assert fused.tolist() == [[0.0, 1.0, 0.0]]
