from spacerec.config import Config
from spacerec.main import _needs_live_depth_estimator


def test_metric_depth_without_hole_fill_does_not_need_live_depth_estimator():
    cfg = Config()
    cfg.depth.oak_fill_missing = False

    assert _needs_live_depth_estimator(cfg, source_has_metric_depth=True) is False


def test_metric_depth_with_hole_fill_needs_live_depth_estimator():
    cfg = Config()
    cfg.depth.oak_fill_missing = True

    assert _needs_live_depth_estimator(cfg, source_has_metric_depth=True) is True


def test_non_metric_source_needs_live_depth_estimator_even_when_oak_fill_disabled():
    cfg = Config()
    cfg.depth.oak_fill_missing = False

    assert _needs_live_depth_estimator(cfg, source_has_metric_depth=False) is True
