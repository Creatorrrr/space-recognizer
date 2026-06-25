from types import SimpleNamespace

from spacerec.config import Config
from spacerec.main import (
    _apply_no_backend_override,
    _check_direct_fusion_alignment,
    _needs_live_depth_estimator,
    prepare_fusion_mode,
    resolve_fusion_mode,
)


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


def test_resolve_fusion_mode_preserves_backend_default():
    cfg = Config()

    assert resolve_fusion_mode(cfg, source_has_metric_depth=True) == "backend"
    assert resolve_fusion_mode(cfg, source_has_metric_depth=False) == "backend"


def test_resolve_fusion_mode_auto_uses_direct_only_for_metric_sources():
    cfg = Config()
    cfg.fusion.mode = "auto"

    assert resolve_fusion_mode(cfg, source_has_metric_depth=True) == "direct"
    assert resolve_fusion_mode(cfg, source_has_metric_depth=False) == "backend"


def test_resolve_fusion_mode_rejects_direct_without_metric_depth():
    cfg = Config()
    cfg.fusion.mode = "direct"

    try:
        resolve_fusion_mode(cfg, source_has_metric_depth=False)
    except ValueError as exc:
        assert "metric-depth" in str(exc)
    else:
        raise AssertionError("direct fusion should require a metric-depth source")


def test_resolve_fusion_mode_accepts_none():
    cfg = Config()
    cfg.fusion.mode = "none"

    assert resolve_fusion_mode(cfg, source_has_metric_depth=True) == "none"


def test_resolve_fusion_mode_treats_disabled_backend_as_none():
    cfg = Config()
    cfg.fusion.mode = "backend"
    cfg.backend.enabled = False

    assert resolve_fusion_mode(cfg, source_has_metric_depth=True) == "none"


def test_no_backend_override_disables_reconstruction_mode():
    cfg = Config()
    cfg.fusion.mode = "backend"

    _apply_no_backend_override(cfg)

    assert cfg.backend.enabled is False
    assert cfg.fusion.mode == "none"


def test_no_backend_override_rejects_direct_fusion_conflict():
    cfg = Config()
    cfg.fusion.mode = "direct"

    try:
        _apply_no_backend_override(cfg)
    except ValueError as exc:
        assert "--fusion direct" in str(exc)
    else:
        raise AssertionError("--no-backend should not be combined with direct fusion")


def test_prepare_direct_fusion_disables_oak_fill_without_depth_estimator():
    cfg = Config()
    cfg.fusion.mode = "direct"
    cfg.depth.oak_fill_missing = True
    source = SimpleNamespace(
        has_metric_depth=True,
        depth_mode="calibrated",
        _depth_to_rgb=object(),
    )

    mode = prepare_fusion_mode(cfg, source)

    assert mode == "direct"
    assert cfg.depth.oak_fill_missing is False
    assert _needs_live_depth_estimator(cfg, source_has_metric_depth=True) is False


def test_direct_fusion_alignment_allows_calibrated_replay_source():
    cfg = Config()
    source = SimpleNamespace(depth_mode="calibrated", _depth_to_rgb=object())

    _check_direct_fusion_alignment(cfg, source)


def test_direct_fusion_alignment_rejects_unaligned_replay_depth():
    cfg = Config()
    source = SimpleNamespace(depth_mode="resize", _depth_to_rgb=object())

    try:
        _check_direct_fusion_alignment(cfg, source)
    except ValueError as exc:
        assert "RGB-aligned" in str(exc)
    else:
        raise AssertionError("direct fusion should require RGB-aligned replay depth")


def test_direct_fusion_alignment_rejects_missing_replay_calibration():
    cfg = Config()
    source = SimpleNamespace(depth_mode="calibrated", _depth_to_rgb=None)

    try:
        _check_direct_fusion_alignment(cfg, source)
    except ValueError as exc:
        assert "depth-to-rgb" in str(exc)
    else:
        raise AssertionError("direct fusion should require recorded depth-to-rgb calibration")


def test_direct_fusion_alignment_rejects_live_oak_without_alignment():
    cfg = Config()
    cfg.source = "oak"
    cfg.capture.source_kind = "oak"
    cfg.capture.oak_align_depth_to_rgb = False

    try:
        _check_direct_fusion_alignment(cfg, SimpleNamespace())
    except ValueError as exc:
        assert "oak_align_depth_to_rgb" in str(exc)
    else:
        raise AssertionError("direct fusion should require live OAK RGB alignment")
