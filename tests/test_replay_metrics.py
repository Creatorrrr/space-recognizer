from argparse import Namespace

from spacerec.config import Config

from benchmarks.replay_smoke import MetricTimer, _apply_runtime_overrides, _summarize_timings


def test_metric_timer_records_count_and_elapsed_ms():
    timings = {}

    with MetricTimer(timings, "stage"):
        pass

    assert timings["stage"]["count"] == 1
    assert timings["stage"]["total_ms"] >= 0.0


def test_summarize_timings_reports_avg_ms():
    summary = _summarize_timings({
        "detect": {"count": 2, "total_ms": 10.0},
        "empty": {"count": 0, "total_ms": 5.0},
    })

    assert summary["detect_count"] == 2
    assert summary["detect_total_ms"] == 10.0
    assert summary["detect_avg_ms"] == 5.0
    assert summary["empty_count"] == 0
    assert summary["empty_avg_ms"] == 0.0


def test_apply_runtime_overrides_updates_compute_and_backend_knobs():
    cfg = Config()
    args = Namespace(
        precision="bf16",
        metric_anchor_every_n_windows=4,
        metric_anchor_process_res=196,
        no_tf32=True,
    )

    _apply_runtime_overrides(cfg, args)

    assert cfg.compute.precision == "bf16"
    assert cfg.compute.tf32 is False
    assert cfg.backend.metric_anchor_every_n_windows == 4
    assert cfg.backend.metric_anchor_process_res == 196
