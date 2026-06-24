from spacerec.perf import PerfRecorder, summarize_csv


def test_perf_recorder_writes_csv_and_summary(tmp_path):
    path = tmp_path / "perf.csv"
    rec = PerfRecorder(path, stutter_threshold_ms=100.0)

    rec.record(
        {
            "frame": 1,
            "ts": 0.0,
            "wall_s": 0.1,
            "is_keyframe": 0,
            "backend_results": 0,
            "map_points": 0,
            "stable_objects": 0,
            "observations": 0,
            "vo_lost": 0,
            "vo_inlier_ratio": 0.9,
            "vo_tracked": 120,
        },
        {"loop_total_ms": 50.0, "yolo_ms": 20.0},
    )
    rec.record(
        {
            "frame": 2,
            "ts": 0.1,
            "wall_s": 0.2,
            "is_keyframe": 1,
            "backend_results": 1,
            "map_points": 100,
            "stable_objects": 1,
            "observations": 2,
            "vo_lost": 0,
            "vo_inlier_ratio": 0.8,
            "vo_tracked": 100,
        },
        {"loop_total_ms": 120.0, "backend_drain_ms": 80.0},
    )
    rec.close()

    stats = summarize_csv(path, threshold_ms=100.0)

    assert stats["frames"] == 2
    assert stats["budget_miss"] == 1
    assert stats["backend_miss"] == 1
    assert "budget_miss=1/2" in rec.summary()
