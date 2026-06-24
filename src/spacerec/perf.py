"""Lightweight frame timing recorder for live-loop tail latency."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import TextIO

import numpy as np


STAGE_FIELDS = [
    "source_wait_load_ms",
    "yolo_ms",
    "depth_fuse_ms",
    "vo_ms",
    "backend_drain_ms",
    "worldmap_fuse_ms",
    "mesh_ms",
    "log_global_map_ms",
    "log_frame_ms",
    "log_camera_ms",
    "log_calibration_ms",
    "log_live_points_ms",
    "localize_objects_ms",
    "appearance_embed_ms",
    "registry_ms",
    "relocalize_ms",
    "log_objects_ms",
    "loop_total_ms",
]

META_FIELDS = [
    "frame",
    "ts",
    "wall_s",
    "is_keyframe",
    "backend_results",
    "backend_results_deferred",
    "map_points",
    "stable_objects",
    "observations",
    "vo_lost",
    "vo_inlier_ratio",
    "vo_tracked",
]


def add_ms(metrics: dict[str, float], key: str, start: float, end: float | None = None) -> None:
    """Accumulate elapsed milliseconds from a perf_counter start timestamp."""
    if end is None:
        end = time.perf_counter()
    metrics[key] = metrics.get(key, 0.0) + 1e3 * (end - start)


class PerfRecorder:
    """CSV recorder plus stdout stutter/summary reporting."""

    def __init__(self, path: str | Path | None = None,
                 stutter_threshold_ms: float = 100.0):
        self.path = Path(path) if path else None
        self.stutter_threshold_ms = float(stutter_threshold_ms)
        self._fh: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._rows: list[dict[str, float | int | str]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._fh,
                fieldnames=META_FIELDS + STAGE_FIELDS,
                extrasaction="ignore",
            )
            self._writer.writeheader()

    def record(self, meta: dict, metrics: dict[str, float]) -> None:
        row: dict[str, float | int | str] = {}
        for key in META_FIELDS:
            row[key] = meta.get(key, 0)
        for key in STAGE_FIELDS:
            row[key] = float(metrics.get(key, 0.0))
        self._rows.append(row)
        if self._writer is not None:
            self._writer.writerow(row)

        total = float(row["loop_total_ms"])
        if total > self.stutter_threshold_ms:
            ranked = sorted(
                ((key, float(row[key])) for key in STAGE_FIELDS if key != "loop_total_ms"),
                key=lambda item: item[1],
                reverse=True,
            )
            top = " ".join(f"{key.removesuffix('_ms')}={value:.0f}ms"
                           for key, value in ranked[:5] if value > 0.05)
            backend = int(row["backend_results"])
            print(f"[stutter] frame={row['frame']} ts={float(row['ts']):.2f}s "
                  f"loop={total:.0f}ms backend_results={backend} {top}")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def rows(self) -> list[dict[str, float | int | str]]:
        return self._rows

    def summary(self) -> str:
        if not self._rows:
            return "[perf] no frames recorded"
        lines = [f"[perf] frames={len(self._rows)} "
                 f"stutter_threshold={self.stutter_threshold_ms:.0f}ms"]
        for key in STAGE_FIELDS:
            vals = np.array([float(row[key]) for row in self._rows], dtype=np.float64)
            if not np.any(vals > 0):
                continue
            p50, p90, p95, p99 = np.percentile(vals, [50, 90, 95, 99])
            lines.append(
                f"[perf] {key.removesuffix('_ms')}: "
                f"p50={p50:.1f} p90={p90:.1f} p95={p95:.1f} "
                f"p99={p99:.1f} max={vals.max():.1f} ms"
            )
        totals = np.array([float(row["loop_total_ms"]) for row in self._rows])
        miss = int(np.sum(totals > self.stutter_threshold_ms))
        miss_pct = 100.0 * miss / max(1, len(totals))
        backend_miss = sum(
            1 for row in self._rows
            if float(row["loop_total_ms"]) > self.stutter_threshold_ms
            and int(row["backend_results"]) > 0
        )
        lines.append(f"[perf] budget_miss={miss}/{len(totals)} "
                     f"({miss_pct:.2f}%) backend_miss={backend_miss}")
        if self.path is not None:
            lines.append(f"[perf] csv={self.path}")
        return "\n".join(lines)


def summarize_csv(path: str | Path, threshold_ms: float = 100.0) -> dict[str, float | int]:
    """Return compact benchmark stats from a perf CSV."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"frames": 0}
    totals = np.array([float(row["loop_total_ms"]) for row in rows], dtype=np.float64)
    result: dict[str, float | int] = {
        "frames": len(rows),
        "p50_ms": float(np.percentile(totals, 50)),
        "p90_ms": float(np.percentile(totals, 90)),
        "p95_ms": float(np.percentile(totals, 95)),
        "p99_ms": float(np.percentile(totals, 99)),
        "max_ms": float(totals.max()),
        "budget_miss": int(np.sum(totals > threshold_ms)),
        "budget_miss_pct": float(100.0 * np.sum(totals > threshold_ms) / len(totals)),
        "backend_miss": int(sum(
            float(row["loop_total_ms"]) > threshold_ms
            and int(float(row.get("backend_results", 0))) > 0
            for row in rows
        )),
    }
    fps_vals = [
        1.0 / (float(row["loop_total_ms"]) / 1000.0)
        for row in rows
        if float(row["loop_total_ms"]) > 0 and math.isfinite(float(row["loop_total_ms"]))
    ]
    if fps_vals:
        result["mean_processing_fps"] = float(np.mean(fps_vals))
    return result
