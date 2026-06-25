"""Run a small latency attribution matrix for spacerec.main.

Example:
  .venv/bin/python benchmarks/perf_matrix.py \
    sources/session_20260624_054529_194430108151D05A00 \
    --frames 120 --out-dir artifacts/perf --headless
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spacerec.perf import summarize_csv  # noqa: E402


def _set_nested(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


VARIANTS: list[tuple[str, dict[tuple[str, ...], Any], list[str]]] = [
    ("baseline", {}, []),
    ("fusion_direct", {
        ("fusion", "mode"): "direct",
        ("mesh", "enabled"): False,
        ("backend", "metric_anchor"): False,
    }, ["--fusion", "direct"]),
    ("fusion_none", {
        ("fusion", "mode"): "none",
        ("mesh", "enabled"): False,
    }, ["--fusion", "none"]),
    ("mesh_off", {("mesh", "enabled"): False}, []),
    ("metric_anchor_off", {
        ("mesh", "enabled"): False,
        ("backend", "metric_anchor"): False,
    }, []),
    ("lighter_backend", {
        ("mesh", "enabled"): False,
        ("backend", "metric_anchor"): False,
        ("backend", "period_s"): 10.0,
        ("backend", "window_size"): 8,
        ("backend", "overlap"): 4,
    }, []),
    ("appearance_off", {
        ("mesh", "enabled"): False,
        ("backend", "metric_anchor"): False,
        ("backend", "period_s"): 10.0,
        ("backend", "window_size"): 8,
        ("backend", "overlap"): 4,
        ("objects", "appearance"): False,
    }, []),
    ("realtime_profile", {("runtime_profile",): "realtime"}, ["--runtime-profile", "realtime"]),
    ("backend_off", {("mesh", "enabled"): False}, ["--no-backend"]),
    ("viz_off", {("mesh", "enabled"): False}, ["--no-viz"]),
]


def _load_base_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _write_variant_config(base: dict[str, Any], out: Path,
                          overrides: dict[tuple[str, ...], Any]) -> None:
    cfg = deepcopy(base)
    for path, value in overrides.items():
        _set_nested(cfg, path, value)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="spacerec performance matrix")
    ap.add_argument("source", help="recorded OAK session or video source")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out-dir", default="artifacts/perf")
    ap.add_argument("--threshold-ms", type=float, default=100.0)
    ap.add_argument("--realtime", action="store_true",
                    help="keep realtime pacing; default adds --no-realtime")
    ap.add_argument("--headless", action="store_true",
                    help="add --no-viz to all variants except viz_off")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _load_base_config(Path(args.config))
    rows = []
    for run_idx in range(args.runs):
        for name, overrides, extra_args in VARIANTS:
            cfg_path = out_dir / f"{run_idx:02d}_{name}.yaml"
            csv_path = out_dir / f"{run_idx:02d}_{name}.csv"
            log_path = out_dir / f"{run_idx:02d}_{name}.stdout.txt"
            _write_variant_config(base, cfg_path, overrides)
            cmd = [
                sys.executable, "-m", "spacerec.main",
                "--config", str(cfg_path),
                "--source", args.source,
                "--max-frames", str(args.frames),
                "--perf-log", str(csv_path),
                "--stutter-threshold-ms", str(args.threshold_ms),
            ]
            if not args.realtime:
                cmd.append("--no-realtime")
            if args.headless and "--no-viz" not in extra_args:
                cmd.append("--no-viz")
            cmd.extend(extra_args)
            print("[matrix]", name, " ".join(cmd))
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(SRC) if not env.get("PYTHONPATH")
                else str(SRC) + os.pathsep + env["PYTHONPATH"]
            )
            with log_path.open("w", encoding="utf-8") as log_fh:
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            stats = summarize_csv(csv_path, threshold_ms=args.threshold_ms)
            stats.update({
                "run": run_idx,
                "variant": name,
                "exit_code": proc.returncode,
                "csv": str(csv_path),
                "stdout": str(log_path),
            })
            rows.append(stats)

    summary_path = out_dir / "summary.csv"
    fields = [
        "run", "variant", "exit_code", "frames",
        "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms",
        "budget_miss", "budget_miss_pct", "backend_miss",
        "mean_processing_fps", "csv", "stdout",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("[matrix] summary:", summary_path)
    for row in rows:
        print(
            f"{row['variant']:>18} exit={row['exit_code']} "
            f"p95={float(row.get('p95_ms', 0)):.1f} "
            f"p99={float(row.get('p99_ms', 0)):.1f} "
            f"max={float(row.get('max_ms', 0)):.1f} "
            f"miss={row.get('budget_miss', 0)}"
        )


if __name__ == "__main__":
    main()
