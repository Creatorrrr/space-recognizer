#!/usr/bin/env python3
"""Recorded OAK replay -> TSDF mesh smoke.

Examples:
  .venv/bin/python benchmarks/mesh_smoke.py sources/session_... --frames 120
  .venv/bin/python benchmarks/mesh_smoke.py sources/session_... --frames 120 --fusion direct
  .venv/bin/python benchmarks/mesh_smoke.py sources/session_* --frames 60 --out-dir artifacts/mesh
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from spacerec.config import Config
from spacerec.mesh import MeshMap
from spacerec.replay import RecordedOakSource
from spacerec.vo import VisualOdometry


def _scaled_K(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    out = K.copy()
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    out[0] *= sx
    out[1] *= sy
    return out


def _readable_mesh(path: Path) -> tuple[int, int]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    return len(mesh.vertices), len(mesh.triangles)


def run_session(path: Path, args, cfg: Config) -> dict:
    src = RecordedOakSource(
        path,
        proc_width=args.proc_width,
        realtime=False,
        depth_mode=cfg.capture.replay_depth_mode,
        max_pair_dt_ms=cfg.capture.replay_pair_tolerance_ms,
    )
    meshmap = MeshMap(cfg.mesh)
    depths, colors, valids, poses, Ks, ids = [], [], [], [], [], []
    vo = None
    depth_frames = 0
    lost = 0
    t0 = time.perf_counter()
    try:
        for idx, frame in enumerate(src.frames()):
            if idx >= args.frames:
                break
            if frame.depth_m is None or frame.K is None:
                continue
            h, w = frame.depth_m.shape
            if vo is None:
                vo = VisualOdometry(frame.K, cfg.vo)
            gray = (frame.gray_track if frame.gray_track is not None
                    and frame.gray_track.shape == (frame.bgr.shape[0], frame.bgr.shape[1])
                    else cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY))
            pose = vo.process(gray, frame.depth_m, frame.ts, None)
            if pose.lost:
                lost += 1
            depth_frames += 1
            if idx % args.keyframe_step != 0:
                continue

            mesh_w = min(args.mesh_width, w)
            mesh_h = max(1, int(round(h * mesh_w / w)))
            depth_small = cv2.resize(frame.depth_m, (mesh_w, mesh_h),
                                     interpolation=cv2.INTER_NEAREST).astype(np.float32)
            valid_small = np.isfinite(depth_small) & (depth_small > 0)
            rgb_small = cv2.resize(frame.bgr[:, :, ::-1], (mesh_w, mesh_h),
                                   interpolation=cv2.INTER_AREA)
            depths.append(depth_small)
            colors.append(rgb_small)
            valids.append(valid_small)
            poses.append(pose.T_wc.copy())
            Ks.append(_scaled_K(frame.K, (w, h), (mesh_w, mesh_h)))
            ids.append(idx)
    finally:
        src.release()

    if depths:
        meshmap.integrate_views(
            np.stack(depths),
            np.stack(colors),
            np.stack(valids),
            np.stack(poses),
            np.stack(Ks),
            window_ids=ids,
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.name}.ply"
    mesh = meshmap.export_ply(out_path)
    raw_mesh = meshmap.raw_combined_mesh()
    read_vertices, read_faces = _readable_mesh(out_path)
    runtime_s = time.perf_counter() - t0
    return {
        "session": path.name,
        "frames": min(args.frames, depth_frames),
        "depth_frames": depth_frames,
        "keyframes": len(depths),
        "lost": lost,
        "submaps": len(meshmap.submaps),
        "raw_vertices": raw_mesh.n_vertices,
        "raw_faces": raw_mesh.n_faces,
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "read_vertices": read_vertices,
        "read_faces": read_faces,
        "runtime_s": runtime_s,
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recorded OAK TSDF mesh smoke")
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--keyframe-step", type=int, default=5)
    parser.add_argument("--mesh-width", type=int, default=160)
    parser.add_argument("--proc-width", type=int, default=640)
    parser.add_argument("--out-dir", default="artifacts/mesh")
    parser.add_argument("--fusion", choices=["direct"], default="direct",
                        help="accepted for parity with spacerec.main; mesh_smoke always uses OAK RGB-D direct fusion")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    for session in args.sessions:
        result = run_session(session, args, cfg)
        print(
            "MESH_SMOKE "
            f"session={result['session']} "
            f"frames={result['frames']} depth_frames={result['depth_frames']} "
            f"keyframes={result['keyframes']} lost={result['lost']} "
            f"submaps={result['submaps']} raw_vertices={result['raw_vertices']} "
            f"raw_faces={result['raw_faces']} vertices={result['vertices']} "
            f"faces={result['faces']} read_vertices={result['read_vertices']} "
            f"read_faces={result['read_faces']} runtime_s={result['runtime_s']:.2f} "
            f"output={result['output']}"
        )
        if result["vertices"] == 0 or result["faces"] == 0:
            raise SystemExit(f"mesh was empty for {session}")
        if result["read_vertices"] == 0 or result["read_faces"] == 0:
            raise SystemExit(f"exported mesh was unreadable for {session}")


if __name__ == "__main__":
    main()
