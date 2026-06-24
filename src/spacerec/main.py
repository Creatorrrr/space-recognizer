"""Orchestrator: live depth + detection + visual odometry + 3D visualization."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import spacerec  # noqa: F401  (env vars must be set before heavy imports)

logging.disable(logging.INFO)  # DA3 래퍼가 추론마다 INFO 3줄을 출력해 콘솔이 가려짐

import cv2
import numpy as np

from .backend import BackendKeyframe, ReconstructionBackend
from .calib import DepthCalibration
from .capture import VideoSource
from .config import Config
from .depth import DepthEstimator, fuse_metric_depth
from .detect import Detection, ObjectDetector
from .geometry import sim3_apply, sim3_inverse
from .graph import build_graph
from .imu import estimate_camera_rotation, should_accept_backend_keyframe
from .mesh import MeshMap
from .objects import ObjectRegistry, localize_objects
from .viz import Visualizer
from .vo import VisualOdometry, default_intrinsics
from .worldmap import GlobalMap


def _source_from_config(cfg: Config):
    if isinstance(cfg.source, str):
        from .replay import RecordedOakSource, is_recorded_oak_session

        if is_recorded_oak_session(cfg.source):
            return RecordedOakSource(
                cfg.source,
                proc_width=cfg.proc_width,
                realtime=cfg.realtime,
                depth_mode=cfg.capture.replay_depth_mode,
                max_pair_dt_ms=cfg.capture.replay_pair_tolerance_ms,
            )

    oak_mode = cfg.capture.source_kind.lower() == "oak" or cfg.source == "oak"
    if oak_mode:
        from .oak import OakSource

        return OakSource(cfg.capture, proc_width=cfg.proc_width)
    return VideoSource(cfg.source, proc_width=cfg.proc_width,
                       realtime=cfg.realtime)


def _drain_backend_results(backend: ReconstructionBackend, worldmap: GlobalMap,
                           viz: Visualizer, calib: DepthCalibration,
                           vo: VisualOdometry, frame_wh: tuple[int, int],
                           apply_calib: bool = True,
                           meshmap: MeshMap | None = None,
                           ) -> DepthCalibration:
    while True:
        try:
            res = backend.results.get_nowait()
        except Exception:
            return calib
        worldmap.fuse(res.points, res.colors,
                      origins=res.view_origins, view_idx=res.point_view_idx)
        worldmap.set_correction_target(res.T_global_live)
        if meshmap is not None and getattr(res, "view_depths", None) is not None:
            try:
                meshmap.integrate_backend_result(res)
                viz.log_mesh_submaps(meshmap.changed_submaps())
            except Exception as exc:
                print(f"[mesh] TSDF integration skipped: {exc}")
        if apply_calib and res.calib.inlier_frac > 0.3:
            calib = res.calib
        if res.meters_per_unit is not None:
            viz.meters_per_unit = res.meters_per_unit
        # 주의: 여기서 vo.set_intrinsics()를 호출하면 안 된다. 실행 도중 K가
        # 바뀌면 VO 병진/키프레임 3D의 스케일이 전환 시점 전후로 달라져,
        # 기존 지도 위에 다른 크기의 공간이 겹쳐 그려진다 (8초 시점에 공간이
        # 갑자기 커지던 버그). K는 시작 시 첫 프레임의 DA3 추정으로 고정한다.
        viz.log_global_map(worldmap.points, worldmap.colors)
        mpu = f" 1unit={res.meters_per_unit:.2f}m" if res.meters_per_unit else ""
        print(f"[backend] window={len(res.window_ids)}kf "
              f"{res.runtime_s:.1f}s map={len(worldmap.points)}pts "
              f"calib a={calib.a:.3f} b={calib.b:.3f} "
              f"scale={res.T_global_live[0]:.3f}{mpu}")


def dynamic_mask(detections: list[Detection], shape: tuple[int, int],
                 dynamic_classes: set[str]) -> np.ndarray | None:
    """Union mask of objects that are likely to move (excluded from VO/map)."""
    mask = None
    for det in detections:
        if det.cls_name in dynamic_classes and det.mask is not None:
            mask = det.mask if mask is None else (mask | det.mask)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="spacerec")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None,
                        help="video file path or webcam index (overrides config)")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--no-realtime", action="store_true",
                        help="process every frame instead of wall-clock pacing")
    parser.add_argument("--profile", action="store_true",
                        help="print per-stage timings every 10 frames")
    parser.add_argument("--map", default=None, metavar="PATH",
                        help="세션 간 지도/객체 영속화 파일 (.npz). 있으면 불러와 "
                             "재위치추정 후 이어서 누적, 종료 시 저장")
    parser.add_argument("--mesh-out", default=None, metavar="PATH",
                        help="종료 시 생성된 mesh를 .ply로 export")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.source is not None:
        if args.source.lower() == "oak":
            cfg.capture.source_kind = "oak"
            cfg.source = "oak"
        else:
            cfg.source = int(args.source) if args.source.isdigit() else args.source
    if args.no_realtime:
        cfg.realtime = False

    print("loading models...")
    detector = ObjectDetector(cfg.detect.model, conf=cfg.detect.conf,
                              vocabulary=cfg.detect.vocabulary)
    depth_est = DepthEstimator(cfg.depth.model, process_res=cfg.depth.process_res)
    embedder = None
    if cfg.objects.appearance:
        from .appearance import AppearanceEmbedder
        embedder = AppearanceEmbedder(depth_est.device)
    viz = Visualizer(memory_limit=cfg.viz.memory_limit)
    source = _source_from_config(cfg)
    source_has_metric_depth = bool(getattr(source, "has_metric_depth", False))
    if source_has_metric_depth:
        viz.meters_per_unit = 1.0
    if getattr(source, "metadata", None):
        print(f"source metadata: {source.metadata if not isinstance(source.metadata, dict) else source.metadata.get('device', source.metadata)}")
    W, H = source.proc_width, source.proc_height
    source_K = getattr(source, "K", None)
    K = source_K if source_K is not None else default_intrinsics(W, H)
    vo = VisualOdometry(K, cfg.vo)
    R_cam_imu = getattr(source, "R_cam_imu", None)
    imu_since_keyframe = []
    imu_max_angle_rad = np.radians(cfg.imu.max_rotation_deg)
    prev_frame_ts_for_imu: float | None = None
    imu_keyframe_ts: float | None = None
    worldmap = GlobalMap(cfg.backend)
    meshmap = MeshMap(cfg.mesh) if cfg.mesh.enabled else None
    if not cfg.realtime:
        cfg.backend.period_s = 0.0
    backend = ReconstructionBackend(cfg.backend, cfg.depth.model,
                                    depth_est.device, cfg.depth.process_res,
                                    metric_model=cfg.depth.metric_model)
    backend.start()
    print("waiting for backend process...")
    backend.wait_ready()
    calib = DepthCalibration()
    frame_scale = 1.0  # 키프레임 3D 기준의 프레임별 mono depth 스케일 보정
    # DA3의 프레임별 intrinsics 추정은 출렁임이 크다(fx 740~1015 관측).
    # 처음 K_WARMUP 프레임 동안 표본을 모아 중앙값으로 확정한 뒤 VO를 시작해야
    # 실행 도중 K가 바뀌며 지도 스케일이 갈라지는 일이 없다.
    K_WARMUP = 0 if source_K is not None else 10
    K_samples: list[np.ndarray] = []
    registry = ObjectRegistry(cfg.objects)

    saved_state = None
    saved_mesh_state = None
    reloc_done = False
    if args.map:
        from .persistence import load_state
        saved_state = load_state(args.map)
        if saved_state is not None:
            print(f"loaded saved world: {len(saved_state.points)} pts, "
                  f"{len(saved_state.obj_classes)} objects — 재위치추정 대기")
        else:
            print(f"no saved world at {args.map} (새로 시작, 종료 시 저장)")
        mesh_state_path = Path(args.map).with_suffix(".mesh.npz")
        if meshmap is not None and mesh_state_path.exists():
            from .persistence import load_mesh_state
            saved_mesh_state = load_mesh_state(mesh_state_path, cfg.mesh)
            print(f"loaded saved mesh: {len(saved_mesh_state.submaps)} submaps")
    dyn_classes = set(cfg.detect.dynamic_classes)
    sub = cfg.viz.point_subsample
    bw = cfg.depth.process_res  # 백엔드 입력 가로 해상도
    bh = int(H * bw / W)
    kf_counter = 0
    last_backend_keyframe_ts: float | None = None

    # 모델 첫 호출은 커널 컴파일로 수 초가 걸린다 (YOLOE ~3s 실측). 실시간
    # 페이싱이 시작되기 전에 더미 프레임으로 전부 워밍업해 두지 않으면
    # 짧은 영상에서는 워밍업이 재생 시간을 다 잡아먹는다.
    print("warming up models...")
    dummy = np.zeros((H, W, 3), np.uint8)
    detector.track(dummy)
    if not source_has_metric_depth or cfg.depth.oak_fill_missing:
        depth_est.infer(dummy)
    if embedder is not None:
        embedder.warmup()

    print(f"running on {cfg.source!r} ({W}x{H})")
    frame_count, t_start = 0, time.monotonic()
    t_loop_end = t_start
    try:
        for frame in source.frames():
            if args.max_seconds and frame.ts > args.max_seconds:
                break
            viz.set_time(frame.ts)
            t0 = time.perf_counter()
            if frame.K is not None:
                vo.set_intrinsics(frame.K)
            detections = detector.track(frame.bgr)
            t1 = time.perf_counter()
            raw_depth = None
            has_metric_depth = frame.depth_m is not None
            if has_metric_depth:
                fallback = (depth_est.infer(frame.bgr)
                            if cfg.depth.oak_fill_missing else None)
                depth, oak_calib, _ = fuse_metric_depth(
                    frame.depth_m, fallback, frame.depth_m > 0,
                    min_depth_m=cfg.capture.oak_depth_min_m,
                    max_depth_m=cfg.capture.oak_depth_max_m,
                    min_valid=cfg.depth.oak_fill_min_valid,
                )
                raw_depth = depth
                calib = oak_calib
                frame_scale = 1.0
            else:
                raw_depth = depth_est.infer(frame.bgr)

            if K_WARMUP and len(K_samples) < K_WARMUP:
                if depth_est.last_K is not None:
                    K_samples.append(depth_est.last_K)
                if len(K_samples) == K_WARMUP:
                    vo.set_intrinsics(np.median(np.stack(K_samples), axis=0))
                    print(f"intrinsics fixed (median of {K_WARMUP}): "
                          f"fx={vo.K[0, 0]:.0f} fy={vo.K[1, 1]:.0f}")
                else:
                    # K 확정 전에는 VO/지도를 시작하지 않는다 (2D 패널만 갱신)
                    viz.log_frame(frame.bgr, calib.apply(raw_depth), detections)
                    frame_count += 1
                    continue
            t2 = time.perf_counter()
            if has_metric_depth:
                depth = raw_depth
            else:
                depth = calib.apply(raw_depth) * frame_scale
            gray = (frame.gray_track if frame.gray_track is not None
                    and frame.gray_track.shape == (H, W)
                    else cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY))
            excl = dynamic_mask(detections, (H, W), dyn_classes)
            if R_cam_imu is None and frame.metadata is not None:
                meta_R = frame.metadata.get("imu_to_camera_rotation")
                if meta_R is not None:
                    R_cam_imu = np.asarray(meta_R, dtype=np.float64)
            imu_samples = list(frame.imu_samples or [])
            imu_candidate = imu_since_keyframe + imu_samples
            imu_time_aligned = True
            if frame.metadata is not None:
                imu_time_aligned = bool(frame.metadata.get("imu_timestamp_aligned", True))
            delta_t0 = prev_frame_ts_for_imu if imu_time_aligned else None
            delta_t1 = frame.ts if imu_time_aligned else None
            keyframe_t0 = imu_keyframe_ts if imu_time_aligned else None
            keyframe_t1 = frame.ts if imu_time_aligned else None
            delta_prior = None
            since_prior = None
            if cfg.imu.enabled and R_cam_imu is not None:
                if cfg.imu.use_lk_prior:
                    delta_prior = estimate_camera_rotation(
                        imu_samples,
                        R_cam_imu,
                        min_samples=cfg.imu.min_rotation_samples,
                        max_angle_rad=imu_max_angle_rad,
                        t0=delta_t0,
                        t1=delta_t1,
                    )
                if cfg.imu.use_pnp_prior:
                    since_prior = estimate_camera_rotation(
                        imu_candidate,
                        R_cam_imu,
                        min_samples=cfg.imu.min_rotation_samples,
                        max_angle_rad=imu_max_angle_rad,
                        t0=keyframe_t0,
                        t1=keyframe_t1,
                    )
            omega_norm = None
            if delta_prior is not None:
                omega_norm = delta_prior.omega_norm
            elif since_prior is not None:
                omega_norm = since_prior.omega_norm
            pose = vo.process(
                gray,
                depth,
                frame.ts,
                excl,
                R_delta_prev=None if delta_prior is None else delta_prior.R,
                R_since_keyframe=None if since_prior is None else since_prior.R,
                omega_norm=omega_norm,
            )
            if pose.is_keyframe:
                imu_since_keyframe = []
                imu_keyframe_ts = frame.ts
            else:
                imu_since_keyframe = imu_candidate
            prev_frame_ts_for_imu = frame.ts
            t3 = time.perf_counter()

            # 프레임별 depth 스케일 보정: 키프레임에 고정된 3D 특징점의 예측
            # z와 이번 프레임 depth 맵의 측정 z를 비교해, mono depth의 프레임별
            # 스케일 요동을 다음 프레임부터 상쇄한다 (log-EMA, 한 프레임 지연).
            if (not has_metric_depth
                    and pose.feat_uv is not None and len(pose.feat_uv) >= 20):
                u = pose.feat_uv[:, 0].astype(int).clip(0, W - 1)
                v = pose.feat_uv[:, 1].astype(int).clip(0, H - 1)
                z_meas = depth[v, u]
                ok = (z_meas > 1e-6) & (pose.feat_z > 1e-6)
                if ok.sum() >= 20:
                    ratio = float(np.clip(
                        np.median(pose.feat_z[ok] / z_meas[ok]), 0.8, 1.25))
                    frame_scale = float(np.clip(frame_scale * ratio ** 0.3,
                                                0.5, 2.0))

            accept_backend_keyframe = should_accept_backend_keyframe(
                frame_ts=frame.ts,
                last_backend_keyframe_ts=last_backend_keyframe_ts,
                omega_norm=omega_norm,
                blur_omega_rad_s=cfg.imu.keyframe_blur_omega_rad_s,
                max_delay_s=cfg.imu.keyframe_max_delay_s,
            )
            if pose.is_keyframe and accept_backend_keyframe:
                small = cv2.resize(frame.bgr, (bw, bh), interpolation=cv2.INTER_AREA)
                backend.add_keyframe(BackendKeyframe(
                    kf_id=kf_counter, ts=frame.ts,
                    rgb=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                    T_wc_live=pose.T_wc.copy(),
                    raw_depth=cv2.resize(raw_depth, (bw, bh)),
                    dyn_mask=None if excl is None else
                             cv2.resize(excl.astype(np.uint8), (bw, bh),
                                        interpolation=cv2.INTER_NEAREST).astype(bool),
                    calib_ab=((1.0, 0.0) if has_metric_depth
                              else (calib.a * frame_scale, calib.b * frame_scale)),
                ))
                kf_counter += 1
                last_backend_keyframe_ts = frame.ts

            # 백엔드 결과 반영 (논블로킹)
            new_calib = _drain_backend_results(backend, worldmap, viz, calib,
                                               vo, (W, H),
                                               apply_calib=not has_metric_depth,
                                               meshmap=meshmap)
            if new_calib is not calib:
                # 새 calib은 키프레임 시점의 frame_scale까지 흡수해 피팅된 값.
                # frame_scale을 리셋하지 않으면 같은 보정이 이중 적용되어
                # 스케일이 복리로 표류한다 (calib a가 2.2+까지 증식했던 버그).
                calib = new_calib
                frame_scale = 1.0
            worldmap.step_correction()

            T_wc_global = worldmap.to_global_pose(pose.T_wc)
            viz.log_frame(frame.bgr, depth, detections, vo.K)
            viz.log_camera(T_wc_global, vo.K, W, H)
            viz.log_calibration(calib.a, calib.b, frame_scale)
            if pose.is_keyframe:
                # 키프레임마다 현재 프레임 포인트클라우드 미리보기를 *전역 좌표*로
                # 변환해 로깅 (카메라 좌표로 두면 다음 키프레임까지 카메라를 따라
                # 움직여 지도와 어긋나 보인다)
                scale = worldmap.T_global_live[0]
                d = depth[::sub, ::sub] * scale
                Kv = vo.K
                vs, us = np.mgrid[0:H:sub, 0:W:sub].astype(np.float32)
                pts = np.stack([(us - Kv[0, 2]) / Kv[0, 0] * d,
                                (vs - Kv[1, 2]) / Kv[1, 1] * d, d], axis=-1)
                pts_global = (pts.reshape(-1, 3) @ T_wc_global[:3, :3].T
                              + T_wc_global[:3, 3])
                cols = frame.bgr[::sub, ::sub, ::-1]
                viz.log_live_points(pts_global, cols.reshape(-1, 3))
            observations = localize_objects(detections, depth, vo.K, pose.T_wc,
                                            frame.depth_conf)
            for obs in observations:
                obs.position = worldmap.to_global_points(obs.position[None])[0]
            if embedder is not None and observations:
                embedder.embed(frame.bgr, observations)
            visible = registry.update(observations, frame.ts)

            # 부재 증거: 보여야 하는 위치인데 안 보이는 노드는 신뢰를 깎는다
            T_lg = sim3_inverse(worldmap.T_global_live)
            positions_live = {o.obj_id: sim3_apply(T_lg, o.position[None])[0]
                              for o in registry.objects.values()}
            registry.decay_absent(visible, observations, positions_live,
                                  pose.T_wc, vo.K, depth,
                                  cfg.objects.absence_limit)

            # 이전 세션 지도가 있으면 임베딩 매칭으로 재위치추정을 시도
            if saved_state is not None and not reloc_done and frame_count % 10 == 0:
                from .persistence import (icp_refine, merge_into_session,
                                          merge_mesh_into_session, relocalize)
                result = relocalize(saved_state, registry)
                if result is not None:
                    T, matches, rms = result
                    T = icp_refine(T, saved_state.points, worldmap.points,
                                   cfg.backend.voxel_size)
                    merge_into_session(saved_state, T, matches, worldmap,
                                       registry, frame.ts)
                    if meshmap is not None and saved_mesh_state is not None:
                        n_mesh = merge_mesh_into_session(saved_mesh_state, T, meshmap)
                        viz.log_mesh_submaps(meshmap.changed_submaps())
                    else:
                        n_mesh = 0
                    viz.log_global_map(worldmap.points, worldmap.colors)
                    print(f"[reloc] 이전 지도 정렬 성공: 매칭 {len(matches)}개, "
                          f"rms={rms:.3f}, scale={T[0]:.3f} "
                          f"mesh_submaps={n_mesh} → 병합 완료")
                    reloc_done = True

            objects = registry.stable_objects(frame.ts)
            viz.log_objects(objects, build_graph(objects, cfg.graph), visible)
            t4 = time.perf_counter()
            frame_count += 1
            t_loop_end = time.monotonic()
            if args.profile and frame_count % 10 == 0:
                print(f"  [prof] yolo={1e3 * (t1 - t0):.0f} depth={1e3 * (t2 - t1):.0f} "
                      f"vo={1e3 * (t3 - t2):.0f} viz+rest={1e3 * (t4 - t3):.0f} ms")
            if frame_count % 30 == 0:
                fps = frame_count / (time.monotonic() - t_start)
                p = pose.T_wc[:3, 3]
                print(f"t={frame.ts:5.1f}s processed={frame_count} avg {fps:.1f} FPS | "
                      f"pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"inliers={pose.inlier_ratio:.2f} n={pose.n_tracked} "
                      f"fscale={frame_scale:.3f}"
                      f"{' LOST' if pose.lost else ''}")
        # 영상 종료 후 진행 중인 백엔드 윈도 결과를 기다려 지도에 반영
        print("video ended; draining backend...")
        idle_since = time.monotonic()
        while time.monotonic() - idle_since < 8.0:
            before = len(worldmap.points)
            calib = _drain_backend_results(backend, worldmap, viz, calib, vo,
                                           (W, H),
                                           apply_calib=not source_has_metric_depth,
                                           meshmap=meshmap)
            for _ in range(10):
                worldmap.step_correction()
            if len(worldmap.points) != before:
                idle_since = time.monotonic()
            time.sleep(0.3)
        if args.map:
            from .persistence import save_mesh_state, save_state
            if saved_state is not None and not reloc_done:
                # 이전 지도와 정렬하지 못함 — 덮어쓰면 이전 공간이 사라지므로
                # 별도 파일로 저장해 보존한다
                alt = str(Path(args.map).with_suffix(".unmerged.npz"))
                n = save_state(alt, worldmap, registry, viz.meters_per_unit)
                if meshmap is not None:
                    save_mesh_state(Path(alt).with_suffix(".mesh.npz"), meshmap)
                print(f"[reloc] 정렬 실패 — 이전 지도 보존, 이번 세션은 {alt}에 "
                      f"저장 ({n} objects)")
            else:
                n = save_state(args.map, worldmap, registry, viz.meters_per_unit)
                if meshmap is not None:
                    save_mesh_state(Path(args.map).with_suffix(".mesh.npz"), meshmap)
                print(f"world saved to {args.map} "
                      f"({len(worldmap.points)} pts, {n} objects)")
        mesh_out = args.mesh_out
        if mesh_out is None and cfg.mesh.enabled and cfg.mesh.export_on_exit:
            mesh_out = "artifacts/mesh/latest.ply"
        if mesh_out and meshmap is not None:
            mesh = meshmap.export_ply(mesh_out)
            print(f"mesh saved to {mesh_out} "
                  f"({mesh.n_vertices} vertices, {mesh.n_faces} faces)")
    finally:
        backend.stop()
        source.release()
        elapsed = t_loop_end - t_start
        if frame_count and elapsed > 0:
            print(f"done: {frame_count} frames in {elapsed:.1f}s "
                  f"({frame_count / elapsed:.1f} FPS)")
        if registry.objects:
            print(f"world objects ({len(registry.objects)}):")
            for o in registry.objects.values():
                print(f"  {o.label:>16} pos=({o.position[0]:+.2f},"
                      f"{o.position[1]:+.2f},{o.position[2]:+.2f}) "
                      f"obs={o.n_obs} last_seen={o.last_seen:.1f}s")


if __name__ == "__main__":
    main()
