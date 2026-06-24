"""Replay DepthAI/OAK recording sessions as Frame streams."""

from __future__ import annotations

import bisect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from .capture import Frame
from .imu import ImuSample


class ReplayFormatError(RuntimeError):
    """Raised when a recorded session does not match the expected schema."""


@dataclass(frozen=True)
class ReplayEvent:
    stream: str
    seq: int
    ts_device_ns: int
    ts_host_ns: int
    payload_path: Path | None
    shape: tuple[int, ...] | None
    dtype: str | None
    metadata: dict[str, Any]
    raw: dict[str, Any]


def is_recorded_oak_session(path: str | Path) -> bool:
    root = Path(path)
    return (root.is_dir()
            and (root / "metadata.json").is_file()
            and (root / "events.jsonl").is_file()
            and (root / "streams").is_dir())


def crop_scaled_intrinsics(meta: dict[str, Any], camera: str,
                           target_width: int, target_height: int) -> np.ndarray:
    """Scale DepthAI calibration intrinsics to a center-cropped target frame.

    DepthAI preview/video outputs commonly preserve FOV by center-cropping the
    sensor frame to the requested aspect ratio before resizing. A plain axis
    scale would make `fy` wrong for 16:9 RGB recordings from a 4:3 sensor.
    """
    try:
        K0, src_width, src_height = meta["calibration"]["cameras"][camera][
            "default_intrinsics"]
    except KeyError as exc:
        raise ReplayFormatError(f"missing intrinsics for {camera}") from exc

    src_width = float(src_width)
    src_height = float(src_height)
    if src_width <= 0 or src_height <= 0 or target_width <= 0 or target_height <= 0:
        raise ReplayFormatError("invalid intrinsic source or target size")

    target_aspect = float(target_width) / float(target_height)
    source_aspect = src_width / src_height
    if source_aspect < target_aspect:
        crop_width = src_width
        crop_height = src_width / target_aspect
        x0 = 0.0
        y0 = (src_height - crop_height) * 0.5
    else:
        crop_height = src_height
        crop_width = src_height * target_aspect
        x0 = (src_width - crop_width) * 0.5
        y0 = 0.0

    sx = float(target_width) / crop_width
    sy = float(target_height) / crop_height
    K = np.asarray(K0, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - x0) * sx
    K[1, 2] = (K[1, 2] - y0) * sy
    return K


def _safe_payload_path(root: Path, rel: str | None) -> Path | None:
    if not rel:
        return None
    path = Path(rel)
    if path.is_absolute():
        raise ReplayFormatError(f"absolute payload path is not allowed: {rel}")
    full = (root / path).resolve()
    root_resolved = root.resolve()
    try:
        full.relative_to(root_resolved)
    except ValueError as exc:
        raise ReplayFormatError(f"payload path escapes session root: {rel}") from exc
    return full


def _event_ts(ev: dict[str, Any], key: str) -> int:
    value = ev.get(key)
    if value is None:
        return 0
    return int(value)


def _load_metadata(root: Path) -> dict[str, Any]:
    try:
        meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReplayFormatError(f"cannot read metadata.json: {exc}") from exc
    if not isinstance(meta, dict):
        raise ReplayFormatError("metadata.json must contain an object")
    if "calibration" not in meta or "viewer_config" not in meta:
        raise ReplayFormatError("metadata.json missing calibration/viewer_config")
    return meta


def _load_events(root: Path) -> dict[str, list[ReplayEvent]]:
    by_stream: dict[str, list[ReplayEvent]] = {}
    try:
        lines = (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReplayFormatError(f"cannot read events.jsonl: {exc}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayFormatError(f"invalid events.jsonl line {line_no}") from exc
        stream = raw.get("stream")
        if not isinstance(stream, str):
            raise ReplayFormatError(f"missing stream at events.jsonl line {line_no}")
        event = ReplayEvent(
            stream=stream,
            seq=int(raw.get("seq", 0)),
            ts_device_ns=_event_ts(raw, "ts_device_ns"),
            ts_host_ns=_event_ts(raw, "ts_host_ns"),
            payload_path=_safe_payload_path(root, raw.get("payload_path")),
            shape=tuple(raw["shape"]) if "shape" in raw else None,
            dtype=str(raw["dtype"]) if "dtype" in raw else None,
            metadata=dict(raw.get("metadata") or {}),
            raw=raw,
        )
        by_stream.setdefault(stream, []).append(event)
    for events in by_stream.values():
        events.sort(key=lambda e: (e.ts_device_ns, e.seq))
    return by_stream


def _nearest(events: list[ReplayEvent], ts_device_ns: int,
             max_delta_ns: int | None = None) -> ReplayEvent | None:
    if not events:
        return None
    keys = [e.ts_device_ns for e in events]
    idx = bisect.bisect_left(keys, ts_device_ns)
    candidates: list[ReplayEvent] = []
    if idx < len(events):
        candidates.append(events[idx])
    if idx > 0:
        candidates.append(events[idx - 1])
    event = min(candidates, key=lambda e: abs(e.ts_device_ns - ts_device_ns))
    if max_delta_ns is not None and abs(event.ts_device_ns - ts_device_ns) > max_delta_ns:
        return None
    return event


def _load_npy(event: ReplayEvent, expected_dtype: str | None = None) -> np.ndarray:
    if event.payload_path is None:
        raise ReplayFormatError(f"{event.stream} event has no payload")
    if not event.payload_path.is_file():
        raise ReplayFormatError(f"missing payload: {event.payload_path}")
    arr = np.load(event.payload_path, allow_pickle=False)
    if expected_dtype is not None and str(arr.dtype) != expected_dtype:
        raise ReplayFormatError(
            f"{event.stream} payload dtype {arr.dtype}, expected {expected_dtype}")
    if event.shape is not None and tuple(arr.shape) != event.shape:
        raise ReplayFormatError(
            f"{event.stream} payload shape {arr.shape}, expected {event.shape}")
    return arr


def _camera_data(meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    data = meta.get("calibration", {}).get("eeprom", {}).get("cameraData", [])
    result: dict[int, dict[str, Any]] = {}
    for item in data:
        if isinstance(item, list) and len(item) == 2:
            result[int(item[0])] = item[1]
    return result


def _transform_between_recorded_cameras(meta: dict[str, Any],
                                        source_socket: int,
                                        target_socket: int
                                        ) -> tuple[np.ndarray, np.ndarray] | None:
    """Return transform from source camera coords to target camera coords.

    DepthAI stores translations in centimeters. The transform direction in the
    calibration JSON is interpreted as camera socket -> `toCameraSocket`.
    """
    data = _camera_data(meta)
    if source_socket == target_socket:
        return np.eye(3), np.zeros(3)
    path: list[int] = []
    current = source_socket
    visited = set()
    R_total = np.eye(3)
    t_total = np.zeros(3)
    while current != target_socket and current not in visited:
        visited.add(current)
        entry = data.get(current)
        if not entry:
            return None
        ext = entry.get("extrinsics") or {}
        nxt = int(ext.get("toCameraSocket", -1))
        if nxt < 0:
            return None
        R = np.asarray(ext.get("rotationMatrix"), dtype=np.float64)
        tr = ext.get("translation") or {}
        t = np.array([float(tr.get("x", 0.0)),
                      float(tr.get("y", 0.0)),
                      float(tr.get("z", 0.0))], dtype=np.float64) * 0.01
        if R.shape != (3, 3):
            return None
        R_total = R @ R_total
        t_total = R @ t_total + t
        current = nxt
        path.append(current)
    if current != target_socket:
        return None
    return R_total, t_total


def imu_to_camera_rotation_from_metadata(meta: dict[str, Any],
                                         target_socket: int = 0) -> np.ndarray | None:
    """Return recorded OAK IMU-frame to target camera-frame rotation.

    DepthAI recordings store IMU extrinsics to one camera socket. If the target
    camera differs, compose through the recorded camera extrinsics path. Missing
    metadata returns None so downstream IMU priors can stay disabled.
    """

    imu_ext = meta.get("calibration", {}).get("eeprom", {}).get("imuExtrinsics")
    if not isinstance(imu_ext, dict):
        return None
    R_imu_to_source = np.asarray(imu_ext.get("rotationMatrix"), dtype=np.float64)
    if R_imu_to_source.shape != (3, 3):
        return None
    source_socket = int(imu_ext.get("toCameraSocket", target_socket))
    if source_socket == target_socket:
        return R_imu_to_source
    source_to_target = _transform_between_recorded_cameras(
        meta, source_socket=source_socket, target_socket=target_socket)
    if source_to_target is None:
        return None
    return source_to_target[0] @ R_imu_to_source


def reproject_depth_to_rgb(depth_m: np.ndarray, K_depth: np.ndarray,
                           K_rgb: np.ndarray,
                           depth_to_rgb: tuple[np.ndarray, np.ndarray],
                           rgb_shape: tuple[int, int],
                           splat_radius: int = 0) -> np.ndarray:
    """Project a metric depth map into the RGB camera plane with z buffering."""
    h_rgb, w_rgb = rgb_shape
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros((h_rgb, w_rgb), np.float32)

    vs, us = np.nonzero(valid)
    z = depth[vs, us].astype(np.float64)
    x = (us.astype(np.float64) - K_depth[0, 2]) / K_depth[0, 0] * z
    y = (vs.astype(np.float64) - K_depth[1, 2]) / K_depth[1, 1] * z
    pts = np.stack([x, y, z], axis=0)
    R, t = depth_to_rgb
    pts_rgb = R @ pts + t[:, None]
    z_rgb = pts_rgb[2]
    ok = z_rgb > 1e-6
    if not np.any(ok):
        return np.zeros((h_rgb, w_rgb), np.float32)
    u_rgb = np.rint(K_rgb[0, 0] * pts_rgb[0, ok] / z_rgb[ok] + K_rgb[0, 2]).astype(np.int64)
    v_rgb = np.rint(K_rgb[1, 1] * pts_rgb[1, ok] / z_rgb[ok] + K_rgb[1, 2]).astype(np.int64)
    z_valid = z_rgb[ok].astype(np.float32)
    inside = ((0 <= u_rgb) & (u_rgb < w_rgb)
              & (0 <= v_rgb) & (v_rgb < h_rgb)
              & np.isfinite(z_valid))
    if not np.any(inside):
        return np.zeros((h_rgb, w_rgb), np.float32)
    flat = v_rgb[inside] * w_rgb + u_rgb[inside]
    z_inside = z_valid[inside]
    order = np.argsort(z_inside, kind="stable")
    flat_sorted = flat[order]
    z_sorted = z_inside[order]
    unique_flat, first = np.unique(flat_sorted, return_index=True)
    out = np.zeros(h_rgb * w_rgb, np.float32)
    out[unique_flat] = z_sorted[first]
    projected = out.reshape(h_rgb, w_rgb)
    if splat_radius <= 0:
        return projected
    large = np.where(projected > 0, projected, 1_000_000.0).astype(np.float32)
    kernel = np.ones((2 * splat_radius + 1, 2 * splat_radius + 1), np.uint8)
    nearest = cv2.erode(large, kernel)
    fill = (projected <= 0) & (nearest < 1_000_000.0)
    projected[fill] = nearest[fill]
    return projected


class RecordedOakSource:
    """Replay an OAK-D-Lite recording directory through the Frame contract."""

    def __init__(self, source: str | Path, proc_width: int = 1280,
                 realtime: bool = False, depth_mode: str = "calibrated",
                 max_pair_dt_ms: float = 20.0):
        self.root = Path(source)
        if not is_recorded_oak_session(self.root):
            raise ReplayFormatError(f"not a recorded OAK session: {self.root}")
        self.metadata = _load_metadata(self.root)
        self.events = _load_events(self.root)
        self.rgb_events = self.events.get("rgb", [])
        if not self.rgb_events:
            raise ReplayFormatError("recording has no rgb frames")
        self.depth_events = self.events.get("depth", [])
        self.left_events = self.events.get("left", [])
        self.imu_events = self.events.get("imu", [])
        self._imu_event_times = [event.ts_device_ns for event in self.imu_events]
        self.realtime = bool(realtime)
        self.depth_mode = depth_mode
        self.max_pair_delta_ns = int(max_pair_dt_ms * 1_000_000)
        self.is_file = True
        self.has_metric_depth = bool(self.depth_events)
        self._closed = False
        self._start_ts_ns = self.rgb_events[0].ts_device_ns
        self.fps = float(self.metadata.get("viewer_config", {}).get("fps", 10.0) or 10.0)

        first = _load_npy(self.rgb_events[0], expected_dtype="uint8")
        if first.ndim != 3 or first.shape[2] != 3:
            raise ReplayFormatError(f"rgb payload must be HxWx3, got {first.shape}")
        src_h, src_w = first.shape[:2]
        if proc_width and src_w > proc_width:
            self.proc_width = int(proc_width)
            self.proc_height = int(round(src_h * self.proc_width / src_w / 2) * 2)
        else:
            self.proc_width = int(src_w)
            self.proc_height = int(src_h)
        self.K = crop_scaled_intrinsics(self.metadata, "CAM_A",
                                        self.proc_width, self.proc_height)
        self._depth_to_rgb = _transform_between_recorded_cameras(
            self.metadata, source_socket=1, target_socket=0)
        self.R_cam_imu = imu_to_camera_rotation_from_metadata(
            self.metadata, target_socket=0)

    def _resize_bgr(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.shape[1] == self.proc_width and bgr.shape[0] == self.proc_height:
            return bgr
        return cv2.resize(bgr, (self.proc_width, self.proc_height),
                          interpolation=cv2.INTER_AREA)

    def _depth_intrinsics(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        return crop_scaled_intrinsics(self.metadata, "CAM_B", w, h)

    def _load_depth_for(self, rgb_event: ReplayEvent,
                        bgr_shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
        depth_event = _nearest(self.depth_events, rgb_event.ts_device_ns,
                               self.max_pair_delta_ns)
        if depth_event is None:
            return None, None
        depth_mm = _load_npy(depth_event, expected_dtype="uint16")
        if depth_mm.ndim != 2:
            raise ReplayFormatError(f"depth payload must be HxW, got {depth_mm.shape}")
        depth_m = depth_mm.astype(np.float32) * 0.001
        h_rgb, w_rgb = bgr_shape
        aligned = None
        mode = self.depth_mode.lower()
        if mode in {"calibrated", "reproject"} and self._depth_to_rgb is not None:
            aligned = reproject_depth_to_rgb(
                depth_m, self._depth_intrinsics(depth_m.shape), self.K,
                self._depth_to_rgb, (h_rgb, w_rgb), splat_radius=1)
            # If the calibration interpretation is wrong for a legacy recording,
            # fail open to the smoke path while marking the limitation in metadata.
            if int(np.count_nonzero(aligned)) < max(50, int(0.01 * h_rgb * w_rgb)):
                aligned = None
        if aligned is None:
            aligned = cv2.resize(depth_m, (w_rgb, h_rgb),
                                 interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(aligned) & (aligned > 0)
        return aligned.astype(np.float32), valid.astype(np.uint8)

    def _load_gray_for(self, rgb_event: ReplayEvent,
                       bgr: np.ndarray) -> np.ndarray:
        # Use RGB-derived gray by default. Recorded left mono is not guaranteed to
        # be RGB-aligned, so feeding it directly to VO can break depth backprojection.
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _load_imu_for(self, rgb_event: ReplayEvent) -> dict[str, Any] | None:
        event = _nearest(self.imu_events, rgb_event.ts_device_ns,
                         max_delta_ns=50_000_000)
        if event is None:
            return None
        records = event.raw.get("records") or []
        if not records:
            return None
        rec = records[-1]
        sample: dict[str, Any] = {}
        accel = rec.get("accel")
        gyro = rec.get("gyro")
        if accel:
            sample["accel"] = np.array([accel.get("x", 0.0),
                                        accel.get("y", 0.0),
                                        accel.get("z", 0.0)], dtype=np.float32)
        if gyro:
            sample["gyro"] = np.array([gyro.get("x", 0.0),
                                       gyro.get("y", 0.0),
                                       gyro.get("z", 0.0)], dtype=np.float32)
        ts = rec.get("ts_device_ns") or event.ts_device_ns
        sample["timestamp_s"] = (int(ts) - self._start_ts_ns) / 1e9
        return sample or None

    def _record_to_imu_sample(self, event: ReplayEvent,
                              rec: dict[str, Any]) -> ImuSample:
        accel = rec.get("accel") or {}
        gyro = rec.get("gyro") or {}
        ts = int(rec.get("ts_device_ns") or event.ts_device_ns)
        return ImuSample(
            t=(ts - self._start_ts_ns) / 1e9,
            accel=np.array([accel.get("x", 0.0),
                            accel.get("y", 0.0),
                            accel.get("z", 0.0)], dtype=np.float64),
            gyro=np.array([gyro.get("x", 0.0),
                           gyro.get("y", 0.0),
                           gyro.get("z", 0.0)], dtype=np.float64),
        )

    def _load_imu_window(self, prev_rgb_ts_ns: int | None,
                         rgb_ts_ns: int) -> list[ImuSample]:
        if not self.imu_events:
            return []
        if prev_rgb_ts_ns is None:
            start_idx = bisect.bisect_left(self._imu_event_times, rgb_ts_ns)
        else:
            start_idx = bisect.bisect_right(self._imu_event_times, prev_rgb_ts_ns)
        end_idx = bisect.bisect_right(self._imu_event_times, rgb_ts_ns)
        samples: list[ImuSample] = []
        lower = rgb_ts_ns if prev_rgb_ts_ns is None else prev_rgb_ts_ns
        for event in self.imu_events[start_idx:end_idx]:
            for rec in event.raw.get("records") or []:
                ts = int(rec.get("ts_device_ns") or event.ts_device_ns)
                if prev_rgb_ts_ns is not None and ts <= lower:
                    continue
                if ts > rgb_ts_ns:
                    continue
                samples.append(self._record_to_imu_sample(event, rec))
        return samples

    def frames(self) -> Iterator[Frame]:
        wall_start = time.monotonic()
        prev_rgb_ts_ns: int | None = None
        for index, rgb_event in enumerate(self.rgb_events):
            if self._closed:
                return
            ts = (rgb_event.ts_device_ns - self._start_ts_ns) / 1e9
            if self.realtime:
                delay = ts - (time.monotonic() - wall_start)
                if delay > 0:
                    time.sleep(delay)
            bgr = _load_npy(rgb_event, expected_dtype="uint8")
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ReplayFormatError(f"rgb payload must be HxWx3, got {bgr.shape}")
            bgr = self._resize_bgr(bgr)
            depth_m, depth_conf = self._load_depth_for(rgb_event, bgr.shape[:2])
            imu_samples = self._load_imu_window(prev_rgb_ts_ns,
                                                rgb_event.ts_device_ns)
            yield Frame(
                ts=ts,
                bgr=bgr,
                index=index,
                depth_m=depth_m,
                depth_conf=depth_conf,
                K=self.K.copy(),
                gray_track=self._load_gray_for(rgb_event, bgr),
                imu=self._load_imu_for(rgb_event),
                imu_samples=imu_samples,
                metadata={
                    "recording": str(self.root),
                    "source_metadata": self.metadata,
                    "depth_mode": self.depth_mode,
                    "has_depth_to_rgb_calibration": self._depth_to_rgb is not None,
                    "imu_to_camera_rotation": (
                        None if self.R_cam_imu is None else self.R_cam_imu.tolist()),
                },
            )
            prev_rgb_ts_ns = rgb_event.ts_device_ns

    def release(self) -> None:
        self._closed = True
