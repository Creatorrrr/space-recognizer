"""DepthAI OAK-D-Lite source.

Provides RGB frames plus metric stereo depth aligned to the RGB preview when
DepthAI supports it. The rest of the pipeline can consume it through the same
Frame iterator contract as VideoSource.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import cv2
import numpy as np

from .capture import Frame
from .config import CaptureCfg


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _socket(dai: Any, name: str, fallback: str) -> Any:
    sockets = dai.CameraBoardSocket
    if hasattr(sockets, name):
        return getattr(sockets, name)
    return getattr(sockets, fallback)


def _create_xout(dai: Any, pipeline: Any, stream_name: str) -> Any:
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(stream_name)
    return xout


def _depth_resolution(dai: Any, value: str) -> Any:
    res = dai.MonoCameraProperties.SensorResolution
    key = value.strip().lower().replace("_", "").replace("-", "")
    if key in {"400p", "400"}:
        return res.THE_400_P
    if key in {"480p", "480"} and hasattr(res, "THE_480_P"):
        return res.THE_480_P
    if key in {"720p", "720"} and hasattr(res, "THE_720_P"):
        return res.THE_720_P
    return res.THE_400_P


def _median_filter(dai: Any, value: str) -> Any:
    filt = dai.MedianFilter
    key = value.strip().lower().replace("_", "").replace("-", "")
    mapping = {
        "off": "MEDIAN_OFF",
        "3x3": "KERNEL_3x3",
        "5x5": "KERNEL_5x5",
        "7x7": "KERNEL_7x7",
    }
    return getattr(filt, mapping.get(key, "KERNEL_7x7"), filt.KERNEL_7x7)


class OakSource:
    """Live OAK-D-Lite RGB + stereo-depth source."""

    has_metric_depth = True

    def __init__(self, cfg: CaptureCfg, proc_width: int = 1280):
        try:
            import depthai as dai
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "depthai is required for OAK input. Install with "
                '`uv pip install -p .venv -e ".[oak]"` or '
                "`uv pip install -p .venv depthai`."
            ) from exc

        self.dai = dai
        self.cfg = cfg
        self.proc_width = int(cfg.oak_rgb_width or proc_width)
        self.proc_height = int(cfg.oak_rgb_height)
        self.fps = float(cfg.oak_fps)
        self.realtime = True
        self.is_file = False
        self._closed = False
        self._device = None
        self._queues: dict[str, Any] = {}
        self._stream_names: tuple[str, ...] = ()
        self._imu_enabled = False
        self.metadata: dict[str, Any] = {}
        self.K: np.ndarray | None = None

        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            raise RuntimeError("No OAK device detected")

        last_exc: Exception | None = None
        attempts = [bool(cfg.oak_enable_imu), False] if cfg.oak_enable_imu else [False]
        for enable_imu in dict.fromkeys(attempts):
            pipeline, stream_names = self._build_pipeline(enable_imu=enable_imu)
            for attempt in range(2):
                try:
                    self._device = dai.Device(pipeline)
                    self._stream_names = tuple(stream_names)
                    self._imu_enabled = enable_imu
                    break
                except RuntimeError as exc:
                    last_exc = exc
                    msg = str(exc)
                    transient = ("Failed to boot device" in msg
                                 or "X_LINK_DEVICE_NOT_FOUND" in msg)
                    if attempt == 0 and transient:
                        time.sleep(2.0)
                        continue
                    break
            if self._device is not None:
                break
        if self._device is None:
            raise RuntimeError(f"Failed to open OAK device: {last_exc}")

        self.metadata = self._collect_metadata()
        self.K = self._read_intrinsics()
        self._queues = {
            name: self._device.getOutputQueue(
                name=name, maxSize=int(cfg.oak_queue_size), blocking=False)
            for name in self._stream_names
        }

    def _build_pipeline(self, enable_imu: bool) -> tuple[Any, list[str]]:
        dai, cfg = self.dai, self.cfg
        pipeline = dai.Pipeline()
        streams = ["rgb", "depth", "left"]

        color = pipeline.create(dai.node.ColorCamera)
        color.setBoardSocket(_socket(dai, "CAM_A", "RGB"))
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setPreviewSize(self.proc_width, self.proc_height)
        color.setInterleaved(False)
        color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        color.setFps(self.fps)
        color.preview.link(_create_xout(dai, pipeline, "rgb").input)

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(_socket(dai, "CAM_B", "LEFT"))
        mono_right.setBoardSocket(_socket(dai, "CAM_C", "RIGHT"))
        mono_res = _depth_resolution(dai, cfg.oak_depth_resolution)
        mono_left.setResolution(mono_res)
        mono_right.setResolution(mono_res)
        mono_left.setFps(self.fps)
        mono_right.setFps(self.fps)
        mono_left.out.link(_create_xout(dai, pipeline, "left").input)

        stereo = pipeline.create(dai.node.StereoDepth)
        try:
            stereo_preset = dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        except AttributeError:
            stereo_preset = dai.node.StereoDepth.PresetMode.DEFAULT
        stereo.setDefaultProfilePreset(stereo_preset)
        stereo.initialConfig.setMedianFilter(_median_filter(dai, cfg.oak_median_filter))
        stereo.setLeftRightCheck(bool(cfg.oak_lr_check))
        stereo.setExtendedDisparity(bool(cfg.oak_extended_disparity))
        stereo.setSubpixel(bool(cfg.oak_subpixel))
        if cfg.oak_align_depth_to_rgb and hasattr(stereo, "setDepthAlign"):
            stereo.setDepthAlign(_socket(dai, "CAM_A", "RGB"))
            if hasattr(stereo, "setOutputSize"):
                stereo.setOutputSize(self.proc_width, self.proc_height)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        stereo.depth.link(_create_xout(dai, pipeline, "depth").input)

        if enable_imu:
            imu = pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(
                [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
                int(cfg.oak_imu_rate_hz),
            )
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            imu.out.link(_create_xout(dai, pipeline, "imu").input)
            streams.append("imu")

        return pipeline, streams

    def _latest(self, name: str) -> Any:
        q = self._queues.get(name)
        latest = None
        while q is not None:
            msg = q.tryGet()
            if msg is None:
                return latest
            latest = msg
        return latest

    def _collect_metadata(self) -> dict[str, Any]:
        device = self._device
        assert device is not None
        meta = {
            "name": device.getDeviceName() if hasattr(device, "getDeviceName") else "Unknown",
            "mxid": device.getMxId() if hasattr(device, "getMxId") else "Unknown",
            "usb_speed": "Unknown",
            "imu_model": "Unknown",
            "imu_stream": self._imu_enabled,
        }
        if hasattr(device, "getUsbSpeed"):
            meta["usb_speed"] = _enum_name(device.getUsbSpeed())
        if hasattr(device, "getConnectedIMU"):
            try:
                meta["imu_model"] = _enum_name(device.getConnectedIMU())
            except RuntimeError:
                meta["imu_model"] = "Unavailable"
        return meta

    def _read_intrinsics(self) -> np.ndarray | None:
        device = self._device
        assert device is not None
        try:
            calib = device.readCalibration()
            K = calib.getCameraIntrinsics(
                _socket(self.dai, "CAM_A", "RGB"), self.proc_width, self.proc_height)
            return np.asarray(K, dtype=np.float64)
        except Exception:
            return None

    def _read_imu(self) -> dict[str, Any] | None:
        msg = self._latest("imu")
        if msg is None:
            return None
        packets = getattr(msg, "packets", None)
        if not packets:
            return None
        packet = packets[-1]
        sample: dict[str, Any] = {}
        accel = getattr(packet, "acceleroMeter", None)
        gyro = getattr(packet, "gyroscope", None)
        if accel is not None:
            sample["accel"] = np.array([accel.x, accel.y, accel.z], dtype=np.float32)
        if gyro is not None:
            sample["gyro"] = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float32)
        ts = getattr(packet, "timestamp", None)
        if ts is not None:
            try:
                sample["timestamp_s"] = float(ts.total_seconds())
            except AttributeError:
                sample["timestamp_s"] = float(ts)
        return sample or None

    def frames(self) -> Iterator[Frame]:
        start = time.monotonic()
        index = -1
        last_depth_m: np.ndarray | None = None
        last_gray: np.ndarray | None = None
        last_imu: dict[str, Any] | None = None
        while not self._closed:
            rgb_msg = self._queues["rgb"].get()
            bgr = rgb_msg.getCvFrame()
            if bgr.shape[1] != self.proc_width or bgr.shape[0] != self.proc_height:
                bgr = cv2.resize(bgr, (self.proc_width, self.proc_height),
                                 interpolation=cv2.INTER_AREA)

            depth_msg = self._latest("depth")
            if depth_msg is not None:
                depth = depth_msg.getFrame().astype(np.float32) * 0.001
                if depth.shape[:2] != bgr.shape[:2]:
                    depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                last_depth_m = depth

            left_msg = self._latest("left")
            if left_msg is not None:
                last_gray = left_msg.getCvFrame()

            imu_sample = self._read_imu()
            if imu_sample is not None:
                last_imu = imu_sample

            index += 1
            yield Frame(
                ts=time.monotonic() - start,
                bgr=bgr,
                index=index,
                depth_m=None if last_depth_m is None else last_depth_m.copy(),
                depth_conf=None,
                K=None if self.K is None else self.K.copy(),
                gray_track=None if last_gray is None else last_gray.copy(),
                imu=None if last_imu is None else dict(last_imu),
                metadata=self.metadata,
            )

    def release(self) -> None:
        self._closed = True
        if self._device is not None:
            self._device.close()
            self._device = None
