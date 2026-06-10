"""Video input: real webcam, or a video file emulating a webcam.

File mode with realtime=True drops frames against the wall clock, so the
pipeline sees the same "latest frame only" behaviour as a live camera.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    ts: float            # seconds since stream start (video time)
    bgr: np.ndarray      # processing-resolution BGR image
    index: int           # source frame index


class VideoSource:
    def __init__(self, source: str | int, proc_width: int = 1280, realtime: bool = True):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source: {source!r}")
        self.is_file = isinstance(source, str)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.proc_width = proc_width
        self.realtime = realtime
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.proc_height = int(round(h * proc_width / w / 2) * 2)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] == self.proc_width:
            return frame
        return cv2.resize(frame, (self.proc_width, self.proc_height),
                          interpolation=cv2.INTER_AREA)

    def frames(self) -> Iterator[Frame]:
        if self.is_file and self.realtime:
            yield from self._frames_file_realtime()
        else:
            yield from self._frames_sequential()

    def _frames_sequential(self) -> Iterator[Frame]:
        """Webcam, or file without pacing (process every frame)."""
        start = time.monotonic()
        index = -1
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            index += 1
            ts = index / self.fps if self.is_file else time.monotonic() - start
            yield Frame(ts=ts, bgr=self._resize(frame), index=index)

    def _frames_file_realtime(self) -> Iterator[Frame]:
        """Skip file frames so playback follows the wall clock."""
        start = time.monotonic()
        index = -1
        while True:
            target = int((time.monotonic() - start) * self.fps)
            if target <= index:  # ahead of the clock: wait for the next frame slot
                time.sleep(max(0.0, (index + 1) / self.fps - (time.monotonic() - start)))
                target = index + 1
            # grab() (decode-light) until we catch up with the wall clock
            while index < target:
                if not self.cap.grab():
                    return
                index += 1
            ok, frame = self.cap.retrieve()
            if not ok:
                return
            yield Frame(ts=index / self.fps, bgr=self._resize(frame), index=index)

    def release(self) -> None:
        self.cap.release()
