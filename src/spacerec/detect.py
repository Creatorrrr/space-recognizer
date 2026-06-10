"""YOLO26-seg detection + ByteTrack tracking wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class Detection:
    track_id: int            # -1 if the tracker has not confirmed the object yet
    cls_name: str
    conf: float
    box: np.ndarray          # xyxy, frame pixels
    mask: np.ndarray | None  # bool HxW at frame resolution


class ObjectDetector:
    def __init__(self, model_path: str, conf: float = 0.35, device: str | None = None):
        from ultralytics import YOLO

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = self.model.names

    def track(self, bgr: np.ndarray) -> list[Detection]:
        result = self.model.track(
            bgr, device=self.device, conf=self.conf, persist=True,
            tracker="bytetrack.yaml", verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        h, w = bgr.shape[:2]
        ids = boxes.id.int().tolist() if boxes.id is not None else [-1] * len(boxes)
        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # (N, mh, mw) float

        detections = []
        for i in range(len(boxes)):
            mask = None
            if masks is not None:
                mask = cv2.resize(masks[i], (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            detections.append(Detection(
                track_id=ids[i],
                cls_name=self.names[int(boxes.cls[i])],
                conf=float(boxes.conf[i]),
                box=boxes.xyxy[i].cpu().numpy(),
                mask=mask,
            ))
        return detections
