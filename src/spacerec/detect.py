"""Detection + ByteTrack tracking wrapper.

두 가지 모드를 지원한다:
- 고정 클래스(COCO): YOLO26-seg 등 — vocabulary 미지정 시
- 오픈 보캐뷸러리: YOLOE-seg + 텍스트 어휘 — COCO에 없는 실내 물체
  (rug, wardrobe 등)를 올바른 라벨로 검출. COCO 모델은 이런 물체를
  가장 비슷한 클래스(bed 등)로 오인하는 문제가 있었다.
"""

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
    def __init__(self, model_path: str, conf: float = 0.35, device: str | None = None,
                 vocabulary: list[str] | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        if vocabulary:
            from ultralytics import YOLOE

            self.model = YOLOE(model_path)
            self.model.set_classes(vocabulary, self.model.get_text_pe(vocabulary))
        else:
            from ultralytics import YOLO

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
