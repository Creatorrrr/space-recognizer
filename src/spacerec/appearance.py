"""Appearance embeddings for object re-identification (DINOv2-small, MPS).

같은 클래스의 이웃 물체(예: 한 방의 침대 두 대)는 위치만으로 구분할 수 없다.
검출 crop의 시각 임베딩을 노드에 저장해 두면, 화면 밖으로 나갔다 돌아온
물체를 외형으로 같은 노드에 복원할 수 있다.
"""

from __future__ import annotations

import numpy as np
import torch

from .device import select_torch_device

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_INPUT = 224  # 14의 배수 (DINOv2 patch 14)


class AppearanceEmbedder:
    def __init__(self, device: str | None = None):
        self.device = select_torch_device(device)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def warmup(self) -> None:
        """첫 추론의 커널 컴파일을 미리 치러 실시간 루프의 지연을 막는다."""
        dummy = torch.zeros(1, 3, _INPUT, _INPUT, device=self.device)
        self.model(dummy)

    @torch.no_grad()
    def embed_frame(self, rgb: np.ndarray) -> np.ndarray:
        """전체 프레임의 전역 임베딩 (L2 정규화) — 키프레임 place recognition용."""
        import cv2
        img = cv2.resize(rgb.astype(np.float32) / 255.0, (_INPUT, _INPUT),
                         interpolation=cv2.INTER_AREA)
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        batch = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(self.device)
        feat = self.model(batch).float().cpu().numpy()[0]
        return feat / (np.linalg.norm(feat) + 1e-9)

    @torch.no_grad()
    def embed(self, bgr: np.ndarray, observations: list) -> None:
        """각 Observation의 crop을 배치로 임베딩해 obs.emb에 채운다 (L2 정규화)."""
        if not observations:
            return
        h, w = bgr.shape[:2]
        crops = []
        for obs in observations:
            x0, y0, x1, y1 = obs.det.box
            # 약간의 여백을 두고 crop
            mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
            x0 = int(np.clip(x0 - mx, 0, w - 2))
            x1 = int(np.clip(x1 + mx, x0 + 1, w))
            y0 = int(np.clip(y0 - my, 0, h - 2))
            y1 = int(np.clip(y1 + my, y0 + 1, h))
            crop = bgr[y0:y1, x0:x1, ::-1].astype(np.float32) / 255.0  # RGB
            if obs.det.mask is not None:
                # 배경을 중간 회색으로 지워 주변 물체/벽의 영향 차단
                m = obs.det.mask[y0:y1, x0:x1]
                crop = np.where(m[..., None], crop, 0.45)
            import cv2
            crop = cv2.resize(crop, (_INPUT, _INPUT), interpolation=cv2.INTER_AREA)
            crops.append((crop - _IMAGENET_MEAN) / _IMAGENET_STD)

        batch = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2)).to(self.device)
        feats = self.model(batch).float().cpu().numpy()  # (N, 384) CLS tokens
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
        for obs, f in zip(observations, feats):
            obs.emb = f
