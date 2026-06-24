import numpy as np
import torch

from spacerec.appearance import AppearanceEmbedder
from spacerec.detect import Detection
from spacerec.objects import Observation


class _FakeDino:
    def __init__(self):
        self.inference_modes = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return torch.ones((batch.shape[0], 384), dtype=torch.float32, device=batch.device)


def _observation() -> Observation:
    det = Detection(
        track_id=1,
        cls_name="chair",
        conf=0.9,
        box=np.array([0, 0, 4, 4], dtype=np.float32),
        mask=None,
    )
    return Observation(det=det, position=np.zeros(3, dtype=np.float32), size=0.5)


def test_appearance_embedder_uses_inference_mode_for_warmup_and_embed(monkeypatch):
    model = _FakeDino()
    monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: model)
    embedder = AppearanceEmbedder(device="cpu")
    obs = _observation()

    embedder.warmup()
    embedder.embed(np.zeros((4, 4, 3), dtype=np.uint8), [obs])

    assert model.inference_modes == [True, True]
    assert obs.emb.shape == (384,)
