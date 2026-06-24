import sys
import types

import numpy as np
import torch

from spacerec.depth import DepthEstimator


class _FakePrediction:
    def __init__(self):
        self.depth = np.ones((1, 2, 2), dtype=np.float32)
        self.intrinsics = np.repeat(
            np.array([[[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]]], dtype=np.float64),
            1,
            axis=0,
        )


class _FakeDepthAnything3:
    def __init__(self):
        self.inference_modes = []

    @classmethod
    def from_pretrained(cls, _model_name):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def inference(self, _images, process_res):
        assert process_res == 2
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return _FakePrediction()


def test_depth_estimator_runs_da3_inference_in_inference_mode(monkeypatch):
    parent = types.ModuleType("depth_anything_3")
    api = types.ModuleType("depth_anything_3.api")
    api.DepthAnything3 = _FakeDepthAnything3
    monkeypatch.setitem(sys.modules, "depth_anything_3", parent)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api)

    estimator = DepthEstimator("fake-da3", process_res=2, device="cpu")

    depth = estimator.infer(np.zeros((2, 2, 3), dtype=np.uint8))

    assert depth.shape == (2, 2)
    assert estimator.model.inference_modes == [True]
