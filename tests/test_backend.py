import numpy as np
import torch

from spacerec.backend import BackendKeyframe, _Worker
from spacerec.config import BackendCfg


class _Pred:
    def __init__(self, n):
        self.depth = np.ones((n, 2, 2), dtype=np.float32)
        self.conf = np.ones((n, 2, 2), dtype=np.float32)
        self.intrinsics = np.repeat(
            np.array([[[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]]], dtype=np.float64),
            n,
            axis=0,
        )


class _Model:
    def __init__(self):
        self.inference_modes = []
        self.process_res_values = []

    def inference(self, images, process_res):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.process_res_values.append(process_res)
        return _Pred(len(images))


class _ProcessResShapeModel(_Model):
    def inference(self, images, process_res):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.process_res_values.append(process_res)
        pred = _Pred(len(images))
        pred.depth = np.ones((len(images), process_res, process_res), dtype=np.float32)
        pred.conf = np.ones((len(images), process_res, process_res), dtype=np.float32)
        pred.intrinsics = np.repeat(
            np.array([[[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]]], dtype=np.float64),
            len(images),
            axis=0,
        )
        return pred


def _kf(kf_id):
    T = np.eye(4)
    T[0, 3] = 0.1 * kf_id
    return BackendKeyframe(
        kf_id=kf_id,
        ts=float(kf_id),
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        T_wc_live=T,
        raw_depth=np.ones((2, 2), dtype=np.float32),
        dyn_mask=None,
    )


def test_backend_consumes_pending_keyframes_fifo_without_dropping():
    worker = _Worker.__new__(_Worker)
    worker.cfg = BackendCfg(window_size=4, overlap=2, metric_anchor=False)
    worker.process_res = 2
    worker.model = _Model()
    worker.metric_model = None
    worker._meters_per_unit = None
    worker.kf_global_poses = {}
    worker._pending = [_kf(i) for i in range(8)]
    worker._reconstructed = []

    first = worker._run_window()
    second = worker._run_window()

    assert first.window_ids == [0, 1]
    assert first.view_depths.shape == (2, 2, 2)
    assert first.view_valid.shape == (2, 2, 2)
    assert first.view_colors.shape == (2, 2, 2, 3)
    assert first.view_poses.shape == (2, 4, 4)
    assert first.view_intrinsics.shape == (2, 3, 3)
    assert first.anchor_kf_id == 0
    assert second.window_ids == [0, 1, 2, 3]
    assert [kf.kf_id for kf in worker._pending] == [4, 5, 6, 7]


def test_backend_window_runs_da3_models_in_inference_mode():
    worker = _Worker.__new__(_Worker)
    worker.cfg = BackendCfg(window_size=2, overlap=0, metric_anchor=True)
    worker.process_res = 2
    worker.model = _Model()
    worker.metric_model = _Model()
    worker._meters_per_unit = None
    worker._windows_run = 0
    worker.kf_global_poses = {}
    worker._pending = [_kf(0), _kf(1)]
    worker._reconstructed = []
    worker.device = "cpu"
    worker.precision = "fp32"

    worker._run_window()

    assert worker.model.inference_modes == [True]
    assert worker.metric_model.inference_modes == [True]


def test_backend_metric_anchor_respects_cadence_and_process_res():
    worker = _Worker.__new__(_Worker)
    worker.cfg = BackendCfg(
        window_size=2,
        overlap=0,
        metric_anchor=True,
        metric_anchor_every_n_windows=2,
        metric_anchor_process_res=3,
    )
    worker.process_res = 2
    worker.model = _Model()
    worker.metric_model = _ProcessResShapeModel()
    worker._meters_per_unit = None
    worker._windows_run = 0
    worker.kf_global_poses = {}
    worker._pending = [_kf(0), _kf(1), _kf(2), _kf(3)]
    worker._reconstructed = []
    worker.device = "cpu"
    worker.precision = "fp32"

    first = worker._run_window()
    second = worker._run_window()

    assert first.window_ids == [0, 1]
    assert second.window_ids == [2, 3]
    assert worker.model.process_res_values == [2, 2]
    assert worker.metric_model.process_res_values == [3]
