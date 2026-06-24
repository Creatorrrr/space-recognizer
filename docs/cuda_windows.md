# Windows CUDA Runtime Notes

This repo was originally documented around Apple MPS, but the current Windows
workstation can run the torch model paths on CUDA. Verified on 2026-06-24 with:

- GPU: NVIDIA GeForce RTX 4080, 16 GB VRAM
- Driver: 591.86
- PyTorch: `2.12.1+cu126`
- Torch CUDA runtime: `12.6`
- torchvision: `0.27.1+cu126`
- xformers: `0.0.35`

## Install

Use the repo virtualenv from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch==2.12.1+cu126 torchvision==0.27.1+cu126
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 xformers==0.0.35
```

The project installs `depth-anything-3` separately with `--no-deps`, because its
declared dependency set pins `numpy<2` and declares CUDA-centric extras. A
`pip check` warning for `depth-anything-3` can remain even when the runtime smoke
tests pass.

## Smoke Check

```powershell
@'
import json, torch
x = torch.randn((2048, 2048), device="cuda")
y = x @ x.T
torch.cuda.synchronize()
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0),
    "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
    "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
    "result_mean": float(y.mean().detach().cpu()),
}, indent=2))
'@ | .\.venv\Scripts\python.exe -
```

Expected shape of the output:

```json
{
  "torch": "2.12.1+cu126",
  "torch_cuda": "12.6",
  "cuda_available": true,
  "device": "NVIDIA GeForce RTX 4080"
}
```

## Replay Validation

The repo currently has recorded OAK sessions under `sources\session_*`, not
`sources\sample_720p.mp4`. Use replay smoke for a no-camera CUDA path check:

```powershell
.\.venv\Scripts\python.exe benchmarks\replay_smoke.py sources\session_20260624_054320_194430108151D05A00 --frames 60 --backend
```

Observed result on 2026-06-24:

```text
REPLAY_SMOKE session=session_20260624_054320_194430108151D05A00 imu=on payload_missing=0 pairs=750 pair_median_ms=0.7 pair_max_ms=2.6 frames=60 depth_frames=60 depth_valid_mean=0.932 lost=0 keyframes=11 avg_tracked=255.0 avg_inlier=0.93 imu_prior_frames=59 imu_lk_priors=59 imu_pnp_priors=59 imu_no_extrinsics=0 imu_no_samples=1 imu_blur_skipped_kf=0 imu_blur_forced_kf=0 detections=0 object_observations=0 objects=0 backend_keyframes=6 backend_points=148176 top_classes=[]
```

The command output is saved in `artifacts\cuda_activation\replay_backend_stdout.txt`
when using the local validation wrapper.

## CUDA Bottleneck Pass

The risk-adjusted CUDA pass adds three controls:

- `compute.tf32`: enables CUDA TF32 matmul/conv and cuDNN benchmark in the main
  process and backend worker.
- `compute.precision`: defaults to `fp32`; `bf16` is a CUDA-only opt-in wrapped
  around DA3 live depth, backend DA3/metric DA3, and DINOv2 inference.
- `backend.metric_anchor_every_n_windows` and
  `backend.metric_anchor_process_res`: control the optional DA3METRIC-LARGE
  forward used for display-scale anchoring.

Before/after replay artifacts:

```powershell
.\.venv\Scripts\python.exe benchmarks\replay_smoke.py sources\session_20260624_054320_194430108151D05A00 --frames 60 --backend --backend-metric-anchor --precision fp32 --no-tf32 --metric-anchor-every-n-windows 1 --metrics-out artifacts\cuda_bottleneck\baseline_fp32_no_tf32_metric_every1.json
.\.venv\Scripts\python.exe benchmarks\replay_smoke.py sources\session_20260624_054320_194430108151D05A00 --frames 60 --backend --backend-metric-anchor --precision bf16 --metric-anchor-every-n-windows 3 --metric-anchor-process-res 196 --metrics-out artifacts\cuda_bottleneck\optimized_bf16_tf32_metric_every3_res196.json
```

Observed on 2026-06-24:

| Run | backend_runtime_s | DA3 window ms | metric anchor ms | backend_points | lost | avg_inlier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32, TF32 off, metric every window | 12.80 | 11482 | 1309 | 148177 | 0 | 0.93 |
| bf16, TF32 on, metric res 196 | 10.14 | 9076 | 1048 | 148176 | 0 | 0.93 |

The optimized run preserved the replay quality counters on this session while
reducing direct backend runtime. Keep `compute.precision: fp32` as the safe
default until a longer office-loop or target-scene run confirms bf16 quality.

SDPA/xformers diagnostic artifact:
`artifacts\cuda_bottleneck\sdpa_xformers_diagnostic.json`.

Current local package evidence:

- DA3 attention code uses `torch.nn.functional.scaled_dot_product_attention` in
  `depth_anything_3\model\utils\attention.py` and
  `depth_anything_3\model\dinov2\layers\attention.py`.
- No DA3 call site directly uses `xformers.ops.memory_efficient_attention`.
- Torch CUDA reports flash, memory-efficient, math, and cuDNN SDPA backends
  enabled on the RTX 4080.

## CUDA Path Classification

Currently solved:

- `src/spacerec/device.py` selects `cuda` before `mps` and `cpu` when available.
- `src/spacerec/device.py` applies low-risk CUDA TF32/cuDNN runtime switches
  through `configure_torch_runtime()`.
- `src/spacerec/depth.py` runs DA3 live monocular depth under
  `torch.inference_mode()` and the configured inference autocast context.
- `src/spacerec/backend.py` runs backend DA3 and optional metric-anchor DA3 under
  `torch.inference_mode()` and the configured inference autocast context.
- `src/spacerec/appearance.py` runs DINOv2 warmup and embedding under
  `torch.inference_mode()` and the configured inference autocast context.
- `src/spacerec/detect.py` passes the selected torch device into Ultralytics
  `model.track(..., device=self.device)`, so YOLOE can use CUDA.
- `benchmarks/replay_smoke.py` can emit JSON timing/VRAM artifacts and can run
  the optional metric-anchor backend path for before/after comparisons.
- `backend.metric_anchor_process_res` safely resizes metric depth back to the
  backend window depth shape before scale estimation.

Not proven by this pass:

- `xformers` is installed and importable, but DA3 uses PyTorch SDPA in the
  inspected package. Do not claim xformers attention acceleration unless a later
  profiler proves an xformers operator is actually used.
- The replay sample uses only a small part of the RTX 4080 VRAM budget. That is
  good headroom, not evidence that GPU throughput is saturated or fully tuned.
- `bf16` is validated only on the recorded OAK replay smoke above; keep it
  opt-in until longer target-scene validation passes.

Currently inactive in `config.yaml`:

- `mesh.enabled: false`, so TSDF mesh rebuild is not on the hot path for the
  current default run.

Separate refactors needed for stronger CUDA utilization:

- Object detection still synchronizes masks and boxes back to CPU via
  `.cpu().numpy()` before OpenCV resize and object localization.
- `backend.metric_anchor: true` still performs an additional metric DA3 forward,
  but its cadence and process resolution are now configurable.
- Visual odometry, dynamic masks, local object localization, and most geometry
  fusion remain OpenCV/NumPy CPU paths.
- When mesh is enabled, the TSDF mesh path is Open3D/NumPy-style CPU work rather
  than an Open3D Tensor CUDA pipeline.
- The backend process structure was originally added for MPS isolation. On CUDA
  it avoids shared-process contention, but it can duplicate model memory and does
  not coordinate CUDA streams across live/backend work.
- YOLOE mask GPU resize remains unimplemented because the current replay sample
  has `detections=0`; use a detection-heavy sample before changing this path.
