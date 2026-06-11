"""gsplat 게이트: JIT 컴파일 + 최소 렌더 + 미니 최적화 루프 동작 확인 (CUDA).

Usage: python benchmarks/bench_gsplat.py
"""

import time

import numpy as np
import torch


def main() -> None:
    assert torch.cuda.is_available()
    t0 = time.perf_counter()
    from gsplat import rasterization
    print(f"import gsplat: {time.perf_counter() - t0:.1f}s")

    dev = "cuda"
    N, H, W = 5000, 283, 504
    rng = np.random.default_rng(0)
    means = torch.tensor(rng.uniform(-1, 1, (N, 3)), dtype=torch.float32,
                         device=dev, requires_grad=True)
    means.data[:, 2] += 3.0
    quats = torch.zeros(N, 4, device=dev)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)
    log_scales = torch.full((N, 3), -3.5, device=dev, requires_grad=True)
    logit_op = torch.zeros(N, device=dev, requires_grad=True)
    colors = torch.rand(N, 3, device=dev, requires_grad=True)
    K = torch.tensor([[[400.0, 0, W / 2], [0, 400.0, H / 2], [0, 0, 1]]],
                     device=dev)
    viewmat = torch.eye(4, device=dev)[None]

    t0 = time.perf_counter()
    render, alpha, _ = rasterization(
        means, torch.nn.functional.normalize(quats, dim=-1),
        log_scales.exp(), torch.sigmoid(logit_op), colors,
        viewmat, K, W, H, render_mode="RGB+ED")
    torch.cuda.synchronize()
    print(f"first render (JIT compile 포함): {time.perf_counter() - t0:.1f}s, "
          f"out={tuple(render.shape)}")

    target = torch.rand(1, H, W, 3, device=dev)
    opt = torch.optim.Adam([means, quats, log_scales, logit_op, colors], lr=1e-2)
    t0 = time.perf_counter()
    for _ in range(50):
        render, alpha, _ = rasterization(
            means, torch.nn.functional.normalize(quats, dim=-1),
            log_scales.exp(), torch.sigmoid(logit_op), colors,
            viewmat, K, W, H, render_mode="RGB+ED")
        loss = (render[..., :3] - target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 50
    print(f"opt step (5k gaussians, {W}x{H}): {dt * 1e3:.1f} ms/step")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("GATE: PASS")


if __name__ == "__main__":
    main()
