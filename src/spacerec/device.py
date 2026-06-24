"""Torch device selection helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def select_torch_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_torch_runtime(
    device: str | None = None,
    *,
    tf32: bool = True,
    cudnn_benchmark: bool = True,
) -> str:
    """Resolve a torch device and apply low-risk CUDA runtime switches."""
    resolved = select_torch_device(device)
    if resolved == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    return resolved


def autocast_context(device: str, precision: str = "fp32") -> ContextManager:
    """Return the configured inference autocast context for a torch device."""
    precision = (precision or "fp32").lower()
    if precision in ("fp32", "float32"):
        return nullcontext()
    if precision in ("bf16", "bfloat16") and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision in ("bf16", "bfloat16"):
        return nullcontext()
    raise ValueError(f"unsupported compute precision: {precision!r}")
