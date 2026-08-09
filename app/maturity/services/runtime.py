from __future__ import annotations

import contextlib
import gc
from time import perf_counter
from typing import Callable

import torch

from app.maturity.config import DEVICE_MODE


def format_elapsed(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:05.2f}s"


def log_elapsed(log: Callable[[str], None], label: str, started_at: float) -> None:
    log(f"[timing] {label}: {format_elapsed(perf_counter() - started_at)}")


def select_device(log: Callable[[str], None] | None = None) -> torch.device:
    requested = DEVICE_MODE

    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested in {"cpu", "cuda", "mps"}:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if log is not None:
        if device.type == "cuda":
            log(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            log(f"Device: {device.type.upper()} (CUDA unavailable; running locally without GPU acceleration)")

    return device


def amp_dtype(device: torch.device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device: torch.device, enabled: bool = True):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype(device), enabled=enabled)


def model_load_kwargs(device: torch.device) -> dict:
    kwargs = {"low_cpu_mem_usage": True}
    if device.type == "cuda":
        kwargs["torch_dtype"] = amp_dtype(device)
    return kwargs


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
