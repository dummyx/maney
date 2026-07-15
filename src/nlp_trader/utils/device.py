from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class LlamaCppDevice:
    """Effective llama.cpp compute placement."""

    name: Literal["metal", "cpu"]
    gpu_layers: int


def select_llama_cpp_device(llama_cpp: Any, requested_gpu_layers: int) -> LlamaCppDevice:
    """Use Metal offload when the native binding supports it, otherwise use CPU."""

    native = getattr(llama_cpp, "llama_cpp", llama_cpp)
    supports_gpu_offload = getattr(native, "llama_supports_gpu_offload", None)
    has_gpu_offload = bool(supports_gpu_offload()) if callable(supports_gpu_offload) else False
    if requested_gpu_layers != 0 and has_gpu_offload:
        return LlamaCppDevice(name="metal", gpu_layers=requested_gpu_layers)
    return LlamaCppDevice(name="cpu", gpu_layers=0)


def get_torch_device() -> Any:
    """Return MPS on Apple Silicon when available, otherwise CPU.

    PyTorch is imported lazily so the baseline package remains usable without the
    optional ``nlp`` dependency group.
    """

    try:
        torch = import_module("torch")
    except ImportError as exc:  # pragma: no cover - exercised only with optional extras
        raise RuntimeError("PyTorch is optional; install it with `uv sync --extra nlp`") from exc
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
