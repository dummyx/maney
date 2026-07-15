from __future__ import annotations

from importlib import import_module
from typing import Any


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
