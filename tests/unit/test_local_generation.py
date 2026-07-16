from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nlp_trader.nlp import local_generation
from nlp_trader.nlp.local_generation import (
    GenerationRequest,
    GenerationResponse,
    LlamaCppGenerationSession,
    LocalGenerationConfig,
)


def _config(tmp_path: Path) -> LocalGenerationConfig:
    model_path = tmp_path / "fixture.gguf"
    model_path.write_bytes(b"synthetic GGUF transport fixture")
    return LocalGenerationConfig(
        model_path=model_path,
        expected_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        context_tokens=128,
        prompt_batch_tokens=32,
        max_input_tokens=64,
        max_new_tokens=32,
    )


def test_local_config_validates_identity_without_reading_model_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_generation,
        "file_sha256",
        lambda _path: pytest.fail("config construction must not hash a multi-GB model"),
    )

    config = _config(tmp_path)

    assert config.model_path.suffix == ".gguf"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        LocalGenerationConfig(
            model_path=config.model_path,
            expected_model_sha256=config.expected_model_sha256.upper(),
            context_tokens=128,
            prompt_batch_tokens=32,
            max_input_tokens=64,
            max_new_tokens=32,
        )


def test_session_hashes_model_immediately_before_and_after_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    hash_calls: list[Path] = []
    events: list[str] = []

    def record_hash(path: Path) -> str:
        hash_calls.append(path)
        events.append("hash")
        return config.expected_model_sha256

    monkeypatch.setattr(local_generation, "file_sha256", record_hash)

    class FakeModel:
        metadata = {"tokenizer.chat_template": "{{ messages }}"}

        def __init__(self, **_options: object) -> None:
            events.append("load")

        @staticmethod
        def token_eos() -> int:
            return 2

        @staticmethod
        def token_bos() -> int:
            return 1

        @staticmethod
        def detokenize(_tokens: list[int], *, special: bool) -> bytes:
            assert special is True
            return b"token"

    class FakeFormatter:
        def __init__(self, **_options: object) -> None:
            pass

    runtime = SimpleNamespace(
        Llama=FakeModel,
        llama_chat_format=SimpleNamespace(Jinja2ChatFormatter=FakeFormatter),
        llama_cpp=SimpleNamespace(llama_supports_gpu_offload=lambda: False),
    )

    LlamaCppGenerationSession(
        config,
        {"type": "object"},
        module_loader=lambda _name: runtime,
        runtime_version="test-runtime",
    )

    assert hash_calls == [config.model_path, config.model_path]
    assert events == ["hash", "load", "hash"]


def test_session_rejects_model_mutation_during_loading(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class MutatingModel:
        metadata = {"tokenizer.chat_template": "{{ messages }}"}

        def __init__(self, **_options: Any) -> None:
            config.model_path.write_bytes(b"mutated while loading")

    runtime = SimpleNamespace(
        Llama=MutatingModel,
        llama_cpp=SimpleNamespace(llama_supports_gpu_offload=lambda: False),
    )

    with pytest.raises(ValueError, match="changed while model files were loading"):
        LlamaCppGenerationSession(
            config,
            {"type": "object"},
            module_loader=lambda _name: runtime,
            runtime_version="test-runtime",
        )


def test_generation_transport_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GenerationRequest(request_id="", prompt="prompt")
    with pytest.raises(ValueError, match="cannot contain"):
        GenerationResponse(request_id="request", generated_text="bad", input_too_long=True)
