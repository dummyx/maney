from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_trader.config import (
    LLMAnnotationsConfig,
    ResearchConfig,
    TransformerConfig,
    validate_config,
)
from nlp_trader.research import input_manifest


def _enabled_llm_config(
    generated_config: ResearchConfig,
    model_path: Path,
    *,
    feature_mode: str = "sidecar",
) -> ResearchConfig:
    payload = generated_config.model_dump(mode="python")
    payload["paths"]["llm_model"] = model_path
    payload["llm_annotations"] = {
        "enabled": True,
        "feature_mode": feature_mode,
        "model_id": "local-test-causal-lm",
        "model_revision": "immutable-test-revision",
        "model_license_or_terms_ref": "redistributable-test-fixture",
    }
    if feature_mode == "augment":
        payload["models"]["families"] = [
            "traditional",
            "text",
            "combined",
            "llm",
            "traditional_llm",
            "all",
        ]
    return ResearchConfig.model_validate(payload)


def test_enabled_llm_gguf_file_is_validated_and_hashed(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    first_bytes = b"first immutable GGUF model bytes"
    model_path = tmp_path / "local-model.gguf"
    model_path.write_bytes(first_bytes)
    config = _enabled_llm_config(generated_config, model_path)

    assert validate_config(config) == []
    first = next(entry for entry in input_manifest(config) if entry["role"] == "llm_model")
    assert first["input_kind"] == "local_gguf_file"
    assert first["model_id"] == "local-test-causal-lm"
    assert first["model_revision"] == "immutable-test-revision"
    assert first["license_or_terms_ref"] == "redistributable-test-fixture"
    assert first["bytes"] == len(first_bytes)
    assert first["sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert "files" not in first
    assert "file_count" not in first

    model_path.write_bytes(b"different immutable GGUF model bytes")
    second = next(entry for entry in input_manifest(config) if entry["role"] == "llm_model")
    assert second["sha256"] != first["sha256"]


def test_llm_defaults_select_pinned_qwen_gguf_and_llama_cpp_runtime() -> None:
    config = LLMAnnotationsConfig()

    assert config.backend == "llama_cpp_gguf"
    assert config.model_id == "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL"
    assert config.model_revision == "5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf"
    assert config.model_license_or_terms_ref == (
        "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/tree/"
        "5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf"
    )
    assert config.context_tokens == 8192
    assert config.prompt_batch_tokens == 512
    assert config.gpu_layers == -1
    assert config.flash_attention is True
    assert config.use_mmap is True


def test_llm_application_requires_enabled_mode_and_valid_runtime_settings() -> None:
    with pytest.raises(ValidationError, match="enabled must be true"):
        LLMAnnotationsConfig(feature_mode="augment")

    with pytest.raises(ValidationError, match="model_revision"):
        LLMAnnotationsConfig(model_revision=" ")

    with pytest.raises(ValidationError, match="local_files_only"):
        LLMAnnotationsConfig.model_validate({"local_files_only": False})

    with pytest.raises(ValidationError, match="trust_remote_code"):
        LLMAnnotationsConfig.model_validate({"trust_remote_code": True})

    with pytest.raises(ValidationError, match="backend"):
        LLMAnnotationsConfig.model_validate({"backend": "transformers_causal_lm"})

    with pytest.raises(ValidationError, match="context_tokens"):
        LLMAnnotationsConfig(
            max_input_tokens=100,
            max_new_tokens=50,
            context_tokens=149,
        )

    with pytest.raises(ValidationError, match="prompt_batch_tokens"):
        LLMAnnotationsConfig(
            max_input_tokens=8,
            max_new_tokens=8,
            context_tokens=32,
            prompt_batch_tokens=33,
        )

    with pytest.raises(ValidationError, match="gpu_layers"):
        LLMAnnotationsConfig(gpu_layers=-2)


def test_enabled_llm_rejects_directory_and_non_gguf_file(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "model-directory"
    directory.mkdir()
    directory_config = _enabled_llm_config(generated_config, directory)
    assert validate_config(directory_config) == [
        f"llm_model must be an existing local GGUF file: {directory}"
    ]

    non_gguf = tmp_path / "model.bin"
    non_gguf.write_bytes(b"not a GGUF fixture")
    non_gguf_config = _enabled_llm_config(generated_config, non_gguf)
    assert validate_config(non_gguf_config) == [
        f"llm_model must have a .gguf extension: {non_gguf}"
    ]


def test_llm_augmentation_requires_ablation_families_and_can_coexist_with_transformer(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "local-model.gguf"
    model_path.write_bytes(b"GGUF test fixture")
    configured = _enabled_llm_config(
        generated_config,
        model_path,
        feature_mode="augment",
    )
    payload = configured.model_dump(mode="python")
    payload["transformer"] = TransformerConfig(
        enabled=True,
        model_name="local-transformer",
    )

    validated = ResearchConfig.model_validate(payload)
    assert validated.transformer.enabled is True
    assert validated.llm_annotations.feature_mode == "augment"

    payload["models"]["families"] = ["traditional", "text", "combined"]
    with pytest.raises(ValidationError, match="canonical LLM ablation"):
        ResearchConfig.model_validate(payload)


def test_llm_cost_rates_must_be_configured_as_a_pair() -> None:
    with pytest.raises(ValidationError, match="configured together"):
        LLMAnnotationsConfig(input_cost_per_million_tokens_usd=1.0)
