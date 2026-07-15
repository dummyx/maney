from __future__ import annotations

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
    apply_to_features: bool = False,
) -> ResearchConfig:
    payload = generated_config.model_dump(mode="python")
    payload["paths"]["llm_model"] = model_path
    payload["llm_annotations"] = {
        "enabled": True,
        "apply_to_features": apply_to_features,
        "model_id": "local-test-causal-lm",
        "model_revision": "immutable-test-revision",
        "model_license_or_terms_ref": "redistributable-test-fixture",
    }
    return ResearchConfig.model_validate(payload)


def test_enabled_llm_model_directory_is_validated_and_hashed(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    weights = model_path / "weights.bin"
    weights.write_bytes(b"first immutable model bytes")
    config = _enabled_llm_config(generated_config, model_path)

    assert validate_config(config) == []
    first = next(entry for entry in input_manifest(config) if entry["role"] == "llm_model")
    assert first["input_kind"] == "local_model_directory"
    assert first["model_id"] == "local-test-causal-lm"
    assert first["model_revision"] == "immutable-test-revision"
    assert first["license_or_terms_ref"] == "redistributable-test-fixture"
    assert first["file_count"] == 2
    assert len(str(first["sha256"])) == 64

    weights.write_bytes(b"different immutable model bytes")
    second = next(entry for entry in input_manifest(config) if entry["role"] == "llm_model")
    assert second["sha256"] != first["sha256"]


def test_llm_application_requires_enabled_local_identity_and_safe_loading() -> None:
    with pytest.raises(ValidationError, match="enabled must be true"):
        LLMAnnotationsConfig(apply_to_features=True)

    with pytest.raises(ValidationError, match="model_revision"):
        LLMAnnotationsConfig(
            enabled=True,
            model_id="local-model",
            model_license_or_terms_ref="local-license",
        )

    with pytest.raises(ValidationError, match="local_files_only"):
        LLMAnnotationsConfig.model_validate({"local_files_only": False})

    with pytest.raises(ValidationError, match="trust_remote_code"):
        LLMAnnotationsConfig.model_validate({"trust_remote_code": True})


def test_applied_llm_and_transformer_sentiment_cannot_compete(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    configured = _enabled_llm_config(
        generated_config,
        model_path,
        apply_to_features=True,
    )
    payload = configured.model_dump(mode="python")
    payload["transformer"] = TransformerConfig(
        enabled=True,
        model_name="local-transformer",
    )

    with pytest.raises(ValidationError, match="cannot both be enabled"):
        ResearchConfig.model_validate(payload)
