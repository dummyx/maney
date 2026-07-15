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
        LLMAnnotationsConfig(feature_mode="augment")

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


def test_llm_augmentation_requires_ablation_families_and_can_coexist_with_transformer(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
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
