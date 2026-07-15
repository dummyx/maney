from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pytest

from nlp_trader.config import (
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_MODEL_LICENSE_OR_TERMS_REF,
    DEFAULT_LLM_MODEL_REVISION,
    DEFAULT_LLM_MODEL_SHA256,
)
from nlp_trader.nlp.llm_annotations import (
    AssetCandidate,
    CachedLocalLLMAnnotator,
    LLMAnnotationConfig,
    build_annotation_request,
)
from nlp_trader.schemas import TextItem

_EXPECTED_MODEL_SHA256 = "4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095"
_EXPECTED_LLAMA_CPP_PYTHON_VERSION = "0.3.34"
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models/llm/Qwen3.6-27B-UD-Q4_K_XL.gguf"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
pytestmark = pytest.mark.acceptance


def test_real_qwen_gguf_annotation_and_cache_replay(tmp_path: Path) -> None:
    if os.environ.get("NLP_TRADER_RUN_REAL_LLM") != "1":
        pytest.skip("set NLP_TRADER_RUN_REAL_LLM=1 to run local GGUF acceptance")

    model_path = Path(os.environ.get("NLP_TRADER_LLM_MODEL_PATH", _DEFAULT_MODEL_PATH))
    assert model_path.is_file(), (
        "real LLM acceptance requires the pre-provisioned GGUF at "
        f"{model_path}; this test never downloads a model"
    )
    assert DEFAULT_LLM_MODEL_SHA256 == _EXPECTED_MODEL_SHA256

    config = LLMAnnotationConfig(
        model_path=model_path,
        model_id=DEFAULT_LLM_MODEL_ID,
        model_revision=DEFAULT_LLM_MODEL_REVISION,
        model_license_or_terms_ref=DEFAULT_LLM_MODEL_LICENSE_OR_TERMS_REF,
        prompt_version="semantic-evidence-v2",
        schema_version="semantic-signal-v2",
        verifier_version="semantic-evidence-verifier-v1",
        cache_dir=tmp_path / "cache",
    )
    available_at = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    item = TextItem(
        item_id="real-qwen-acceptance-item",
        source="local-acceptance-fixture",
        source_type="news",
        language="en",
        title="Issuer A raises full-year revenue guidance",
        body="Issuer A raised its full-year revenue guidance after stronger demand.",
        published_at=available_at,
        vendor_received_at=available_at,
        ingested_at=available_at,
        available_at=available_at,
        license_or_terms_ref="synthetic-acceptance-fixture",
    )
    request = build_annotation_request(
        item,
        (AssetCandidate(asset_id="asset_a", symbol="AAA", name="Issuer A"),),
    )

    annotator = CachedLocalLLMAnnotator(config)
    response = annotator.annotate([request])

    assert len(response) == 1
    assert response[0].item_id == item.item_id
    assert len(response[0].annotations) == 1
    annotation = response[0].annotations[0]
    assert annotation.asset_id == "asset_a"
    assert annotation.stance_label == "positive"
    assert annotation.semantic_signal > 0
    assert annotation.supporting_evidence_span_ids
    assert annotator.verification_for(request, response[0]).valid is True
    assert annotator.generated_input_token_count is not None
    assert annotator.generated_input_token_count > 0
    assert annotator.generated_output_token_count is not None
    assert annotator.generated_output_token_count > 0

    provenance = annotator.provenance_payload
    assert provenance["backend"] == "llama_cpp_gguf"
    assert provenance["generator_mode"] == "llama_cpp"
    assert provenance["model_file_sha256"] == _EXPECTED_MODEL_SHA256
    assert provenance["llama_cpp_python_version"] == version("llama-cpp-python")
    assert provenance["llama_cpp_python_version"] == _EXPECTED_LLAMA_CPP_PYTHON_VERSION
    assert provenance["device"] in {"metal", "cpu"}
    assert provenance["device"] != "not_loaded"
    assert provenance["effective_gpu_layers"] in {-1, 0}
    assert provenance["chat_template_source"] == "embedded_gguf_metadata"
    assert isinstance(provenance["chat_template_sha256"], str)
    assert _SHA256.fullmatch(provenance["chat_template_sha256"])
    assert provenance["mtp_speculative_decoding"] is False

    cache_record = annotator.cache_record_for(request)
    assert cache_record["generation"]["input_token_count"] > 0
    assert cache_record["generation"]["output_token_count"] > 0

    replay = CachedLocalLLMAnnotator(config)
    replayed_response = replay.annotate([request])

    assert replayed_response == response
    assert replay.cache_hit_count == 1
    assert replay.generation_request_count == 0
    assert replay.generated_input_token_count == 0
    assert replay.generated_output_token_count == 0
    assert replay.inference_source_for(request) == "cache"
    assert replay.provenance_payload["device"] == provenance["device"]
    assert replay.provenance_payload["chat_template_sha256"] == provenance["chat_template_sha256"]
