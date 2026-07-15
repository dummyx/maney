from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from nlp_trader.config import ResearchConfig
from nlp_trader.data.parquet import read_partitioned_parquet
from nlp_trader.nlp.llm_annotations import GenerationRequest, GenerationResponse
from nlp_trader.nlp.llm_decision_rounds import replay_decision_rounds_jsonl
from nlp_trader.pipeline import backtest, build_features
from nlp_trader.timestamps import parse_utc


def _enabled_llm_config(
    config: ResearchConfig,
    tmp_path: Path,
    *,
    feature_mode: str,
) -> ResearchConfig:
    model_path = tmp_path / "local-llm.gguf"
    model_path.write_bytes(b"tiny synthetic GGUF fixture\n")
    paths = config.paths.model_copy(update={"llm_model": model_path})
    annotations = config.llm_annotations.model_copy(
        update={
            "enabled": True,
            "feature_mode": feature_mode,
            "model_id": "local-test-llm",
            "model_revision": "test-revision-1",
            "model_license_or_terms_ref": "synthetic-test-only",
            "batch_size": 2,
            "max_input_tokens": 512,
            "max_new_tokens": 256,
        }
    )
    models = config.models
    if feature_mode == "augment":
        models = models.model_copy(
            update={
                "families": (
                    "traditional",
                    "text",
                    "combined",
                    "llm",
                    "traditional_llm",
                    "all",
                )
            }
        )
    return config.model_copy(
        update={"paths": paths, "llm_annotations": annotations, "models": models}
    )


def _input_payload(request: GenerationRequest) -> dict[str, Any]:
    _, separator, tail = request.prompt.partition("REQUEST_JSON:")
    assert separator
    payload, _ = json.JSONDecoder().raw_decode(tail.lstrip())
    assert isinstance(payload, dict)
    return payload


def _response(
    request: GenerationRequest,
    annotations: list[dict[str, object]],
) -> GenerationResponse:
    return GenerationResponse(
        request_id=request.request_id,
        generated_text=json.dumps({"annotations": annotations}, sort_keys=True),
    )


def _positive_annotation(asset_id: str) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "stance_label": "positive",
        "semantic_signal": 2,
        "raw_confidence": 0.8,
        "uncertainty": 0.1,
        "horizon_days": 1,
        "primary_event_type": None,
        "event_confidence": 0.0,
        "supporting_evidence_span_ids": ["S1"],
        "counterevidence_span_ids": [],
        "mechanism": "The operating update supports a positive semantic assessment.",
        "invalidation_conditions": ["The operating update is withdrawn."],
        "abstain_reason": None,
    }


def _partition_rows(outputs: dict[str, object], key: str) -> list[dict[str, object]]:
    paths = outputs[key]
    assert isinstance(paths, list) and paths
    first = Path(str(paths[0]))
    return read_partitioned_parquet(first.parents[2])


def _sorted_signals(outputs: dict[str, object]) -> list[dict[str, object]]:
    return sorted(
        _partition_rows(outputs, "signals"),
        key=lambda row: (str(row["item_id"]), str(row["asset_id"])),
    )


def test_disabled_sample_path_has_no_llm_artifacts_or_optional_imports(
    generated_config: ResearchConfig,
) -> None:
    optional_before = {
        module
        for module in sys.modules
        if module.split(".", 1)[0] in {"torch", "transformers", "llama_cpp"}
    }

    outputs = build_features(generated_config)

    optional_after = {
        module
        for module in sys.modules
        if module.split(".", 1)[0] in {"torch", "transformers", "llama_cpp"}
    }
    assert optional_after == optional_before
    assert not any(key.startswith("llm_") for key in outputs)
    assert not (generated_config.paths.models_dir / "_cache" / "llm_annotations").exists()
    run_id = str(outputs["run_id"])
    assert not (
        generated_config.paths.processed_dir / run_id / "silver" / "llm_annotations"
    ).exists()


def test_sidecar_annotations_are_audited_without_changing_signals_and_replay_from_cache(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    baseline_outputs = build_features(generated_config)
    configured = _enabled_llm_config(
        generated_config,
        tmp_path,
        feature_mode="sidecar",
    )
    calls: list[list[str]] = []

    def fake_generator(requests: list[GenerationRequest]) -> list[GenerationResponse]:
        calls.append([request.request_id for request in requests])
        responses: list[GenerationResponse] = []
        for request in requests:
            payload = _input_payload(request)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            responses.append(
                _response(
                    request,
                    [_positive_annotation(str(candidate["asset_id"])) for candidate in candidates],
                )
            )
        return responses

    sidecar_outputs = build_features(configured, llm_generator=fake_generator)

    assert calls
    assert _sorted_signals(sidecar_outputs) == _sorted_signals(baseline_outputs)
    for key in (
        "llm_annotation_prompt",
        "llm_annotation_schema",
        "llm_annotation_provenance",
        "llm_annotation_responses",
        "llm_annotation_generation_attempts",
        "llm_annotations",
        "llm_annotation_summary",
        "llm_verification_summary",
        "llm_decision_rounds",
    ):
        assert key in sidecar_outputs
    provenance = json.loads(
        Path(sidecar_outputs["llm_annotation_provenance"]).read_text(encoding="utf-8")
    )
    summary = json.loads(
        Path(sidecar_outputs["llm_annotation_summary"]).read_text(encoding="utf-8")
    )
    annotation_rows = _partition_rows(sidecar_outputs, "llm_annotations")
    assert provenance["retrospective_parser"] is True
    assert provenance["feature_mode"] == "sidecar"
    assert provenance["backend"] == "llama_cpp_gguf"
    assert len(provenance["model_file_sha256"]) == 64
    assert summary["request_count"] == 3
    assert summary["annotation_count"] == 3
    assert summary["feature_mode"] == "sidecar"
    assert summary["generated_input_token_count"] is None
    assert summary["generated_output_token_count"] is None
    assert summary["estimated_inference_cost_usd"] is None
    assert len(annotation_rows) == 3
    rounds = replay_decision_rounds_jsonl(Path(str(sidecar_outputs["llm_decision_rounds"])))
    assert len(rounds) == 3
    assert {round.inference_source for round in rounds} == {"generated"}
    assert all(round.source_available_at <= round.decision_time for round in rounds)
    assert all(round.verifier.passed for round in rounds)
    cache_files = sorted(
        (configured.paths.models_dir / "_cache" / "llm_annotations").glob("*.json")
    )
    assert len(cache_files) == 3

    def fail_if_called(_requests: list[GenerationRequest]) -> list[GenerationResponse]:
        raise AssertionError("a complete cache replay must not call the generator")

    replay_outputs = build_features(configured, llm_generator=fail_if_called)

    assert _sorted_signals(replay_outputs) == _sorted_signals(baseline_outputs)
    assert len(replay_outputs["llm_annotation_responses"]) == 3
    replay_rounds = replay_decision_rounds_jsonl(Path(str(replay_outputs["llm_decision_rounds"])))
    assert {round.inference_source for round in replay_rounds} == {"cache"}


def test_augment_mode_keeps_conventional_text_and_adds_per_entity_llm_features(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "multi-entity.jsonl"
    text_path.write_text(
        json.dumps(
            {
                "item_id": "multi-after-hours",
                "source": "synthetic_news",
                "source_type": "news",
                "language": "en",
                "title": "Synthetic AAA, BBB, and CCC operating update",
                "body": (
                    "Synthetic AAA beat targets. Synthetic BBB cut guidance. "
                    "Synthetic CCC reported stable demand."
                ),
                "published_at": "2026-07-06T21:55:00Z",
                "vendor_received_at": "2026-07-06T22:00:00Z",
                "ingested_at": "2026-07-06T22:01:00Z",
                "available_at": "2026-07-06T22:00:00Z",
                "license_or_terms_ref": "synthetic-fixture-v1",
                "relationship_type": "original",
                "content_status": "active",
                "retention_permitted": True,
                "entities": [
                    {
                        "asset_id": f"asset_{symbol.casefold()}",
                        "symbol": symbol,
                        "name": f"Synthetic {symbol} Corporation",
                        "relevance": 1.0,
                        "mention_type": "primary",
                        "confidence": 0.99,
                    }
                    for symbol in ("AAA", "BBB", "CCC")
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    custom_paths = generated_config.paths.model_copy(update={"text_items": text_path})
    baseline_config = generated_config.model_copy(update={"paths": custom_paths})
    baseline_outputs = build_features(baseline_config)
    configured = _enabled_llm_config(
        baseline_config,
        tmp_path,
        feature_mode="augment",
    )

    def fake_generator(requests: list[GenerationRequest]) -> list[GenerationResponse]:
        assert len(requests) == 1
        request = requests[0]
        payload = _input_payload(request)
        assert [candidate["asset_id"] for candidate in payload["candidates"]] == [
            "asset_aaa",
            "asset_bbb",
            "asset_ccc",
        ]
        return [
            _response(
                request,
                [
                    {
                        "asset_id": "asset_aaa",
                        "stance_label": "positive",
                        "semantic_signal": 2,
                        "raw_confidence": 0.91,
                        "uncertainty": 0.05,
                        "horizon_days": 1,
                        "primary_event_type": "guidance",
                        "event_confidence": 0.8,
                        "supporting_evidence_span_ids": ["S2"],
                        "counterevidence_span_ids": [],
                        "mechanism": "Beating targets supports a positive near-term assessment.",
                        "invalidation_conditions": ["The reported target beat is withdrawn."],
                        "abstain_reason": None,
                    },
                    {
                        "asset_id": "asset_bbb",
                        "stance_label": "negative",
                        "semantic_signal": -2,
                        "raw_confidence": 0.82,
                        "uncertainty": 0.1,
                        "horizon_days": 1,
                        "primary_event_type": "guidance",
                        "event_confidence": 0.9,
                        "supporting_evidence_span_ids": ["S3"],
                        "counterevidence_span_ids": [],
                        "mechanism": "Cut guidance supports a negative near-term assessment.",
                        "invalidation_conditions": ["The guidance cut is reversed."],
                        "abstain_reason": None,
                    },
                    {
                        "asset_id": "asset_ccc",
                        "stance_label": "abstain",
                        "semantic_signal": 0,
                        "raw_confidence": 0.0,
                        "uncertainty": 1.0,
                        "horizon_days": 1,
                        "primary_event_type": None,
                        "event_confidence": 0.0,
                        "supporting_evidence_span_ids": [],
                        "counterevidence_span_ids": [],
                        "mechanism": None,
                        "invalidation_conditions": [],
                        "abstain_reason": "insufficient entity-specific evidence",
                    },
                ],
            )
        ]

    applied_outputs = build_features(configured, llm_generator=fake_generator)
    baseline = {row["asset_id"]: row for row in _sorted_signals(baseline_outputs)}
    applied = {row["asset_id"]: row for row in _sorted_signals(applied_outputs)}

    conventional_fields = (
        "sentiment_score",
        "sentiment_label",
        "sentiment_confidence",
        "event_type",
        "model_version",
    )
    for asset_id in ("asset_aaa", "asset_bbb", "asset_ccc"):
        for field in conventional_fields:
            assert applied[asset_id][field] == baseline[asset_id][field]

    assert applied["asset_aaa"]["llm_semantic_signal"] == 2
    assert applied["asset_aaa"]["llm_raw_confidence"] == 0.91
    assert applied["asset_aaa"]["llm_uncertainty"] == 0.05
    assert applied["asset_aaa"]["llm_abstained"] is False
    assert applied["asset_bbb"]["llm_semantic_signal"] == -2
    assert applied["asset_bbb"]["llm_raw_confidence"] == 0.82
    assert applied["asset_ccc"]["llm_semantic_signal"] == 0
    assert applied["asset_ccc"]["llm_raw_confidence"] == 0.0
    assert applied["asset_ccc"]["llm_uncertainty"] == 1.0
    assert applied["asset_ccc"]["llm_abstained"] is True

    feature_rows = read_partitioned_parquet(Path(str(applied_outputs["features"])))
    assert feature_rows
    assert all("llm_semantic_mean_1d" in row for row in feature_rows)
    assert all("llm_raw_confidence_mean_1d" in row for row in feature_rows)
    assert all("llm_uncertainty_mean_1d" in row for row in feature_rows)

    expected_available_at = parse_utc("2026-07-06T22:00:00Z")
    expected_decision = parse_utc("2026-07-07T20:00:00Z")
    assert {
        parse_utc(str(row["available_at"]))
        for row in _partition_rows(applied_outputs, "llm_annotations")
    } == {expected_available_at}
    assert {
        parse_utc(str(row["decision_time"]))
        for row in _partition_rows(applied_outputs, "llm_annotations")
    } == {expected_decision}
    for signal in applied.values():
        available_at = parse_utc(str(signal["available_at"]))
        asof_ts = parse_utc(str(signal["asof_ts"]))
        assert available_at == expected_available_at
        assert asof_ts == expected_decision
        assert available_at <= asof_ts


def test_identical_text_items_share_one_generation_and_one_run_response_artifact(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "copied-items.jsonl"
    records = []
    for item_id in ("copy-a", "copy-b"):
        records.append(
            {
                "item_id": item_id,
                "source": "synthetic_news",
                "source_type": "news",
                "language": "en",
                "title": "Synthetic AAA operating update",
                "body": "Synthetic AAA reported improved demand.",
                "published_at": "2026-07-06T14:00:00Z",
                "vendor_received_at": "2026-07-06T14:01:00Z",
                "ingested_at": "2026-07-06T14:02:00Z",
                "available_at": "2026-07-06T14:01:00Z",
                "license_or_terms_ref": "synthetic-fixture-v1",
                "entities": [
                    {
                        "asset_id": "asset_aaa",
                        "symbol": "AAA",
                        "name": "Synthetic AAA Corporation",
                        "relevance": 1.0,
                        "mention_type": "primary",
                        "confidence": 0.99,
                    }
                ],
            }
        )
    text_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    paths = generated_config.paths.model_copy(update={"text_items": text_path})
    configured = _enabled_llm_config(
        generated_config.model_copy(update={"paths": paths}),
        tmp_path,
        feature_mode="sidecar",
    )
    generated_request_counts: list[int] = []

    def fake_generator(requests: list[GenerationRequest]) -> list[GenerationResponse]:
        generated_request_counts.append(len(requests))
        return [_response(request, [_positive_annotation("asset_aaa")]) for request in requests]

    outputs = build_features(configured, llm_generator=fake_generator)

    assert generated_request_counts == [1]
    assert len(outputs["llm_annotation_responses"]) == 1
    assert len(_partition_rows(outputs, "llm_annotations")) == 2
    rounds = replay_decision_rounds_jsonl(Path(str(outputs["llm_decision_rounds"])))
    assert [round.inference_source for round in rounds] == ["generated", "deduplicated"]


def test_augment_backtest_emits_matched_llm_ablation_without_promotion_claim(
    generated_config: ResearchConfig,
    tmp_path: Path,
) -> None:
    configured = _enabled_llm_config(
        generated_config,
        tmp_path,
        feature_mode="augment",
    )

    def fake_generator(requests: list[GenerationRequest]) -> list[GenerationResponse]:
        responses: list[GenerationResponse] = []
        for request in requests:
            payload = _input_payload(request)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            responses.append(
                _response(
                    request,
                    [_positive_annotation(str(candidate["asset_id"])) for candidate in candidates],
                )
            )
        return responses

    outputs = backtest(configured, llm_generator=fake_generator)
    comparison = json.loads(
        Path(str(outputs["llm_ablation_comparison"])).read_text(encoding="utf-8")
    )

    assert "not evidence" in comparison["interpretation"]
    assert set(comparison["family_semantics"]) == {
        "traditional",
        "text",
        "combined",
        "llm",
        "traditional_llm",
        "all",
    }
    assert set(comparison["development"]) == {
        "llm_only_vs_conventional_text",
        "numeric_plus_llm_vs_numeric",
        "all_vs_numeric_plus_conventional_text",
    }
    assert set(comparison["final_holdout"]) == set(comparison["development"])
    for window in ("development", "final_holdout"):
        for values in comparison[window].values():
            assert "enhanced_minus_baseline" in values
