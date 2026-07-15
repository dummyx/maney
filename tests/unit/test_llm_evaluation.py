from __future__ import annotations

import pytest

from nlp_trader.nlp.llm_annotations import AnnotationResponse, EntityAnnotation
from nlp_trader.nlp.llm_evaluation import (
    GoldEntityAnnotation,
    evaluate_annotation_set,
)


def test_annotation_quality_metrics_are_independent_of_market_outcomes() -> None:
    gold = [
        GoldEntityAnnotation(
            item_id="item-1",
            asset_id="asset-a",
            stance_label="positive",
            primary_event_type="guidance",
            supporting_evidence_span_ids=frozenset({"S2"}),
        ),
        GoldEntityAnnotation(
            item_id="item-1",
            asset_id="asset-b",
            stance_label="negative",
            primary_event_type=None,
            supporting_evidence_span_ids=frozenset({"S3"}),
        ),
    ]
    predictions = [
        AnnotationResponse(
            item_id="item-1",
            annotations=(
                EntityAnnotation(
                    asset_id="asset-a",
                    stance_label="positive",
                    semantic_signal=2,
                    raw_confidence=0.9,
                    uncertainty=0.1,
                    horizon_days=1,
                    primary_event_type="guidance",
                    event_confidence=0.8,
                    supporting_evidence_span_ids=("S2", "S4"),
                    counterevidence_span_ids=(),
                    mechanism="Raised guidance supports a stronger outlook.",
                    invalidation_conditions=("The guidance is withdrawn.",),
                    abstain_reason=None,
                ),
                EntityAnnotation(
                    asset_id="asset-b",
                    stance_label="abstain",
                    semantic_signal=0,
                    raw_confidence=0.0,
                    uncertainty=1.0,
                    horizon_days=1,
                    primary_event_type=None,
                    event_confidence=0.0,
                    supporting_evidence_span_ids=(),
                    counterevidence_span_ids=(),
                    mechanism=None,
                    invalidation_conditions=(),
                    abstain_reason="insufficient evidence",
                ),
            ),
        )
    ]

    metrics = evaluate_annotation_set(gold, predictions, calibration_bins=2)

    assert metrics["stance_macro_f1"] == 0.5
    assert metrics["event_macro_f1"] == 0.5
    assert metrics["evidence_precision"] == 0.5
    assert metrics["supporting_evidence_precision"] == 0.5
    assert metrics["counterevidence_precision"] == 0.0
    assert metrics["horizon_accuracy"] == 1.0
    assert metrics["abstention_rate"] == 0.5
    assert metrics["invalid_response_rate"] == 0.0
    assert metrics["stance_confidence_brier"] == pytest.approx(0.005)
    assert metrics["stance_expected_calibration_error"] == pytest.approx(0.05)
    assert metrics["raw_confidence_brier"] == pytest.approx(0.005)


def test_invalid_items_are_explicit_and_partial_valid_coverage_fails() -> None:
    gold = [
        GoldEntityAnnotation(
            item_id="invalid-item",
            asset_id="asset-a",
            stance_label="neutral",
            primary_event_type=None,
            supporting_evidence_span_ids=frozenset({"S1"}),
        )
    ]

    metrics = evaluate_annotation_set(
        gold,
        [],
        invalid_item_ids=frozenset({"invalid-item"}),
    )
    assert metrics["invalid_response_rate"] == 1.0
    assert metrics["stance_macro_f1"] == 0.0

    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_annotation_set(gold, [])
