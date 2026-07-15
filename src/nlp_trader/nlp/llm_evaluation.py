from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from nlp_trader.nlp.llm_annotations import (
    AnnotationResponse,
    EntityAnnotation,
    EventType,
)

type GoldStanceLabel = Literal["positive", "negative", "neutral"]


@dataclass(frozen=True, slots=True)
class GoldEntityAnnotation:
    """Frozen human label for evaluating the extraction task independently of returns."""

    item_id: str
    asset_id: str
    stance_label: GoldStanceLabel
    primary_event_type: EventType | None
    supporting_evidence_span_ids: frozenset[str]
    counterevidence_span_ids: frozenset[str] = frozenset()
    horizon_days: int = 1

    def __post_init__(self) -> None:
        if not self.item_id.strip() or not self.asset_id.strip():
            raise ValueError("gold item_id and asset_id must not be empty")
        if self.stance_label not in {"positive", "negative", "neutral"}:
            raise ValueError("gold stance_label must be positive, negative, or neutral")
        evidence_ids = self.supporting_evidence_span_ids | self.counterevidence_span_ids
        if any(not value.strip() for value in evidence_ids):
            raise ValueError("gold evidence span IDs must not be empty")
        if self.supporting_evidence_span_ids & self.counterevidence_span_ids:
            raise ValueError("gold supporting and counterevidence must be disjoint")
        if not 1 <= self.horizon_days <= 252:
            raise ValueError("gold horizon_days must be between 1 and 252")


def _macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted(set(expected))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            expected_value == label and predicted_value == label
            for expected_value, predicted_value in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            expected_value != label and predicted_value == label
            for expected_value, predicted_value in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            expected_value == label and predicted_value != label
            for expected_value, predicted_value in zip(expected, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def evaluate_annotation_set(
    gold: list[GoldEntityAnnotation],
    predictions: list[AnnotationResponse],
    *,
    invalid_item_ids: frozenset[str] = frozenset(),
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate a frozen labeled set without prices, returns, or portfolio outcomes."""

    if calibration_bins < 1:
        raise ValueError("calibration_bins must be positive")
    gold_by_key: dict[tuple[str, str], GoldEntityAnnotation] = {}
    for row in gold:
        key = (row.item_id, row.asset_id)
        if key in gold_by_key:
            raise ValueError(f"duplicate gold annotation key: {key}")
        gold_by_key[key] = row
    gold_item_ids = {row.item_id for row in gold}
    if not invalid_item_ids <= gold_item_ids:
        raise ValueError("invalid_item_ids must refer to items in the gold set")

    predicted_by_key: dict[tuple[str, str], EntityAnnotation] = {}
    predicted_item_ids: set[str] = set()
    for response in predictions:
        if response.item_id not in gold_item_ids:
            raise ValueError(f"prediction item_id is not in the gold set: {response.item_id}")
        if response.item_id in predicted_item_ids:
            raise ValueError(f"duplicate prediction item_id: {response.item_id}")
        predicted_item_ids.add(response.item_id)
        if response.item_id in invalid_item_ids:
            raise ValueError("an invalid item cannot also have a validated prediction")
        for annotation in response.annotations:
            key = (response.item_id, annotation.asset_id)
            if key in predicted_by_key:
                raise ValueError(f"duplicate prediction annotation key: {key}")
            predicted_by_key[key] = annotation

    expected_prediction_keys = {key for key in gold_by_key if key[0] not in invalid_item_ids}
    if set(predicted_by_key) != expected_prediction_keys:
        missing = sorted(expected_prediction_keys - set(predicted_by_key))
        extra = sorted(set(predicted_by_key) - expected_prediction_keys)
        raise ValueError(f"prediction coverage mismatch; missing={missing}, extra={extra}")

    expected_stances: list[str] = []
    predicted_stances: list[str] = []
    expected_events: list[str] = []
    predicted_events: list[str] = []
    correctness: list[float] = []
    confidences: list[float] = []
    cited = 0
    correct_citations = 0
    counterevidence_cited = 0
    correct_counterevidence = 0
    correct_horizons = 0
    abstentions = 0
    for key, expected in sorted(gold_by_key.items()):
        predicted = predicted_by_key.get(key)
        expected_stances.append(expected.stance_label)
        expected_events.append(expected.primary_event_type or "none")
        if predicted is None:
            predicted_stances.append("invalid")
            predicted_events.append("invalid")
            correctness.append(0.0)
            confidences.append(0.0)
            continue
        predicted_stances.append(predicted.stance_label)
        predicted_events.append(
            "abstain"
            if predicted.stance_label == "abstain"
            else predicted.primary_event_type or "none"
        )
        is_correct = predicted.stance_label == expected.stance_label
        correctness.append(float(is_correct))
        confidences.append(predicted.raw_confidence)
        abstentions += predicted.stance_label == "abstain"
        cited += len(predicted.supporting_evidence_span_ids)
        correct_citations += len(
            set(predicted.supporting_evidence_span_ids) & expected.supporting_evidence_span_ids
        )
        counterevidence_cited += len(predicted.counterevidence_span_ids)
        correct_counterevidence += len(
            set(predicted.counterevidence_span_ids) & expected.counterevidence_span_ids
        )
        correct_horizons += predicted.horizon_days == expected.horizon_days

    example_count = len(gold)
    brier = (
        sum(
            (confidence - correct) ** 2
            for confidence, correct in zip(confidences, correctness, strict=True)
        )
        / example_count
        if example_count
        else 0.0
    )
    calibration_rows: list[dict[str, float | int]] = []
    calibration_error = 0.0
    for index in range(calibration_bins):
        lower = index / calibration_bins
        upper = (index + 1) / calibration_bins
        indices = [
            row_index
            for row_index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == calibration_bins - 1 and confidence == 1.0)
        ]
        if not indices:
            continue
        mean_confidence = sum(confidences[row_index] for row_index in indices) / len(indices)
        accuracy = sum(correctness[row_index] for row_index in indices) / len(indices)
        calibration_error += abs(accuracy - mean_confidence) * len(indices) / example_count
        calibration_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(indices),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    item_count = len(gold_item_ids)
    return {
        "artifact_schema_version": "llm-semantic-signal-evaluation-v2",
        "item_count": item_count,
        "entity_count": example_count,
        "stance_macro_f1": _macro_f1(expected_stances, predicted_stances),
        "event_macro_f1": _macro_f1(expected_events, predicted_events),
        "evidence_precision": correct_citations / cited if cited else 0.0,
        "supporting_evidence_precision": correct_citations / cited if cited else 0.0,
        "counterevidence_precision": (
            correct_counterevidence / counterevidence_cited if counterevidence_cited else 0.0
        ),
        "horizon_accuracy": correct_horizons / example_count if example_count else 0.0,
        "abstention_count": abstentions,
        "abstention_rate": abstentions / example_count if example_count else 0.0,
        "invalid_item_count": len(invalid_item_ids),
        "invalid_response_rate": len(invalid_item_ids) / item_count if item_count else 0.0,
        "stance_confidence_brier": brier,
        "stance_expected_calibration_error": calibration_error,
        "raw_confidence_brier": brier,
        "raw_confidence_expected_calibration_error": calibration_error,
        "calibration_bins": calibration_rows,
    }
