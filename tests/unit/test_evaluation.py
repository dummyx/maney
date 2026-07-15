from __future__ import annotations

from math import isclose
from typing import Any

import pytest

from nlp_trader.models.evaluation import evaluate_predictions, prediction_metrics


def _evaluation_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = [
        (
            "2026-07-01T20:00:00Z",
            "a",
            "A",
            1.0,
            0.8,
            0.02,
            1,
            "Tech",
            2_000_000.0,
            0.5,
            "news",
            "earnings",
        ),
        (
            "2026-07-01T20:00:00Z",
            "b",
            "B",
            -1.0,
            0.2,
            -0.01,
            0,
            "Finance",
            100_000_000.0,
            1.5,
            None,
            None,
        ),
        (
            "2026-07-02T20:00:00Z",
            "a",
            "A",
            0.5,
            0.8,
            0.01,
            1,
            "Tech",
            10_000_000.0,
            1.0,
            "social",
            "guidance",
        ),
        (
            "2026-07-02T20:00:00Z",
            "b",
            "B",
            -0.5,
            0.2,
            -0.02,
            0,
            "Finance",
            60_000_000.0,
            1.4,
            "news",
            None,
        ),
    ]
    predictions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    for (
        asof_ts,
        asset_id,
        symbol,
        score,
        probability,
        forward_return,
        binary_up,
        sector,
        dollar_volume,
        volatility_regime,
        source_type,
        event_type,
    ) in definitions:
        key = {
            "asset_id": asset_id,
            "symbol": symbol,
            "asof_ts": asof_ts,
            "horizon": "1d",
        }
        predictions.append(
            {
                **key,
                "score": score,
                "probability_up": probability,
                "model_family": "combined",
            }
        )
        labels.append({**key, "forward_return": forward_return, "binary_up": binary_up})
        feature = {
            **key,
            "sector": sector,
            "dollar_volume": dollar_volume,
            "volatility_regime": volatility_regime,
            "event_item_count_1d": int(event_type is not None),
            "source_identity_missing_1d": source_type is None,
        }
        if source_type is not None:
            feature[f"attention_source_{source_type}_count_1d"] = 1
        if event_type is not None:
            feature[f"event_{event_type}_count_1d"] = 1
        context.append(feature)
    return predictions, labels, context


def test_existing_predictions_get_date_safe_segmented_diagnostics() -> None:
    predictions, labels, context = _evaluation_rows()

    result = evaluate_predictions(
        {"combined": predictions},
        labels,
        context_rows=context,
        top_k=1,
        calibration_bin_count=5,
    )

    metrics = result["families"]["combined"]
    assert metrics["dates"] == 2
    assert metrics["ic_dates"] == 2
    assert metrics["mean_daily_spearman_ic"] == 1.0
    assert metrics["mean_daily_precision_at_k"] == 1.0
    assert "mean_squared_error" not in metrics
    assert isclose(metrics["classification"]["brier_score"], 0.04)
    assert metrics["classification"]["probability_source"] == "probability_up"
    assert metrics["classification"]["calibration_bins"]

    segments = result["segments"]["combined"]
    assert set(segments["sector"]) == {"Finance", "Tech"}
    assert set(segments["liquidity_bucket"]) == {"high", "low", "medium"}
    assert set(segments["volatility_regime"]) == {"high", "low", "normal"}
    assert set(segments["source_availability"]) == {"available", "missing"}
    assert set(segments["source_type"]) == {"news", "social"}
    assert set(segments["event_availability"]) == {"available", "missing"}
    assert set(segments["event_type"]) == {"earnings", "guidance"}
    assert result["segment_definitions"]["date_safety"].startswith("fixed thresholds")


def test_classification_metrics_are_omitted_without_binary_target() -> None:
    predictions, labels, _ = _evaluation_rows()
    regression_only = [
        {key: value for key, value in row.items() if key != "binary_up"} for row in labels
    ]

    metrics = prediction_metrics(predictions, regression_only)

    assert metrics["rows"] == 4
    assert "classification" not in metrics


def test_calibrated_metrics_require_explicit_prediction_fields() -> None:
    predictions, labels, _ = _evaluation_rows()
    rank_only = [
        {key: value for key, value in row.items() if key != "probability_up"} for row in predictions
    ]

    metrics = prediction_metrics(rank_only, labels)

    assert metrics["rows"] == 4
    assert "classification" not in metrics
    assert "mean_squared_error" not in metrics

    expected = []
    for prediction, label in zip(rank_only, labels, strict=True):
        expected.append(
            {
                **prediction,
                "expected_return": float(label["forward_return"]) + 0.01,
            }
        )
    calibrated = prediction_metrics(expected, labels)

    assert isclose(calibrated["mean_squared_error"], 0.0001)

    partial_expected = [dict(row) for row in expected]
    partial_expected[0]["expected_return"] = None
    with pytest.raises(ValueError, match="expected_return must be supplied for every"):
        prediction_metrics(partial_expected, labels)

    partial_probability = [dict(row) for row in predictions]
    partial_probability[0]["probability_up"] = None
    with pytest.raises(ValueError, match="probability_up must be supplied for every"):
        prediction_metrics(partial_probability, labels)

    partial_targets = [dict(row) for row in labels]
    partial_targets[0]["binary_up"] = None
    with pytest.raises(ValueError, match="binary_up must be supplied for every"):
        prediction_metrics(predictions, partial_targets)


def test_optional_metric_fields_must_cover_development_and_holdout() -> None:
    predictions, labels, context = _evaluation_rows()
    boundary_safe_labels = [dict(label) for label in labels]
    for label in boundary_safe_labels[:2]:
        label["label_available_at"] = "2026-07-02T19:59:59Z"
    frozen_training = {
        "name": "frozen_pre_holdout_snapshot_v1",
        "enabled": True,
        "final_holdout_periods": 1,
        "final_holdout_start": "2026-07-02T20:00:00Z",
        "frozen_snapshot_asof_ts": "2026-07-02T20:00:00Z",
        "training_cutoff_exclusive": "2026-07-02T20:00:00Z",
        "training_key_rule": "training key asof_ts < final_holdout_start",
        "update_rule": "no training updates at or after final_holdout_start",
    }
    development_only_probabilities = [dict(row) for row in predictions]
    for row in development_only_probabilities[2:]:
        row["probability_up"] = None

    with pytest.raises(ValueError, match="probability_up must be supplied for every"):
        evaluate_predictions(
            {"combined": development_only_probabilities},
            boundary_safe_labels,
            context_rows=context,
            final_holdout_periods=1,
            final_holdout_training=frozen_training,
        )


def test_prediction_metrics_requires_unique_matching_labels() -> None:
    predictions, labels, _ = _evaluation_rows()

    with pytest.raises(ValueError, match="predictions have no matching labels"):
        prediction_metrics(predictions, labels[1:])

    with pytest.raises(ValueError, match="duplicate label row"):
        prediction_metrics(predictions, [*labels, dict(labels[0])])


def test_precision_at_k_is_permutation_invariant_at_the_cutoff_tie() -> None:
    predictions = [
        {
            "asset_id": asset_id,
            "symbol": asset_id.upper(),
            "asof_ts": "2026-07-01T20:00:00Z",
            "horizon": "1d",
            "score": 1.0,
        }
        for asset_id in ("a", "b")
    ]
    labels = [
        {
            "asset_id": asset_id,
            "symbol": asset_id.upper(),
            "asof_ts": "2026-07-01T20:00:00Z",
            "horizon": "1d",
            "forward_return": outcome,
        }
        for asset_id, outcome in (("a", 0.01), ("b", -0.01))
    ]

    first = prediction_metrics(predictions, labels, top_k=1)
    reversed_rows = prediction_metrics(list(reversed(predictions)), labels, top_k=1)

    assert first["precision_at_k"] == 0.5
    assert first["mean_daily_precision_at_k"] == 0.5
    assert reversed_rows["precision_at_k"] == first["precision_at_k"]
    assert reversed_rows["mean_daily_precision_at_k"] == first["mean_daily_precision_at_k"]


def test_prediction_metrics_rejects_partial_cross_section_label_coverage() -> None:
    predictions, labels, _ = _evaluation_rows()
    partial = [dict(label) for label in labels]
    partial[2]["forward_return"] = None

    with pytest.raises(ValueError, match="partial forward-label coverage"):
        prediction_metrics(predictions, partial)


def test_prediction_metrics_omits_only_wholly_censored_trailing_cross_sections() -> None:
    predictions, labels, _ = _evaluation_rows()
    trailing = [dict(label) for label in labels]
    for label in trailing[2:]:
        label["forward_return"] = None
        label["binary_up"] = None
        label["expected_label_end_ts"] = "2026-07-03T20:00:00Z"

    metrics = prediction_metrics(predictions, trailing)

    assert metrics["rows"] == 2
    assert metrics["dates"] == 1


def test_evaluation_reserves_a_chronological_final_holdout() -> None:
    predictions, labels, context = _evaluation_rows()
    boundary_safe_labels = [dict(label) for label in labels]
    for label in boundary_safe_labels[:2]:
        label["label_available_at"] = "2026-07-02T19:59:59Z"
    frozen_training = {
        "name": "frozen_pre_holdout_snapshot_v1",
        "enabled": True,
        "final_holdout_periods": 1,
        "final_holdout_start": "2026-07-02T20:00:00Z",
        "frozen_snapshot_asof_ts": "2026-07-02T20:00:00Z",
        "training_cutoff_exclusive": "2026-07-02T20:00:00Z",
        "eligible_training_rows": 2,
        "training_key_count": 2,
        "training_key_digest": "fixture-digest",
        "label_eligibility_rule": "label availability timestamp < training_cutoff_exclusive",
        "training_key_rule": "training key asof_ts < final_holdout_start",
        "update_rule": "no training updates at or after final_holdout_start",
    }
    result = evaluate_predictions(
        {"combined": predictions},
        boundary_safe_labels,
        context_rows=context,
        top_k=1,
        final_holdout_periods=1,
        final_holdout_training=frozen_training,
    )

    assert result["families"]["combined"]["dates"] == 1
    assert result["final_holdout"]["families"]["combined"]["dates"] == 1
    assert result["evaluation_protocol"] == {
        "name": "chronological_frozen_final_holdout_v2",
        "selection_scope": "development_only",
        "development_periods": 1,
        "pre_holdout_periods": 1,
        "purged_development_periods": 0,
        "purged_development_times": [],
        "final_holdout_periods": 1,
        "final_holdout_start": "2026-07-02T20:00:00Z",
        "final_holdout_end": "2026-07-02T20:00:00Z",
        "purge_rule": "training labels become eligible only when their availability timestamp "
        "is strictly before the effective cutoff",
        "development_purge_rule": "remove the contiguous pre-holdout suffix beginning at the "
        "first cross-section not wholly available before final_holdout_start",
        "holdout_update_rule": "no training updates at or after final_holdout_start",
        "final_holdout_training": {**frozen_training, "verified_untouched": True},
    }


def test_final_holdout_refuses_unverified_training_metadata() -> None:
    predictions, labels, _ = _evaluation_rows()
    boundary_safe_labels = [dict(label) for label in labels]
    for label in boundary_safe_labels[:2]:
        label["label_available_at"] = "2026-07-02T19:59:59Z"

    with pytest.raises(ValueError, match="requires frozen training protocol metadata"):
        evaluate_predictions(
            {"combined": predictions},
            boundary_safe_labels,
            final_holdout_periods=1,
        )


def test_evaluation_purges_labels_that_overlap_the_holdout_boundary() -> None:
    predictions, labels, _ = _evaluation_rows()

    with pytest.raises(ValueError, match="after purging overlapping labels"):
        evaluate_predictions(
            {"combined": predictions},
            labels,
            final_holdout_periods=1,
        )


def test_non_monotonic_label_availability_purges_a_contiguous_suffix() -> None:
    decision_times = [f"2026-07-{day:02d}T20:00:00Z" for day in range(1, 6)]
    predictions = [
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": asof_ts,
            "horizon": "2d",
            "model_family": "combined",
            "score": 1.0,
        }
        for asof_ts in decision_times
    ]
    availability = [
        "2026-07-02T20:01:00Z",
        "2026-07-03T20:01:00Z",
        "2026-07-06T20:00:00Z",
        "2026-07-04T20:01:00Z",
        "2026-07-07T20:00:00Z",
    ]
    labels = [
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": asof_ts,
            "horizon": "2d",
            "forward_return": 0.01,
            "label_available_at": available_at,
        }
        for asof_ts, available_at in zip(decision_times, availability, strict=True)
    ]
    frozen_training = {
        "name": "frozen_pre_holdout_snapshot_v1",
        "enabled": True,
        "final_holdout_periods": 1,
        "final_holdout_start": decision_times[-1],
        "frozen_snapshot_asof_ts": decision_times[-1],
        "training_cutoff_exclusive": decision_times[-1],
        "training_key_rule": "training key asof_ts < final_holdout_start",
        "update_rule": "no training updates at or after final_holdout_start",
    }

    result = evaluate_predictions(
        {"combined": predictions},
        labels,
        final_holdout_periods=1,
        final_holdout_training=frozen_training,
    )

    protocol = result["evaluation_protocol"]
    assert protocol["development_periods"] == 2
    assert protocol["purged_development_times"] == decision_times[2:4]


def test_final_holdout_must_leave_a_development_period() -> None:
    predictions, labels, _ = _evaluation_rows()

    with pytest.raises(ValueError, match="leave at least one"):
        evaluate_predictions(
            {"combined": predictions},
            labels,
            final_holdout_periods=2,
        )
