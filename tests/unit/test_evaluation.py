from __future__ import annotations

from math import isclose
from typing import Any

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
