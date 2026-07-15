from __future__ import annotations

from typing import Any

from nlp_trader.features.build import finite_float
from nlp_trader.models.baselines import predict_all_families, predict_with_model, train_baselines
from nlp_trader.models.evaluation import evaluate_families


def test_boolean_missingness_indicators_are_numeric_model_inputs() -> None:
    assert finite_float(True) == 1.0
    assert finite_float(False) == 0.0


def _rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for day in range(1, 6):
        asof = f"2026-07-{day:02d}T20:00:00Z"
        for symbol, direction in (("AAA", 1.0), ("BBB", -1.0)):
            asset_id = f"asset_{symbol.lower()}"
            features.append(
                {
                    "asset_id": asset_id,
                    "symbol": symbol,
                    "asof_ts": asof,
                    "horizon": "1d",
                    "return_1d": direction * day / 100.0,
                    "return_3d": direction * day / 50.0,
                    "text_count_1d": day,
                    "sentiment_mean_1d": direction,
                    "close": 10.0,
                    "dollar_volume": 1_000_000.0,
                }
            )
            labels.append(
                {
                    "asset_id": asset_id,
                    "symbol": symbol,
                    "asof_ts": asof,
                    "horizon": "1d",
                    "forward_return": direction * 0.01 if day < 5 else None,
                }
            )
    return features, labels


def test_walk_forward_snapshots_use_only_strictly_available_labels() -> None:
    features, labels = _rows()
    model = train_baselines(
        features,
        labels,
        model_version="test-v1",
        record_training_keys=True,
    )

    snapshot = next(
        row for row in model["walk_forward_snapshots"] if row["asof_ts"].startswith("2026-07-03")
    )

    assert snapshot["families"]["combined"]["training_rows"] == 2
    assert {key[1] for key in snapshot["training_keys"]} == {"2026-07-01T20:00:00Z"}
    assert all(key[1] < snapshot["training_cutoff_exclusive"] for key in snapshot["training_keys"])


def test_all_model_and_naive_families_are_deterministic() -> None:
    features, labels = _rows()
    first = train_baselines(features, labels, model_version="test-v1")
    second = train_baselines(features, labels, model_version="test-v1")

    assert first == second
    predictions = predict_all_families(features, first)
    assert set(predictions) == {
        "traditional",
        "text",
        "combined",
        "equal_weight",
        "momentum_only",
        "no_trade",
    }
    assert all(row["score"] == 0.0 for row in predictions["no_trade"])
    assert all(row["score"] == 1.0 for row in predictions["equal_weight"])
    combined = predict_with_model(features, first)
    assert all(row["training_rows"] == 0 for row in combined if "2026-07-01" in row["asof_ts"])

    evaluation = evaluate_families(features, labels, model_version="test-v1", top_k=2)
    assert evaluation["families"]["combined"]["rows"] == 8
    assert "spearman_ic" in evaluation["families"]["momentum_only"]
