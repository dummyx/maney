from __future__ import annotations

from typing import Any

import pytest

from nlp_trader.features.build import finite_float
from nlp_trader.models.baselines import (
    label_availability_by_key,
    predict_all_families,
    predict_with_model,
    train_baselines,
)
from nlp_trader.models.evaluation import evaluate_families
from nlp_trader.timestamps import parse_utc


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
            label = {
                "asset_id": asset_id,
                "symbol": symbol,
                "asof_ts": asof,
                "horizon": "1d",
                "forward_return": direction * 0.01 if day < 5 else None,
            }
            if day == 5:
                label["expected_label_end_ts"] = "2026-07-06T20:00:00Z"
            labels.append(label)
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


def test_label_availability_prefers_actual_delivery_over_outcome_end() -> None:
    base = {
        "asset_id": "a",
        "symbol": "A",
        "asof_ts": "2026-07-01T20:00:00Z",
        "horizon": "1d",
        "forward_return": 0.01,
        "label_end_ts": "2026-07-02T20:00:00Z",
        "available_at": "2026-07-03T20:00:00Z",
        "label_available_at": "2026-07-04T20:00:00Z",
    }
    key = ("a", "2026-07-01T20:00:00Z", "1d")

    assert label_availability_by_key([base])[key] == "2026-07-04T20:00:00Z"
    without_label_specific = {
        key: value for key, value in base.items() if key != "label_available_at"
    }
    assert label_availability_by_key([without_label_specific])[key] == "2026-07-03T20:00:00Z"
    outcome_only = {
        key: value for key, value in without_label_specific.items() if key != "available_at"
    }
    assert label_availability_by_key([outcome_only])[key] == "2026-07-02T20:00:00Z"


def test_final_holdout_freezes_pre_boundary_training_membership() -> None:
    features, labels = _rows()
    frozen = train_baselines(
        features,
        labels,
        model_version="test-v1",
        final_holdout_periods=2,
        record_training_keys=True,
    )
    expanding = train_baselines(
        features,
        labels,
        model_version="test-v1",
        record_training_keys=True,
    )

    protocol = frozen["final_holdout_training"]
    boundary = str(protocol["final_holdout_start"])
    assert protocol["name"] == "frozen_pre_holdout_snapshot_v1"
    assert protocol["training_cutoff_exclusive"] == boundary
    assert protocol["update_rule"] == "no training updates at or after final_holdout_start"

    frozen_snapshots = [
        snapshot
        for snapshot in frozen["walk_forward_snapshots"]
        if parse_utc(str(snapshot["asof_ts"])) >= parse_utc(boundary)
    ]
    assert frozen_snapshots
    assert {snapshot["training_snapshot_role"] for snapshot in frozen_snapshots} == {
        "frozen_final_holdout"
    }
    assert len({snapshot["training_key_digest"] for snapshot in frozen_snapshots}) == 1
    assert len({snapshot["training_key_count"] for snapshot in frozen_snapshots}) == 1
    assert all(
        snapshot["families"] == frozen_snapshots[0]["families"] for snapshot in frozen_snapshots
    )

    availability = label_availability_by_key(labels)
    for key_values in frozen_snapshots[0]["training_keys"]:
        key = (str(key_values[0]), str(key_values[1]), str(key_values[2]))
        assert parse_utc(availability[key]) < parse_utc(boundary)
        assert parse_utc(key[1]) < parse_utc(boundary)

    development_snapshots = [
        snapshot
        for snapshot in frozen["walk_forward_snapshots"]
        if parse_utc(str(snapshot["asof_ts"])) < parse_utc(boundary)
    ]
    expanding_development = [
        snapshot
        for snapshot in expanding["walk_forward_snapshots"]
        if parse_utc(str(snapshot["asof_ts"])) < parse_utc(boundary)
    ]
    assert development_snapshots == expanding_development
    assert (
        expanding["walk_forward_snapshots"][-1]["eligible_training_rows"]
        > frozen_snapshots[-1]["eligible_training_rows"]
    )

    evaluation = evaluate_families(
        features,
        labels,
        model_version="test-v1",
        final_holdout_periods=2,
    )
    holdout_protocol = evaluation["evaluation_protocol"]["final_holdout_training"]
    assert holdout_protocol["verified_untouched"] is True
    assert holdout_protocol["training_key_digest"] == protocol["training_key_digest"]


def test_training_rejects_forward_label_availability_before_its_decision() -> None:
    features, labels = _rows()
    malformed = [dict(label) for label in labels]
    malformed[4]["label_available_at"] = "2026-07-01T20:00:00Z"

    with pytest.raises(ValueError, match="availability must be strictly after"):
        train_baselines(
            features,
            malformed,
            model_version="test-v1",
            final_holdout_periods=2,
        )


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


def test_training_requires_unique_matching_labels() -> None:
    features, labels = _rows()

    with pytest.raises(ValueError, match="features have no matching labels"):
        train_baselines(features, labels[1:], model_version="test-v1")

    with pytest.raises(ValueError, match="duplicate label row"):
        train_baselines(features, [*labels, dict(labels[0])], model_version="test-v1")


def test_training_rejects_partial_or_non_terminal_censored_cross_sections() -> None:
    features, labels = _rows()
    partial = [dict(label) for label in labels]
    partial[2]["forward_return"] = None

    with pytest.raises(ValueError, match="partial forward-label coverage"):
        train_baselines(features, partial, model_version="test-v1")

    non_terminal = [dict(label) for label in labels]
    for label in non_terminal[2:4]:
        label["forward_return"] = None
        label["expected_label_end_ts"] = "2026-07-03T20:00:00Z"

    with pytest.raises(ValueError, match="non-terminal decision has no forward-label coverage"):
        train_baselines(features, non_terminal, model_version="test-v1")


def test_model_discovery_includes_engineered_values_but_not_availability_provenance() -> None:
    asof_ts = "2026-07-01T20:00:00Z"
    features = [
        {
            "asset_id": "asset_aaa",
            "symbol": "AAA",
            "asof_ts": asof_ts,
            "horizon": "1d",
            "short_term_reversal_1d": 0.1,
            "amihud_illiquidity_20d": 0.2,
            "high_low_spread_estimate": 0.3,
            "market_residual_return_1d": 0.4,
            "fundamental_size_proxy_log_market_cap": 12.0,
            "fundamental_available_at": "2026-06-30T20:00:00Z",
            "earnings_calendar_available_at": "2026-06-29T20:00:00Z",
            "source_credibility_mean_1d": 0.9,
            "spam_score_mean_1d": 0.1,
            "credible_attention_1d": 1.5,
            "author_disagreement_1d": 0.2,
            "latest_text_age_hours_1d": 2.0,
            "latest_text_available_at_1d": "2026-07-01T18:00:00Z",
            "llm_semantic_mean_1d": 1.0,
            "llm_raw_confidence_mean_1d": 0.8,
            "llm_missing_1d": False,
        }
    ]
    labels = [
        {
            "asset_id": "asset_aaa",
            "symbol": "AAA",
            "asof_ts": asof_ts,
            "horizon": "1d",
            "forward_return": 0.01,
        }
    ]

    model = train_baselines(features, labels, model_version="test-v1")
    traditional = set(model["families"]["traditional"]["features"])
    text = set(model["families"]["text"]["features"])
    combined = set(model["families"]["combined"]["features"])

    assert {
        "short_term_reversal_1d",
        "amihud_illiquidity_20d",
        "high_low_spread_estimate",
        "market_residual_return_1d",
        "fundamental_size_proxy_log_market_cap",
    } <= traditional
    assert {
        "source_credibility_mean_1d",
        "spam_score_mean_1d",
        "credible_attention_1d",
        "author_disagreement_1d",
        "latest_text_age_hours_1d",
    } <= text
    assert "fundamental_available_at" not in traditional
    assert "earnings_calendar_available_at" not in traditional
    assert "latest_text_available_at_1d" not in text
    assert not {"llm_semantic_mean_1d", "llm_raw_confidence_mean_1d", "llm_missing_1d"} & (
        traditional | text | combined
    )


def test_llm_model_families_are_explicit_and_do_not_change_default_semantics() -> None:
    features, labels = _rows()
    enriched_features = [
        {
            **row,
            "llm_semantic_mean_1d": 1.0 if row["symbol"] == "AAA" else -1.0,
            "llm_semantic_conf_weighted_1d": 0.8 if row["symbol"] == "AAA" else -0.8,
            "llm_raw_confidence_mean_1d": 0.8,
            "llm_missing_1d": False,
        }
        for row in features
    ]

    default_model = train_baselines(enriched_features, labels, model_version="default-v1")
    assert set(default_model["families"]) == {"traditional", "text", "combined"}
    assert all(
        not any(column.startswith("llm_") for column in spec["features"])
        for spec in default_model["families"].values()
    )

    llm_model = train_baselines(
        enriched_features,
        labels,
        model_version="llm-v1",
        families=("llm", "traditional_llm", "all"),
    )
    assert list(llm_model["families"]) == ["llm", "traditional_llm", "all"]
    llm_columns = set(llm_model["families"]["llm"]["features"])
    traditional_llm_columns = set(llm_model["families"]["traditional_llm"]["features"])
    all_columns = set(llm_model["families"]["all"]["features"])
    assert llm_columns == {
        "llm_missing_1d",
        "llm_raw_confidence_mean_1d",
        "llm_semantic_mean_1d",
    }
    assert traditional_llm_columns > llm_columns
    assert all_columns > traditional_llm_columns
    assert "return_1d" in traditional_llm_columns
    assert "sentiment_mean_1d" not in traditional_llm_columns
    assert "sentiment_mean_1d" in all_columns

    predictions = predict_all_families(enriched_features, llm_model)
    assert set(predictions) == {
        "llm",
        "traditional_llm",
        "all",
        "equal_weight",
        "momentum_only",
        "no_trade",
    }


@pytest.mark.parametrize(
    ("families", "message"),
    [
        ((), "must not be empty"),
        (("llm", "llm"), "must be unique"),
        (("unknown",), "unknown model families"),
    ],
)
def test_training_rejects_invalid_model_family_selection(
    families: tuple[str, ...],
    message: str,
) -> None:
    features, labels = _rows()

    with pytest.raises(ValueError, match=message):
        train_baselines(features, labels, model_version="test-v1", families=families)
