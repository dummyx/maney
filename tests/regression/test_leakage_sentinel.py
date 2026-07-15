from __future__ import annotations

from datetime import UTC, datetime

from nlp_trader.config import ResearchConfig
from nlp_trader.features.build import build_feature_rows
from nlp_trader.schemas import MarketBar, TextSignal


def test_text_is_excluded_until_its_proven_availability(
    generated_config: ResearchConfig,
) -> None:
    asof_ts = datetime(2026, 7, 1, 20, tzinfo=UTC)
    bar = MarketBar(
        asset_id="asset_aaa",
        symbol="AAA",
        ts=asof_ts,
        bar_size="1d",
        open=10.0,
        high=10.5,
        low=9.5,
        close=10.0,
        volume=1_000_000,
        vwap=10.0,
        adjusted_close=10.0,
        corporate_action_adjusted=True,
        adjustment_vintage_at=asof_ts,
    )
    signal = TextSignal(
        item_id="future-vendor-delivery",
        asset_id="asset_aaa",
        symbol="AAA",
        asof_ts=asof_ts,
        available_at=datetime(2026, 7, 1, 20, 0, 1, tzinfo=UTC),
        sentiment_score=1.0,
        sentiment_label="positive",
        sentiment_confidence=1.0,
        relevance=1.0,
        novelty=1.0,
        source_credibility=1.0,
        model_version="sentinel-v1",
    )
    config = generated_config.model_copy(
        update={
            "features": generated_config.features.model_copy(
                update={"windows_days": (1,)},
            )
        }
    )

    rows = build_feature_rows([bar], [signal], config)

    assert rows[0]["text_count_1d"] == 0
    assert rows[0]["latest_text_available_at_1d"] is None


def test_text_window_age_uses_original_availability_not_mapped_decision(
    generated_config: ResearchConfig,
) -> None:
    decision_ts = datetime(2026, 7, 6, 20, tzinfo=UTC)
    bar = MarketBar(
        asset_id="asset_aaa",
        symbol="AAA",
        ts=decision_ts,
        bar_size="1d",
        open=10.0,
        high=10.5,
        low=9.5,
        close=10.0,
        volume=1_000_000,
        vwap=10.0,
        adjusted_close=10.0,
        corporate_action_adjusted=True,
        adjustment_vintage_at=decision_ts,
    )
    signal = TextSignal(
        item_id="holiday-weekend-item",
        asset_id="asset_aaa",
        symbol="AAA",
        asof_ts=decision_ts,
        available_at=datetime(2026, 7, 2, 21, tzinfo=UTC),
        sentiment_score=1.0,
        sentiment_label="positive",
        sentiment_confidence=1.0,
        relevance=1.0,
        novelty=1.0,
        source_credibility=1.0,
        model_version="sentinel-v1",
    )
    config = generated_config.model_copy(
        update={
            "features": generated_config.features.model_copy(
                update={"windows_days": (1, 5)},
            )
        }
    )

    row = build_feature_rows([bar], [signal], config)[0]

    assert row["text_count_1d"] == 0
    assert row["text_count_5d"] == 1
    assert row["latest_text_available_at_5d"] == "2026-07-02T21:00:00Z"
    assert row["latest_text_age_hours_5d"] == 95.0
