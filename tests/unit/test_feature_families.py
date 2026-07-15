from __future__ import annotations

import json
from datetime import UTC, date, datetime
from math import isclose
from pathlib import Path

from nlp_trader.config import ResearchConfig, load_config
from nlp_trader.features.build import build_feature_rows, build_label_rows
from nlp_trader.schemas import (
    Asset,
    CorporateAction,
    EarningsCalendarEvent,
    FundamentalRecord,
    MarketBar,
    TextSignal,
)


def _config(tmp_path: Path) -> ResearchConfig:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "mode": "sample",
                "paths": {
                    "assets": "missing.csv",
                    "market_bars": "missing.csv",
                    "text_items": "missing.jsonl",
                    "interim_dir": "interim",
                    "processed_dir": "processed",
                    "models_dir": "models",
                    "reports_dir": "reports",
                },
                "features": {
                    "windows_days": [3],
                    "horizon_days": 1,
                    "feature_set_version": "families-test",
                    "label_version": "families-test",
                    "model_version": "families-test",
                },
                "backtest": {
                    "commission_bps": 1.0,
                    "half_spread_bps": 1.0,
                    "slippage_bps": 1.0,
                    "borrow_bps_per_year": 0.0,
                    "max_position_weight": 0.5,
                    "max_gross_exposure": 1.0,
                    "max_net_exposure": 1.0,
                    "max_daily_turnover": 1.0,
                    "max_participation_rate": 0.05,
                    "min_price": 1.0,
                    "min_dollar_volume": 1.0,
                    "shorting_allowed": False,
                    "hard_to_borrow_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return load_config(config_path)


def _bar(
    asset_id: str,
    symbol: str,
    day: int,
    close: float,
    adjusted_close: float,
    volume: int,
) -> MarketBar:
    return MarketBar(
        asset_id=asset_id,
        symbol=symbol,
        ts=datetime(2026, 7, day, 20, tzinfo=UTC),
        bar_size="1d",
        open=close * 0.99,
        high=close * 1.02,
        low=close * 0.98,
        close=close,
        volume=volume,
        vwap=close,
        adjusted_close=adjusted_close,
        corporate_action_adjusted=True,
        adjustment_vintage_at=datetime(2026, 7, day, 20, tzinfo=UTC),
    )


def _signal(
    item_id: str,
    day: int,
    score: float,
    *,
    novelty: float,
    event_type: str,
) -> TextSignal:
    return TextSignal(
        item_id=item_id,
        asset_id="alpha",
        symbol="AAA",
        asof_ts=datetime(2026, 7, day, 12, tzinfo=UTC),
        sentiment_score=score,
        sentiment_label="positive" if score > 0 else "negative",
        sentiment_confidence=1.0,
        relevance=1.0,
        novelty=novelty,
        source_credibility=0.9,
        model_version="test",
        event_type=event_type,
        spam_score=0.0,
    )


def _bars() -> list[MarketBar]:
    alpha_close = [50.0, 51.0, 52.0, 54.0, 53.0, 56.0]
    alpha_adjusted = list(alpha_close)
    beta_close = [20.0, 21.0, 20.5, 22.0, 23.0, 24.0]
    session_days = [1, 2, 6, 7, 8, 9]
    bars: list[MarketBar] = []
    for index, (day, close, adjusted) in enumerate(
        zip(session_days, alpha_close, alpha_adjusted, strict=True), start=1
    ):
        bars.append(_bar("alpha", "AAA", day, close, adjusted, 1_000_000 + index * 10_000))
    for index, (day, close) in enumerate(zip(session_days, beta_close, strict=True), start=1):
        bars.append(_bar("beta", "BBB", day, close, close, 2_000_000 + index * 20_000))
    return bars


def test_feature_builder_emits_text_and_traditional_families(tmp_path: Path) -> None:
    config = _config(tmp_path)
    signals = [
        _signal("older", 8, 1.0, novelty=1.0, event_type="earnings"),
        _signal("recent", 9, -1.0, novelty=0.0, event_type="guidance"),
        _signal("future", 10, 1.0, novelty=1.0, event_type="earnings"),
    ]

    rows = build_feature_rows(_bars(), signals, config)
    row = next(
        value
        for value in rows
        if value["symbol"] == "AAA" and value["asof_ts"].startswith("2026-07-09")
    )

    assert row["text_count_3d"] == 2
    assert row["text_missing_3d"] is False
    assert row["latest_text_available_at_3d"] == "2026-07-09T12:00:00Z"
    assert row["text_decay_half_life_days_3d"] == 1.0
    assert row["time_since_first_seen_hours_3d"] == 32.0
    assert row["sentiment_mean_3d"] == 0.0
    assert row["sentiment_decay_weighted_3d"] < 0.0
    assert row["sentiment_dispersion_3d"] > 0.0
    assert row["attention_item_count_3d"] == 2
    assert row["novelty_share_3d"] == 0.5
    assert row["duplicate_item_count_3d"] == 1
    assert row["source_credibility_mean_3d"] == 0.9
    assert row["event_item_count_3d"] == 2
    assert row["event_earnings_count_3d"] == 1
    assert row["event_guidance_count_3d"] == 1

    assert row["price_basis"] == "causal_return_adjustment_factor"
    assert row["return_5d_missing"] is False
    assert row["amihud_illiquidity_20d"] >= 0.0
    assert row["realized_volatility_3d"] > 0.0
    assert row["market_beta_60d_missing"] is False
    assert row["sector_data_missing"] is True
    assert row["earnings_proximity_missing"] is True


def test_labels_use_adjusted_prices_and_future_only_supported_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)

    labels = build_label_rows(_bars(), config)
    first_alpha = next(
        value
        for value in labels
        if value["symbol"] == "AAA" and value["asof_ts"].startswith("2026-07-01")
    )

    assert first_alpha["price_basis"] == "causal_return_adjustment_factor"
    assert isclose(float(first_alpha["forward_return"]), 51.0 / (51.0 * 0.99) - 1.0)
    assert first_alpha["label_start_ts"] == "2026-07-02T13:30:00Z"
    assert first_alpha["binary_up_1d"] == 1
    assert first_alpha["rank_1d"] == 0.5
    assert first_alpha["volatility_1d"] is not None
    assert first_alpha["volume_1d"] is not None
    assert first_alpha["forward_abnormal_return_1d"] is not None
    assert first_alpha["forward_sector_neutral_return_1d"] is None

    final_alpha = next(
        value
        for value in labels
        if value["symbol"] == "AAA" and value["asof_ts"].startswith("2026-07-09")
    )
    assert final_alpha["forward_return"] is None
    assert final_alpha["label_start_ts"] is None


def test_known_fundamentals_and_event_calendars_are_point_in_time(tmp_path: Path) -> None:
    config = _config(tmp_path)
    asset = Asset(
        asset_id="alpha",
        symbol="AAA",
        exchange="XNAS",
        currency="USD",
        name="Alpha",
        sector="Technology",
        active_from=None,
        active_to=None,
    )
    bar = _bar("alpha", "AAA", 6, 50.0, 50.0, 1_000_000)
    known_at = datetime(2026, 7, 1, 12, tzinfo=UTC)
    fundamental = FundamentalRecord(
        asset_id="alpha",
        symbol="AAA",
        period_end=date(2026, 6, 30),
        available_at=known_at,
        values={"book_to_market": 0.4, "return_on_equity": 0.2, "market_cap": 1e9},
    )
    earnings = EarningsCalendarEvent(
        asset_id="alpha",
        symbol="AAA",
        event_ts=datetime(2026, 7, 7, 20, tzinfo=UTC),
        available_at=known_at,
        status="confirmed",
    )
    unavailable_revision = EarningsCalendarEvent(
        asset_id="alpha",
        symbol="AAA",
        event_ts=datetime(2026, 7, 7, 19, tzinfo=UTC),
        available_at=datetime(2026, 7, 7, 12, tzinfo=UTC),
        status="confirmed",
    )
    dividend = CorporateAction(
        asset_id="alpha",
        symbol="AAA",
        event_ts=datetime(2026, 7, 8, 13, 30, tzinfo=UTC),
        available_at=known_at,
        action_type="ex_dividend",
        value=0.25,
    )

    row = build_feature_rows(
        [bar],
        [],
        config,
        [asset],
        fundamentals=[fundamental],
        earnings_events=[earnings, unavailable_revision],
        corporate_actions=[dividend],
    )[0]

    assert row["value_proxy"] == 0.4
    assert row["quality_proxy"] == 0.2
    assert row["fundamental_available_at"] == "2026-07-01T12:00:00Z"
    assert row["earnings_calendar_available_at"] == "2026-07-01T12:00:00Z"
    assert row["earnings_blackout"] is True
    assert row["ex_dividend_proximity_missing"] is False
    assert row["known_event_blackout"] is True
