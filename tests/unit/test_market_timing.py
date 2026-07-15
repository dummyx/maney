from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nlp_trader.config import ResearchConfig, load_config
from nlp_trader.features.build import build_feature_rows
from nlp_trader.schemas import MarketBar, TextSignal


def _config(tmp_path: Path) -> ResearchConfig:
    path = tmp_path / "xjpx.yaml"
    path.write_text(
        """
mode: sample
paths:
  assets: missing.csv
  market_bars: missing.csv
  text_items: missing.jsonl
  interim_dir: interim
  processed_dir: processed
  models_dir: models
  reports_dir: reports
features:
  windows_days: [1]
  horizon_days: 1
  feature_set_version: timing-test
  label_version: timing-test
  model_version: timing-test
backtest:
  commission_bps: 1
  half_spread_bps: 1
  slippage_bps: 1
  borrow_bps_per_year: 0
  max_position_weight: 0.5
  max_gross_exposure: 1
  max_net_exposure: 1
  max_daily_turnover: 1
  max_participation_rate: 0.05
  min_price: 1
  min_dollar_volume: 1
  shorting_allowed: false
  hard_to_borrow_allowed: false
data:
  calendar: XJPX
  market_contract: japan_cash_equity_v1
""",
        encoding="utf-8",
    )
    return load_config(path)


def _bar(
    symbol: str,
    day: int,
    *,
    available_at: datetime,
    close: float = 100.0,
) -> MarketBar:
    close_ts = datetime(2026, 7, day, 6, 30, tzinfo=UTC)
    return MarketBar(
        asset_id=f"asset_{symbol.lower()}",
        symbol=symbol,
        ts=close_ts,
        bar_size="1d",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100,
        vwap=None,
        adjusted_close=None,
        corporate_action_adjusted=True,
        adjustment_vintage_at=available_at,
        return_adjustment_factor=1.0,
        exchange="XJPX",
        currency="JPY",
        trading_unit=100,
        session_date=close_ts.date(),
        available_at=available_at,
        price_basis="raw_tradable",
    )


def test_session_cross_section_waits_for_slowest_market_bar(tmp_path: Path) -> None:
    first_available = datetime(2026, 7, 15, 7, 15, tzinfo=UTC)
    complete_available = datetime(2026, 7, 15, 7, 30, tzinfo=UTC)
    signal_available = datetime(2026, 7, 15, 7, 20, tzinfo=UTC)
    bars = [
        _bar("7203", 15, available_at=first_available),
        _bar("6758", 15, available_at=complete_available),
    ]
    signal = TextSignal(
        item_id="known-before-complete-cross-section",
        asset_id="asset_7203",
        symbol="7203",
        asof_ts=signal_available,
        available_at=signal_available,
        sentiment_score=0.5,
        sentiment_label="positive",
        sentiment_confidence=1.0,
        relevance=1.0,
        novelty=1.0,
        source_credibility=1.0,
        model_version="test",
    )

    rows = build_feature_rows(bars, [signal], _config(tmp_path))

    assert {row["asof_ts"] for row in rows} == {"2026-07-15T07:30:00Z"}
    toyota = next(row for row in rows if row["symbol"] == "7203")
    assert toyota["session_close_ts"] == "2026-07-15T06:30:00Z"
    assert toyota["market_bar_available_at"] == "2026-07-15T07:15:00Z"
    assert toyota["text_count_1d"] == 1


def test_late_prior_session_payload_is_rejected(tmp_path: Path) -> None:
    bars = [
        _bar(
            "7203",
            15,
            available_at=datetime(2026, 7, 16, 8, 0, tzinfo=UTC),
        ),
        _bar(
            "7203",
            16,
            available_at=datetime(2026, 7, 16, 7, 30, tzinfo=UTC),
        ),
    ]

    with pytest.raises(ValueError, match="decision times must be strictly increasing"):
        build_feature_rows(bars, [], _config(tmp_path))
