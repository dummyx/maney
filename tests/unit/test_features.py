from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from nlp_trader.config import load_config
from nlp_trader.features.build import build_feature_rows
from nlp_trader.schemas import MarketBar, TextSignal


def test_future_text_is_not_used_in_past_features(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
{
  "mode": "sample",
  "paths": {
    "assets": "missing.csv",
    "market_bars": "missing.csv",
    "text_items": "missing.jsonl",
    "interim_dir": "interim",
    "processed_dir": "processed",
    "models_dir": "models",
    "reports_dir": "reports"
  },
  "features": {
    "windows_days": [1],
    "horizon_days": 1,
    "feature_set_version": "test",
    "label_version": "test",
    "model_version": "test"
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
    "shorting_allowed": false,
    "hard_to_borrow_allowed": false
  }
}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    bars = [
        MarketBar(
            asset_id="asset_aaa",
            symbol="AAA",
            ts=datetime(2026, 7, 1, 20, tzinfo=UTC),
            bar_size="1d",
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.0,
            volume=100,
            vwap=None,
            adjusted_close=10.0,
            corporate_action_adjusted=True,
            adjustment_vintage_at=datetime(2026, 7, 1, 20, tzinfo=UTC),
        )
    ]
    signals = [
        TextSignal(
            item_id="future",
            asset_id="asset_aaa",
            symbol="AAA",
            asof_ts=datetime(2026, 7, 1, 21, tzinfo=UTC),
            sentiment_score=1.0,
            sentiment_label="positive",
            sentiment_confidence=1.0,
            relevance=1.0,
            novelty=1.0,
            source_credibility=1.0,
            model_version="test",
        )
    ]

    rows = build_feature_rows(bars, signals, config)

    assert rows[0]["text_count_1d"] == 0
    assert rows[0]["latest_text_available_at_1d"] is None


def test_causal_return_adjustment_factor_removes_split_discontinuity(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
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
  feature_set_version: split-test
  label_version: split-test
  model_version: split-test
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
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    bars = [
        MarketBar(
            asset_id="asset_aaa",
            symbol="AAA",
            ts=datetime(2026, 7, day, 20, tzinfo=UTC),
            bar_size="1d",
            open=price,
            high=price,
            low=price,
            close=price,
            volume=100,
            vwap=price,
            adjusted_close=None,
            corporate_action_adjusted=True,
            adjustment_vintage_at=datetime(2026, 7, day, 20, tzinfo=UTC),
            return_adjustment_factor=factor,
        )
        for day, price, factor in ((1, 10.0, 1.0), (2, 5.0, 2.0))
    ]

    rows = build_feature_rows(bars, [], config)

    assert rows[1]["return_1d"] == 0.0
    assert rows[1]["gap_return_1d"] == 0.0
    assert rows[1]["price_basis"] == "causal_return_adjustment_factor"
