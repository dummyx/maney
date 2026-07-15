from __future__ import annotations

from datetime import UTC, datetime
from math import isclose
from pathlib import Path

import pytest

from nlp_trader.backtest.costs import CostModel, transaction_cost_return
from nlp_trader.config import load_config
from nlp_trader.features.build import build_label_rows
from nlp_trader.schemas import MarketBar


def _bar(day: int, close: float, *, open_price: float | None = None) -> MarketBar:
    opening = close if open_price is None else open_price
    return MarketBar(
        asset_id="asset_aaa",
        symbol="AAA",
        ts=datetime(2026, 7, day, 20, tzinfo=UTC),
        bar_size="1d",
        open=opening,
        high=max(opening, close),
        low=min(opening, close),
        close=close,
        volume=100,
        vwap=None,
        adjusted_close=close,
        corporate_action_adjusted=True,
        adjustment_vintage_at=datetime(2026, 7, day, 20, tzinfo=UTC),
    )


def _jpx_bar(
    day: int,
    close: float,
    *,
    open_price: float | None = None,
    available_at: datetime | None = None,
) -> MarketBar:
    opening = close if open_price is None else open_price
    close_ts = datetime(2026, 7, day, 6, 30, tzinfo=UTC)
    known_at = available_at or close_ts
    return MarketBar(
        asset_id="asset_7203",
        symbol="7203",
        ts=close_ts,
        bar_size="1d",
        open=opening,
        high=max(opening, close),
        low=min(opening, close),
        close=close,
        volume=100,
        vwap=None,
        adjusted_close=close,
        corporate_action_adjusted=True,
        adjustment_vintage_at=known_at,
        available_at=known_at,
    )


def test_labels_use_next_bar_after_asof(tmp_path: Path) -> None:
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
    labels = build_label_rows(
        [_bar(1, 10.0), _bar(2, 11.0, open_price=10.0)],
        load_config(config_path),
    )

    assert isclose(float(labels[0]["forward_return"]), 0.1)
    assert labels[0]["label_start_ts"] == "2026-07-02T13:30:00Z"
    assert labels[1]["forward_return"] is None


def test_xjpx_labels_start_after_asof_and_skip_exchange_holiday(tmp_path: Path) -> None:
    config_path = tmp_path / "xjpx-config.yaml"
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
  feature_set_version: test
  label_version: test
  model_version: test
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

    labels = build_label_rows(
        [_jpx_bar(17, 100.0), _jpx_bar(21, 110.0, open_price=100.0)],
        load_config(config_path),
    )

    assert labels[0]["asof_ts"] == "2026-07-17T06:30:00Z"
    assert labels[0]["label_start_ts"] == "2026-07-21T00:00:00Z"
    assert labels[0]["label_end_ts"] == "2026-07-21T06:30:00Z"
    assert isclose(float(labels[0]["forward_return"]), 0.1)
    assert labels[1]["forward_return"] is None


def test_xjpx_label_rolls_past_an_open_missed_by_delayed_data(tmp_path: Path) -> None:
    config_path = tmp_path / "xjpx-delayed.yaml"
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
  feature_set_version: test
  label_version: test
  model_version: test
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
    delayed_until_after_tuesday_open = datetime(2026, 7, 21, 0, 1, tzinfo=UTC)
    labels = build_label_rows(
        [
            _jpx_bar(17, 100.0, available_at=delayed_until_after_tuesday_open),
            _jpx_bar(
                21,
                101.0,
                open_price=99.0,
                available_at=datetime(2026, 7, 21, 7, 30, tzinfo=UTC),
            ),
            _jpx_bar(
                22,
                110.0,
                open_price=100.0,
                available_at=datetime(2026, 7, 22, 7, 30, tzinfo=UTC),
            ),
        ],
        load_config(config_path),
    )

    assert labels[0]["asof_ts"] == "2026-07-21T00:01:00Z"
    assert labels[0]["label_start_ts"] == "2026-07-22T00:00:00Z"
    assert labels[0]["label_end_ts"] == "2026-07-22T06:30:00Z"
    assert labels[0]["label_available_at"] == "2026-07-22T07:30:00Z"
    assert isclose(float(labels[0]["forward_return"]), 0.1)


def test_labels_fail_instead_of_stretching_across_a_missing_exchange_session(
    tmp_path: Path,
) -> None:
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
  feature_set_version: test
  label_version: test
  model_version: test
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

    with pytest.raises(ValueError, match="missing daily market session"):
        build_label_rows([_bar(1, 10.0), _bar(6, 11.0)], load_config(config_path))


def test_transaction_cost_return_uses_turnover_and_bps() -> None:
    assert transaction_cost_return(0.5, CostModel(1.0, 2.0, 3.0, 0.0)) == 0.0003
