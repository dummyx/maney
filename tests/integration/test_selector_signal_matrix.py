from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nlp_trader.config import BacktestConfig
from nlp_trader.research_templates.selector_signal_matrix import (
    SelectorInput,
    build_selector_signal_matrix,
    evaluate_selector_signal_matrix,
    holm_adjust,
    moving_block_bootstrap_mean,
)
from nlp_trader.timestamps import format_utc


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        commission_bps=1.0,
        half_spread_bps=2.0,
        slippage_bps=3.0,
        borrow_bps_per_year=0.0,
        max_position_weight=0.5,
        max_gross_exposure=1.0,
        max_net_exposure=0.5,
        max_daily_turnover=1.0,
        max_participation_rate=0.01,
        min_price=1.0,
        min_dollar_volume=1_000.0,
        shorting_allowed=False,
        hard_to_borrow_allowed=False,
    )


def test_selector_matrix_is_causal_seeded_cost_aware_and_dependence_aware() -> None:
    decisions = (
        datetime(2026, 7, 1, 20, tzinfo=UTC),
        datetime(2026, 7, 2, 20, tzinfo=UTC),
    )
    inputs = tuple(
        SelectorInput(
            asset_id=asset,
            asof_ts=decision,
            available_at=decision - timedelta(minutes=1),
            eligible=True,
            momentum_score=float(index),
            volatility_score=0.1 * (3 - index),
        )
        for decision in decisions
        for index, asset in enumerate(("asset-a", "asset-b", "asset-c"), start=1)
    )
    matrix = build_selector_signal_matrix(
        inputs,
        selected_k=2,
        random_seeds=(7, 23),
        momentum_lookback_sessions=20,
        momentum_skip_sessions=5,
    )
    reversed_matrix = build_selector_signal_matrix(
        tuple(reversed(inputs)),
        selected_k=2,
        random_seeds=(23, 7),
        momentum_lookback_sessions=20,
        momentum_skip_sessions=5,
    )
    assert matrix == reversed_matrix
    assert len(matrix.decisions) == 10

    predictions: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for decision in decisions:
        for asset_index, asset in enumerate(("asset-a", "asset-b", "asset-c"), start=1):
            row = {
                "asset_id": asset,
                "symbol": asset.upper(),
                "asof_ts": format_utc(decision),
                "horizon": "1d",
                "score": float(asset_index),
                "close": 10.0,
                "dollar_volume": 100_000_000.0,
                "sector": "Mixed",
                "beta": 0.0,
                "volatility": 0.01,
            }
            predictions.append(row)
            labels.append(
                {
                    **{key: row[key] for key in ("asset_id", "symbol", "asof_ts", "horizon")},
                    "forward_return": 0.01 * asset_index,
                    "label_start_ts": format_utc(decision + timedelta(hours=17, minutes=30)),
                    "label_end_ts": format_utc(decision + timedelta(days=1)),
                    "execution_price": 10.0,
                    "exit_price": 10.0,
                    "execution_dollar_volume": 100_000_000.0,
                    "exit_dollar_volume": 100_000_000.0,
                }
            )
    evaluated = evaluate_selector_signal_matrix(
        matrix,
        {"combined": predictions},
        labels,
        _backtest_config(),
        top_k=2,
    )

    assert "causal_momentum" in evaluated
    assert "metrics" in evaluated["causal_momentum"]["combined"]
    interval = moving_block_bootstrap_mean(
        (0.1, -0.1, 0.2, 0.0, 0.3, -0.2),
        block_size=2,
        repetitions=200,
        seed=7,
    )
    assert interval.lower <= interval.estimate <= interval.upper
    assert holm_adjust((0.01, 0.04, 0.20)) == (0.03, 0.08, 0.2)
