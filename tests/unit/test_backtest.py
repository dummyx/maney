from __future__ import annotations

from math import isclose

import pytest

from nlp_trader.backtest.costs import CostModel, transaction_cost_return
from nlp_trader.backtest.engine import run_backtest
from nlp_trader.config import BacktestConfig
from nlp_trader.portfolio.constraints import constraints_from_config, round_trip_entry_constraints
from nlp_trader.portfolio.construction import construct_portfolio


def _config(*, shorts: bool = False, borrow_bps: float = 0.0) -> BacktestConfig:
    return BacktestConfig(
        commission_bps=1.0,
        half_spread_bps=2.0,
        slippage_bps=3.0,
        borrow_bps_per_year=borrow_bps,
        max_position_weight=0.5,
        max_gross_exposure=1.0,
        max_net_exposure=0.5,
        max_daily_turnover=1.0,
        max_participation_rate=0.01,
        min_price=1.0,
        min_dollar_volume=1_000.0,
        shorting_allowed=shorts,
        hard_to_borrow_allowed=False,
    )


def _execution_fields(index: int) -> dict[str, float | str]:
    day = index + 2
    return {
        "label_start_ts": f"2026-07-{day:02d}T13:30:00Z",
        "label_end_ts": f"2026-07-{day:02d}T20:00:00Z",
        "execution_price": 10.0,
        "exit_price": 10.0,
        "execution_dollar_volume": 100_000_000.0,
        "exit_dollar_volume": 100_000_000.0,
    }


def test_round_trip_entry_constraints_have_explicit_horizon_formulas() -> None:
    config = _config()
    base = constraints_from_config(config)
    one_session = round_trip_entry_constraints(config, horizon_steps=1)
    multi_session = round_trip_entry_constraints(config, horizon_steps=2)

    assert one_session.max_daily_turnover == pytest.approx(
        config.max_daily_turnover / (2.0 + config.same_day_exit_notional_buffer)
    )
    assert one_session.max_participation_rate == pytest.approx(
        config.max_participation_rate / (1.0 + config.same_day_exit_notional_buffer)
    )
    assert multi_session == base
    with pytest.raises(ValueError, match="horizon_steps"):
        round_trip_entry_constraints(config, horizon_steps=0)


def test_costs_increase_with_volatility_participation_and_borrow() -> None:
    model = CostModel(1.0, 2.0, 3.0, 365.0, 0.05, 50.0, 0.10)
    base = transaction_cost_return(0.5, model)
    stressed = transaction_cost_return(
        0.5,
        model,
        volatility=0.03,
        participation_rate=0.05,
        short_exposure=-0.5,
        holding_period_days=10.0,
    )
    assert isclose(base, 0.0003)
    assert stressed > base


def test_backtest_uses_conservative_missing_risk_estimates() -> None:
    prediction = {
        "asset_id": "a",
        "symbol": "A",
        "asof_ts": "2026-07-01T20:00:00Z",
        "horizon": "1d",
        "score": 1.0,
        "close": 10.0,
        "dollar_volume": 10_000_000.0,
        "sector": "Tech",
    }
    label = {
        **{key: prediction[key] for key in ("asset_id", "symbol", "asof_ts", "horizon")},
        "forward_return": 0.01,
        **_execution_fields(0),
    }
    config = _config().model_copy(update={"max_beta_exposure": 0.1, "missing_beta_fallback": 1.0})

    result = run_backtest([prediction], [label], config)
    entry = next(
        trade for trade in result["trades"] if trade["execution_phase"] == "entry_next_session_open"
    )

    assert entry["target_weight"] <= 0.1 + 1e-12
    assert "missing_beta_conservative_fallback" in entry["risk_flags"]
    assert "missing_volatility_conservative_fallback" in entry["risk_flags"]


def test_backtest_uses_the_same_top_k_selection_as_portfolio_construction() -> None:
    predictions = [
        {
            "asset_id": asset_id,
            "symbol": asset_id.upper(),
            "asof_ts": "2026-07-01T20:00:00Z",
            "horizon": "1d",
            "score": score,
            "close": 10.0,
            "dollar_volume": 100_000_000.0,
            "sector": "Mixed",
            "beta": 0.0,
            "volatility": 0.01,
        }
        for asset_id, score in (("a", 0.9), ("b", 0.5), ("c", 0.1))
    ]
    labels = [
        {
            **{key: row[key] for key in ("asset_id", "symbol", "asof_ts", "horizon")},
            "forward_return": 0.01,
            **_execution_fields(0),
        }
        for row in predictions
    ]

    result = run_backtest(predictions, labels, _config(), top_k=1)
    entries = [
        trade for trade in result["trades"] if trade["execution_phase"] == "entry_next_session_open"
    ]
    paper_decision = construct_portfolio(
        predictions,
        {},
        round_trip_entry_constraints(_config(), horizon_steps=1),
        equity=_config().initial_capital,
        top_k=1,
    )

    assert [trade["asset_id"] for trade in entries] == ["a"]
    assert {trade["asset_id"]: trade["target_weight"] for trade in entries} == (
        paper_decision.target_weights
    )
    assert result["assumptions"]["top_k"] == 1


def test_backtest_enforces_participation_and_logs_drift_and_metrics() -> None:
    predictions = [
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": "2026-07-01T20:00:00Z",
            "horizon": "1d",
            "score": 1.0,
            "close": 10.0,
            "dollar_volume": 10_000_000.0,
            "sector": "Tech",
            "beta": 1.0,
            "volatility": 0.02,
        },
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": "2026-07-02T20:00:00Z",
            "horizon": "1d",
            "score": 1.0,
            "close": 11.0,
            "dollar_volume": 10_000_000.0,
            "sector": "Tech",
            "beta": 1.0,
            "volatility": 0.02,
        },
    ]
    labels = [
        {
            **{key: row[key] for key in ("asset_id", "symbol", "asof_ts", "horizon")},
            "forward_return": value,
        }
        | _execution_fields(index)
        for index, (row, value) in enumerate(zip(predictions, (0.10, -0.05), strict=True))
    ]

    result = run_backtest(predictions, labels, _config())

    assert result["periods"]
    assert result["trades"]
    assert result["positions"]
    assert result["final_positions"] == {}
    assert {trade["execution_phase"] for trade in result["trades"]} == {
        "entry_next_session_open",
        "forced_horizon_exit",
    }
    assert result["periods"][0]["max_participation_rate"] <= 0.01 + 1e-12
    assert result["positions"][0]["post_return_weight"] != result["positions"][0]["target_weight"]
    assert result["metrics"]["total_cost_return"] > 0
    assert "sortino" in result["metrics"]
    assert "tail_loss_5pct" in result["metrics"]
    assert result["assumptions"]["initial_capital"] == 1_000_000.0


def test_short_backtest_charges_borrow() -> None:
    prediction = {
        "asset_id": "a",
        "symbol": "A",
        "asof_ts": "2026-07-01T20:00:00Z",
        "horizon": "1d",
        "score": -1.0,
        "close": 10.0,
        "dollar_volume": 100_000_000.0,
        "short_available": True,
        "hard_to_borrow": False,
    }
    label = {
        "asset_id": "a",
        "symbol": "A",
        "asof_ts": prediction["asof_ts"],
        "horizon": "1d",
        "forward_return": -0.02,
        **_execution_fields(0),
    }

    result = run_backtest([prediction], [label], _config(shorts=True, borrow_bps=365.0))

    assert result["periods"][0]["borrow_return"] > 0
    assert result["trades"][0]["target_weight"] < 0


def test_backtest_rejects_drifted_exit_participation_above_limit() -> None:
    prediction = {
        "asset_id": "a",
        "symbol": "A",
        "asof_ts": "2026-07-01T20:00:00Z",
        "horizon": "1d",
        "score": 1.0,
        "close": 10.0,
        "dollar_volume": 100_000_000.0,
    }
    label = {
        "asset_id": "a",
        "symbol": "A",
        "asof_ts": prediction["asof_ts"],
        "horizon": "1d",
        "forward_return": 2.0,
        **_execution_fields(0),
    }

    with pytest.raises(ValueError, match="horizon-close exit"):
        run_backtest([prediction], [label], _config())


def test_backtest_rejects_realized_same_day_turnover_beyond_buffer() -> None:
    prediction = {
        "asset_id": "a",
        "symbol": "A",
        "asof_ts": "2026-07-01T20:00:00Z",
        "horizon": "1d",
        "score": 1.0,
        "close": 10.0,
        "dollar_volume": 100_000_000.0,
    }
    label = {
        "asset_id": "a",
        "symbol": "A",
        "asof_ts": prediction["asof_ts"],
        "horizon": "1d",
        "forward_return": 1.0,
        **_execution_fields(0),
    }
    config = _config().model_copy(update={"max_participation_rate": 0.10})

    with pytest.raises(ValueError, match="same-session round-trip turnover"):
        run_backtest([prediction], [label], config)


def test_multi_day_labels_are_replayed_without_overlap() -> None:
    predictions = [
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": f"2026-07-{day:02d}T20:00:00Z",
            "horizon": "2d",
            "score": 1.0,
            "close": 10.0,
            "dollar_volume": 100_000_000.0,
        }
        for day in range(1, 5)
    ]
    labels = [
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": row["asof_ts"],
            "horizon": "2d",
            "forward_return": 0.02,
        }
        | {
            "label_start_ts": f"2026-07-{day + 1:02d}T13:30:00Z",
            "label_end_ts": f"2026-07-{day + 2:02d}T20:00:00Z",
            "execution_price": 10.0,
            "exit_price": 10.2,
            "execution_dollar_volume": 100_000_000.0,
            "exit_dollar_volume": 100_000_000.0,
        }
        for day, row in enumerate(predictions, start=1)
    ]

    result = run_backtest(predictions, labels, _config())
    offset = run_backtest(predictions, labels, _config(), rebalance_offset=1)

    assert result["metrics"]["periods"] == 2
    assert result["assumptions"]["overlapping_labels"].startswith("multi-session")
    assert [row["asof_ts"] for row in offset["periods"]] == [
        predictions[1]["asof_ts"],
        predictions[3]["asof_ts"],
    ]
    with pytest.raises(ValueError, match="rebalance_offset"):
        run_backtest(predictions, labels, _config(), rebalance_offset=2)


def test_split_multi_day_replay_preserves_the_global_rebalance_phase() -> None:
    predictions = [
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": f"2026-07-{day:02d}T20:00:00Z",
            "horizon": "3d",
            "score": 1.0,
            "close": 10.0,
            "dollar_volume": 100_000_000.0,
        }
        for day in range(1, 9)
    ]
    labels = [
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": row["asof_ts"],
            "horizon": "3d",
            "forward_return": 0.03,
            "label_start_ts": f"2026-07-{day + 1:02d}T13:30:00Z",
            "label_end_ts": f"2026-07-{day + 3:02d}T20:00:00Z",
            "execution_price": 10.0,
            "exit_price": 10.3,
            "execution_dollar_volume": 100_000_000.0,
            "exit_dollar_volume": 100_000_000.0,
        }
        for day, row in enumerate(predictions, start=1)
    ]

    unsplit = run_backtest(predictions, labels, _config())
    pre_holdout_periods = 5
    development = run_backtest(predictions[:pre_holdout_periods], labels, _config())
    holdout = run_backtest(
        predictions[pre_holdout_periods:],
        labels,
        _config(),
        rebalance_offset=(-pre_holdout_periods) % 3,
    )

    unsplit_times = [row["asof_ts"] for row in unsplit["periods"]]
    split_times = [row["asof_ts"] for row in development["periods"] + holdout["periods"]]
    assert split_times == unsplit_times
    assert development["periods"][-1]["exit_ts"] < holdout["periods"][0]["execution_ts"]


def test_backtest_never_selects_assets_by_partial_future_label_coverage() -> None:
    predictions = [
        {
            "asset_id": asset_id,
            "symbol": asset_id.upper(),
            "asof_ts": "2026-07-01T20:00:00Z",
            "horizon": "1d",
            "score": score,
            "close": 10.0,
            "dollar_volume": 100_000_000.0,
        }
        for asset_id, score in (("a", 1.0), ("b", 0.5))
    ]
    labels = [
        {
            "asset_id": "a",
            "symbol": "A",
            "asof_ts": predictions[0]["asof_ts"],
            "horizon": "1d",
            "forward_return": 0.01,
            **_execution_fields(0),
        },
        {
            "asset_id": "b",
            "symbol": "B",
            "asof_ts": predictions[1]["asof_ts"],
            "horizon": "1d",
            "forward_return": None,
            "expected_label_end_ts": "2026-07-02T20:00:00Z",
        },
    ]

    with pytest.raises(ValueError, match="partial forward-label coverage"):
        run_backtest(predictions, labels, _config())
