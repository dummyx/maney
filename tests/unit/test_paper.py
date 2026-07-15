from __future__ import annotations

import pytest

from nlp_trader.config import BacktestConfig
from nlp_trader.paper.simulator import PaperOrderIntent, PaperSimulator


def _config() -> BacktestConfig:
    return BacktestConfig(
        commission_bps=1.0,
        half_spread_bps=1.0,
        slippage_bps=1.0,
        borrow_bps_per_year=0.0,
        max_position_weight=0.5,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_daily_turnover=1.0,
        max_participation_rate=0.05,
        min_price=1.0,
        min_dollar_volume=1_000.0,
        shorting_allowed=False,
        hard_to_borrow_allowed=False,
    )


def test_paper_simulator_is_in_memory_constrained_and_marked_simulation_only() -> None:
    simulator = PaperSimulator(_config(), initial_capital=100_000.0)
    intent = PaperOrderIntent(
        strategy_id="research",
        asof_ts="2026-07-01T20:00:00Z",
        asset_id="a",
        symbol="A",
        target_weight=0.4,
        side="BUY",
        reason_codes=("model_positive",),
    )
    event = simulator.rebalance(
        [intent],
        [
            {
                "asset_id": "a",
                "symbol": "A",
                "close": 10.0,
                "dollar_volume": 10_000_000.0,
                "sector": "Tech",
            }
        ],
    )
    mark = simulator.mark_to_market("2026-07-02T20:00:00Z", {"a": 0.10})
    snapshot = simulator.snapshot()

    assert event["simulation_only"] is True
    assert event["trades"][0]["simulation_only"] is True
    assert "missing_beta_conservative_fallback" in event["risk_flags"]
    assert "missing_volatility_conservative_fallback" in event["risk_flags"]
    assert mark["gross_return"] > 0
    assert snapshot["equity"] > 100_000.0
    assert not hasattr(simulator, "route_order")


def test_paper_simulator_rejects_invalid_side_and_out_of_order_events() -> None:
    with pytest.raises(ValueError, match="negative target"):
        PaperOrderIntent(
            strategy_id="research",
            asof_ts="2026-07-01T20:00:00Z",
            asset_id="a",
            symbol="A",
            target_weight=-0.2,
            side="BUY",
        )

    simulator = PaperSimulator(_config())
    simulator.rebalance(
        [
            PaperOrderIntent(
                strategy_id="research",
                asof_ts="2026-07-02T20:00:00Z",
                asset_id="a",
                symbol="A",
                target_weight=0.1,
                side="BUY",
            )
        ],
        [{"asset_id": "a", "symbol": "A", "close": 10.0, "dollar_volume": 1_000_000.0}],
    )
    with pytest.raises(ValueError, match="timestamp order"):
        simulator.mark_to_market("2026-07-01T20:00:00Z", {"a": 0.01})
