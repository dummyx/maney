from __future__ import annotations

from nlp_trader.portfolio.constraints import PortfolioConstraints
from nlp_trader.portfolio.construction import constrain_target_weights
from nlp_trader.portfolio.risk import calculate_exposures


def _constraints(*, shorting: bool = False) -> PortfolioConstraints:
    return PortfolioConstraints(
        max_position_weight=0.4,
        max_gross_exposure=0.6,
        max_net_exposure=0.4,
        max_sector_weight=0.3,
        max_beta_exposure=0.35,
        max_daily_turnover=0.5,
        max_participation_rate=1.0,
        min_price=5.0,
        min_dollar_volume=100.0,
        shorting_allowed=shorting,
        hard_to_borrow_allowed=False,
    )


def test_target_constraints_enforce_exposure_limits() -> None:
    rows = [
        {
            "asset_id": "a",
            "symbol": "A",
            "sector": "Tech",
            "beta": 2.0,
            "close": 10.0,
            "dollar_volume": 10_000.0,
        },
        {
            "asset_id": "b",
            "symbol": "B",
            "sector": "Tech",
            "beta": 1.0,
            "close": 10.0,
            "dollar_volume": 10_000.0,
        },
        {
            "asset_id": "c",
            "symbol": "C",
            "sector": "Finance",
            "beta": 1.0,
            "close": 10.0,
            "dollar_volume": 10_000.0,
        },
    ]
    decision = constrain_target_weights(
        {"a": 0.4, "b": 0.4, "c": 0.4},
        rows,
        {},
        _constraints(),
        equity=1_000.0,
    )
    exposure = calculate_exposures(decision.target_weights, {row["asset_id"]: row for row in rows})

    assert max(abs(weight) for weight in decision.target_weights.values()) <= 0.4
    assert exposure.gross <= 0.6
    assert abs(exposure.net) <= 0.4
    assert max(exposure.sectors.values()) <= 0.3
    assert abs(exposure.beta) <= 0.35
    assert decision.turnover <= 0.5
    assert not decision.risk_flags


def test_short_and_participation_controls_reject_or_clip_orders() -> None:
    row = {
        "asset_id": "a",
        "symbol": "A",
        "sector": "Tech",
        "close": 10.0,
        "dollar_volume": 1_000.0,
        "short_available": False,
    }
    rejected = constrain_target_weights({"a": -0.3}, [row], {}, _constraints(), equity=1_000.0)
    assert rejected.target_weights == {}
    assert "shorting_disabled" in rejected.rejected["a"]

    constraints = PortfolioConstraints(
        max_position_weight=0.5,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_sector_weight=1.0,
        max_beta_exposure=1.0,
        max_daily_turnover=1.0,
        max_participation_rate=0.01,
        min_price=1.0,
        min_dollar_volume=1.0,
        shorting_allowed=False,
        hard_to_borrow_allowed=False,
    )
    clipped = constrain_target_weights({"a": 0.5}, [row], {}, constraints, equity=1_000.0)
    assert clipped.target_weights["a"] == 0.01
    assert clipped.participation["a"] == 0.01


def test_participation_clipping_cannot_break_beta_constraint() -> None:
    constraints = PortfolioConstraints(
        max_position_weight=0.5,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_sector_weight=1.0,
        max_beta_exposure=0.1,
        max_daily_turnover=1.0,
        max_participation_rate=0.01,
        min_price=1.0,
        min_dollar_volume=1.0,
        shorting_allowed=False,
        hard_to_borrow_allowed=False,
    )
    rows = [
        {
            "asset_id": "a",
            "symbol": "A",
            "sector": "Tech",
            "beta": 2.0,
            "close": 10.0,
            "dollar_volume": 1_000_000.0,
        },
        {
            "asset_id": "b",
            "symbol": "B",
            "sector": "Tech",
            "beta": -2.0,
            "close": 10.0,
            "dollar_volume": 1_000.0,
        },
    ]

    decision = constrain_target_weights({"a": 0.5, "b": 0.5}, rows, {}, constraints, equity=1_000.0)
    exposure = calculate_exposures(
        decision.target_weights, {str(row["asset_id"]): row for row in rows}
    )

    assert abs(exposure.beta) <= constraints.max_beta_exposure + 1e-12
    assert not decision.risk_flags
    assert max(decision.participation.values()) <= constraints.max_participation_rate + 1e-12
