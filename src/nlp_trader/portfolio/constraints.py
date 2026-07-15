from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from nlp_trader.config import BacktestConfig


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    max_position_weight: float
    max_gross_exposure: float
    max_net_exposure: float
    max_sector_weight: float
    max_beta_exposure: float
    max_daily_turnover: float
    max_participation_rate: float
    min_price: float
    min_dollar_volume: float
    shorting_allowed: bool
    hard_to_borrow_allowed: bool

    def __post_init__(self) -> None:
        positive = {
            "max_position_weight": self.max_position_weight,
            "max_gross_exposure": self.max_gross_exposure,
            "max_sector_weight": self.max_sector_weight,
            "max_daily_turnover": self.max_daily_turnover,
            "max_participation_rate": self.max_participation_rate,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_net_exposure < 0 or self.max_beta_exposure < 0:
            raise ValueError("net and beta exposure limits must be non-negative")
        if self.min_price < 0 or self.min_dollar_volume < 0:
            raise ValueError("price and liquidity floors must be non-negative")


def constraints_from_config(config: BacktestConfig) -> PortfolioConstraints:
    """Build complete constraints while retaining compatibility with the initial config."""

    max_gross = float(config.max_gross_exposure)
    max_net = float(config.max_net_exposure)
    return PortfolioConstraints(
        max_position_weight=float(config.max_position_weight),
        max_gross_exposure=max_gross,
        max_net_exposure=max_net,
        max_sector_weight=float(getattr(config, "max_sector_weight", max_gross)),
        max_beta_exposure=float(getattr(config, "max_beta_exposure", max_net)),
        max_daily_turnover=float(config.max_daily_turnover),
        max_participation_rate=float(config.max_participation_rate),
        min_price=float(config.min_price),
        min_dollar_volume=float(config.min_dollar_volume),
        shorting_allowed=bool(config.shorting_allowed),
        hard_to_borrow_allowed=bool(config.hard_to_borrow_allowed),
    )


def round_trip_entry_constraints(
    config: BacktestConfig,
    *,
    horizon_steps: int,
) -> PortfolioConstraints:
    """Return entry limits that reserve capacity for the planned liquidation leg."""

    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    constraints = constraints_from_config(config)
    if horizon_steps != 1:
        return constraints
    return replace(
        constraints,
        max_daily_turnover=config.max_daily_turnover / (2.0 + config.same_day_exit_notional_buffer),
        max_participation_rate=constraints.max_participation_rate
        / (1.0 + config.same_day_exit_notional_buffer),
    )


def constraint_snapshot(constraints: PortfolioConstraints) -> dict[str, Any]:
    return {
        field: getattr(constraints, field) for field in PortfolioConstraints.__dataclass_fields__
    }
