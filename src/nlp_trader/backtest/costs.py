from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from nlp_trader.config import BacktestConfig


@dataclass(frozen=True, slots=True)
class CostModel:
    commission_bps: float
    half_spread_bps: float
    slippage_bps: float
    borrow_bps_per_year: float
    volatility_slippage_multiplier: float = 0.0
    participation_slippage_bps: float = 0.0
    market_impact_multiplier: float = 0.0

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.commission_bps,
                self.half_spread_bps,
                self.slippage_bps,
                self.borrow_bps_per_year,
                self.volatility_slippage_multiplier,
                self.participation_slippage_bps,
                self.market_impact_multiplier,
            )
        ):
            raise ValueError("cost model inputs must be non-negative")


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    commission: float
    spread: float
    slippage: float
    market_impact: float
    borrow: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.slippage + self.market_impact + self.borrow

    def to_dict(self) -> dict[str, float]:
        return {
            "commission": self.commission,
            "spread": self.spread,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "borrow": self.borrow,
            "total": self.total,
        }


def cost_model_from_config(config: BacktestConfig) -> CostModel:
    return CostModel(
        commission_bps=float(config.commission_bps),
        half_spread_bps=float(config.half_spread_bps),
        slippage_bps=float(config.slippage_bps),
        borrow_bps_per_year=float(config.borrow_bps_per_year),
        volatility_slippage_multiplier=float(
            getattr(config, "volatility_slippage_multiplier", 0.05)
        ),
        participation_slippage_bps=float(getattr(config, "participation_slippage_bps", 50.0)),
        market_impact_multiplier=float(getattr(config, "market_impact_multiplier", 0.10)),
    )


def cost_breakdown(
    turnover: float,
    model: CostModel,
    *,
    volatility: float = 0.0,
    participation_rate: float = 0.0,
    short_exposure: float = 0.0,
    holding_period_days: float = 1.0,
) -> CostBreakdown:
    traded = abs(turnover)
    volatility = max(0.0, volatility)
    participation = max(0.0, participation_rate)
    commission = traded * model.commission_bps / 10_000.0
    spread = traded * model.half_spread_bps / 10_000.0
    dynamic_slippage_bps = (
        model.slippage_bps
        + volatility * 10_000.0 * model.volatility_slippage_multiplier
        + participation * model.participation_slippage_bps
    )
    slippage = traded * dynamic_slippage_bps / 10_000.0
    impact_bps = volatility * sqrt(participation) * 10_000.0 * model.market_impact_multiplier
    market_impact = traded * impact_bps / 10_000.0
    borrow = (
        abs(min(0.0, short_exposure))
        * model.borrow_bps_per_year
        / 10_000.0
        * max(0.0, holding_period_days)
        / 365.0
    )
    return CostBreakdown(commission, spread, slippage, market_impact, borrow)


def transaction_cost_return(
    turnover: float,
    model: CostModel,
    *,
    volatility: float = 0.0,
    participation_rate: float = 0.0,
    short_exposure: float = 0.0,
    holding_period_days: float = 1.0,
) -> float:
    if (
        volatility == 0.0
        and participation_rate == 0.0
        and short_exposure == 0.0
        and model.volatility_slippage_multiplier == 0.0
        and model.participation_slippage_bps == 0.0
        and model.market_impact_multiplier == 0.0
    ):
        traded_bps = model.commission_bps + model.half_spread_bps + model.slippage_bps
        return abs(turnover) * traded_bps / 10_000.0
    return cost_breakdown(
        turnover,
        model,
        volatility=volatility,
        participation_rate=participation_rate,
        short_exposure=short_exposure,
        holding_period_days=holding_period_days,
    ).total
