from __future__ import annotations

import statistics
from math import pow, sqrt
from typing import Any

from nlp_trader.features.build import finite_float
from nlp_trader.timestamps import parse_utc


def max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 1.0
    drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            drawdown = min(drawdown, equity / peak - 1.0)
    return drawdown


def annualized_return(equity_curve: list[float], periods_per_year: float = 252.0) -> float:
    if len(equity_curve) < 2:
        return 0.0
    periods = len(equity_curve) - 1
    if equity_curve[0] <= 0 or equity_curve[-1] <= 0:
        return -1.0
    return pow(equity_curve[-1] / equity_curve[0], periods_per_year / periods) - 1.0


def _tail_loss(returns: list[float], quantile: float = 0.05) -> float:
    if not returns:
        return 0.0
    ordered = sorted(returns)
    count = max(1, int(len(ordered) * quantile))
    return statistics.fmean(ordered[:count])


def _average_holding_days(trades: list[dict[str, Any]]) -> float:
    opened: dict[str, str] = {}
    durations: list[float] = []
    for trade in sorted(trades, key=lambda row: str(row["asof_ts"])):
        asset_id = str(trade["asset_id"])
        previous = finite_float(trade.get("previous_weight"))
        target = finite_float(trade.get("target_weight"))
        if abs(previous) <= 1e-12 and abs(target) > 1e-12:
            opened[asset_id] = str(trade["asof_ts"])
        elif previous * target < 0:
            if asset_id in opened:
                duration = parse_utc(str(trade["asof_ts"])) - parse_utc(opened.pop(asset_id))
                durations.append(max(0.0, duration.total_seconds() / 86_400.0))
            opened[asset_id] = str(trade["asof_ts"])
        elif abs(previous) > 1e-12 and abs(target) <= 1e-12 and asset_id in opened:
            duration = parse_utc(str(trade["asof_ts"])) - parse_utc(opened.pop(asset_id))
            durations.append(max(0.0, duration.total_seconds() / 86_400.0))
    return statistics.fmean(durations) if durations else 0.0


def summarize_backtest(
    periods: list[dict[str, Any]],
    equity_curve: list[float],
    trades: list[dict[str, Any]],
    *,
    periods_per_year: float = 252.0,
) -> dict[str, float | int]:
    returns = [finite_float(row.get("net_return")) for row in periods]
    gross_returns = [finite_float(row.get("gross_return")) for row in periods]
    mean_return = statistics.fmean(returns) if returns else 0.0
    daily_volatility = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    downside = [min(0.0, value) for value in returns]
    downside_deviation = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    gross_equity = 1.0
    for value in gross_returns:
        gross_equity *= max(0.0, 1.0 + value)
    total_cost = sum(finite_float(row.get("cost_return")) for row in periods)
    metrics: dict[str, float | int] = {
        "periods": len(periods),
        "total_return": equity_curve[-1] / equity_curve[0] - 1.0 if equity_curve else 0.0,
        "gross_total_return": gross_equity - 1.0,
        "cost_adjusted_return": equity_curve[-1] / equity_curve[0] - 1.0 if equity_curve else 0.0,
        "annualized_return": annualized_return(equity_curve, periods_per_year),
        "annualized_volatility": daily_volatility * sqrt(periods_per_year),
        "sharpe": mean_return / daily_volatility * sqrt(periods_per_year)
        if daily_volatility
        else 0.0,
        "sortino": mean_return / downside_deviation * sqrt(periods_per_year)
        if downside_deviation
        else 0.0,
        "max_drawdown": max_drawdown(equity_curve),
        "hit_rate": statistics.fmean(1.0 if value > 0 else 0.0 for value in returns)
        if returns
        else 0.0,
        "tail_loss_5pct": _tail_loss(returns),
        "average_turnover": statistics.fmean(finite_float(row.get("turnover")) for row in periods)
        if periods
        else 0.0,
        "average_cost_return": total_cost / len(periods) if periods else 0.0,
        "total_cost_return": total_cost,
        "average_gross_exposure": statistics.fmean(
            finite_float(row.get("gross_exposure")) for row in periods
        )
        if periods
        else 0.0,
        "average_net_exposure": statistics.fmean(
            finite_float(row.get("net_exposure")) for row in periods
        )
        if periods
        else 0.0,
        "average_beta_exposure": statistics.fmean(
            finite_float(row.get("beta_exposure")) for row in periods
        )
        if periods
        else 0.0,
        "max_participation_rate": max(
            (finite_float(row.get("max_participation_rate")) for row in periods), default=0.0
        ),
        "minimum_capacity_proxy_equity": min(
            (
                finite_float(row.get("capacity_proxy_equity"))
                for row in periods
                if finite_float(row.get("capacity_proxy_equity")) > 0
            ),
            default=0.0,
        ),
        "average_holding_period_days": _average_holding_days(trades),
        "final_equity": equity_curve[-1] if equity_curve else 1.0,
        "trades": len(trades),
    }
    for component in ("commission", "spread", "slippage", "market_impact", "borrow"):
        metrics[f"total_{component}_return"] = sum(
            finite_float(row.get(f"{component}_return")) for row in periods
        )
    return metrics
