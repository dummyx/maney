from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from nlp_trader.backtest.costs import (
    CostBreakdown,
    cost_breakdown,
    cost_model_from_config,
)
from nlp_trader.backtest.metrics import summarize_backtest
from nlp_trader.config import BacktestConfig
from nlp_trader.features.build import finite_float
from nlp_trader.portfolio.constraints import constraints_from_config
from nlp_trader.portfolio.construction import construct_portfolio
from nlp_trader.portfolio.risk import (
    calculate_exposures,
    conservative_risk_estimates,
    drift_weights,
    risk_estimate_flags,
)
from nlp_trader.timestamps import parse_utc


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["asset_id"]), str(row["asof_ts"]), str(row["horizon"])


def _unique_rows(
    rows: list[dict[str, Any]], *, name: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate {name} row: {key}")
        indexed[key] = row
    return indexed


def _volatility(row: dict[str, Any]) -> float:
    if "volatility" in row:
        return max(0.0, finite_float(row["volatility"]))
    return max(0.0, finite_float(row.get("realized_volatility_3d")))


def _holding_days(start: str, end: str) -> float:
    days = (parse_utc(end) - parse_utc(start)).total_seconds() / 86_400.0
    if days <= 0:
        raise ValueError("label exit must be strictly after next-session execution")
    return max(1.0, days)


def _horizon_steps(horizon: object) -> int:
    digits = "".join(character for character in str(horizon) if character.isdigit())
    return max(1, int(digits)) if digits else 1


def _sum_costs(costs: list[CostBreakdown], borrow: CostBreakdown) -> dict[str, float]:
    return {
        "commission": sum(cost.commission for cost in costs),
        "spread": sum(cost.spread for cost in costs),
        "slippage": sum(cost.slippage for cost in costs),
        "market_impact": sum(cost.market_impact for cost in costs),
        "borrow": borrow.borrow,
    }


def _scaled_cost(cost: CostBreakdown, scale: float) -> CostBreakdown:
    return CostBreakdown(
        commission=cost.commission * scale,
        spread=cost.spread * scale,
        slippage=cost.slippage * scale,
        market_impact=cost.market_impact * scale,
        borrow=cost.borrow * scale,
    )


def _execution_window(
    asof_ts: str,
    rows: list[dict[str, Any]],
    labels_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[str, str]:
    starts = {str(labels_by_key[_key(row)].get("label_start_ts") or "") for row in rows}
    ends = {str(labels_by_key[_key(row)].get("label_end_ts") or "") for row in rows}
    if "" in starts or "" in ends:
        raise ValueError("backtest labels require label_start_ts and label_end_ts")
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("all assets at a decision must share one execution window")
    start = next(iter(starts))
    end = next(iter(ends))
    if parse_utc(start) <= parse_utc(asof_ts):
        raise ValueError("backtest execution must begin strictly after the decision timestamp")
    _holding_days(start, end)
    return start, end


def _prediction_groups_with_complete_cross_sections(
    predictions: list[dict[str, Any]],
    labels_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep only wholly observed decision dates without selecting assets by outcomes."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["asof_ts"])].append(row)
    if not grouped:
        return {}
    final_prediction_ts = max(parse_utc(value) for value in grouped)
    complete: dict[str, list[dict[str, Any]]] = {}
    for asof_ts, rows in grouped.items():
        missing_labels = [_key(row) for row in rows if _key(row) not in labels_by_key]
        if missing_labels:
            raise ValueError(f"predictions have no matching labels at {asof_ts}: {missing_labels}")
        observed = [labels_by_key[_key(row)].get("forward_return") is not None for row in rows]
        if all(observed):
            complete[asof_ts] = rows
            continue
        if any(observed):
            raise ValueError(
                "partial forward-label coverage would select the investable cross-section "
                f"using future availability at {asof_ts}"
            )
        expected_ends = [
            str(labels_by_key[_key(row)].get("expected_label_end_ts") or "") for row in rows
        ]
        if "" in expected_ends or any(
            parse_utc(value) <= final_prediction_ts for value in expected_ends
        ):
            raise ValueError(f"non-terminal decision has no forward-label coverage at {asof_ts}")
        # Every asset is censored beyond the common trailing data boundary.  It is
        # safe to omit the whole decision date, never individual assets.
    return complete


def _known_liquidity(row: Mapping[str, Any], asset_id: str) -> float:
    """Return the decision-time daily dollar-volume proxy used for both legs."""

    value = finite_float(row.get("dollar_volume"))
    if value <= 0:
        raise ValueError(f"decision-time dollar_volume must be positive for {asset_id}")
    return value


def run_backtest(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    config: BacktestConfig,
) -> dict[str, Any]:
    """Replay prediction rows with constrained, cost-aware, deterministic accounting."""

    predictions = [conservative_risk_estimates(row, config) for row in predictions]
    labels_by_key = _unique_rows(labels, name="label")
    _unique_rows(predictions, name="prediction")
    predictions_by_time = _prediction_groups_with_complete_cross_sections(
        predictions, labels_by_key
    )

    constraints = constraints_from_config(config)
    model = cost_model_from_config(config)
    initial_capital = float(getattr(config, "initial_capital", 1_000_000.0))
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    metadata: dict[str, dict[str, Any]] = {}
    equity = 1.0
    equity_curve = [equity]
    periods: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    position_log: list[dict[str, Any]] = []
    horizons = {str(row["horizon"]) for rows in predictions_by_time.values() for row in rows}
    if len(horizons) > 1:
        raise ValueError("one backtest run must use a single prediction horizon")
    horizon_steps = _horizon_steps(next(iter(horizons), "1d"))
    all_times = sorted(predictions_by_time, key=parse_utc)
    times = all_times[::horizon_steps]
    for asof_ts in times:
        period_start_equity = initial_capital * equity
        rows = sorted(predictions_by_time[asof_ts], key=lambda row: str(row["asset_id"]))
        for row in rows:
            metadata[str(row["asset_id"])] = row
        execution_ts, exit_ts = _execution_window(asof_ts, rows, labels_by_key)
        before: dict[str, float] = {}
        # A one-session round trip enters and exits on the same trading day, so
        # reserve half of the daily turnover budget for each planned leg.  Longer
        # horizons apply the full budget separately on their distinct execution days.
        portfolio_constraints = constraints
        if horizon_steps == 1:
            portfolio_constraints = replace(
                constraints,
                max_daily_turnover=config.max_daily_turnover
                / (2.0 + config.same_day_exit_notional_buffer),
                max_participation_rate=constraints.max_participation_rate
                / (1.0 + config.same_day_exit_notional_buffer),
            )
        decision = construct_portfolio(
            rows,
            before,
            portfolio_constraints,
            equity=period_start_equity,
        )
        target = decision.target_weights
        before_exposure = calculate_exposures(before, metadata)
        target_exposure = calculate_exposures(target, metadata)

        execution_costs: list[CostBreakdown] = []
        entry_participation: dict[str, float] = {}
        for asset_id in sorted(target):
            target_weight = target[asset_id]
            row = metadata.get(asset_id, {})
            label = labels_by_key[_key(row)]
            execution_price = finite_float(label.get("execution_price"))
            if execution_price <= 0:
                raise ValueError(f"label execution_price must be positive for {asset_id}")
            liquidity = _known_liquidity(row, asset_id)
            participation = abs(target_weight) * period_start_equity / liquidity
            if participation > constraints.max_participation_rate + 1e-12:
                raise ValueError(
                    f"ex-ante next-open participation for {asset_id} exceeds max_participation_rate"
                )
            entry_participation[asset_id] = participation
            breakdown = cost_breakdown(
                abs(target_weight),
                model,
                volatility=_volatility(row),
                participation_rate=participation,
            )
            execution_costs.append(breakdown)
            reason_codes = row.get("reason_codes", [])
            trades.append(
                {
                    "asof_ts": execution_ts,
                    "decision_ts": asof_ts,
                    "execution_phase": "entry_next_session_open",
                    "asset_id": asset_id,
                    "symbol": row.get("symbol", asset_id),
                    "side": "BUY" if target_weight > 0 else "SELL",
                    "previous_weight": 0.0,
                    "target_weight": target_weight,
                    "delta_weight": target_weight,
                    "price": execution_price,
                    "liquidity_proxy_dollar_volume": liquidity,
                    "liquidity_proxy_asof_ts": asof_ts,
                    "participation_rate": participation,
                    "costs": breakdown.to_dict(),
                    "reason_codes": list(reason_codes) if isinstance(reason_codes, list) else [],
                    "risk_flags": sorted(
                        set(decision.rejected.get(asset_id, ())) | set(risk_estimate_flags(row))
                    ),
                }
            )

        holding_days = _holding_days(execution_ts, exit_ts)
        borrow_cost = cost_breakdown(
            0.0,
            model,
            short_exposure=sum(weight for weight in target.values() if weight < 0),
            holding_period_days=holding_days,
        )
        asset_returns = {
            str(row["asset_id"]): finite_float(labels_by_key[_key(row)]["forward_return"])
            for row in rows
        }
        missing_returns = sorted(
            asset_id
            for asset_id, weight in target.items()
            if weight and asset_id not in asset_returns
        )
        gross_return = sum(
            weight * asset_returns.get(asset_id, 0.0) for asset_id, weight in target.items()
        )
        pre_exit = drift_weights(target, asset_returns, portfolio_return=gross_return)
        exit_costs: list[CostBreakdown] = []
        exit_participation: dict[str, float] = {}
        gross_growth = max(0.0, 1.0 + gross_return)
        for asset_id in sorted(pre_exit):
            row = metadata.get(asset_id, {})
            label = labels_by_key[_key(row)]
            exit_price = finite_float(label.get("exit_price"))
            if exit_price <= 0:
                raise ValueError(f"label exit_price must be positive for {asset_id}")
            liquidity = _known_liquidity(row, asset_id)
            participation = abs(pre_exit[asset_id]) * period_start_equity * gross_growth / liquidity
            if participation > constraints.max_participation_rate + 1e-12:
                raise ValueError(
                    f"horizon-close exit for {asset_id} exceeds max_participation_rate"
                )
            exit_participation[asset_id] = participation
            raw_exit_cost = cost_breakdown(
                abs(pre_exit[asset_id]),
                model,
                volatility=_volatility(row),
                participation_rate=participation,
            )
            breakdown = _scaled_cost(raw_exit_cost, gross_growth)
            exit_costs.append(breakdown)
            trades.append(
                {
                    "asof_ts": exit_ts,
                    "decision_ts": asof_ts,
                    "execution_phase": "forced_horizon_exit",
                    "asset_id": asset_id,
                    "symbol": row.get("symbol", asset_id),
                    "side": "SELL" if pre_exit[asset_id] > 0 else "BUY",
                    "previous_weight": pre_exit[asset_id],
                    "target_weight": 0.0,
                    "delta_weight": -pre_exit[asset_id],
                    "price": exit_price,
                    "liquidity_proxy_dollar_volume": liquidity,
                    "liquidity_proxy_asof_ts": asof_ts,
                    "participation_rate": participation,
                    "costs": breakdown.to_dict(),
                    "reason_codes": ["horizon_liquidation"],
                    "risk_flags": list(risk_estimate_flags(row)),
                }
            )
        costs = _sum_costs([*execution_costs, *exit_costs], borrow_cost)
        total_cost = sum(costs.values())
        net_return = gross_return - total_cost
        equity *= max(0.0, 1.0 + net_return)
        equity_curve.append(equity)
        post_return_exposure = calculate_exposures(pre_exit, metadata)
        after_exposure = calculate_exposures({}, metadata)

        period_risk_flags = list(decision.risk_flags)
        for row in rows:
            if str(row["asset_id"]) in target:
                period_risk_flags.extend(risk_estimate_flags(row))
        if missing_returns:
            period_risk_flags.append("missing_forward_return")
        entry_turnover = decision.turnover
        exit_turnover_exit_nav = sum(abs(weight) for weight in pre_exit.values())
        exit_turnover_start_nav = exit_turnover_exit_nav * gross_growth
        round_trip_turnover = entry_turnover + exit_turnover_start_nav
        if horizon_steps == 1 and round_trip_turnover > constraints.max_daily_turnover + 1e-12:
            raise ValueError("realized same-session round-trip turnover exceeds max_daily_turnover")
        if horizon_steps > 1 and exit_turnover_exit_nav > constraints.max_daily_turnover + 1e-12:
            raise ValueError("realized horizon-exit turnover exceeds max_daily_turnover")
        capacity_values: list[float] = []
        for asset_id, weight in target.items():
            if abs(weight) <= 1e-12:
                continue
            liquidity = _known_liquidity(metadata.get(asset_id, {}), asset_id)
            capacity_values.append(constraints.max_participation_rate * liquidity / abs(weight))
            exit_notional_weight = gross_growth * abs(pre_exit.get(asset_id, 0.0))
            if exit_notional_weight > 1e-12:
                capacity_values.append(
                    constraints.max_participation_rate * liquidity / exit_notional_weight
                )
        periods.append(
            {
                "asof_ts": asof_ts,
                "execution_ts": execution_ts,
                "exit_ts": exit_ts,
                "gross_return": gross_return,
                "cost_return": total_cost,
                "net_return": net_return,
                "turnover": round_trip_turnover,
                "entry_turnover_start_nav": entry_turnover,
                "exit_turnover_exit_nav": exit_turnover_exit_nav,
                "exit_turnover_start_nav": exit_turnover_start_nav,
                "gross_exposure": target_exposure.gross,
                "net_exposure": target_exposure.net,
                "beta_exposure": target_exposure.beta,
                "sector_exposure": dict(sorted(target_exposure.sectors.items())),
                "pre_trade_gross_exposure": before_exposure.gross,
                "post_return_gross_exposure": post_return_exposure.gross,
                "post_liquidation_gross_exposure": after_exposure.gross,
                "capacity_proxy_equity": min(capacity_values) if capacity_values else 0.0,
                "max_participation_rate": max(
                    [*entry_participation.values(), *exit_participation.values()],
                    default=0.0,
                ),
                "commission_return": costs["commission"],
                "spread_return": costs["spread"],
                "slippage_return": costs["slippage"],
                "market_impact_return": costs["market_impact"],
                "borrow_return": costs["borrow"],
                "holding_period_days": holding_days,
                "rejected": {
                    asset_id: list(reasons)
                    for asset_id, reasons in sorted(decision.rejected.items())
                },
                "risk_flags": sorted(set(period_risk_flags)),
                "missing_return_assets": missing_returns,
                "equity": equity,
            }
        )
        for asset_id in sorted(set(target) | set(pre_exit)):
            row = metadata.get(asset_id, {})
            position_log.append(
                {
                    "asof_ts": asof_ts,
                    "asset_id": asset_id,
                    "symbol": row.get("symbol", asset_id),
                    "sector": row.get("sector", "UNKNOWN"),
                    "previous_weight": before.get(asset_id, 0.0),
                    "target_weight": target.get(asset_id, 0.0),
                    "post_return_weight": pre_exit.get(asset_id, 0.0),
                    "final_weight": 0.0,
                    "forward_return": asset_returns.get(asset_id),
                    "gross_contribution": target.get(asset_id, 0.0)
                    * asset_returns.get(asset_id, 0.0),
                }
            )

    metrics = summarize_backtest(
        periods,
        equity_curve,
        trades,
        periods_per_year=252.0 / horizon_steps,
    )
    return {
        "metrics": metrics,
        "periods": periods,
        "trades": trades,
        "positions": position_log,
        "final_positions": {},
        "assumptions": {
            "initial_capital": initial_capital,
            "execution": "decision after session close, entry at next session open, forced exit "
            "at the configured horizon close",
            "position_accounting": "independent non-overlapping round trips; no overnight "
            "exposure before the next-session entry",
            "overlapping_labels": "multi-session horizons use non-overlapping rebalances",
            "label_coverage": "only whole trailing decision dates with no realized horizon are "
            "omitted; partial cross-sectional label coverage fails the run",
            "liquidity_proxy": "both entry and exit use dollar volume observed at the decision "
            "close; future session volume is not used for sizing or costs",
            "turnover_denominator": "entry and total round-trip turnover use period-start NAV; "
            "exit_turnover_exit_nav also reports the contemporaneous pre-exit NAV basis",
            "same_day_exit_notional_buffer": config.same_day_exit_notional_buffer,
            "unmodeled": [
                "queue priority",
                "intraday path between decision timestamps",
                "forced locate recalls",
            ],
        },
    }


class DeterministicBacktestEngine:
    """Local implementation of the research-only backtest provider contract."""

    def run(
        self,
        predictions: list[dict[str, Any]],
        labels: list[dict[str, Any]],
        config: BacktestConfig,
    ) -> dict[str, Any]:
        return run_backtest(predictions, labels, config)

    def summarize(self, result: Mapping[str, Any]) -> dict[str, Any]:
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("backtest result is missing metrics")
        return {str(key): value for key, value in metrics.items()}
