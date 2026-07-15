from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from nlp_trader.backtest.costs import (
    CostBreakdown,
    cost_breakdown,
    cost_model_from_config,
)
from nlp_trader.config import BacktestConfig
from nlp_trader.features.build import finite_float
from nlp_trader.paper.ledger import PaperEventLedger
from nlp_trader.portfolio.constraints import constraints_from_config
from nlp_trader.portfolio.construction import constrain_target_weights
from nlp_trader.portfolio.risk import (
    calculate_exposures,
    conservative_risk_estimates,
    drift_weights,
    risk_estimate_flags,
)
from nlp_trader.timestamps import format_utc, parse_utc


@dataclass(frozen=True, slots=True)
class PaperOrderIntent:
    strategy_id: str
    asof_ts: str
    asset_id: str
    symbol: str
    target_weight: float
    side: str
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        parse_utc(self.asof_ts)
        normalized_side = self.side.upper()
        if self.target_weight < 0 and normalized_side not in {"SELL", "SHORT"}:
            raise ValueError("negative target weight requires SELL or SHORT side")
        if self.target_weight > 0 and normalized_side in {"SHORT"}:
            raise ValueError("positive target weight cannot use SHORT side")

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> PaperOrderIntent:
        target = finite_float(row.get("target_weight"))
        side = str(row.get("side") or ("BUY" if target > 0 else "SHORT" if target < 0 else "FLAT"))
        reasons = row.get("reason_codes", [])
        return cls(
            strategy_id=str(row["strategy_id"]),
            asof_ts=str(row["asof_ts"]),
            asset_id=str(row.get("asset_id") or row["symbol"]),
            symbol=str(row["symbol"]),
            target_weight=target,
            side=side,
            reason_codes=tuple(str(reason) for reason in reasons)
            if isinstance(reasons, list)
            else (),
        )


class PaperSimulator:
    """Deterministic in-memory simulator; it deliberately exposes no broker adapter."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        initial_capital: float = 1_000_000.0,
        ledger: PaperEventLedger | None = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if ledger is not None and ledger.replay():
            raise ValueError(
                "PaperSimulator requires an empty ledger; resuming existing paper state "
                "is not supported"
            )
        self._constraints = constraints_from_config(config)
        self._cost_model = cost_model_from_config(config)
        self._config = config
        self._initial_capital = float(initial_capital)
        self._equity = float(initial_capital)
        self._positions: dict[str, float] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._last_ts: str | None = None
        self._events: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []
        self._ledger = ledger

    def _check_time(self, asof_ts: str) -> str:
        current = parse_utc(asof_ts)
        if self._last_ts is not None and current < parse_utc(self._last_ts):
            raise ValueError("paper events must be submitted in timestamp order")
        return format_utc(current)

    def _persist_event(self, event: dict[str, Any]) -> dict[str, Any]:
        if self._ledger is None:
            return event
        return self._ledger.append(event)

    @staticmethod
    def _volatility(row: dict[str, Any]) -> float:
        return max(
            0.0,
            finite_float(row.get("volatility", row.get("realized_volatility_3d", 0.0))),
        )

    def rebalance(
        self,
        intents: Sequence[PaperOrderIntent | dict[str, Any]],
        market_rows: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        parsed = [
            intent if isinstance(intent, PaperOrderIntent) else PaperOrderIntent.from_record(intent)
            for intent in intents
        ]
        if not parsed:
            raise ValueError("paper rebalance requires at least one intent")
        asof_times = {format_utc(parse_utc(intent.asof_ts)) for intent in parsed}
        if len(asof_times) != 1:
            raise ValueError("all paper intents in a rebalance must share asof_ts")
        asof_ts = self._check_time(next(iter(asof_times)))

        rows_by_asset: dict[str, dict[str, Any]] = {}
        next_metadata = dict(self._metadata)
        for row in market_rows:
            asset_id = str(row["asset_id"])
            if asset_id in rows_by_asset:
                raise ValueError(f"duplicate paper market row: {asset_id}")
            resolved = conservative_risk_estimates(row, self._config)
            rows_by_asset[asset_id] = resolved
            next_metadata[asset_id] = resolved
        desired: dict[str, float] = {}
        reasons: dict[str, tuple[str, ...]] = {}
        for intent in parsed:
            if intent.asset_id in desired:
                raise ValueError(f"duplicate paper intent: {intent.asset_id}")
            desired[intent.asset_id] = intent.target_weight
            reasons[intent.asset_id] = intent.reason_codes
        for asset_id in self._positions:
            if asset_id not in rows_by_asset and asset_id in next_metadata:
                rows_by_asset[asset_id] = next_metadata[asset_id]

        decision = constrain_target_weights(
            desired,
            list(rows_by_asset.values()),
            self._positions,
            self._constraints,
            equity=self._equity,
        )
        execution_costs: list[CostBreakdown] = []
        trade_records: list[dict[str, Any]] = []
        for asset_id in sorted(set(self._positions) | set(decision.target_weights)):
            previous = self._positions.get(asset_id, 0.0)
            target = decision.target_weights.get(asset_id, 0.0)
            delta = target - previous
            if abs(delta) <= 1e-12:
                continue
            row = next_metadata.get(asset_id, {})
            costs = cost_breakdown(
                abs(delta),
                self._cost_model,
                volatility=self._volatility(row),
                participation_rate=decision.participation.get(asset_id, 0.0),
            )
            execution_costs.append(costs)
            trade = {
                "asof_ts": asof_ts,
                "asset_id": asset_id,
                "symbol": row.get("symbol", asset_id),
                "side": "BUY" if delta > 0 else "SELL",
                "previous_weight": previous,
                "target_weight": target,
                "delta_weight": delta,
                "participation_rate": decision.participation.get(asset_id, 0.0),
                "costs": costs.to_dict(),
                "reason_codes": list(reasons.get(asset_id, ())),
                "risk_flags": sorted(
                    set(decision.rejected.get(asset_id, ())) | set(risk_estimate_flags(row))
                ),
                "simulation_only": True,
            }
            trade_records.append(trade)
        cost_return = sum(cost.total for cost in execution_costs)
        next_equity = self._equity * max(0.0, 1.0 - cost_return)
        next_positions = dict(decision.target_weights)
        exposure = calculate_exposures(next_positions, next_metadata)
        event = {
            "event_type": "paper_rebalance",
            "asof_ts": asof_ts,
            "simulation_only": True,
            "cost_return": cost_return,
            "equity": next_equity,
            "turnover": decision.turnover,
            "exposures": exposure.to_dict(),
            "rejected": {
                asset_id: list(flags) for asset_id, flags in sorted(decision.rejected.items())
            },
            "risk_flags": sorted(
                set(decision.risk_flags)
                | {
                    flag
                    for asset_id in decision.target_weights
                    for flag in risk_estimate_flags(next_metadata.get(asset_id, {}))
                }
            ),
            "trades": trade_records,
        }
        persisted = self._persist_event(event)
        self._equity = next_equity
        self._positions = next_positions
        self._metadata = next_metadata
        self._last_ts = asof_ts
        self._trades.extend(trade_records)
        self._events.append(persisted)
        return persisted

    def mark_to_market(self, asof_ts: str, asset_returns: dict[str, float]) -> dict[str, Any]:
        asof_ts = self._check_time(asof_ts)
        gross_return = sum(
            weight * finite_float(asset_returns.get(asset_id, 0.0))
            for asset_id, weight in self._positions.items()
        )
        next_equity = self._equity * max(0.0, 1.0 + gross_return)
        next_positions = drift_weights(
            self._positions,
            asset_returns,
            portfolio_return=gross_return,
        )
        exposure = calculate_exposures(next_positions, self._metadata)
        event = {
            "event_type": "paper_mark_to_market",
            "asof_ts": asof_ts,
            "simulation_only": True,
            "gross_return": gross_return,
            "equity": next_equity,
            "exposures": exposure.to_dict(),
        }
        persisted = self._persist_event(event)
        self._equity = next_equity
        self._positions = next_positions
        self._last_ts = asof_ts
        self._events.append(persisted)
        return persisted

    def snapshot(self) -> dict[str, Any]:
        return {
            "simulation_only": True,
            "initial_capital": self._initial_capital,
            "equity": self._equity,
            "total_return": self._equity / self._initial_capital - 1.0,
            "positions": dict(sorted(self._positions.items())),
            "events": list(self._events),
            "trades": list(self._trades),
        }
