from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nlp_trader.config import BacktestConfig
from nlp_trader.features.build import finite_float
from nlp_trader.portfolio.constraints import PortfolioConstraints


@dataclass(frozen=True, slots=True)
class ExposureSnapshot:
    gross: float
    net: float
    beta: float
    sectors: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gross": self.gross,
            "net": self.net,
            "beta": self.beta,
            "sectors": dict(sorted(self.sectors.items())),
        }


def conservative_risk_estimates(row: Mapping[str, Any], config: BacktestConfig) -> dict[str, Any]:
    """Return a row with finite, conservative beta and volatility estimates."""

    resolved = dict(row)

    def finite(value: object, *, non_negative: bool = False) -> float | None:
        try:
            number = float(str(value))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or (non_negative and number < 0):
            return None
        return number

    beta = finite(resolved.get("beta"))
    beta_missing = bool(resolved.get("market_beta_60d_missing", False)) or beta is None
    if beta_missing:
        beta = config.missing_beta_fallback
    resolved["beta"] = beta
    resolved["beta_fallback_used"] = beta_missing

    volatility = finite(resolved.get("volatility"), non_negative=True)
    volatility_missing = (
        bool(resolved.get("realized_volatility_20d_missing", False)) or volatility is None
    )
    if volatility_missing:
        candidates = [config.missing_volatility_floor]
        for field in ("realized_volatility_3d", "high_low_volatility_20d"):
            candidate = finite(resolved.get(field), non_negative=True)
            if candidate is not None:
                candidates.append(candidate)
        volatility = max(candidates)
    resolved["volatility"] = volatility
    resolved["volatility_fallback_used"] = volatility_missing
    return resolved


def risk_estimate_flags(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return auditable flags for conservative missing-risk substitutions."""

    flags: list[str] = []
    if bool(row.get("beta_fallback_used", False)):
        flags.append("missing_beta_conservative_fallback")
    if bool(row.get("volatility_fallback_used", False)):
        flags.append("missing_volatility_conservative_fallback")
    return tuple(flags)


def _beta(row: dict[str, Any]) -> float:
    if "beta" in row:
        return finite_float(row["beta"])
    if "market_beta" in row:
        return finite_float(row["market_beta"])
    return 1.0


def calculate_exposures(
    weights: dict[str, float], metadata: dict[str, dict[str, Any]]
) -> ExposureSnapshot:
    sector_exposure: dict[str, float] = {}
    for asset_id, weight in weights.items():
        row = metadata.get(asset_id, {})
        sector = str(row.get("sector") or "UNKNOWN")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(weight)
    return ExposureSnapshot(
        gross=sum(abs(weight) for weight in weights.values()),
        net=sum(weights.values()),
        beta=sum(
            weight * _beta(metadata.get(asset_id, {})) for asset_id, weight in weights.items()
        ),
        sectors=sector_exposure,
    )


def constraint_violations(
    weights: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    constraints: PortfolioConstraints,
    *,
    tolerance: float = 1e-10,
) -> tuple[str, ...]:
    exposure = calculate_exposures(weights, metadata)
    violations: list[str] = []
    if any(
        abs(weight) > constraints.max_position_weight + tolerance for weight in weights.values()
    ):
        violations.append("max_position_weight")
    if exposure.gross > constraints.max_gross_exposure + tolerance:
        violations.append("max_gross_exposure")
    if abs(exposure.net) > constraints.max_net_exposure + tolerance:
        violations.append("max_net_exposure")
    if abs(exposure.beta) > constraints.max_beta_exposure + tolerance:
        violations.append("max_beta_exposure")
    if any(
        sector_weight > constraints.max_sector_weight + tolerance
        for sector_weight in exposure.sectors.values()
    ):
        violations.append("max_sector_weight")
    return tuple(violations)


def drift_weights(
    weights: dict[str, float],
    asset_returns: dict[str, float],
    *,
    portfolio_return: float | None = None,
) -> dict[str, float]:
    """Mark fractional positions to market after returns and cash costs."""

    if portfolio_return is None:
        portfolio_return = sum(
            weight * finite_float(asset_returns.get(asset_id, 0.0))
            for asset_id, weight in weights.items()
        )
    denominator = 1.0 + portfolio_return
    if denominator <= 0:
        return {}
    return {
        asset_id: weight * (1.0 + finite_float(asset_returns.get(asset_id, 0.0))) / denominator
        for asset_id, weight in weights.items()
        if abs(weight) > 1e-12
    }
