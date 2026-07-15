from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nlp_trader.features.build import finite_float
from nlp_trader.portfolio.constraints import PortfolioConstraints
from nlp_trader.portfolio.risk import calculate_exposures, constraint_violations


@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    target_weights: dict[str, float]
    turnover: float
    participation: dict[str, float]
    rejected: dict[str, tuple[str, ...]]
    risk_flags: tuple[str, ...]


def _metadata_by_asset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        asset_id = str(row["asset_id"])
        if asset_id in metadata:
            raise ValueError(f"duplicate portfolio candidate: {asset_id}")
        metadata[asset_id] = row
    return metadata


def _eligibility_reasons(
    weight: float, row: dict[str, Any] | None, constraints: PortfolioConstraints
) -> list[str]:
    if row is None:
        return ["missing_market_data"]
    reasons: list[str] = []
    if finite_float(row.get("close")) < constraints.min_price:
        reasons.append("min_price")
    if finite_float(row.get("dollar_volume")) < constraints.min_dollar_volume:
        reasons.append("min_dollar_volume")
    if weight < 0:
        if not constraints.shorting_allowed:
            reasons.append("shorting_disabled")
        if not bool(row.get("short_available", False)):
            reasons.append("short_unavailable")
        if bool(row.get("hard_to_borrow", False)) and not constraints.hard_to_borrow_allowed:
            reasons.append("hard_to_borrow")
    return reasons


def _scale_sector_weights(
    target: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    max_sector_weight: float,
) -> None:
    by_sector: dict[str, list[str]] = {}
    for asset_id in target:
        sector = str(metadata.get(asset_id, {}).get("sector") or "UNKNOWN")
        by_sector.setdefault(sector, []).append(asset_id)
    for asset_ids in by_sector.values():
        gross = sum(abs(target[asset_id]) for asset_id in asset_ids)
        if gross > max_sector_weight:
            scale = max_sector_weight / gross
            for asset_id in asset_ids:
                target[asset_id] *= scale


def _scale_net_weights(target: dict[str, float], max_net: float) -> None:
    long_gross = sum(weight for weight in target.values() if weight > 0)
    short_gross = -sum(weight for weight in target.values() if weight < 0)
    net = long_gross - short_gross
    if net > max_net and long_gross:
        allowed_long = max(0.0, max_net + short_gross)
        scale = min(1.0, allowed_long / long_gross)
        for asset_id, weight in target.items():
            if weight > 0:
                target[asset_id] = weight * scale
    elif net < -max_net and short_gross:
        allowed_short = max(0.0, max_net + long_gross)
        scale = min(1.0, allowed_short / short_gross)
        for asset_id, weight in target.items():
            if weight < 0:
                target[asset_id] = weight * scale


def _homogeneous_exposure_scale(
    target: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    constraints: PortfolioConstraints,
) -> float:
    """Return a zero-anchored scale that makes every homogeneous exposure safe."""

    exposure = calculate_exposures(target, metadata)
    scales = [1.0]

    def add_limit(observed: float, limit: float) -> None:
        if observed > limit and observed > 0:
            scales.append(limit / observed)

    for weight in target.values():
        add_limit(abs(weight), constraints.max_position_weight)
    add_limit(exposure.gross, constraints.max_gross_exposure)
    add_limit(abs(exposure.net), constraints.max_net_exposure)
    add_limit(abs(exposure.beta), constraints.max_beta_exposure)
    for sector_gross in exposure.sectors.values():
        add_limit(sector_gross, constraints.max_sector_weight)
    return max(0.0, min(scales))


def constrain_target_weights(
    desired_weights: dict[str, float],
    rows: list[dict[str, Any]],
    current_weights: dict[str, float],
    constraints: PortfolioConstraints,
    *,
    equity: float,
) -> PortfolioDecision:
    """Apply eligibility, exposure, turnover, and participation limits deterministically."""

    metadata = _metadata_by_asset(rows)
    rejected: dict[str, tuple[str, ...]] = {}
    target: dict[str, float] = {}
    for asset_id, requested in sorted(desired_weights.items()):
        weight = max(
            -constraints.max_position_weight, min(constraints.max_position_weight, requested)
        )
        reasons = _eligibility_reasons(weight, metadata.get(asset_id), constraints)
        if reasons and abs(weight) > 1e-12:
            rejected[asset_id] = tuple(reasons)
            continue
        if abs(weight) > 1e-12:
            target[asset_id] = weight

    _scale_sector_weights(target, metadata, constraints.max_sector_weight)
    gross = sum(abs(weight) for weight in target.values())
    if gross > constraints.max_gross_exposure:
        scale = constraints.max_gross_exposure / gross
        target = {asset_id: weight * scale for asset_id, weight in target.items()}
    _scale_net_weights(target, constraints.max_net_exposure)
    beta = calculate_exposures(target, metadata).beta
    if abs(beta) > constraints.max_beta_exposure:
        scale = constraints.max_beta_exposure / abs(beta)
        target = {asset_id: weight * scale for asset_id, weight in target.items()}

    all_assets = set(current_weights) | set(target)
    raw_turnover = sum(
        abs(target.get(asset_id, 0.0) - current_weights.get(asset_id, 0.0))
        for asset_id in all_assets
    )
    turnover_scale = (
        min(1.0, constraints.max_daily_turnover / raw_turnover) if raw_turnover else 1.0
    )
    limited = {
        asset_id: current_weights.get(asset_id, 0.0)
        + (target.get(asset_id, 0.0) - current_weights.get(asset_id, 0.0)) * turnover_scale
        for asset_id in all_assets
    }

    participation: dict[str, float] = {}
    for asset_id in sorted(all_assets):
        previous = current_weights.get(asset_id, 0.0)
        delta = limited.get(asset_id, 0.0) - previous
        if abs(delta) <= 1e-12:
            participation[asset_id] = 0.0
            continue
        dollar_volume = finite_float(metadata.get(asset_id, {}).get("dollar_volume"))
        if dollar_volume <= 0:
            limited[asset_id] = previous
            participation[asset_id] = 0.0
            rejected[asset_id] = tuple(
                sorted(set(rejected.get(asset_id, ())) | {"missing_dollar_volume"})
            )
            continue
        requested_participation = abs(delta) * max(0.0, equity) / dollar_volume
        if requested_participation > constraints.max_participation_rate:
            max_delta = constraints.max_participation_rate * dollar_volume / max(equity, 1e-12)
            limited[asset_id] = previous + (max_delta if delta > 0 else -max_delta)
            participation[asset_id] = constraints.max_participation_rate
        else:
            participation[asset_id] = requested_participation

    limited = {asset_id: weight for asset_id, weight in limited.items() if abs(weight) > 1e-12}
    # Per-asset participation clipping is asymmetric and can destroy a beta/net hedge that was
    # valid before clipping. Re-project toward cash; this cannot increase absolute target weights.
    # For a non-flat starting book it can increase a trade delta, so the execution constraints are
    # recomputed below and fail closed if the corrected target is not reachable safely.
    exposure_scale = _homogeneous_exposure_scale(limited, metadata, constraints)
    if exposure_scale < 1.0:
        limited = {
            asset_id: weight * exposure_scale
            for asset_id, weight in limited.items()
            if abs(weight * exposure_scale) > 1e-12
        }
    turnover = sum(
        abs(limited.get(asset_id, 0.0) - current_weights.get(asset_id, 0.0))
        for asset_id in set(current_weights) | set(limited)
    )
    if turnover > constraints.max_daily_turnover + 1e-10:
        raise ValueError(
            "post-participation exposure correction exceeds max_daily_turnover; "
            "the requested target cannot be reached safely"
        )
    participation = {}
    for asset_id in sorted(set(current_weights) | set(limited)):
        delta = limited.get(asset_id, 0.0) - current_weights.get(asset_id, 0.0)
        if abs(delta) <= 1e-12:
            participation[asset_id] = 0.0
            continue
        dollar_volume = finite_float(metadata.get(asset_id, {}).get("dollar_volume"))
        if dollar_volume <= 0:
            raise ValueError(
                f"post-participation exposure correction lacks dollar volume for {asset_id}"
            )
        actual = abs(delta) * max(0.0, equity) / dollar_volume
        if actual > constraints.max_participation_rate + 1e-10:
            raise ValueError(
                "post-participation exposure correction exceeds max_participation_rate "
                f"for {asset_id}; the requested target cannot be reached safely"
            )
        participation[asset_id] = actual
    risk_flags = constraint_violations(limited, metadata, constraints)
    if risk_flags:
        raise ValueError(
            "portfolio construction could not enforce constraints after participation: "
            + ", ".join(risk_flags)
        )
    return PortfolioDecision(limited, turnover, participation, rejected, risk_flags)


def construct_portfolio(
    rows: list[dict[str, Any]],
    current_weights: dict[str, float],
    constraints: PortfolioConstraints,
    *,
    equity: float,
) -> PortfolioDecision:
    """Convert ranked model scores into constrained target weights."""

    metadata = _metadata_by_asset(rows)
    desired: dict[str, float] = {}
    ranked = sorted(
        metadata.items(),
        key=lambda pair: (-abs(finite_float(pair[1].get("score"))), pair[0]),
    )
    for asset_id, row in ranked:
        score = finite_float(row.get("score"))
        if score > 0:
            desired[asset_id] = constraints.max_position_weight
        elif score < 0 and constraints.shorting_allowed:
            desired[asset_id] = -constraints.max_position_weight
    return constrain_target_weights(desired, rows, current_weights, constraints, equity=equity)
