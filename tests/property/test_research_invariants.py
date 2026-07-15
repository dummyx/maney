from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nlp_trader.backtest import CostModel, cost_breakdown
from nlp_trader.data.stores import PointInTimeFeatureStore, PointInTimeViolation
from nlp_trader.portfolio import (
    PortfolioConstraints,
    calculate_exposures,
    constrain_target_weights,
)

_ASSET_IDS = tuple(f"asset_{index}" for index in range(6))
_FINITE_WEIGHT = st.floats(
    min_value=-1.5,
    max_value=1.5,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_DESIRED_WEIGHTS = st.dictionaries(
    keys=st.sampled_from(_ASSET_IDS),
    values=_FINITE_WEIGHT,
    max_size=len(_ASSET_IDS),
)


def _non_negative_float(max_value: float) -> st.SearchStrategy[float]:
    return st.floats(
        min_value=0.0,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


@settings(max_examples=40, deadline=None)
@given(
    turnover=_non_negative_float(1.0),
    turnover_increase=_non_negative_float(1.0),
    volatility=_non_negative_float(0.10),
    volatility_increase=_non_negative_float(0.10),
    participation=_non_negative_float(0.50),
    participation_increase=_non_negative_float(0.50),
    short_exposure=_non_negative_float(1.0),
    short_exposure_increase=_non_negative_float(1.0),
    holding_days=_non_negative_float(365.0),
    holding_days_increase=_non_negative_float(365.0),
)
def test_costs_are_non_negative_and_monotone_under_worse_execution_conditions(
    turnover: float,
    turnover_increase: float,
    volatility: float,
    volatility_increase: float,
    participation: float,
    participation_increase: float,
    short_exposure: float,
    short_exposure_increase: float,
    holding_days: float,
    holding_days_increase: float,
) -> None:
    model = CostModel(
        commission_bps=1.0,
        half_spread_bps=2.0,
        slippage_bps=3.0,
        borrow_bps_per_year=365.0,
        volatility_slippage_multiplier=0.05,
        participation_slippage_bps=50.0,
        market_impact_multiplier=0.10,
    )
    baseline = cost_breakdown(
        turnover,
        model,
        volatility=volatility,
        participation_rate=participation,
        short_exposure=-short_exposure,
        holding_period_days=holding_days,
    )
    stressed = cost_breakdown(
        turnover + turnover_increase,
        model,
        volatility=volatility + volatility_increase,
        participation_rate=participation + participation_increase,
        short_exposure=-(short_exposure + short_exposure_increase),
        holding_period_days=holding_days + holding_days_increase,
    )

    assert all(value >= 0.0 for value in baseline.to_dict().values())
    assert stressed.total >= baseline.total - 1e-12


@settings(max_examples=60, deadline=None)
@given(desired_weights=_DESIRED_WEIGHTS)
def test_constrained_targets_are_deterministic_and_respect_every_exposure_limit(
    desired_weights: dict[str, float],
) -> None:
    constraints = PortfolioConstraints(
        max_position_weight=0.30,
        max_gross_exposure=0.90,
        max_net_exposure=0.35,
        max_sector_weight=0.50,
        max_beta_exposure=0.35,
        max_daily_turnover=0.75,
        max_participation_rate=0.05,
        min_price=1.0,
        min_dollar_volume=1_000_000.0,
        shorting_allowed=True,
        hard_to_borrow_allowed=False,
    )
    rows = [
        {
            "asset_id": asset_id,
            "symbol": asset_id.upper(),
            "close": 20.0,
            # Low enough that generated large targets exercise participation clipping.
            "dollar_volume": 2_000_000.0,
            "sector": f"sector_{index % 2}",
            "beta": 1.0,
            "short_available": True,
            "hard_to_borrow": False,
        }
        for index, asset_id in enumerate(sorted(desired_weights))
    ]

    decision = constrain_target_weights(
        desired_weights,
        rows,
        {},
        constraints,
        equity=1_000_000.0,
    )
    repeated = constrain_target_weights(
        desired_weights,
        rows,
        {},
        constraints,
        equity=1_000_000.0,
    )
    metadata = {str(row["asset_id"]): row for row in rows}
    exposure = calculate_exposures(decision.target_weights, metadata)
    tolerance = 1e-10

    assert repeated == decision
    assert all(
        abs(weight) <= constraints.max_position_weight + tolerance
        for weight in decision.target_weights.values()
    )
    assert exposure.gross <= constraints.max_gross_exposure + tolerance
    assert abs(exposure.net) <= constraints.max_net_exposure + tolerance
    assert abs(exposure.beta) <= constraints.max_beta_exposure + tolerance
    assert all(
        sector_gross <= constraints.max_sector_weight + tolerance
        for sector_gross in exposure.sectors.values()
    )
    assert decision.turnover <= constraints.max_daily_turnover + tolerance
    assert all(
        rate <= constraints.max_participation_rate + tolerance
        for rate in decision.participation.values()
    )


@settings(max_examples=40, deadline=None)
@given(
    asof_offset_seconds=st.integers(min_value=0, max_value=365 * 86_400),
    past_lag_seconds=st.integers(min_value=0, max_value=7 * 86_400),
    future_lead_seconds=st.integers(min_value=1, max_value=7 * 86_400),
)
def test_feature_provenance_rejects_every_input_after_the_decision_time(
    asof_offset_seconds: int,
    past_lag_seconds: int,
    future_lead_seconds: int,
) -> None:
    store = PointInTimeFeatureStore(Path("unused-for-validation"))
    asof_ts = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=asof_offset_seconds)
    known_at = asof_ts - timedelta(seconds=past_lag_seconds)
    safe_record = {
        "asset_id": "asset_0",
        "symbol": "ASSET_0",
        "asof_ts": asof_ts,
        "horizon": "1d",
        "feature_set_version": "property-test-v1",
        "input_available_at": [known_at],
        "fundamental_available_at": known_at,
    }

    store.validate_point_in_time([safe_record])

    future_record = {
        **safe_record,
        "latest_text_available_at_1d": asof_ts + timedelta(seconds=future_lead_seconds),
    }
    with pytest.raises(PointInTimeViolation, match="after asof_ts"):
        store.validate_point_in_time([future_record])
