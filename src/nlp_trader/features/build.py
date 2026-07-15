from __future__ import annotations

import math
import re
import statistics
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from nlp_trader.calendars import USEquityCalendar
from nlp_trader.config import ResearchConfig
from nlp_trader.market_timing import (
    market_bar_available_at,
    market_decision_time_for_bar,
    market_decision_times_by_session,
)
from nlp_trader.schemas import (
    Asset,
    CorporateAction,
    EarningsCalendarEvent,
    FundamentalRecord,
    MarketBar,
    TextSignal,
)
from nlp_trader.timestamps import format_utc, parse_utc

RETURN_LOOKBACKS = (1, 3, 5, 20, 60)
VOLATILITY_LOOKBACKS = (3, 20, 60)


def _return(now: float, before: float) -> float:
    if before == 0:
        return 0.0
    return now / before - 1.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(statistics.fmean(value * value for value in values))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if not values or total <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def _price_series(bars: list[MarketBar]) -> tuple[list[float], str]:
    """Use only point-in-time-safe, consistently adjusted tradable OHLC fields."""

    if any(
        not bar.corporate_action_adjusted
        or bar.adjustment_vintage_at is None
        or bar.adjustment_vintage_at > market_bar_available_at(bar)
        for bar in bars
    ):
        raise ValueError(
            "daily baseline requires point-in-time corporate-action-adjusted OHLC with "
            "adjustment_vintage_at <= market availability for every bar"
        )
    return [
        bar.close * bar.return_adjustment_factor for bar in bars
    ], "causal_return_adjustment_factor"


def _period_returns(prices: list[float]) -> list[float]:
    return [_return(prices[index], prices[index - 1]) for index in range(1, len(prices))]


def _trailing_returns(prices: list[float], index: int, lookback: int) -> list[float]:
    start = max(1, index - lookback + 1)
    return [_return(prices[position], prices[position - 1]) for position in range(start, index + 1)]


def _lookback_return(prices: list[float], index: int, lookback: int) -> tuple[float, bool]:
    if index < lookback:
        return 0.0, True
    return _return(prices[index], prices[index - lookback]), False


def _covariance(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    return statistics.fmean(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _beta(asset_returns: list[float], market_returns: list[float]) -> tuple[float, bool]:
    if len(asset_returns) < 2 or len(asset_returns) != len(market_returns):
        return 0.0, True
    market_variance = _covariance(market_returns, market_returns)
    if market_variance <= 0.0:
        return 0.0, True
    return _covariance(asset_returns, market_returns) / market_variance, False


def _event_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "other"


def _daily_bar_calendar(
    bars: list[MarketBar],
    config: ResearchConfig,
) -> USEquityCalendar | None:
    """Validate that complete daily OHLC rows are timestamped at official closes."""

    if not bars:
        return None
    invalid_sizes = sorted({bar.bar_size for bar in bars if bar.bar_size != "1d"})
    if invalid_sizes:
        raise ValueError(
            "daily feature and label builders require bar_size=1d; got " + ", ".join(invalid_sizes)
        )
    if any(bar.ts.tzinfo is None for bar in bars):
        raise ValueError("daily market-bar timestamps must be timezone-aware")
    session_dates = [bar.ts.astimezone(UTC).date() for bar in bars]
    availability_dates = [market_bar_available_at(bar).date() for bar in bars]
    padding_days = max(14, config.features.horizon_days * 4 + 14)
    calendar = USEquityCalendar(
        calendar_name=config.data.calendar,
        start=min(session_dates) - timedelta(days=14),
        end=max(max(session_dates), max(availability_dates)) + timedelta(days=padding_days),
    )
    for bar in bars:
        session_date = bar.ts.astimezone(calendar.timezone).date()
        if not calendar.is_session(session_date):
            raise ValueError(
                f"daily market bar {bar.symbol} at {format_utc(bar.ts)} is not an "
                f"{config.data.calendar} session"
            )
        expected_close = calendar.session_close(session_date)
        if bar.ts.astimezone(UTC) != expected_close:
            raise ValueError(
                f"daily market bar {bar.symbol} must be timestamped at the official "
                f"session close {format_utc(expected_close)}, got {format_utc(bar.ts)}"
            )
    return calendar


def _label_session_closes(
    calendar: USEquityCalendar,
    asof_ts: datetime,
    count: int,
) -> tuple[datetime, tuple[datetime, ...]]:
    """Return the first strictly future open and its inclusive horizon closes."""

    entry_open = calendar.next_open_decision_time(asof_ts)
    if entry_open <= asof_ts.astimezone(UTC):
        entry_date = entry_open.astimezone(calendar.timezone).date()
        entry_open = calendar.session_open(calendar.next_session(entry_date))
    session_date = entry_open.astimezone(calendar.timezone).date()
    values: list[datetime] = []
    for index in range(count):
        values.append(calendar.session_close(session_date))
        if index + 1 < count:
            session_date = calendar.next_session(session_date)
    return entry_open, tuple(values)


def _validate_contiguous_sessions(
    asset_id: str,
    bars: list[MarketBar],
    calendar: USEquityCalendar,
) -> None:
    closes = [bar.ts.astimezone(UTC) for bar in bars]
    if len(closes) != len(set(closes)):
        raise ValueError(f"duplicate market-bar timestamp for asset {asset_id}")
    for previous, current in zip(bars, bars[1:], strict=False):
        previous_date = previous.ts.astimezone(calendar.timezone).date()
        current_date = current.ts.astimezone(calendar.timezone).date()
        expected = calendar.next_session(previous_date)
        if current_date != expected:
            raise ValueError(
                f"missing daily market session for asset {asset_id}: expected "
                f"{expected.isoformat()} after {previous_date.isoformat()}, got "
                f"{current_date.isoformat()}"
            )


def _index_assets(assets: list[Asset]) -> dict[str, Asset]:
    assets_by_id: dict[str, Asset] = {}
    for master_asset in assets:
        if master_asset.asset_id in assets_by_id:
            raise ValueError(f"duplicate asset_id in asset master: {master_asset.asset_id}")
        assets_by_id[master_asset.asset_id] = master_asset
    return assets_by_id


def _asset_is_active(asset: Asset, session_date: date) -> bool:
    return (asset.active_from is None or session_date >= asset.active_from) and (
        asset.active_to is None or session_date <= asset.active_to
    )


def _validate_asset_bar_contracts(
    bars: list[MarketBar],
    assets_by_id: dict[str, Asset],
    calendar: USEquityCalendar,
) -> None:
    if not assets_by_id:
        return
    asset_ids_by_session: dict[date, set[str]] = defaultdict(set)
    for bar in bars:
        asset = assets_by_id.get(bar.asset_id)
        if asset is None:
            raise ValueError(f"market bar references unknown asset_id: {bar.asset_id}")
        if bar.symbol != asset.symbol:
            raise ValueError(
                f"market bar symbol {bar.symbol} does not match asset master "
                f"{asset.symbol} for {bar.asset_id}"
            )
        session_date = bar.ts.astimezone(calendar.timezone).date()
        if asset.active_from is not None and session_date < asset.active_from:
            raise ValueError(f"market bar for {bar.asset_id} predates active_from")
        if asset.active_to is not None and session_date > asset.active_to:
            raise ValueError(f"market bar for {bar.asset_id} is after active_to")
        asset_ids_by_session[session_date].add(bar.asset_id)

    if not asset_ids_by_session:
        return
    first_session = min(asset_ids_by_session)
    last_session = max(asset_ids_by_session)
    for session_date in calendar.sessions(first_session, last_session):
        expected = {
            asset_id
            for asset_id, asset in assets_by_id.items()
            if _asset_is_active(asset, session_date)
        }
        missing = sorted(expected - asset_ids_by_session.get(session_date, set()))
        if missing:
            raise ValueError(
                "missing active asset market bars for exchange session "
                f"{session_date.isoformat()}: {', '.join(missing)}"
            )


def _market_returns(
    bars_by_asset: dict[str, list[MarketBar]], prices_by_asset: dict[str, list[float]]
) -> dict[datetime, float]:
    returns_by_ts: dict[datetime, list[float]] = defaultdict(list)
    for asset_id, asset_bars in bars_by_asset.items():
        prices = prices_by_asset[asset_id]
        for index in range(1, len(asset_bars)):
            returns_by_ts[asset_bars[index].ts].append(_return(prices[index], prices[index - 1]))
    return {timestamp: statistics.fmean(values) for timestamp, values in returns_by_ts.items()}


def _sector_returns(
    bars_by_asset: dict[str, list[MarketBar]],
    prices_by_asset: dict[str, list[float]],
    sectors_by_asset: dict[str, str],
) -> dict[tuple[str, datetime], float]:
    returns: dict[tuple[str, datetime], list[float]] = defaultdict(list)
    for asset_id, bars in bars_by_asset.items():
        sector = sectors_by_asset.get(asset_id)
        if not sector:
            continue
        prices = prices_by_asset[asset_id]
        for index in range(1, len(bars)):
            returns[(sector, bars[index].ts)].append(_return(prices[index], prices[index - 1]))
    return {key: statistics.fmean(values) for key, values in returns.items()}


def _populate_traditional_features(
    row: dict[str, Any],
    bars: list[MarketBar],
    prices: list[float],
    index: int,
    market_returns: dict[datetime, float],
    sector: str | None,
    sector_returns: dict[tuple[str, datetime], float],
) -> None:
    bar = bars[index]
    current_return, current_missing = _lookback_return(prices, index, 1)
    for lookback in RETURN_LOOKBACKS:
        value, missing = _lookback_return(prices, index, lookback)
        row[f"return_{lookback}d"] = value
        row[f"return_{lookback}d_missing"] = missing

    row["short_term_reversal_1d"] = -current_return
    row["momentum_5d"] = row["return_5d"] - current_return if not row["return_5d_missing"] else 0.0
    row["momentum_20d"] = row["return_20d"] if not row["return_20d_missing"] else 0.0
    row["gap_return_1d"] = (
        _return(
            bar.open * bar.return_adjustment_factor,
            bars[index - 1].close * bars[index - 1].return_adjustment_factor,
        )
        if index >= 1
        else 0.0
    )
    row["gap_return_1d_missing"] = index < 1
    row["intraday_return"] = _return(bar.close, bar.open)

    prior_volumes = [old.volume for old in bars[max(0, index - 20) : index]]
    for lookback in (3, 20):
        baseline = [old.volume for old in bars[max(0, index - lookback) : index]]
        average = statistics.fmean(baseline) if baseline else float(bar.volume)
        row[f"abnormal_volume_{lookback}d"] = bar.volume / average if average else 1.0
        row[f"abnormal_volume_{lookback}d_missing"] = not baseline
    row["dollar_volume"] = bar.close * bar.volume
    row["average_dollar_volume_20d"] = statistics.fmean(
        old.close * old.volume for old in bars[max(0, index - 20) : index + 1]
    )
    row["turnover"] = None
    row["turnover_missing"] = True

    amihud_values: list[float] = []
    for position in range(max(1, index - 19), index + 1):
        dollar_volume = bars[position].close * bars[position].volume
        if dollar_volume > 0:
            amihud_values.append(
                abs(_return(prices[position], prices[position - 1])) / dollar_volume
            )
    row["amihud_illiquidity_20d"] = statistics.fmean(amihud_values) if amihud_values else 0.0
    row["amihud_illiquidity_20d_missing"] = not amihud_values
    high_low_midpoint = (bar.high + bar.low) / 2.0
    row["high_low_spread_estimate"] = (
        (bar.high - bar.low) / high_low_midpoint if high_low_midpoint > 0 else 0.0
    )

    for lookback in VOLATILITY_LOOKBACKS:
        window_returns = _trailing_returns(prices, index, lookback)
        row[f"realized_volatility_{lookback}d"] = _std(window_returns)
        row[f"realized_volatility_{lookback}d_missing"] = len(window_returns) < 2
    downside = [min(value, 0.0) for value in _trailing_returns(prices, index, 20)]
    row["downside_volatility_20d"] = _std(downside)
    log_ranges = [
        math.log(old.high / old.low) ** 2
        for old in bars[max(0, index - 19) : index + 1]
        if old.high > 0 and old.low > 0
    ]
    row["high_low_volatility_20d"] = (
        math.sqrt(statistics.fmean(log_ranges) / (4.0 * math.log(2.0))) if log_ranges else 0.0
    )
    long_volatility = float(row["realized_volatility_60d"])
    row["volatility_regime"] = (
        float(row["realized_volatility_20d"]) / long_volatility if long_volatility > 0 else 0.0
    )

    paired_asset: list[float] = []
    paired_market: list[float] = []
    for position in range(max(1, index - 59), index + 1):
        market_value = market_returns.get(bars[position].ts)
        if market_value is not None:
            paired_asset.append(_return(prices[position], prices[position - 1]))
            paired_market.append(market_value)
    beta, beta_missing = _beta(paired_asset, paired_market)
    market_return = market_returns.get(bar.ts, 0.0)
    row["market_return_1d"] = market_return
    row["market_beta_60d"] = beta
    row["market_beta_60d_missing"] = beta_missing
    row["market_residual_return_1d"] = current_return - beta * market_return
    row["market_residual_return_1d_missing"] = current_missing or beta_missing
    sector_return = sector_returns.get((sector, bar.ts)) if sector else None
    row["sector"] = sector
    row["sector_return_1d"] = sector_return
    row["sector_residual_return_1d"] = (
        current_return - sector_return if sector_return is not None else None
    )
    row["sector_data_missing"] = sector_return is None
    row["size_proxy_log_dollar_volume"] = math.log1p(float(row["average_dollar_volume_20d"]))
    row["value_proxy"] = None
    row["value_proxy_missing"] = True
    row["quality_proxy"] = None
    row["quality_proxy_missing"] = True
    row["fundamental_size_proxy_log_market_cap"] = None
    row["fundamental_size_proxy_missing"] = True
    row["fundamental_available_at"] = None
    row["earnings_proximity_days"] = None
    row["earnings_proximity_missing"] = True
    row["earnings_calendar_available_at"] = None
    row["earnings_blackout"] = False
    row["ex_dividend_proximity_days"] = None
    row["ex_dividend_proximity_missing"] = True
    row["corporate_action_available_at"] = None
    row["known_event_blackout"] = False
    row["volume_history_count"] = len(prior_volumes)


def _first_numeric(values: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = values.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _populate_known_point_in_time_context(
    row: dict[str, Any],
    bar: MarketBar,
    asof_ts: datetime,
    fundamentals: list[FundamentalRecord],
    earnings_events: list[EarningsCalendarEvent],
    corporate_actions: list[CorporateAction],
) -> None:
    session_date = bar.session_date or bar.ts.date()
    known_fundamentals = [
        record
        for record in fundamentals
        if record.available_at <= asof_ts and record.period_end <= session_date
    ]
    if known_fundamentals:
        latest = max(
            known_fundamentals,
            key=lambda record: (record.available_at, record.period_end),
        )
        value = _first_numeric(latest.values, ("book_to_market", "value_proxy", "value"))
        quality = _first_numeric(
            latest.values,
            ("return_on_equity", "quality_proxy", "gross_profitability"),
        )
        market_cap = _first_numeric(latest.values, ("market_cap",))
        row["value_proxy"] = value
        row["value_proxy_missing"] = value is None
        row["quality_proxy"] = quality
        row["quality_proxy_missing"] = quality is None
        row["fundamental_size_proxy_log_market_cap"] = (
            math.log(market_cap) if market_cap is not None and market_cap > 0 else None
        )
        row["fundamental_size_proxy_missing"] = market_cap is None or market_cap <= 0
        row["fundamental_available_at"] = format_utc(latest.available_at)

    known_earnings = [
        event
        for event in earnings_events
        if event.available_at <= asof_ts and event.event_ts > asof_ts
    ]
    if known_earnings:
        event = min(known_earnings, key=lambda value: value.event_ts)
        proximity = (event.event_ts - asof_ts).total_seconds() / 86_400.0
        row["earnings_proximity_days"] = proximity
        row["earnings_proximity_missing"] = False
        row["earnings_calendar_available_at"] = format_utc(event.available_at)
        row["earnings_blackout"] = proximity <= 2.0

    known_dividends = [
        action
        for action in corporate_actions
        if action.available_at <= asof_ts
        and action.event_ts > asof_ts
        and action.action_type.casefold() in {"dividend", "ex_dividend", "cash_dividend"}
    ]
    if known_dividends:
        action = min(known_dividends, key=lambda value: value.event_ts)
        proximity = (action.event_ts - asof_ts).total_seconds() / 86_400.0
        row["ex_dividend_proximity_days"] = proximity
        row["ex_dividend_proximity_missing"] = False
        row["corporate_action_available_at"] = format_utc(action.available_at)
    row["known_event_blackout"] = bool(row["earnings_blackout"])


def _empty_text_features(
    row: dict[str, Any],
    prefix: str,
    event_types: tuple[str, ...],
    source_types: tuple[str, ...],
    decay_half_life_days: float,
) -> None:
    zero_features = (
        "abnormal_mention_volume",
        "attention_abnormal",
        "attention_item_count",
        "attention_mention_velocity",
        "attention_unique_item_count",
        "credible_attention",
        "event_item_count",
        "event_share",
        "event_sentiment_mean",
        "event_type_diversity",
        "mention_velocity",
        "novel_item_count",
        "novelty_share",
        "duplicate_item_count",
        "sentiment_acceleration",
        "sentiment_conf_weighted",
        "sentiment_credibility_weighted",
        "sentiment_decay_weighted",
        "sentiment_disagreement",
        "sentiment_dispersion",
        "sentiment_mean",
        "sentiment_max_negative",
        "sentiment_max_positive",
        "sentiment_positive_negative_ratio",
        "sentiment_relevance_weighted",
        "source_credibility_mean",
        "spam_score_mean",
        "text_decay_weight_sum",
    )
    row[f"text_count_{prefix}"] = 0
    row[f"raw_text_count_{prefix}"] = 0
    row[f"raw_attention_item_count_{prefix}"] = 0
    row[f"raw_mention_velocity_{prefix}"] = 0.0
    row[f"raw_attention_abnormal_{prefix}"] = 0.0
    row[f"text_missing_{prefix}"] = True
    row[f"latest_text_available_at_{prefix}"] = None
    row[f"latest_text_age_hours_{prefix}"] = None
    row[f"time_since_first_seen_hours_{prefix}"] = None
    row[f"source_identity_missing_{prefix}"] = True
    row[f"source_diversity_count_{prefix}"] = None
    row[f"source_diversity_missing_{prefix}"] = True
    row[f"unique_author_count_{prefix}"] = None
    row[f"unique_author_count_missing_{prefix}"] = True
    row[f"author_disagreement_{prefix}"] = None
    row[f"author_disagreement_missing_{prefix}"] = True
    row[f"source_disagreement_{prefix}"] = None
    row[f"source_disagreement_missing_{prefix}"] = True
    row[f"event_source_interaction_{prefix}"] = 0.0
    row[f"event_source_interaction_missing_{prefix}"] = True
    row[f"text_decay_half_life_days_{prefix}"] = decay_half_life_days
    row[f"llm_annotation_count_{prefix}"] = 0
    row[f"llm_non_abstention_count_{prefix}"] = 0
    row[f"llm_annotation_coverage_{prefix}"] = 0.0
    row[f"llm_abstention_rate_{prefix}"] = 0.0
    row[f"llm_semantic_mean_{prefix}"] = 0.0
    row[f"llm_raw_confidence_mean_{prefix}"] = 0.0
    row[f"llm_uncertainty_mean_{prefix}"] = 0.0
    row[f"llm_event_confidence_mean_{prefix}"] = 0.0
    row[f"llm_supporting_evidence_count_{prefix}"] = 0
    row[f"llm_counterevidence_count_{prefix}"] = 0
    row[f"llm_evidence_agreement_{prefix}"] = 0.0
    row[f"llm_missing_{prefix}"] = True
    row[f"llm_semantic_missing_{prefix}"] = True
    row[f"llm_raw_confidence_missing_{prefix}"] = True
    row[f"llm_uncertainty_missing_{prefix}"] = True
    row[f"llm_event_confidence_missing_{prefix}"] = True
    row[f"llm_evidence_missing_{prefix}"] = True
    for name in zero_features:
        row[f"{name}_{prefix}"] = 0.0
    for event_type in event_types:
        row[f"event_{_event_slug(event_type)}_count_{prefix}"] = 0
    for source_type in source_types:
        slug = _event_slug(source_type)
        row[f"attention_source_{slug}_count_{prefix}"] = 0
        row[f"sentiment_source_{slug}_mean_{prefix}"] = 0.0


def _signal_available_at(signal: TextSignal) -> datetime:
    return signal.available_at or signal.asof_ts


def _signal_actionable_at(signal: TextSignal) -> datetime:
    return max(signal.asof_ts, _signal_available_at(signal))


def _populate_raw_text_diagnostics(
    row: dict[str, Any],
    raw_window: list[TextSignal],
    days: int,
    prefix: str,
    raw_prior_count: int,
) -> None:
    """Retain copy volume under explicit raw names without treating it as evidence."""

    row[f"raw_text_count_{prefix}"] = len(raw_window)
    row[f"raw_attention_item_count_{prefix}"] = len(raw_window)
    row[f"raw_mention_velocity_{prefix}"] = len(raw_window) / float(days)
    row[f"raw_attention_abnormal_{prefix}"] = (
        len(raw_window) / raw_prior_count if raw_prior_count else float(len(raw_window))
    )
    row[f"novelty_share_{prefix}"] = statistics.fmean(signal.novelty for signal in raw_window)
    row[f"novel_item_count_{prefix}"] = sum(signal.novelty > 0.5 for signal in raw_window)
    row[f"duplicate_item_count_{prefix}"] = sum(signal.novelty <= 0.5 for signal in raw_window)


def _populate_independent_text_diagnostics(
    row: dict[str, Any],
    window: list[TextSignal],
    asof_ts: datetime,
    days: int,
    prefix: str,
    prior_count: int,
) -> None:
    """Populate model-facing counts and timing from novelty-filtered evidence."""

    latest = max(_signal_available_at(signal) for signal in window)
    row[f"text_count_{prefix}"] = len(window)
    row[f"text_missing_{prefix}"] = False
    row[f"latest_text_available_at_{prefix}"] = format_utc(latest)
    row[f"latest_text_age_hours_{prefix}"] = (asof_ts - latest).total_seconds() / 3600.0
    row[f"time_since_first_seen_hours_{prefix}"] = (
        asof_ts - min(_signal_available_at(signal) for signal in window)
    ).total_seconds() / 3600.0
    row[f"attention_item_count_{prefix}"] = len(window)
    row[f"attention_unique_item_count_{prefix}"] = len({signal.item_id for signal in window})
    row[f"attention_mention_velocity_{prefix}"] = len(window) / float(days)
    row[f"attention_abnormal_{prefix}"] = (
        len(window) / prior_count if prior_count else float(len(window))
    )
    row[f"mention_velocity_{prefix}"] = row[f"attention_mention_velocity_{prefix}"]
    row[f"abnormal_mention_volume_{prefix}"] = row[f"attention_abnormal_{prefix}"]


def _populate_llm_text_features(
    row: dict[str, Any],
    window: list[TextSignal],
    prefix: str,
) -> None:
    """Aggregate only independent, point-in-time-admissible LLM annotations."""

    annotated = [signal for signal in window if signal.llm_abstained is not None]
    non_abstained = [signal for signal in annotated if signal.llm_abstained is False]
    semantic_values = [
        float(signal.llm_semantic_signal)
        for signal in non_abstained
        if signal.llm_semantic_signal is not None
    ]
    raw_confidences = [
        float(signal.llm_raw_confidence)
        for signal in non_abstained
        if signal.llm_raw_confidence is not None
    ]
    uncertainties = [
        float(signal.llm_uncertainty) for signal in annotated if signal.llm_uncertainty is not None
    ]
    event_confidences = [
        float(signal.llm_event_confidence)
        for signal in non_abstained
        if signal.llm_event_confidence is not None
    ]
    supporting_evidence_count = sum(
        signal.llm_supporting_evidence_count or 0 for signal in annotated
    )
    counterevidence_count = sum(signal.llm_counterevidence_count or 0 for signal in annotated)
    evidence_count = supporting_evidence_count + counterevidence_count

    annotation_count = len(annotated)
    row[f"llm_annotation_count_{prefix}"] = annotation_count
    row[f"llm_non_abstention_count_{prefix}"] = len(non_abstained)
    row[f"llm_annotation_coverage_{prefix}"] = annotation_count / len(window)
    row[f"llm_abstention_rate_{prefix}"] = (
        sum(signal.llm_abstained is True for signal in annotated) / annotation_count
        if annotation_count
        else 0.0
    )
    row[f"llm_semantic_mean_{prefix}"] = (
        statistics.fmean(semantic_values) if semantic_values else 0.0
    )
    row[f"llm_raw_confidence_mean_{prefix}"] = (
        statistics.fmean(raw_confidences) if raw_confidences else 0.0
    )
    row[f"llm_uncertainty_mean_{prefix}"] = (
        statistics.fmean(uncertainties) if uncertainties else 0.0
    )
    row[f"llm_event_confidence_mean_{prefix}"] = (
        statistics.fmean(event_confidences) if event_confidences else 0.0
    )
    row[f"llm_supporting_evidence_count_{prefix}"] = supporting_evidence_count
    row[f"llm_counterevidence_count_{prefix}"] = counterevidence_count
    row[f"llm_evidence_agreement_{prefix}"] = (
        supporting_evidence_count / evidence_count if evidence_count else 0.0
    )
    row[f"llm_missing_{prefix}"] = not annotated
    row[f"llm_semantic_missing_{prefix}"] = not semantic_values
    row[f"llm_raw_confidence_missing_{prefix}"] = not raw_confidences
    row[f"llm_uncertainty_missing_{prefix}"] = not uncertainties
    row[f"llm_event_confidence_missing_{prefix}"] = not event_confidences
    row[f"llm_evidence_missing_{prefix}"] = evidence_count == 0


def _populate_text_features(
    row: dict[str, Any],
    signals: list[TextSignal],
    asof_ts: datetime,
    days: int,
    event_types: tuple[str, ...],
    source_types: tuple[str, ...],
    decay_half_life_days: float,
    availability_times: list[datetime],
) -> None:
    if days <= 0:
        raise ValueError("text feature windows must be positive")
    if decay_half_life_days <= 0.0:
        raise ValueError("text decay half-life must be positive")
    prefix = f"{days}d"
    start = asof_ts - timedelta(days=days)
    window_start = bisect_right(availability_times, start)
    window_end = bisect_right(availability_times, asof_ts)
    raw_window = [
        signal
        for signal in signals[window_start:window_end]
        if _signal_actionable_at(signal) <= asof_ts
    ]
    if not raw_window:
        _empty_text_features(row, prefix, event_types, source_types, decay_half_life_days)
        return
    prior_start = start - timedelta(days=days)
    prior_window_start = bisect_right(availability_times, prior_start)
    prior_window_end = bisect_right(availability_times, start)
    raw_prior_window = [
        signal
        for signal in signals[prior_window_start:prior_window_end]
        if _signal_actionable_at(signal) <= asof_ts
    ]
    raw_prior_count = len(raw_prior_window)
    prior_count = sum(signal.novelty > 0.5 for signal in raw_prior_window)
    window = [signal for signal in raw_window if signal.novelty > 0.5]
    if not window:
        _empty_text_features(row, prefix, event_types, source_types, decay_half_life_days)
        _populate_raw_text_diagnostics(row, raw_window, days, prefix, raw_prior_count)
        return

    scores = [signal.sentiment_score for signal in window]
    relevance_weights = [max(0.0, signal.relevance) for signal in window]
    confidence_weights = [
        max(0.0, signal.sentiment_confidence) * max(0.0, signal.relevance) for signal in window
    ]
    credibility_weights = [
        max(0.0, signal.source_credibility)
        * max(0.0, signal.relevance)
        * (1.0 - min(1.0, max(0.0, signal.spam_score or 0.0)))
        for signal in window
    ]
    half_life_seconds = timedelta(days=decay_half_life_days).total_seconds()
    decay_weights = [
        math.exp(
            -math.log(2.0)
            * (asof_ts - _signal_available_at(signal)).total_seconds()
            / half_life_seconds
        )
        for signal in window
    ]
    decay_quality_weights = [
        decay
        * max(0.0, signal.sentiment_confidence)
        * max(0.0, signal.relevance)
        * max(0.0, signal.source_credibility)
        * (1.0 - min(1.0, max(0.0, signal.spam_score or 0.0)))
        for decay, signal in zip(decay_weights, window, strict=True)
    ]
    positive_count = sum(score > 0.05 for score in scores)
    negative_count = sum(score < -0.05 for score in scores)
    recent_start = asof_ts - timedelta(days=days / 2.0)
    recent_scores = [
        signal.sentiment_score for signal in window if _signal_available_at(signal) > recent_start
    ]
    earlier_scores = [
        signal.sentiment_score for signal in window if _signal_available_at(signal) <= recent_start
    ]
    event_window = [signal for signal in window if signal.event_type]

    _populate_raw_text_diagnostics(row, raw_window, days, prefix, raw_prior_count)
    _populate_independent_text_diagnostics(row, window, asof_ts, days, prefix, prior_count)
    _populate_llm_text_features(row, window, prefix)
    sources = {signal.source for signal in window if signal.source}
    authors = {signal.author_hash for signal in window if signal.author_hash}
    author_groups: dict[str, list[float]] = defaultdict(list)
    source_groups: dict[str, list[float]] = defaultdict(list)
    for signal in window:
        if signal.author_hash:
            author_groups[signal.author_hash].append(signal.sentiment_score)
        if signal.source:
            source_groups[signal.source].append(signal.sentiment_score)
    author_means = [statistics.fmean(values) for values in author_groups.values()]
    source_means = [statistics.fmean(values) for values in source_groups.values()]
    row[f"source_identity_missing_{prefix}"] = not sources
    row[f"source_diversity_count_{prefix}"] = len(sources)
    row[f"source_diversity_missing_{prefix}"] = not sources
    row[f"unique_author_count_{prefix}"] = len(authors)
    row[f"unique_author_count_missing_{prefix}"] = not authors
    row[f"author_disagreement_{prefix}"] = _std(author_means)
    row[f"author_disagreement_missing_{prefix}"] = len(author_means) < 2
    row[f"source_disagreement_{prefix}"] = _std(source_means)
    row[f"source_disagreement_missing_{prefix}"] = len(source_means) < 2
    event_source_values = [
        signal.sentiment_score * signal.source_credibility
        for signal in event_window
        if signal.source
    ]
    row[f"event_source_interaction_{prefix}"] = (
        statistics.fmean(event_source_values) if event_source_values else 0.0
    )
    row[f"event_source_interaction_missing_{prefix}"] = not event_source_values
    row[f"text_decay_half_life_days_{prefix}"] = decay_half_life_days
    row[f"sentiment_mean_{prefix}"] = statistics.fmean(scores)
    row[f"sentiment_relevance_weighted_{prefix}"] = _weighted_mean(scores, relevance_weights)
    row[f"sentiment_conf_weighted_{prefix}"] = _weighted_mean(scores, confidence_weights)
    row[f"sentiment_credibility_weighted_{prefix}"] = _weighted_mean(scores, credibility_weights)
    row[f"sentiment_decay_weighted_{prefix}"] = _weighted_mean(scores, decay_quality_weights)
    row[f"sentiment_max_positive_{prefix}"] = max(0.0, max(scores))
    row[f"sentiment_max_negative_{prefix}"] = min(0.0, min(scores))
    row[f"sentiment_dispersion_{prefix}"] = _std(scores)
    supplied_disagreement = [
        float(signal.disagreement) for signal in window if signal.disagreement is not None
    ]
    row[f"sentiment_disagreement_{prefix}"] = (
        statistics.fmean(supplied_disagreement) if supplied_disagreement else _std(scores)
    )
    row[f"sentiment_positive_negative_ratio_{prefix}"] = (positive_count + 1.0) / (
        negative_count + 1.0
    )
    row[f"sentiment_acceleration_{prefix}"] = (
        statistics.fmean(recent_scores) - statistics.fmean(earlier_scores)
        if recent_scores and earlier_scores
        else 0.0
    )

    row[f"source_credibility_mean_{prefix}"] = statistics.fmean(
        signal.source_credibility for signal in window
    )
    row[f"spam_score_mean_{prefix}"] = statistics.fmean(
        signal.spam_score or 0.0 for signal in window
    )
    row[f"credible_attention_{prefix}"] = sum(credibility_weights)
    row[f"text_decay_weight_sum_{prefix}"] = sum(decay_weights)

    row[f"event_item_count_{prefix}"] = len(event_window)
    row[f"event_share_{prefix}"] = len(event_window) / len(window)
    row[f"event_type_diversity_{prefix}"] = len({signal.event_type for signal in event_window})
    row[f"event_sentiment_mean_{prefix}"] = (
        statistics.fmean(signal.sentiment_score for signal in event_window) if event_window else 0.0
    )
    for event_type in event_types:
        row[f"event_{_event_slug(event_type)}_count_{prefix}"] = sum(
            signal.event_type == event_type for signal in window
        )
    for source_type in source_types:
        slug = _event_slug(source_type)
        source_window = [signal for signal in window if signal.source_type == source_type]
        row[f"attention_source_{slug}_count_{prefix}"] = len(source_window)
        row[f"sentiment_source_{slug}_mean_{prefix}"] = (
            statistics.fmean(signal.sentiment_score for signal in source_window)
            if source_window
            else 0.0
        )


def build_feature_rows(
    bars: Iterable[MarketBar],
    signals: Iterable[TextSignal],
    config: ResearchConfig,
    assets: Iterable[Asset] | None = None,
    *,
    fundamentals: Iterable[FundamentalRecord] = (),
    earnings_events: Iterable[EarningsCalendarEvent] = (),
    corporate_actions: Iterable[CorporateAction] = (),
) -> list[dict[str, Any]]:
    """Build point-in-time market and explicit-window text feature rows."""

    bar_list = list(bars)
    asset_list = list(assets or [])
    assets_by_id = _index_assets(asset_list)
    calendar = _daily_bar_calendar(bar_list, config)
    decision_times_by_session = (
        market_decision_times_by_session(bar_list, calendar.timezone)
        if calendar is not None
        else {}
    )
    if calendar is not None:
        _validate_asset_bar_contracts(bar_list, assets_by_id, calendar)
    bars_by_asset: dict[str, list[MarketBar]] = defaultdict(list)
    signals_by_asset: dict[str, list[TextSignal]] = defaultdict(list)
    for bar in bar_list:
        bars_by_asset[bar.asset_id].append(bar)
    signal_list = list(signals)
    fundamentals_by_asset: dict[str, list[FundamentalRecord]] = defaultdict(list)
    earnings_by_asset: dict[str, list[EarningsCalendarEvent]] = defaultdict(list)
    actions_by_asset: dict[str, list[CorporateAction]] = defaultdict(list)
    for record in fundamentals:
        fundamentals_by_asset[record.asset_id].append(record)
    for event in earnings_events:
        earnings_by_asset[event.asset_id].append(event)
    for action in corporate_actions:
        actions_by_asset[action.asset_id].append(action)
    sectors_by_asset = {asset_id: asset.sector for asset_id, asset in assets_by_id.items()}
    for signal in signal_list:
        signals_by_asset[signal.asset_id].append(signal)

    prices_by_asset: dict[str, list[float]] = {}
    price_basis_by_asset: dict[str, str] = {}
    for asset_id, asset_bars in bars_by_asset.items():
        asset_bars.sort(key=lambda value: value.ts)
        if calendar is not None:
            _validate_contiguous_sessions(asset_id, asset_bars, calendar)
        prices, price_basis = _price_series(asset_bars)
        prices_by_asset[asset_id] = prices
        price_basis_by_asset[asset_id] = price_basis
    market_returns = _market_returns(bars_by_asset, prices_by_asset)
    sector_returns = _sector_returns(bars_by_asset, prices_by_asset, sectors_by_asset)
    event_types = tuple(sorted({signal.event_type for signal in signal_list if signal.event_type}))
    source_types = tuple(
        sorted({signal.source_type for signal in signal_list if signal.source_type})
    )

    rows: list[dict[str, Any]] = []
    for asset_id, asset_bars in bars_by_asset.items():
        asset_signals = sorted(signals_by_asset.get(asset_id, []), key=_signal_available_at)
        availability_times = [_signal_available_at(signal) for signal in asset_signals]
        prices = prices_by_asset[asset_id]
        master_asset = assets_by_id.get(asset_id)
        for index, bar in enumerate(asset_bars):
            if calendar is None:
                continue
            decision_ts = market_decision_time_for_bar(
                bar,
                decision_times_by_session,
                calendar.timezone,
            )
            adjustment_available_at = bar.adjustment_vintage_at
            if adjustment_available_at is None:
                raise ValueError("market bar adjustment_vintage_at is required")
            row: dict[str, Any] = {
                "asset_id": bar.asset_id,
                "symbol": bar.symbol,
                "asof_ts": format_utc(decision_ts),
                "horizon": f"{config.features.horizon_days}d",
                "feature_set_version": config.features.feature_set_version,
                "close": bar.close,
                "price_basis": price_basis_by_asset[asset_id],
                "session_close_ts": format_utc(bar.ts),
                "market_bar_available_at": format_utc(market_bar_available_at(bar)),
                "adjustment_available_at": format_utc(adjustment_available_at),
                "short_available": master_asset.short_available if master_asset else False,
                "hard_to_borrow": master_asset.hard_to_borrow if master_asset else False,
            }
            _populate_traditional_features(
                row,
                asset_bars,
                prices,
                index,
                market_returns,
                sectors_by_asset.get(asset_id),
                sector_returns,
            )
            _populate_known_point_in_time_context(
                row,
                bar,
                decision_ts,
                fundamentals_by_asset.get(asset_id, []),
                earnings_by_asset.get(asset_id, []),
                actions_by_asset.get(asset_id, []),
            )
            for days in config.features.windows_days:
                _populate_text_features(
                    row,
                    asset_signals,
                    decision_ts,
                    days,
                    event_types,
                    source_types,
                    config.features.text_decay_half_life_days,
                    availability_times,
                )
            rows.append(row)
    validate_feature_rows(rows, config)
    return sorted(rows, key=lambda row: (row["asof_ts"], row["symbol"]))


def _rank_group(rows: list[dict[str, Any]]) -> dict[int, float]:
    valid = [
        (index, float(row["forward_return"]))
        for index, row in enumerate(rows)
        if row["forward_return"] is not None
    ]
    if not valid:
        return {}
    if len(valid) == 1:
        return {valid[0][0]: 0.5}
    ordered = sorted(valid, key=lambda item: item[1])
    ranks: dict[int, float] = {}
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = ((position + end - 1) / 2.0) / (len(ordered) - 1)
        for offset in range(position, end):
            ranks[ordered[offset][0]] = average_rank
        position = end
    return ranks


def build_label_rows(
    bars: Iterable[MarketBar],
    config: ResearchConfig,
    assets: Iterable[Asset] | None = None,
) -> list[dict[str, Any]]:
    """Build adjusted-price-consistent labels using bars strictly after each as-of time."""

    bar_list = list(bars)
    asset_list = list(assets or [])
    assets_by_id = _index_assets(asset_list)
    calendar = _daily_bar_calendar(bar_list, config)
    decision_times_by_session = (
        market_decision_times_by_session(bar_list, calendar.timezone)
        if calendar is not None
        else {}
    )
    bars_by_asset: dict[str, list[MarketBar]] = defaultdict(list)
    if calendar is not None:
        _validate_asset_bar_contracts(bar_list, assets_by_id, calendar)
    sectors_by_asset = {asset_id: asset.sector for asset_id, asset in assets_by_id.items()}
    for bar in bar_list:
        bars_by_asset[bar.asset_id].append(bar)

    rows: list[dict[str, Any]] = []
    horizon = config.features.horizon_days
    suffix = f"{horizon}d"
    for asset_id, asset_bars in bars_by_asset.items():
        asset_bars.sort(key=lambda value: value.ts)
        if calendar is None:
            continue
        _validate_contiguous_sessions(asset_id, asset_bars, calendar)
        positions_by_close = {bar.ts.astimezone(UTC): index for index, bar in enumerate(asset_bars)}
        prices, price_basis = _price_series(asset_bars)
        for bar in asset_bars:
            decision_ts = market_decision_time_for_bar(
                bar,
                decision_times_by_session,
                calendar.timezone,
            )
            expected_label_start, expected_closes = _label_session_closes(
                calendar,
                decision_ts,
                horizon,
            )
            future_positions = [positions_by_close.get(value) for value in expected_closes]
            complete = all(position is not None for position in future_positions)
            if complete:
                complete_positions = [
                    position for position in future_positions if position is not None
                ]
                entry_position = complete_positions[0]
                target_position = complete_positions[-1]
                entry_bar = asset_bars[entry_position]
                exit_bar = asset_bars[target_position]
                entry_price_value = entry_bar.open
                exit_price_value = exit_bar.close
                entry_return_basis = entry_price_value * entry_bar.return_adjustment_factor
                exit_return_basis = exit_price_value * exit_bar.return_adjustment_factor
                entry_price: float | None = entry_price_value
                exit_price: float | None = exit_price_value
                execution_dollar_volume: float | None = (
                    entry_price_value * asset_bars[entry_position].volume
                )
                exit_dollar_volume: float | None = (
                    exit_price_value * asset_bars[target_position].volume
                )
                forward_return: float | None = _return(
                    exit_return_basis,
                    entry_return_basis,
                )
                step_prices = [
                    entry_return_basis,
                    *(prices[position] for position in complete_positions),
                ]
                future_returns = _period_returns(step_prices)
                forward_volatility: float | None = _rms(future_returns)
                future_average_volume = statistics.fmean(
                    asset_bars[position].volume for position in complete_positions
                )
                forward_volume: float | None = future_average_volume
                forward_volume_change: float | None = _return(
                    future_average_volume, float(bar.volume)
                )
                label_start_ts: str | None = format_utc(expected_label_start)
                label_end_ts: str | None = format_utc(asset_bars[target_position].ts)
                label_available_at: str | None = format_utc(
                    market_decision_time_for_bar(
                        exit_bar,
                        decision_times_by_session,
                        calendar.timezone,
                    )
                )
                label_missing_reason: str | None = None
            else:
                forward_return = None
                forward_volatility = None
                forward_volume = None
                forward_volume_change = None
                label_start_ts = None
                label_end_ts = None
                label_available_at = None
                entry_price = None
                exit_price = None
                execution_dollar_volume = None
                exit_dollar_volume = None
                label_missing_reason = "missing_required_session"
            row: dict[str, Any] = {
                "asset_id": asset_id,
                "symbol": bar.symbol,
                "asof_ts": format_utc(decision_ts),
                "horizon": suffix,
                "label_version": config.features.label_version,
                "sector": sectors_by_asset.get(asset_id),
                "price_basis": price_basis,
                "label_start_ts": label_start_ts,
                "label_end_ts": label_end_ts,
                "label_available_at": label_available_at,
                "expected_label_start_ts": format_utc(expected_label_start),
                "expected_label_end_ts": format_utc(expected_closes[-1]),
                "label_missing_reason": label_missing_reason,
                "execution_price": entry_price,
                "exit_price": exit_price,
                "execution_dollar_volume": execution_dollar_volume,
                "exit_dollar_volume": exit_dollar_volume,
                "forward_return": forward_return,
                f"forward_return_{suffix}": forward_return,
                "forward_abnormal_return": None,
                f"forward_abnormal_return_{suffix}": None,
                "forward_sector_neutral_return": None,
                f"forward_sector_neutral_return_{suffix}": None,
                "sector_data_missing": asset_id not in sectors_by_asset,
                "binary_up": int(forward_return > 0.0) if forward_return is not None else None,
                f"binary_up_{suffix}": (
                    int(forward_return > 0.0) if forward_return is not None else None
                ),
                "rank": None,
                f"rank_{suffix}": None,
                "forward_volatility": forward_volatility,
                f"volatility_{suffix}": forward_volatility,
                "forward_volume": forward_volume,
                "forward_volume_change": forward_volume_change,
                f"volume_{suffix}": forward_volume,
            }
            rows.append(row)

    rows_by_asof: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_asof[str(row["asof_ts"])].append(row)
    for group in rows_by_asof.values():
        valid_returns = [
            float(row["forward_return"]) for row in group if row["forward_return"] is not None
        ]
        market_forward_return = statistics.fmean(valid_returns) if valid_returns else None
        ranks = _rank_group(group)
        sector_values: dict[str, list[float]] = defaultdict(list)
        for row in group:
            if row["sector"] and row["forward_return"] is not None:
                sector_values[str(row["sector"])].append(float(row["forward_return"]))
        sector_means = {
            sector: statistics.fmean(values) for sector, values in sector_values.items()
        }
        for index, row in enumerate(group):
            value = row["forward_return"]
            abnormal = (
                float(value) - market_forward_return
                if value is not None and market_forward_return is not None
                else None
            )
            row["market_forward_return"] = market_forward_return
            row["forward_abnormal_return"] = abnormal
            row[f"forward_abnormal_return_{suffix}"] = abnormal
            sector = str(row["sector"]) if row["sector"] else None
            sector_mean = sector_means.get(sector) if sector else None
            sector_neutral = (
                float(value) - sector_mean
                if value is not None and sector_mean is not None
                else None
            )
            row["forward_sector_neutral_return"] = sector_neutral
            row[f"forward_sector_neutral_return_{suffix}"] = sector_neutral
            row["rank"] = ranks.get(index)
            row[f"rank_{suffix}"] = ranks.get(index)

    for row in rows:
        start = row["label_start_ts"]
        if start is not None and parse_utc(str(start)) <= parse_utc(str(row["asof_ts"])):
            raise ValueError("label window must begin strictly after asof_ts")
    return sorted(rows, key=lambda row: (row["asof_ts"], row["symbol"]))


def validate_feature_rows(rows: list[dict[str, Any]], config: ResearchConfig) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row["asset_id"]),
            str(row["asof_ts"]),
            str(row["horizon"]),
            str(row["feature_set_version"]),
        )
        if key in seen:
            raise ValueError(f"duplicate feature row: {key}")
        seen.add(key)
        asof_ts = parse_utc(str(row["asof_ts"]))
        for days in config.features.windows_days:
            latest = row.get(f"latest_text_available_at_{days}d")
            if latest is not None and parse_utc(str(latest)) > asof_ts:
                raise ValueError(f"feature row uses future text: {key}")
            age = row.get(f"latest_text_age_hours_{days}d")
            if age is not None and float(age) < 0.0:
                raise ValueError(f"feature row has negative text age: {key}")


def finite_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(str(value))
    except ValueError:
        return 0.0
    return number if math.isfinite(number) else 0.0
