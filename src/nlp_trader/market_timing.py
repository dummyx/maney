from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from nlp_trader.schemas import MarketBar


def market_bar_available_at(bar: MarketBar) -> datetime:
    """Return the earliest defensible availability of a complete daily bar."""

    return (bar.available_at or bar.ts).astimezone(UTC)


def market_bar_session_date(bar: MarketBar, timezone: ZoneInfo) -> date:
    """Return and validate the exchange-local session date for a market bar."""

    derived = bar.ts.astimezone(timezone).date()
    if bar.session_date is not None and bar.session_date != derived:
        raise ValueError(
            f"market bar {bar.symbol} session_date {bar.session_date.isoformat()} "
            f"does not match exchange-local timestamp date {derived.isoformat()}"
        )
    return derived


def market_decision_times_by_session(
    bars: Iterable[MarketBar],
    timezone: ZoneInfo,
) -> dict[date, datetime]:
    """Choose one leakage-safe decision timestamp per complete session cross-section.

    A session becomes usable only when its slowest selected asset bar is available.
    Session decisions must advance strictly so a late prior payload can never enter an
    earlier feature row silently.
    """

    availability_by_session: dict[date, list[datetime]] = defaultdict(list)
    for bar in bars:
        available_at = market_bar_available_at(bar)
        close_ts = bar.ts.astimezone(UTC)
        if available_at < close_ts:
            raise ValueError(
                f"daily market bar {bar.symbol} available_at must not be before "
                "the official session close"
            )
        availability_by_session[market_bar_session_date(bar, timezone)].append(available_at)

    decisions: dict[date, datetime] = {}
    previous: datetime | None = None
    for session_date in sorted(availability_by_session):
        decision = max(availability_by_session[session_date])
        if previous is not None and decision <= previous:
            raise ValueError(
                "market session decision times must be strictly increasing; "
                f"{session_date.isoformat()} resolves to {decision.isoformat()}"
            )
        decisions[session_date] = decision
        previous = decision
    return decisions


def market_decision_time_for_bar(
    bar: MarketBar,
    decisions: Mapping[date, datetime],
    timezone: ZoneInfo,
) -> datetime:
    """Resolve the shared cross-section decision timestamp for *bar*."""

    session_date = market_bar_session_date(bar, timezone)
    try:
        return decisions[session_date]
    except KeyError as error:
        raise ValueError(
            f"missing market decision for session {session_date.isoformat()}"
        ) from error
