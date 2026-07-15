from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals  # type: ignore[import-untyped]

_CALENDAR_ALIASES = {
    # ``exchange_calendars`` exposes Tokyo under its operating MIC (XTKS) and
    # the JPX alias. Accept XJPX as the research-facing name used by this
    # project while keeping the upstream calendar data authoritative.
    "XJPX": "XTKS",
    "JPX": "XTKS",
    "XTSE": "XTKS",
}
_DEFAULT_START = date(1990, 1, 1)
_CALENDAR_DEFAULT_STARTS = {"XTKS": date(1997, 1, 1)}


def _date_value(value: Any) -> date:
    converted = value.to_pydatetime()
    if isinstance(converted, datetime):
        return converted.date()
    if isinstance(converted, date):
        return converted
    raise TypeError("exchange calendar did not return a date")


def _utc_datetime(value: Any) -> datetime:
    converted = value.to_pydatetime()
    if not isinstance(converted, datetime):
        raise TypeError("exchange calendar did not return a datetime")
    if converted.tzinfo is None:
        converted = converted.replace(tzinfo=UTC)
    return converted.astimezone(UTC)


class USEquityCalendar:
    """Point-in-time decision calendar backed by a versioned exchange schedule.

    The underlying package includes observed holidays, historical one-off closures,
    daylight-saving transitions, and official early closes. Bounds are explicit so
    research outside them fails instead of silently treating weekdays as sessions.

    The legacy class name is retained for compatibility. ``calendar_name`` may be
    ``XNYS`` or another calendar supported by ``exchange_calendars``; ``XJPX`` is
    normalized to that package's canonical Tokyo Stock Exchange calendar, ``XTKS``.
    """

    timezone: ZoneInfo

    def __init__(
        self,
        *,
        calendar_name: str = "XNYS",
        start: date | None = None,
        end: date = date(2035, 12, 31),
    ) -> None:
        provider_calendar_name = _CALENDAR_ALIASES.get(calendar_name, calendar_name)
        if start is None:
            start = _CALENDAR_DEFAULT_STARTS.get(provider_calendar_name, _DEFAULT_START)
        if end < start:
            raise ValueError("calendar end must be on or after start")
        self.calendar_name = calendar_name
        self.provider_calendar_name = provider_calendar_name
        self.start = start
        self.end = end
        self._calendar: Any = xcals.get_calendar(
            self.provider_calendar_name,
            start=start,
            end=end,
        )
        provider_timezone = self._calendar.tz
        self.timezone = (
            provider_timezone
            if isinstance(provider_timezone, ZoneInfo)
            else ZoneInfo(str(provider_timezone))
        )

    def _require_bounds(self, value: date) -> None:
        if value < self.start or value > self.end:
            raise ValueError(
                f"date {value.isoformat()} is outside configured calendar bounds "
                f"{self.start.isoformat()}..{self.end.isoformat()}"
            )

    def is_session(self, session_date: date) -> bool:
        self._require_bounds(session_date)
        return bool(self._calendar.is_session(session_date))

    def sessions(self, start: date, end: date) -> tuple[date, ...]:
        if end < start:
            raise ValueError("end must be on or after start")
        self._require_bounds(start)
        self._require_bounds(end)
        return tuple(_date_value(value) for value in self._calendar.sessions_in_range(start, end))

    def next_session(self, after: date, *, include_current: bool = False) -> date:
        self._require_bounds(after)
        if include_current and self.is_session(after):
            return after
        session = self._calendar.date_to_session(after, direction="next")
        result = _date_value(session)
        if result == after:
            result = _date_value(self._calendar.next_session(session))
        self._require_bounds(result)
        return result

    def previous_session(self, before: date, *, include_current: bool = False) -> date:
        self._require_bounds(before)
        if include_current and self.is_session(before):
            return before
        session = self._calendar.date_to_session(before, direction="previous")
        result = _date_value(session)
        if result == before:
            result = _date_value(self._calendar.previous_session(session))
        self._require_bounds(result)
        return result

    def is_early_close(self, session_date: date) -> bool:
        if not self.is_session(session_date):
            return False
        return session_date in {_date_value(value) for value in self._calendar.early_closes}

    def session_open(self, session_date: date) -> datetime:
        if not self.is_session(session_date):
            raise ValueError(f"not a {self.calendar_name} session: {session_date.isoformat()}")
        return _utc_datetime(self._calendar.session_open(session_date))

    def session_close(self, session_date: date) -> datetime:
        if not self.is_session(session_date):
            raise ValueError(f"not a {self.calendar_name} session: {session_date.isoformat()}")
        return _utc_datetime(self._calendar.session_close(session_date))

    def decision_times(self, start: date, end: date) -> tuple[datetime, ...]:
        return tuple(self.session_close(day) for day in self.sessions(start, end))

    def next_decision_time(self, available_at: datetime) -> datetime:
        if available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        available_utc = available_at.astimezone(UTC)
        local_date = available_utc.astimezone(self.timezone).date()
        if self.is_session(local_date):
            close = self.session_close(local_date)
            if available_utc <= close:
                return close
        return self.session_close(self.next_session(local_date))

    def next_open_decision_time(self, available_at: datetime) -> datetime:
        """Return the first session open at which the item could be acted upon."""

        if available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        available_utc = available_at.astimezone(UTC)
        local_date = available_utc.astimezone(self.timezone).date()
        if self.is_session(local_date):
            opening = self.session_open(local_date)
            if available_utc <= opening:
                return opening
        return self.session_open(self.next_session(local_date))
