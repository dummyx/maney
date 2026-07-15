from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from nlp_trader.calendars import USEquityCalendar


def test_us_calendar_models_observed_holiday_and_regular_close() -> None:
    calendar = USEquityCalendar()

    assert date(2026, 7, 3) not in calendar.sessions(date(2026, 7, 1), date(2026, 7, 6))
    assert not calendar.is_early_close(date(2026, 7, 2))
    assert calendar.session_close(date(2026, 7, 2)) == datetime(2026, 7, 2, 20, tzinfo=UTC)
    assert calendar.session_close(date(2026, 7, 6)) == datetime(2026, 7, 6, 20, tzinfo=UTC)


def test_after_close_text_moves_to_next_tradable_decision() -> None:
    calendar = USEquityCalendar()

    assert calendar.next_decision_time(datetime(2026, 7, 2, 19, 59, tzinfo=UTC)) == datetime(
        2026, 7, 2, 20, tzinfo=UTC
    )
    assert calendar.next_decision_time(datetime(2026, 7, 2, 20, 1, tzinfo=UTC)) == datetime(
        2026, 7, 6, 20, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.next_decision_time(datetime(2026, 7, 2, 19, 59))


def test_thanksgiving_friday_is_an_early_close() -> None:
    calendar = USEquityCalendar()

    assert not calendar.is_session(date(2026, 11, 26))
    assert calendar.session_close(date(2026, 11, 27)) == datetime(2026, 11, 27, 18, tzinfo=UTC)


def test_open_decisions_roll_forward_after_the_open() -> None:
    calendar = USEquityCalendar()

    assert calendar.next_open_decision_time(datetime(2026, 7, 2, 13, tzinfo=UTC)) == datetime(
        2026, 7, 2, 13, 30, tzinfo=UTC
    )
    assert calendar.next_open_decision_time(datetime(2026, 7, 2, 14, tzinfo=UTC)) == datetime(
        2026, 7, 6, 13, 30, tzinfo=UTC
    )


def test_xjpx_calendar_uses_tokyo_timezone_and_exchange_holidays() -> None:
    calendar = USEquityCalendar(calendar_name="XJPX")

    assert calendar.provider_calendar_name == "XTKS"
    assert calendar.timezone == ZoneInfo("Asia/Tokyo")
    assert calendar.sessions(date(2026, 7, 17), date(2026, 7, 21)) == (
        date(2026, 7, 17),
        date(2026, 7, 21),
    )
    assert not calendar.is_session(date(2026, 7, 20))
    assert calendar.next_session(date(2026, 7, 17)) == date(2026, 7, 21)
    assert calendar.previous_session(date(2026, 7, 21)) == date(2026, 7, 17)


def test_xjpx_open_and_close_decisions_roll_across_holiday() -> None:
    calendar = USEquityCalendar(calendar_name="XJPX")

    assert calendar.session_open(date(2026, 7, 17)) == datetime(2026, 7, 17, 0, tzinfo=UTC)
    assert calendar.session_close(date(2026, 7, 17)) == datetime(2026, 7, 17, 6, 30, tzinfo=UTC)
    assert calendar.next_decision_time(datetime(2026, 7, 17, 6, 29, tzinfo=UTC)) == datetime(
        2026, 7, 17, 6, 30, tzinfo=UTC
    )
    assert calendar.next_decision_time(datetime(2026, 7, 17, 6, 31, tzinfo=UTC)) == datetime(
        2026, 7, 21, 6, 30, tzinfo=UTC
    )
    assert calendar.next_open_decision_time(datetime(2026, 7, 17, 0, 1, tzinfo=UTC)) == datetime(
        2026, 7, 21, 0, tzinfo=UTC
    )


def test_xjpx_models_the_november_2024_tse_close_extension() -> None:
    calendar = USEquityCalendar(calendar_name="XJPX")

    assert calendar.session_close(date(2024, 11, 1)) == datetime(2024, 11, 1, 6, tzinfo=UTC)
    assert calendar.session_close(date(2024, 11, 5)) == datetime(2024, 11, 5, 6, 30, tzinfo=UTC)
    assert not calendar.is_early_close(date(2024, 11, 1))
    assert not calendar.is_early_close(date(2024, 11, 5))
