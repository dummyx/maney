from __future__ import annotations

from datetime import UTC, date, datetime

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
