from __future__ import annotations

from datetime import UTC

import pytest

from nlp_trader.timestamps import parse_utc


def test_parse_utc_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError):
        parse_utc("2026-07-01T12:00:00")


def test_parse_utc_normalizes_to_utc() -> None:
    parsed = parse_utc("2026-07-01T21:00:00+01:00")
    assert parsed.tzinfo == UTC
    assert parsed.hour == 20
