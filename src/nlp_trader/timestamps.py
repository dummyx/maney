from __future__ import annotations

from datetime import UTC, date, datetime


def parse_utc(value: str) -> datetime:
    raw = value.strip()
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value!r}")
    return parsed.astimezone(UTC)


def format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_optional_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)
