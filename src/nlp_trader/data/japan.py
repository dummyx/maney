from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final, Literal
from zoneinfo import ZoneInfo

from nlp_trader.schemas import Asset, MarketBar

type MarketContract = Literal["generic", "japan_cash_equity_v1"]

JAPAN_CASH_EQUITY_CONTRACT: Final = "japan_cash_equity_v1"
JAPAN_EXCHANGE_MIC: Final = "XJPX"
JAPAN_CURRENCY: Final = "JPY"
RAW_TRADABLE_PRICE_BASIS: Final = "raw_tradable"

# SICC permits letters only in the second and fourth positions. The permitted
# alphabet excludes B, E, I, O, Q, V, and Z. Existing numeric codes remain valid.
_JP_LETTERS = "ACDFGHJKLMNPRSTUWXY"
_JP_SECURITY_CODE = re.compile(rf"^[1-9][0-9{_JP_LETTERS}][0-9][0-9{_JP_LETTERS}]$")
_TOKYO = ZoneInfo("Asia/Tokyo")

_ASSET_REQUIRED_FIELDS = frozenset(
    {
        "asset_id",
        "symbol",
        "exchange",
        "currency",
        "name",
        "sector",
        "active_from",
        "active_to",
        "trading_unit",
    }
)
_ASSET_OPTIONAL_FIELDS = frozenset(
    {"cik", "figi", "isin", "industry", "short_available", "hard_to_borrow"}
)
_BAR_REQUIRED_FIELDS = frozenset(
    {
        "asset_id",
        "symbol",
        "exchange",
        "currency",
        "trading_unit",
        "session_date",
        "ts",
        "available_at",
        "bar_size",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "corporate_action_adjusted",
        "adjustment_vintage_at",
        "return_adjustment_factor",
        "price_basis",
    }
)
_BAR_OPTIONAL_FIELDS = frozenset({"vwap", "adjusted_close"})


def is_canonical_japanese_security_code(value: str) -> bool:
    """Return whether *value* is a canonical four-character SICC security code."""

    return _JP_SECURITY_CODE.fullmatch(value) is not None


def strict_integer(value: object, name: str, *, minimum: int) -> int:
    """Parse an integer without accepting booleans or truncating fractional values."""

    if type(value) is int:
        parsed = value
    elif type(value) is float:
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer")
        parsed = int(value)
    elif isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        parsed = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if parsed < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return parsed


def validate_japan_asset_record(record: Mapping[str, Any]) -> None:
    """Validate the exact row-level asset-master contract before coercion."""

    _validate_columns(
        record,
        required=_ASSET_REQUIRED_FIELDS,
        optional=_ASSET_OPTIONAL_FIELDS,
        role="Japanese asset",
    )
    symbol = _required_string(record, "symbol")
    _validate_symbol(symbol)
    _require_equal(record, "exchange", JAPAN_EXCHANGE_MIC)
    _require_equal(record, "currency", JAPAN_CURRENCY)
    strict_integer(record["trading_unit"], "trading_unit", minimum=1)
    for field in ("asset_id", "name", "sector", "active_from"):
        _required_string(record, field)
    for field in ("short_available", "hard_to_borrow"):
        if record.get(field) not in (None, ""):
            _strict_boolean(record[field], field)


def validate_japan_bar_record(record: Mapping[str, Any]) -> None:
    """Validate the exact row-level daily-bar contract before coercion."""

    _validate_columns(
        record,
        required=_BAR_REQUIRED_FIELDS,
        optional=_BAR_OPTIONAL_FIELDS,
        role="Japanese market bar",
    )
    symbol = _required_string(record, "symbol")
    _validate_symbol(symbol)
    _required_string(record, "asset_id")
    _require_equal(record, "exchange", JAPAN_EXCHANGE_MIC)
    _require_equal(record, "currency", JAPAN_CURRENCY)
    _require_equal(record, "bar_size", "1d")
    _require_equal(record, "price_basis", RAW_TRADABLE_PRICE_BASIS)
    strict_integer(record["trading_unit"], "trading_unit", minimum=1)
    strict_integer(record["volume"], "volume", minimum=0)
    if (
        _strict_boolean(record["corporate_action_adjusted"], "corporate_action_adjusted")
        is not True
    ):
        raise ValueError(
            "corporate_action_adjusted must be explicitly true to certify causal "
            "adjustment metadata"
        )
    for field in ("session_date", "ts", "available_at", "adjustment_vintage_at"):
        if record[field] in (None, ""):
            raise ValueError(f"missing required field: {field}")


def validate_japan_assets(assets: Sequence[Asset]) -> None:
    """Validate normalized Japanese assets and reject ambiguous identities."""

    seen_asset_ids: set[str] = set()
    seen_symbols: set[str] = set()
    for asset in assets:
        _validate_symbol(asset.symbol)
        if asset.exchange != JAPAN_EXCHANGE_MIC:
            raise ValueError(f"Japanese asset exchange must be {JAPAN_EXCHANGE_MIC}")
        if asset.currency != JAPAN_CURRENCY:
            raise ValueError(f"Japanese asset currency must be {JAPAN_CURRENCY}")
        if asset.trading_unit is None:
            raise ValueError("Japanese asset trading_unit is required")
        if asset.asset_id in seen_asset_ids:
            raise ValueError(f"duplicate Japanese asset_id: {asset.asset_id}")
        if asset.symbol in seen_symbols:
            raise ValueError(f"duplicate Japanese symbol: {asset.symbol}")
        seen_asset_ids.add(asset.asset_id)
        seen_symbols.add(asset.symbol)


def validate_japan_market_dataset(assets: Sequence[Asset], bars: Sequence[MarketBar]) -> None:
    """Validate bar provenance and identity against the point-in-time asset master."""

    validate_japan_assets(assets)
    assets_by_id = {asset.asset_id: asset for asset in assets}
    seen_bars: set[tuple[str, datetime]] = set()
    for bar in bars:
        _validate_japan_bar(bar)
        asset = assets_by_id.get(bar.asset_id)
        if asset is None:
            raise ValueError(f"Japanese market bar references unknown asset_id: {bar.asset_id}")
        if bar.symbol != asset.symbol:
            raise ValueError(
                f"Japanese market bar symbol {bar.symbol} does not match asset {asset.symbol}"
            )
        if bar.exchange != asset.exchange or bar.currency != asset.currency:
            raise ValueError("Japanese market bar exchange/currency does not match asset master")
        if bar.trading_unit != asset.trading_unit:
            raise ValueError("Japanese market bar trading_unit does not match asset master")
        assert bar.session_date is not None  # established by _validate_japan_bar
        if asset.active_from is not None and bar.session_date < asset.active_from:
            raise ValueError("Japanese market bar is before asset active_from")
        if asset.active_to is not None and bar.session_date > asset.active_to:
            raise ValueError("Japanese market bar is after asset active_to")
        key = (bar.asset_id, bar.ts)
        if key in seen_bars:
            raise ValueError(
                f"duplicate Japanese market bar for {bar.asset_id} at {bar.ts.isoformat()}"
            )
        seen_bars.add(key)


def _validate_japan_bar(bar: MarketBar) -> None:
    _validate_symbol(bar.symbol)
    if bar.exchange != JAPAN_EXCHANGE_MIC:
        raise ValueError(f"Japanese market bar exchange must be {JAPAN_EXCHANGE_MIC}")
    if bar.currency != JAPAN_CURRENCY:
        raise ValueError(f"Japanese market bar currency must be {JAPAN_CURRENCY}")
    if bar.trading_unit is None:
        raise ValueError("Japanese market bar trading_unit is required")
    if bar.session_date is None:
        raise ValueError("Japanese market bar session_date is required")
    if bar.session_date != bar.ts.astimezone(_TOKYO).date():
        raise ValueError("Japanese market bar session_date must match ts in Asia/Tokyo")
    if bar.available_at is None:
        raise ValueError("Japanese market bar available_at is required")
    if bar.available_at < bar.ts:
        raise ValueError("Japanese market bar available_at must not be before ts")
    if not bar.corporate_action_adjusted or bar.adjustment_vintage_at is None:
        raise ValueError("Japanese market bar requires explicit causal adjustment metadata")
    if bar.adjustment_vintage_at < bar.ts:
        raise ValueError("Japanese adjustment_vintage_at must not be before ts")
    if bar.adjustment_vintage_at > bar.available_at:
        raise ValueError("adjustment_vintage_at must not be after available_at")
    if bar.price_basis != RAW_TRADABLE_PRICE_BASIS:
        raise ValueError("Japanese market bar price_basis must be raw_tradable")
    if bar.bar_size != "1d":
        raise ValueError("Japanese cash-equity contract supports only bar_size=1d")


def _validate_symbol(symbol: str) -> None:
    if not is_canonical_japanese_security_code(symbol):
        raise ValueError(f"invalid canonical Japanese security code: {symbol!r}")


def _validate_columns(
    record: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    role: str,
) -> None:
    fields = {str(field) for field in record}
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{role} missing required fields: {', '.join(missing)}")
    unknown = sorted(fields - required - optional)
    if unknown:
        raise ValueError(f"{role} has unsupported fields: {', '.join(unknown)}")


def _required_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain surrounding whitespace")
    return value


def _require_equal(record: Mapping[str, Any], field: str, expected: str) -> None:
    value = _required_string(record, field)
    if value != expected:
        raise ValueError(f"{field} must be {expected}")


def _strict_boolean(value: object, field: str) -> bool:
    if type(value) is bool:
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be a boolean")
