from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "kabus-cash-order-v1"
MAX_CASH_ORDER_INTENT_BYTES = 16 * 1024

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Z0-9]{1,16}$")


class CashOrderIntentJsonError(ValueError):
    """Raised when an intent is not bounded, unambiguous JSON."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError
        result[key] = value
    return result


def _reject_non_finite_constant(_: str) -> None:
    raise ValueError


def _bounded_json_object(payload: object, *, max_bytes: int) -> dict[str, Any]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    try:
        if isinstance(payload, str):
            encoded = payload.encode("utf-8")
        elif isinstance(payload, bytes):
            encoded = payload
        else:
            raise TypeError
    except (TypeError, UnicodeEncodeError) as exc:
        raise CashOrderIntentJsonError("cash order intent must be UTF-8 JSON") from exc
    if len(encoded) > max_bytes:
        raise CashOrderIntentJsonError("cash order intent exceeds the byte limit")
    try:
        parsed = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except _DuplicateJsonKeyError as exc:
        raise CashOrderIntentJsonError("cash order intent repeats a JSON object key") from exc
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise CashOrderIntentJsonError("cash order intent is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise CashOrderIntentJsonError("cash order intent JSON root must be an object")
    return parsed


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class CashOrderIntent(BaseModel):
    """Human-confirmable cash-equity order intent, independent of broker credentials."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal["kabus-cash-order-v1"]
    client_order_id: str = Field(min_length=1, max_length=128)
    strategy_id: str = Field(min_length=1, max_length=128)
    created_at: datetime
    symbol: str = Field(min_length=1, max_length=16)
    exchange: Literal[3, 5, 6, 9, 27]
    reference_exchange: Literal[1, 3, 5, 6]
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)
    order_type: Literal["limit"]
    limit_price: float = Field(gt=0)
    expire_day: int

    @field_validator("client_order_id", "strategy_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("identifier contains unsupported characters")
        return value

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if _SYMBOL.fullmatch(value) is None:
            raise ValueError("symbol must contain only uppercase ASCII letters and digits")
        return value

    @field_validator("created_at", mode="before")
    @classmethod
    def normalize_created_at(cls, value: object) -> datetime:
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("created_at must be an ISO-8601 timestamp") from exc
        elif isinstance(value, datetime):
            parsed = value
        else:
            raise ValueError("created_at must be an ISO-8601 timestamp")
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return parsed.astimezone(UTC)

    @field_validator("quantity", "expire_day", mode="before")
    @classmethod
    def reject_boolean_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("value must be an integer")
        return value

    @field_validator("exchange", "reference_exchange", mode="before")
    @classmethod
    def reject_boolean_exchanges(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("exchange must be an integer")
        return value

    @field_validator("expire_day")
    @classmethod
    def validate_expire_day(cls, value: int) -> int:
        raw = str(value)
        if len(raw) != 8:
            raise ValueError("expire_day must be a valid explicit YYYYMMDD date")
        try:
            date.fromisoformat(f"{raw[:4]}-{raw[4:6]}-{raw[6:]}")
        except ValueError as exc:
            raise ValueError("expire_day must be a valid explicit YYYYMMDD date") from exc
        return value

    @model_validator(mode="after")
    def validate_order_semantics(self) -> CashOrderIntent:
        expected_reference = 1 if self.exchange in (9, 27) else self.exchange
        if self.reference_exchange != expected_reference:
            raise ValueError(
                "reference_exchange must equal the direct exchange, or 1 for SOR/TSE+ routing"
            )
        return self

    @classmethod
    def from_json(
        cls,
        payload: str | bytes,
        *,
        max_bytes: int = MAX_CASH_ORDER_INTENT_BYTES,
    ) -> Self:
        """Load bounded JSON while rejecting duplicate keys at every object level."""
        return cls.model_validate(_bounded_json_object(payload, max_bytes=max_bytes))

    def canonical_json(self) -> str:
        """Return deterministic JSON used for explicit human confirmation."""
        payload = self.model_dump(mode="json")
        payload["created_at"] = _format_utc(self.created_at)
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def confirmation_digest(self) -> str:
        """Return the SHA-256 digest of the complete canonical intent."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_cash_order_intent(
    payload: str | bytes,
    *,
    max_bytes: int = MAX_CASH_ORDER_INTENT_BYTES,
) -> CashOrderIntent:
    """Convenience boundary for loading a cash order intent."""
    return CashOrderIntent.from_json(payload, max_bytes=max_bytes)
