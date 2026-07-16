from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nlp_trader.broker.state import KabuSStateLockError, advisory_file_lock
from nlp_trader.immutable.append import SafeFileError, append_bytes_durable
from nlp_trader.timestamps import format_utc, parse_utc

GENESIS_EVENT_HASH = "0" * 64
_RESERVED_FIELDS = frozenset({"sequence", "previous_event_hash", "event_hash"})
_SENSITIVE_KEY_PARTS = frozenset(
    {"password", "token", "secret", "apikey", "authorization", "xapikey"}
)
_CLIENT_ORDER_EVENTS = frozenset(
    {
        "broker_order_attempt",
        "broker_order_accepted",
        "broker_order_rejected",
        "broker_order_unknown",
        "broker_cancel_attempt",
        "broker_cancel_accepted",
        "broker_cancel_rejected",
        "broker_cancel_unknown",
    }
)
_UNRESOLVED_EVENTS = frozenset(
    {
        "broker_order_attempt",
        "broker_order_unknown",
        "broker_cancel_attempt",
        "broker_cancel_unknown",
    }
)
_RESOLVED_EVENTS = frozenset(
    {
        "broker_order_accepted",
        "broker_order_rejected",
        "broker_cancel_accepted",
        "broker_cancel_rejected",
    }
)
_JAPAN = ZoneInfo("Asia/Tokyo")
_KEY_FORMAT = re.compile(r"[^a-z0-9]")


class BrokerAuditValidationError(ValueError):
    """Raised when broker audit evidence is invalid or unsafe to retain."""


class BrokerAuditLockError(RuntimeError):
    """Raised when exclusive broker audit ledger ownership cannot be obtained."""


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


class _NonFiniteJsonError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _NonFiniteJsonError(value)


class BrokerAuditLedger:
    """Durable, append-only, hash-chained evidence for broker operations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and durably append one broker event under exclusive ownership."""

        with self._exclusive_lock():
            prior_events = self._replay_unlocked()
            normalized = _normalize_event(event)
            sequence = len(prior_events) + 1
            previous_hash = (
                str(prior_events[-1]["event_hash"]) if prior_events else GENESIS_EVENT_HASH
            )
            if prior_events:
                previous_ts = parse_utc(str(prior_events[-1]["event_ts"]))
                current_ts = parse_utc(str(normalized["event_ts"]))
                if current_ts < previous_ts:
                    raise BrokerAuditValidationError(
                        "broker audit events must be appended in event_ts order"
                    )

            record = {
                **normalized,
                "sequence": sequence,
                "previous_event_hash": previous_hash,
            }
            record["event_hash"] = _hash_record(record)
            encoded = (_canonical_json(record) + "\n").encode("utf-8")
            self._append_bytes(encoded)
            return record

    def replay(self) -> list[dict[str, Any]]:
        """Read and validate the complete ledger while holding its exclusive lock."""

        with self._exclusive_lock():
            return self._replay_unlocked()

    def unresolved_order_ids(self) -> tuple[str, ...]:
        """Return order IDs whose latest mutating attempt has no authoritative outcome."""

        unresolved: set[str] = set()
        for event in self.replay():
            event_type = str(event["event_type"])
            if event_type in _UNRESOLVED_EVENTS:
                unresolved.add(str(event["client_order_id"]))
            elif event_type in _RESOLVED_EVENTS:
                unresolved.discard(str(event["client_order_id"]))
        return tuple(sorted(unresolved))

    def seen_client_order_id(self, client_order_id: str) -> bool:
        """Return whether an event has already recorded this client order identity."""

        if not client_order_id:
            raise ValueError("client_order_id must not be empty")
        return any(event.get("client_order_id") == client_order_id for event in self.replay())

    def accepted_notional_for_date(self, order_date: date) -> float:
        """Sum unique accepted-order notional for a Japan trading date."""

        if isinstance(order_date, datetime):
            raise TypeError("order_date must be a date, not a datetime")
        accepted: dict[str, tuple[date, float]] = {}
        total = 0.0
        for event in self.replay():
            if event["event_type"] != "broker_order_accepted":
                continue
            client_order_id = str(event["client_order_id"])
            accepted_value = (_event_order_date(event), _accepted_notional(event))
            prior = accepted.get(client_order_id)
            if prior is not None:
                if prior != accepted_value:
                    raise BrokerAuditValidationError(
                        "duplicate broker_order_accepted events disagree for "
                        f"client_order_id {client_order_id!r}"
                    )
                continue
            accepted[client_order_id] = accepted_value
            if accepted_value[0] == order_date:
                total += accepted_value[1]
        if not math.isfinite(total):
            raise BrokerAuditValidationError("accepted broker notional total is not finite")
        return total

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        try:
            with advisory_file_lock(self.lock_path):
                yield
        except KabuSStateLockError as exc:
            raise BrokerAuditLockError("broker audit ledger lock is unavailable") from exc

    def _append_bytes(self, encoded: bytes) -> None:
        path = self.path.expanduser().absolute()
        try:
            append_bytes_durable(path, encoded)
        except SafeFileError as exc:
            message = (
                "broker audit ledger append failed"
                if exc.operation == "append"
                else "broker audit ledger cannot be opened safely for append"
            )
            raise BrokerAuditValidationError(message) from exc
        except (OSError, ValueError) as exc:
            raise BrokerAuditValidationError(
                "broker audit ledger cannot be opened safely for append"
            ) from exc

    def _replay_unlocked(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        expected_previous_hash = GENESIS_EVENT_HASH
        previous_ts: datetime | None = None
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise BrokerAuditValidationError("broker audit ledger cannot be opened safely") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise BrokerAuditValidationError("broker audit ledger must be a regular file")
            with os.fdopen(descriptor, "r", encoding="utf-8", newline="") as handle:
                descriptor = -1
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        raise BrokerAuditValidationError(
                            f"broker audit ledger line {line_number} is blank"
                        )
                    if not raw_line.endswith("\n"):
                        raise BrokerAuditValidationError(
                            f"broker audit ledger line {line_number} is incomplete"
                        )
                    record = _parse_record(raw_line, line_number=line_number)
                    if raw_line != _canonical_json(record) + "\n":
                        raise BrokerAuditValidationError(
                            f"broker audit ledger line {line_number} is not canonical JSON"
                        )
                    _validate_replayed_record(
                        record,
                        line_number=line_number,
                        expected_sequence=line_number,
                        expected_previous_hash=expected_previous_hash,
                    )

                    current_ts = parse_utc(str(record["event_ts"]))
                    if previous_ts is not None and current_ts < previous_ts:
                        raise BrokerAuditValidationError(
                            f"broker audit ledger line {line_number} regresses event_ts"
                        )
                    previous_ts = current_ts
                    expected_previous_hash = str(record["event_hash"])
                    events.append(record)
        except UnicodeError as exc:
            raise BrokerAuditValidationError("broker audit ledger is not canonical UTF-8") from exc
        except OSError as exc:
            raise BrokerAuditValidationError("broker audit ledger read failed") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return events


def _parse_record(raw_line: str, *, line_number: int) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw_line,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKeyError as exc:
        raise BrokerAuditValidationError(
            f"broker audit ledger line {line_number} repeats JSON key {exc.key!r}"
        ) from exc
    except _NonFiniteJsonError as exc:
        raise BrokerAuditValidationError(
            f"broker audit ledger line {line_number} contains a non-finite number"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BrokerAuditValidationError(
            f"broker audit ledger line {line_number} is not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise BrokerAuditValidationError(
            f"broker audit ledger line {line_number} must contain an object"
        )
    record = _json_value(parsed)
    if not isinstance(record, dict):  # pragma: no cover - guarded above
        raise BrokerAuditValidationError(
            f"broker audit ledger line {line_number} must contain an object"
        )
    return record


def _normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_value(event)
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping guarantees this
        raise BrokerAuditValidationError("broker audit event must be an object")
    reserved = _RESERVED_FIELDS.intersection(normalized)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise BrokerAuditValidationError(f"broker audit fields are assigned internally: {names}")
    _validate_broker_fields(normalized, context="broker audit event")
    normalized["event_ts"] = format_utc(parse_utc(str(normalized["event_ts"])))
    _canonical_json(normalized)
    return normalized


def _validate_broker_fields(record: Mapping[str, Any], *, context: str) -> None:
    _reject_sensitive_keys(record, context=context)
    event_type = record.get("event_type")
    if not isinstance(event_type, str) or not event_type.startswith("broker_"):
        raise BrokerAuditValidationError(f"{context} requires a broker_ event_type")
    if event_type == "broker_":
        raise BrokerAuditValidationError(f"{context} requires a non-empty broker_ event type")
    event_ts = record.get("event_ts")
    if not isinstance(event_ts, str):
        raise BrokerAuditValidationError(f"{context} requires a timezone-aware event_ts")
    try:
        parse_utc(event_ts)
    except (TypeError, ValueError) as exc:
        raise BrokerAuditValidationError(f"{context} requires a timezone-aware event_ts") from exc

    if event_type in _CLIENT_ORDER_EVENTS:
        client_order_id = record.get("client_order_id")
        if not isinstance(client_order_id, str) or not client_order_id.strip():
            raise BrokerAuditValidationError(
                f"{context} {event_type} requires a non-empty client_order_id"
            )
    if event_type == "broker_order_accepted":
        _accepted_notional(record)
        _event_order_date(record)
    if event_type == "broker_reconciliation" and "resolved_client_order_ids" in record:
        raise BrokerAuditValidationError(
            f"{context} reconciliation cannot resolve broker mutations"
        )


def _validate_replayed_record(
    record: dict[str, Any],
    *,
    line_number: int,
    expected_sequence: int,
    expected_previous_hash: str,
) -> None:
    context = f"broker audit ledger line {line_number}"
    _validate_broker_fields(record, context=context)
    canonical_event_ts = format_utc(parse_utc(str(record["event_ts"])))
    if record["event_ts"] != canonical_event_ts:
        raise BrokerAuditValidationError(f"{context} has non-canonical event_ts")

    sequence = record.get("sequence")
    if type(sequence) is not int or sequence != expected_sequence:
        raise BrokerAuditValidationError(
            f"{context} has sequence {sequence!r}; expected {expected_sequence}"
        )
    if record.get("previous_event_hash") != expected_previous_hash:
        raise BrokerAuditValidationError(f"{context} breaks the previous hash link")

    event_hash = record.get("event_hash")
    if not isinstance(event_hash, str) or len(event_hash) != 64:
        raise BrokerAuditValidationError(f"{context} has an invalid event_hash")
    unhashed = {key: value for key, value in record.items() if key != "event_hash"}
    if event_hash != _hash_record(unhashed):
        raise BrokerAuditValidationError(f"{context} has an event_hash mismatch")


def _accepted_notional(event: Mapping[str, Any]) -> float:
    value = _event_or_order_value(event, "notional_jpy")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BrokerAuditValidationError("broker_order_accepted requires numeric notional_jpy")
    try:
        result = float(value)
    except OverflowError as exc:
        raise BrokerAuditValidationError(
            "broker_order_accepted requires finite non-negative notional_jpy"
        ) from exc
    if not math.isfinite(result) or result < 0:
        raise BrokerAuditValidationError(
            "broker_order_accepted requires finite non-negative notional_jpy"
        )
    return result


def _event_order_date(event: Mapping[str, Any]) -> date:
    explicit = _event_or_order_value(event, "order_date")
    if explicit is None:
        return parse_utc(str(event["event_ts"])).astimezone(_JAPAN).date()
    if not isinstance(explicit, str):
        raise BrokerAuditValidationError("broker order_date must be an ISO date string")
    try:
        return date.fromisoformat(explicit)
    except ValueError as exc:
        raise BrokerAuditValidationError("broker order_date must be an ISO date string") from exc


def _event_or_order_value(event: Mapping[str, Any], name: str) -> Any:
    if name in event:
        return event[name]
    order = event.get("order")
    if isinstance(order, Mapping):
        return order.get(name)
    return None


def _reject_sensitive_keys(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _KEY_FORMAT.sub("", key.casefold())
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                raise BrokerAuditValidationError(
                    f"{context} contains forbidden sensitive key {key!r}"
                )
            _reject_sensitive_keys(item, context=context)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_keys(item, context=context)


def _hash_record(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BrokerAuditValidationError(
            "broker audit values must be finite canonical JSON data"
        ) from exc


def _json_value(value: object) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BrokerAuditValidationError("broker audit object keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BrokerAuditValidationError("broker audit numbers must be finite")
        return value
    raise BrokerAuditValidationError(
        f"broker audit value is not JSON serializable: {type(value).__name__}"
    )
