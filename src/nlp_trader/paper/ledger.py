from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nlp_trader.timestamps import format_utc, parse_utc

GENESIS_EVENT_HASH = "0" * 64
_RESERVED_FIELDS = frozenset({"sequence", "previous_event_hash", "event_hash"})


class PaperLedgerValidationError(ValueError):
    """Raised when a paper ledger event or hash chain is invalid."""


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


class PaperEventLedger:
    """Append-only, hash-chained JSONL evidence for paper simulation events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and append one paper-only event without rewriting prior bytes."""
        reserved = _RESERVED_FIELDS.intersection(event)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise PaperLedgerValidationError(f"ledger fields are assigned internally: {names}")

        prior_events = self.replay()
        normalized = _normalize_event(event)
        sequence = len(prior_events) + 1
        previous_hash = str(prior_events[-1]["event_hash"]) if prior_events else GENESIS_EVENT_HASH
        if prior_events:
            previous_ts = parse_utc(str(prior_events[-1]["asof_ts"]))
            current_ts = parse_utc(str(normalized["asof_ts"]))
            if current_ts < previous_ts:
                raise PaperLedgerValidationError(
                    "paper ledger events must be appended in asof_ts order"
                )

        record = {
            **normalized,
            "sequence": sequence,
            "previous_event_hash": previous_hash,
        }
        record["event_hash"] = _hash_record(record)
        encoded = (_canonical_json(record) + "\n").encode("utf-8")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(encoded)
        return record

    def replay(self) -> list[dict[str, Any]]:
        """Read and validate every event, returning records in ledger order."""
        if not self.path.exists():
            return []

        events: list[dict[str, Any]] = []
        expected_previous_hash = GENESIS_EVENT_HASH
        previous_ts = None
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise PaperLedgerValidationError(f"paper ledger line {line_number} is blank")
                if not raw_line.endswith("\n"):
                    raise PaperLedgerValidationError(
                        f"paper ledger line {line_number} is incomplete"
                    )
                try:
                    parsed = json.loads(
                        raw_line,
                        object_pairs_hook=_object_without_duplicate_keys,
                    )
                except _DuplicateJsonKeyError as exc:
                    raise PaperLedgerValidationError(
                        f"paper ledger line {line_number} repeats JSON key {exc.key!r}"
                    ) from exc
                except json.JSONDecodeError as exc:
                    raise PaperLedgerValidationError(
                        f"paper ledger line {line_number} is not valid JSON"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise PaperLedgerValidationError(
                        f"paper ledger line {line_number} must contain an object"
                    )

                record = _json_value(parsed)
                if not isinstance(record, dict):  # pragma: no cover - guarded above
                    raise PaperLedgerValidationError(
                        f"paper ledger line {line_number} must contain an object"
                    )
                if raw_line != _canonical_json(record) + "\n":
                    raise PaperLedgerValidationError(
                        f"paper ledger line {line_number} is not canonical JSON"
                    )
                _validate_replayed_record(
                    record,
                    line_number=line_number,
                    expected_sequence=line_number,
                    expected_previous_hash=expected_previous_hash,
                )

                current_ts = parse_utc(str(record["asof_ts"]))
                if previous_ts is not None and current_ts < previous_ts:
                    raise PaperLedgerValidationError(
                        f"paper ledger line {line_number} regresses asof_ts"
                    )
                previous_ts = current_ts
                expected_previous_hash = str(record["event_hash"])
                events.append(record)

        return events


def _normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_value(event)
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping guarantees this
        raise PaperLedgerValidationError("paper ledger event must be an object")
    _validate_paper_fields(normalized, context="paper ledger event")
    normalized["asof_ts"] = format_utc(parse_utc(str(normalized["asof_ts"])))
    _canonical_json(normalized)
    return normalized


def _validate_paper_fields(record: Mapping[str, Any], *, context: str) -> None:
    event_type = record.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise PaperLedgerValidationError(f"{context} requires a non-empty event_type")
    if not event_type.startswith("paper_"):
        raise PaperLedgerValidationError(f"{context} requires a paper_ event_type")
    if record.get("simulation_only") is not True:
        raise PaperLedgerValidationError(f"{context} requires simulation_only=true")
    asof_ts = record.get("asof_ts")
    if not isinstance(asof_ts, str):
        raise PaperLedgerValidationError(f"{context} requires a timezone-aware asof_ts")
    try:
        parse_utc(asof_ts)
    except (TypeError, ValueError) as exc:
        raise PaperLedgerValidationError(f"{context} requires a timezone-aware asof_ts") from exc


def _validate_replayed_record(
    record: dict[str, Any],
    *,
    line_number: int,
    expected_sequence: int,
    expected_previous_hash: str,
) -> None:
    context = f"paper ledger line {line_number}"
    _validate_paper_fields(record, context=context)
    canonical_asof_ts = format_utc(parse_utc(str(record["asof_ts"])))
    if record["asof_ts"] != canonical_asof_ts:
        raise PaperLedgerValidationError(f"{context} has non-canonical asof_ts")

    sequence = record.get("sequence")
    if type(sequence) is not int or sequence != expected_sequence:
        raise PaperLedgerValidationError(
            f"{context} has sequence {sequence!r}; expected {expected_sequence}"
        )
    if record.get("previous_event_hash") != expected_previous_hash:
        raise PaperLedgerValidationError(f"{context} breaks the previous hash link")

    event_hash = record.get("event_hash")
    if not isinstance(event_hash, str) or len(event_hash) != 64:
        raise PaperLedgerValidationError(f"{context} has an invalid event_hash")
    unhashed = {key: value for key, value in record.items() if key != "event_hash"}
    if event_hash != _hash_record(unhashed):
        raise PaperLedgerValidationError(f"{context} has an event_hash mismatch")


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
        raise PaperLedgerValidationError(
            "paper ledger values must be finite canonical JSON data"
        ) from exc


def _json_value(value: object) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PaperLedgerValidationError("paper ledger object keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise PaperLedgerValidationError(
        f"paper ledger value is not JSON serializable: {type(value).__name__}"
    )
