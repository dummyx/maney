from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import date

import pytest

from nlp_trader.broker.audit import (
    GENESIS_EVENT_HASH,
    BrokerAuditLedger,
    BrokerAuditLockError,
    BrokerAuditValidationError,
)
from nlp_trader.broker.state import advisory_file_lock


def _event(
    event_ts: str = "2026-07-15T09:00:00+09:00",
    *,
    event_type: str = "broker_reconciliation",
    **values: object,
) -> dict[str, object]:
    return {"event_type": event_type, "event_ts": event_ts, **values}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _rehash(record: dict[str, object]) -> None:
    unhashed = {key: value for key, value in record.items() if key != "event_hash"}
    record["event_hash"] = hashlib.sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()


def test_broker_audit_is_canonical_append_only_and_replayable(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = BrokerAuditLedger(path)

    first = ledger.append(_event(details={"z": 2, "a": 1}))
    first_bytes = path.read_bytes()
    second = ledger.append(_event("2026-07-15T01:00:00Z", event_type="broker_kill_switch"))

    assert path.read_bytes().startswith(first_bytes)
    assert first["event_ts"] == "2026-07-15T00:00:00Z"
    assert first["sequence"] == 1
    assert first["previous_event_hash"] == GENESIS_EVENT_HASH
    assert second["sequence"] == 2
    assert second["previous_event_hash"] == first["event_hash"]
    assert ledger.replay() == [first, second]
    assert ledger.lock_path.exists()


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (_event(event_type="paper_order"), "broker_ event_type"),
        (_event(event_type="broker_"), "non-empty"),
        (_event("2026-07-15T09:00:00"), "timezone-aware"),
        (_event(sequence=1), "assigned internally"),
        (_event(previous_event_hash="0" * 64), "assigned internally"),
        (_event(event_hash="0" * 64), "assigned internally"),
        (_event(value=float("nan")), "finite"),
        (_event(value=(1, 2)), "not JSON serializable"),
        (_event(value=date(2026, 7, 15)), "not JSON serializable"),
        (_event(value={1: "bad"}), "keys must be strings"),
    ],
)
def test_broker_audit_rejects_invalid_caller_events(
    tmp_path, event: dict[str, object], message: str
) -> None:
    with pytest.raises(BrokerAuditValidationError, match=message):
        BrokerAuditLedger(tmp_path / "events.jsonl").append(event)


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "password",
        "API-Password",
        "access_token",
        "clientSecret",
        "api_key",
        "ApiKey",
        "Authorization",
        "X_API-Key",
    ],
)
def test_broker_audit_rejects_sensitive_keys_recursively(tmp_path, sensitive_key: str) -> None:
    event = _event(details=[{"safe": {sensitive_key: "must-not-be-written"}}])

    with pytest.raises(BrokerAuditValidationError, match="sensitive key"):
        BrokerAuditLedger(tmp_path / "events.jsonl").append(event)
    assert not (tmp_path / "events.jsonl").exists()


def test_broker_audit_replay_rejects_nested_secret_even_with_recomputed_hash(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = BrokerAuditLedger(path)
    ledger.append(_event())
    record = json.loads(path.read_text(encoding="utf-8"))
    record["response"] = {"headers": {"X-API-KEY": "must-not-be-retained"}}
    _rehash(record)
    path.write_text(_canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(BrokerAuditValidationError, match="sensitive key"):
        ledger.replay()


def test_broker_audit_rejects_tampering_sequence_hash_link_and_timestamp_regression(
    tmp_path,
) -> None:
    tampered_path = tmp_path / "tampered.jsonl"
    tampered = BrokerAuditLedger(tampered_path)
    tampered.append(_event())
    record = json.loads(tampered_path.read_text(encoding="utf-8"))
    record["details"] = {"changed": True}
    tampered_path.write_text(_canonical_json(record) + "\n", encoding="utf-8")
    with pytest.raises(BrokerAuditValidationError, match="event_hash mismatch"):
        tampered.replay()

    sequence_path = tmp_path / "sequence.jsonl"
    sequence = BrokerAuditLedger(sequence_path)
    sequence.append(_event())
    sequence_record = json.loads(sequence_path.read_text(encoding="utf-8"))
    sequence_record["sequence"] = 2
    sequence_path.write_text(_canonical_json(sequence_record) + "\n", encoding="utf-8")
    with pytest.raises(BrokerAuditValidationError, match="expected 1"):
        sequence.replay()

    link_path = tmp_path / "link.jsonl"
    link = BrokerAuditLedger(link_path)
    link.append(_event())
    link.append(_event("2026-07-15T01:00:00Z"))
    records = [json.loads(line) for line in link_path.read_text(encoding="utf-8").splitlines()]
    records[1]["previous_event_hash"] = "f" * 64
    _rehash(records[1])
    link_path.write_text(
        "\n".join(_canonical_json(item) for item in records) + "\n", encoding="utf-8"
    )
    with pytest.raises(BrokerAuditValidationError, match="previous hash link"):
        link.replay()

    timestamp_path = tmp_path / "timestamp.jsonl"
    timestamp = BrokerAuditLedger(timestamp_path)
    timestamp.append(_event("2026-07-15T01:00:00Z"))
    timestamp.append(_event("2026-07-15T02:00:00Z"))
    records = [json.loads(line) for line in timestamp_path.read_text(encoding="utf-8").splitlines()]
    records[1]["event_ts"] = "2026-07-15T00:00:00Z"
    _rehash(records[1])
    timestamp_path.write_text(
        "\n".join(_canonical_json(item) for item in records) + "\n", encoding="utf-8"
    )
    with pytest.raises(BrokerAuditValidationError, match="regresses event_ts"):
        timestamp.replay()


def test_broker_audit_rejects_duplicate_blank_partial_and_noncanonical_lines(tmp_path) -> None:
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate = BrokerAuditLedger(duplicate_path)
    duplicate.append(_event())
    duplicate_path.write_text(
        duplicate_path.read_text(encoding="utf-8").replace(
            "{", '{"event_type":"broker_injected",', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(BrokerAuditValidationError, match="repeats JSON key 'event_type'"):
        duplicate.replay()

    blank_path = tmp_path / "blank.jsonl"
    blank_path.write_text("\n", encoding="utf-8")
    with pytest.raises(BrokerAuditValidationError, match="blank"):
        BrokerAuditLedger(blank_path).replay()

    partial_path = tmp_path / "partial.jsonl"
    partial_path.write_text("{}", encoding="utf-8")
    with pytest.raises(BrokerAuditValidationError, match="incomplete"):
        BrokerAuditLedger(partial_path).replay()

    noncanonical_path = tmp_path / "noncanonical.jsonl"
    noncanonical = BrokerAuditLedger(noncanonical_path)
    noncanonical.append(_event())
    record = json.loads(noncanonical_path.read_text(encoding="utf-8"))
    noncanonical_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(BrokerAuditValidationError, match="not canonical JSON"):
        noncanonical.replay()


def test_broker_audit_lock_is_exclusive_and_stale_metadata_is_replaced(tmp_path) -> None:
    ledger = BrokerAuditLedger(tmp_path / "events.jsonl")
    ledger.lock_path.write_text("stale\n", encoding="utf-8")

    with advisory_file_lock(ledger.lock_path):
        with pytest.raises(BrokerAuditLockError, match="unavailable"):
            ledger.replay()
        with pytest.raises(BrokerAuditLockError, match="unavailable"):
            ledger.append(_event())

    ledger.append(_event())
    assert ledger.lock_path.exists()
    assert b"stale" not in ledger.lock_path.read_bytes()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are not portable")
def test_broker_audit_creates_private_ledger_file(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    BrokerAuditLedger(path).append(_event())

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_broker_audit_derives_ambiguity_identity_and_accepted_notional(tmp_path) -> None:
    ledger = BrokerAuditLedger(tmp_path / "events.jsonl")
    ledger.append(_event(event_type="broker_order_attempt", client_order_id="order-a"))
    ledger.append(
        _event(
            "2026-07-15T00:01:00Z",
            event_type="broker_order_unknown",
            client_order_id="order-a",
        )
    )
    assert ledger.unresolved_order_ids() == ("order-a",)
    ledger.append(
        _event(
            "2026-07-15T00:02:00Z",
            event_type="broker_order_attempt",
            client_order_id="order-b",
        )
    )
    ledger.append(
        _event(
            "2026-07-15T00:03:00Z",
            event_type="broker_order_accepted",
            client_order_id="order-b",
            notional_jpy=1_250.0,
            order_date="2026-07-15",
        )
    )
    assert ledger.unresolved_order_ids() == ("order-a",)
    ledger.append(
        _event(
            "2026-07-15T00:04:00Z",
            event_type="broker_order_accepted",
            client_order_id="order-b",
            notional_jpy=1_250.0,
            order_date="2026-07-15",
        )
    )
    ledger.append(
        _event(
            "2026-07-15T00:05:00Z",
            event_type="broker_cancel_attempt",
            client_order_id="order-b",
        )
    )
    assert ledger.unresolved_order_ids() == ("order-a", "order-b")
    ledger.append(
        _event(
            "2026-07-15T00:06:00Z",
            event_type="broker_cancel_accepted",
            client_order_id="order-b",
        )
    )
    assert ledger.unresolved_order_ids() == ("order-a",)
    ledger.append(
        _event(
            "2026-07-15T15:07:00Z",
            event_type="broker_order_accepted",
            client_order_id="order-c",
            order={"notional_jpy": 2_500, "order_date": "2026-07-16"},
        )
    )
    ledger.append(
        _event(
            "2026-07-15T15:08:00Z",
            event_type="broker_order_rejected",
            client_order_id="order-a",
        )
    )
    assert ledger.unresolved_order_ids() == ()
    ledger.append(
        _event(
            "2026-07-15T15:09:00Z",
            event_type="broker_order_attempt",
            client_order_id="order-d",
        )
    )
    assert ledger.unresolved_order_ids() == ("order-d",)
    with pytest.raises(BrokerAuditValidationError, match="cannot resolve"):
        ledger.append(
            _event(
                "2026-07-15T15:10:00Z",
                event_type="broker_reconciliation",
                resolved_client_order_ids=["order-d"],
            )
        )
    ledger.append(_event("2026-07-15T15:11:00Z", event_type="broker_kill_switch"))

    assert ledger.unresolved_order_ids() == ("order-d",)
    assert ledger.seen_client_order_id("order-a") is True
    assert ledger.seen_client_order_id("missing") is False
    assert ledger.accepted_notional_for_date(date(2026, 7, 15)) == 1_250.0
    assert ledger.accepted_notional_for_date(date(2026, 7, 16)) == 2_500.0


def test_broker_audit_rejects_conflicting_duplicate_acceptance_for_risk_totals(
    tmp_path,
) -> None:
    ledger = BrokerAuditLedger(tmp_path / "events.jsonl")
    ledger.append(
        _event(
            event_type="broker_order_accepted",
            client_order_id="order-a",
            notional_jpy=1_000,
            order_date="2026-07-15",
        )
    )
    ledger.append(
        _event(
            "2026-07-15T00:01:00Z",
            event_type="broker_order_accepted",
            client_order_id="order-a",
            notional_jpy=2_000,
            order_date="2026-07-15",
        )
    )

    with pytest.raises(BrokerAuditValidationError, match="disagree"):
        ledger.accepted_notional_for_date(date(2026, 7, 15))


def test_broker_audit_refuses_a_symlink_ledger(tmp_path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform does not expose no-follow file opens")
    target = tmp_path / "redirected.jsonl"
    target.write_text("", encoding="utf-8")
    path = tmp_path / "events.jsonl"
    path.symlink_to(target)
    ledger = BrokerAuditLedger(path)

    with pytest.raises(BrokerAuditValidationError, match="opened safely"):
        ledger.replay()
    with pytest.raises(BrokerAuditValidationError, match="opened safely"):
        ledger.append(_event())
