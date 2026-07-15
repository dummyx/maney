from __future__ import annotations

import json

import pytest

from nlp_trader.paper.ledger import (
    GENESIS_EVENT_HASH,
    PaperEventLedger,
    PaperLedgerValidationError,
)


def _event(asof_ts: str, *, equity: float = 100_000.0) -> dict[str, object]:
    return {
        "event_type": "paper_mark_to_market",
        "asof_ts": asof_ts,
        "simulation_only": True,
        "equity": equity,
        "details": {"z": 2, "a": 1},
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_paper_ledger_is_append_only_deterministic_and_replayable(tmp_path) -> None:
    path = tmp_path / "paper_events.jsonl"
    ledger = PaperEventLedger(path)

    first = ledger.append(_event("2026-07-02T05:00:00+09:00"))
    first_bytes = path.read_bytes()
    second = ledger.append(_event("2026-07-02T20:00:00Z", equity=101_000.0))

    assert path.read_bytes().startswith(first_bytes)
    assert first["sequence"] == 1
    assert first["asof_ts"] == "2026-07-01T20:00:00Z"
    assert first["previous_event_hash"] == GENESIS_EVENT_HASH
    assert second["sequence"] == 2
    assert second["previous_event_hash"] == first["event_hash"]
    assert ledger.replay() == [first, second]

    equivalent = {
        "details": {"a": 1, "z": 2},
        "equity": 100_000.0,
        "simulation_only": True,
        "asof_ts": "2026-07-01T20:00:00Z",
        "event_type": "paper_mark_to_market",
    }
    other = PaperEventLedger(tmp_path / "equivalent.jsonl").append(equivalent)
    assert other["event_hash"] == first["event_hash"]


@pytest.mark.parametrize(
    ("event", "message"),
    [
        ({"event_type": "paper_test", "asof_ts": "2026-07-01T00:00:00Z"}, "simulation_only"),
        (
            {
                "event_type": "paper_test",
                "asof_ts": "2026-07-01T00:00:00Z",
                "simulation_only": False,
            },
            "simulation_only",
        ),
        (
            {
                "event_type": "paper_test",
                "asof_ts": "2026-07-01T00:00:00",
                "simulation_only": True,
            },
            "timezone-aware",
        ),
        (
            {
                "event_type": "rebalance",
                "asof_ts": "2026-07-01T00:00:00Z",
                "simulation_only": True,
            },
            "paper_ event_type",
        ),
    ],
)
def test_paper_ledger_requires_paper_safety_fields(
    tmp_path, event: dict[str, object], message: str
) -> None:
    ledger = PaperEventLedger(tmp_path / "paper_events.jsonl")

    with pytest.raises(PaperLedgerValidationError, match=message):
        ledger.append(event)


def test_paper_ledger_rejects_timestamp_regression_and_tampering(tmp_path) -> None:
    path = tmp_path / "paper_events.jsonl"
    ledger = PaperEventLedger(path)
    ledger.append(_event("2026-07-02T20:00:00Z"))

    with pytest.raises(PaperLedgerValidationError, match="asof_ts order"):
        ledger.append(_event("2026-07-01T20:00:00Z"))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["equity"] = 1.0
    path.write_text(
        "\n".join(_canonical_json(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PaperLedgerValidationError, match="event_hash mismatch"):
        ledger.replay()
    with pytest.raises(PaperLedgerValidationError, match="event_hash mismatch"):
        ledger.append(_event("2026-07-03T20:00:00Z"))


def test_paper_ledger_replay_rejects_sequence_gaps_and_partial_lines(tmp_path) -> None:
    path = tmp_path / "paper_events.jsonl"
    ledger = PaperEventLedger(path)
    ledger.append(_event("2026-07-01T20:00:00Z"))
    ledger.append(_event("2026-07-02T20:00:00Z"))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1]["sequence"] = 3
    path.write_text(
        "\n".join(_canonical_json(record) for record in records) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PaperLedgerValidationError, match="expected 2"):
        ledger.replay()

    partial_path = tmp_path / "partial.jsonl"
    partial_path.write_text(json.dumps(records[0]), encoding="utf-8")
    with pytest.raises(PaperLedgerValidationError, match="incomplete"):
        PaperEventLedger(partial_path).replay()


def test_paper_ledger_replay_rejects_duplicate_keys_and_noncanonical_json(tmp_path) -> None:
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_ledger = PaperEventLedger(duplicate_path)
    duplicate_ledger.append(_event("2026-07-01T20:00:00Z"))
    duplicate_path.write_text(
        duplicate_path.read_text(encoding="utf-8").replace(
            "{",
            '{"simulation_only":false,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(PaperLedgerValidationError, match="repeats JSON key 'simulation_only'"):
        duplicate_ledger.replay()

    noncanonical_path = tmp_path / "noncanonical.jsonl"
    noncanonical_ledger = PaperEventLedger(noncanonical_path)
    noncanonical_ledger.append(_event("2026-07-01T20:00:00Z"))
    record = json.loads(noncanonical_path.read_text(encoding="utf-8"))
    noncanonical_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(PaperLedgerValidationError, match="not canonical JSON"):
        noncanonical_ledger.replay()

    crlf_path = tmp_path / "crlf.jsonl"
    crlf_ledger = PaperEventLedger(crlf_path)
    crlf_ledger.append(_event("2026-07-01T20:00:00Z"))
    crlf_path.write_bytes(crlf_path.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(PaperLedgerValidationError, match="not canonical JSON"):
        crlf_ledger.replay()
