from __future__ import annotations

from pathlib import Path

import pytest

from nlp_trader.broker.audit import BrokerAuditLedger
from nlp_trader.broker.state import KabuSStateLockError, advisory_file_lock


def test_broker_public_lock_errors_and_audit_append_remain_compatible(tmp_path: Path) -> None:
    lock = (tmp_path / "operation.lock").resolve()
    with (
        advisory_file_lock(lock),
        pytest.raises(
            KabuSStateLockError,
            match="another broker operation already holds the lock",
        ),
        advisory_file_lock(lock),
    ):
        pytest.fail("contended broker lock body must not run")

    ledger = BrokerAuditLedger((tmp_path / "audit.jsonl").resolve())
    event = ledger.append(
        {
            "event_type": "broker_reconciliation",
            "event_ts": "2026-07-16T00:00:00Z",
        }
    )
    assert ledger.replay() == [event]
