from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nlp_trader.immutable.locking import advisory_file_lock
from nlp_trader.research_agents.contracts import (
    GENESIS_HASH,
    GenerationDiagnostics,
    ResearchAgentRound,
    StudyDefinition,
    canonical_json,
)
from nlp_trader.research_agents.ledger import (
    ResearchAgentRoundLedger,
    ResearchAgentRoundLedgerError,
    ResearchAgentRoundLedgerLockError,
)


def _round(
    study: StudyDefinition,
    *,
    step: int = 1,
    previous_round_hash: str = GENESIS_HASH,
    agent_run_id: str = "agent-run-1",
    attempt_id: str = "2" * 64,
) -> ResearchAgentRound:
    return ResearchAgentRound(
        agent_run_id=agent_run_id,
        study_id=study.study_id,
        attempt_id=attempt_id,
        step=step,
        previous_round_hash=previous_round_hash,
        model=study.model,
        prompt_contract=study.prompt_contract,
        action_schema_contract=study.action_schema_contract,
        proposal_schema_contract=study.proposal_schema_contract,
        verifier_contract=study.verifier_contract,
        tool_catalog_contract=study.tool_catalog_contract,
        bundle_id="3" * 64,
        input_snapshot_hash="4" * 64,
        attempt_reservation_event_hash="5" * 64,
        reserved_study_state_hash="6" * 64,
        context_hash="7" * 64,
        raw_generation='{"action_type":"abstention","abstention":{"reason":"insufficient_evidence"}}',
        parse_status="passed",
        parsed_action={
            "action_type": "abstention",
            "abstention": {
                "study_id": study.study_id,
                "attempt_id": attempt_id,
                "bundle_id": "3" * 64,
                "input_snapshot_hash": "4" * 64,
                "reason": "insufficient_evidence",
                "explanation": "The sealed snapshot has insufficient retained evidence.",
                "missing_input_ids": (),
                "tool_query_ids": (),
                "resolvable_in_new_study": True,
            },
        },
        origin="generated",
        diagnostics=GenerationDiagnostics(
            input_tokens=100,
            output_tokens=10,
            latency_ms=25.0,
            throughput_tokens_per_second=400.0,
            peak_memory_bytes=1024,
            requested_gpu_layers=0,
            effective_gpu_layers=0,
            device_path="injected",
        ),
        termination_reason="abstention",
    )


def test_round_ledger_appends_replays_and_binds_one_attempt(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchAgentRoundLedger((tmp_path / "rounds.jsonl").resolve())
    first = _round(research_study_definition)
    second = _round(
        research_study_definition,
        step=2,
        previous_round_hash=first.round_id,
    )

    ledger.append(first)
    first_bytes = ledger.path.read_bytes()
    ledger.append(second)

    assert ledger.path.read_bytes().startswith(first_bytes)
    assert ledger.replay() == (first, second)
    assert second.round_id == second.computed_round_id()
    with pytest.raises(ResearchAgentRoundLedgerError, match="run, study, and attempt"):
        ledger.append(
            _round(
                research_study_definition,
                step=3,
                previous_round_hash=second.round_id,
                attempt_id="8" * 64,
            )
        )


def test_round_ledger_rejects_step_and_previous_hash_mismatches(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchAgentRoundLedger((tmp_path / "rounds.jsonl").resolve())
    with pytest.raises(ResearchAgentRoundLedgerError, match="step must be 1"):
        ledger.append(_round(research_study_definition, step=2))
    with pytest.raises(ResearchAgentRoundLedgerError, match="previous hash"):
        ledger.append(_round(research_study_definition, previous_round_hash="9" * 64))


def test_round_ledger_replay_rejects_tamper_duplicate_partial_and_invalid_utf8(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    tampered = ResearchAgentRoundLedger((tmp_path / "tampered.jsonl").resolve())
    tampered.append(_round(research_study_definition))
    record = json.loads(tampered.path.read_text(encoding="utf-8"))
    record["raw_generation"] = "changed"
    tampered.path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    with pytest.raises(ResearchAgentRoundLedgerError, match="contract"):
        tampered.replay()

    duplicate = ResearchAgentRoundLedger((tmp_path / "duplicate.jsonl").resolve())
    duplicate.append(_round(research_study_definition))
    raw = duplicate.path.read_text(encoding="utf-8")
    duplicate.path.write_text(raw.replace("{", '{"step":1,', 1), encoding="utf-8")
    with pytest.raises(ResearchAgentRoundLedgerError, match="repeats JSON key"):
        duplicate.replay()

    partial = ResearchAgentRoundLedger((tmp_path / "partial.jsonl").resolve())
    partial.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ResearchAgentRoundLedgerError, match="incomplete"):
        partial.replay()

    invalid_utf8 = ResearchAgentRoundLedger((tmp_path / "invalid.jsonl").resolve())
    invalid_utf8.path.write_bytes(b"\xff\n")
    with pytest.raises(ResearchAgentRoundLedgerError, match="UTF-8"):
        invalid_utf8.replay()


def test_round_ledger_rejects_symlink_and_lock_contention(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchAgentRoundLedger((tmp_path / "rounds.jsonl").resolve())
    with (
        advisory_file_lock(ledger.lock_path),
        pytest.raises(ResearchAgentRoundLedgerLockError, match="unavailable"),
    ):
        ledger.replay()

    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform does not expose no-follow file opens")
    target = (tmp_path / "target.jsonl").resolve()
    target.write_text("", encoding="utf-8")
    ledger.path.symlink_to(target)
    with pytest.raises(ResearchAgentRoundLedgerError, match="opened safely"):
        ledger.append(_round(research_study_definition))
    assert target.read_text(encoding="utf-8") == ""
