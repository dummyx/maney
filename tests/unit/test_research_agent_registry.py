from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from nlp_trader.immutable.locking import advisory_file_lock
from nlp_trader.research_agents.contracts import (
    HoldoutIdentity,
    ProposalAttemptCompletedPayload,
    RegistryEvent,
    StudyDefinition,
    canonical_json,
)
from nlp_trader.research_agents.registry import (
    ResearchRegistryError,
    ResearchRegistryLedger,
    ResearchRegistryLockError,
    StaleRegistryHeadError,
)


def _reserve_worker(
    artifact_root: str,
    study_id: str,
    expected_head_hash: str,
    start: Any,
    output: Any,
) -> None:
    start.wait(10)
    try:
        event = ResearchRegistryLedger(artifact_root).reserve_proposal_attempt(
            study_id,
            expected_head_hash=expected_head_hash,
            actor_label=f"worker-{os.getpid()}",
        )
        output.put(("ok", event.payload.attempt_id))
    except Exception as exc:  # noqa: BLE001 - child reports exact class to its parent test
        output.put((type(exc).__name__, str(exc)))


def _holdout(study: StudyDefinition, *, assets: tuple[str, ...]) -> HoldoutIdentity:
    return HoldoutIdentity(
        data_lineage_id=study.data_lineage_id,
        input_snapshot_hashes=("1" * 64,),
        universe_snapshot_id=study.universe_snapshot_id,
        universe_asset_ids=assets,
        calendar_contract=study.calendar_contract,
        market_data_contract=study.market_data_contract,
        label_contract=study.label_contract,
        target_family=study.target_family,
        horizon_sessions=study.horizon_sessions,
        return_adjustment_contract=study.return_adjustment_contract,
        decision_interval=study.reserved_holdout_decisions,
        outcome_interval=study.reserved_holdout_outcomes,
        study_id=study.study_id,
        candidate_hash="2" * 64,
    )


def _distinct_study(study: StudyDefinition, *, label: str) -> StudyDefinition:
    payload = study.model_dump(mode="python", exclude={"study_id"})
    payload["research_question"] = f"{study.research_question} {label}"
    return StudyDefinition.model_validate(payload)


def _freeze_study_for_holdout(
    ledger: ResearchRegistryLedger,
    study: StudyDefinition,
    *,
    assets: tuple[str, ...],
) -> HoldoutIdentity:
    ledger.register_study(
        study,
        expected_head_hash=ledger.head_hash(),
        actor_label="reviewer",
    )
    reserved = ledger.reserve_proposal_attempt(
        study.study_id,
        expected_head_hash=ledger.head_hash(),
        actor_label="host",
    )
    completed = ledger.complete_proposal_attempt(
        study.study_id,
        reserved.payload.attempt_id,
        outcome="proposal",
        agent_run_id=f"run-{study.study_id[:12]}",
        detail="Synthetic proposal for registry transition coverage.",
        terminal_artifact_hash="3" * 64,
        expected_head_hash=ledger.head_hash(),
        actor_label="host",
    )
    verified = ledger.record_proposal_verification(
        study.study_id,
        reserved.payload.attempt_id,
        terminal_artifact_hash="3" * 64,
        verification_hash="4" * 64,
        passed=True,
        expected_head_hash=completed.event_hash,
        actor_label="verifier",
    )
    approved = ledger.approve_development_execution(
        study.study_id,
        reserved.payload.attempt_id,
        proposal_verification_hash="4" * 64,
        execution_definition_hash="5" * 64,
        reviewer_reason="Approve synthetic registry transition coverage.",
        expected_head_hash=verified.event_hash,
        actor_label="reviewer",
    )
    started = ledger.start_development_run(
        study.study_id,
        execution_definition_hash="5" * 64,
        expected_head_hash=approved.event_hash,
        actor_label="host",
    )
    development = ledger.complete_development_run(
        study.study_id,
        started.payload.development_run_id,
        result_manifest_hash="6" * 64,
        frozen_model_manifest_hash="7" * 64,
        expected_head_hash=started.event_hash,
        actor_label="host",
    )
    identity = _holdout(study, assets=assets)
    ledger.freeze_candidate(
        study.study_id,
        proposal_hash="3" * 64,
        execution_definition_hash="5" * 64,
        development_approval_event_hash=approved.event_hash,
        development_result_manifest_hash="6" * 64,
        frozen_model_manifest_hash="7" * 64,
        candidate_config_hash="2" * 64,
        required_evaluation_contract_hash="8" * 64,
        holdout_identity=identity,
        reviewer_reason="Freeze synthetic registry transition coverage.",
        expected_head_hash=development.event_hash,
        actor_label="reviewer",
    )
    return identity


def test_registry_lifecycle_budget_and_verification_are_authoritative(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    registered = ledger.register_study(
        research_study_definition,
        expected_head_hash="0" * 64,
        actor_label="reviewer",
        event_ts=datetime(2026, 7, 16, 1, 0, tzinfo=UTC),
    )
    first = ledger.reserve_proposal_attempt(
        research_study_definition.study_id,
        expected_head_hash=registered.event_hash,
        actor_label="agent-host",
        event_ts=datetime(2026, 7, 16, 1, 1, tzinfo=UTC),
    )
    completed = ledger.complete_proposal_attempt(
        research_study_definition.study_id,
        first.payload.attempt_id,
        outcome="proposal",
        agent_run_id="agent-run-1",
        detail="One strict proposal was persisted.",
        terminal_artifact_hash="3" * 64,
        expected_head_hash=first.event_hash,
        actor_label="agent-host",
        event_ts=datetime(2026, 7, 16, 1, 2, tzinfo=UTC),
    )
    verified = ledger.record_proposal_verification(
        research_study_definition.study_id,
        first.payload.attempt_id,
        terminal_artifact_hash="3" * 64,
        verification_hash="4" * 64,
        passed=True,
        expected_head_hash=completed.event_hash,
        actor_label="deterministic-verifier",
        event_ts=datetime(2026, 7, 16, 1, 3, tzinfo=UTC),
    )
    second = ledger.reserve_proposal_attempt(
        research_study_definition.study_id,
        expected_head_hash=verified.event_hash,
        actor_label="agent-host",
        event_ts=datetime(2026, 7, 16, 1, 4, tzinfo=UTC),
    )
    crashed = ledger.complete_proposal_attempt(
        research_study_definition.study_id,
        second.payload.attempt_id,
        outcome="crashed",
        agent_run_id="agent-run-2",
        detail="The process stopped after durable reservation.",
        expected_head_hash=second.event_hash,
        actor_label="recovery-host",
        event_ts=datetime(2026, 7, 16, 1, 5, tzinfo=UTC),
    )
    closed = ledger.close_study(
        research_study_definition.study_id,
        reason="Development review is complete.",
        expected_head_hash=crashed.event_hash,
        actor_label="reviewer",
        event_ts=datetime(2026, 7, 16, 1, 6, tzinfo=UTC),
    )

    projection = ledger.project()
    study = projection.studies[research_study_definition.study_id]
    assert len(ledger.replay()) == 7
    assert projection.head_hash == closed.event_hash
    assert study.state == "closed"
    assert study.proposal_budget_consumed == 2
    assert study.proposal_budget_remaining == 0
    assert study.attempts[0].verification_passed is True
    assert study.attempts[1].outcome == "crashed"
    with pytest.raises(ResearchRegistryError, match="development_open"):
        ledger.reserve_proposal_attempt(
            research_study_definition.study_id,
            expected_head_hash=closed.event_hash,
            actor_label="agent-host",
        )


def test_registry_rejects_stale_heads_duplicate_transitions_and_clock_regression(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    registered = ledger.register_study(
        research_study_definition,
        expected_head_hash="0" * 64,
        actor_label="reviewer",
        event_ts=datetime(2026, 7, 16, 2, 0, tzinfo=UTC),
    )
    reserved = ledger.reserve_proposal_attempt(
        research_study_definition.study_id,
        expected_head_hash=registered.event_hash,
        actor_label="host",
        event_ts=datetime(2026, 7, 16, 2, 1, tzinfo=UTC),
    )

    with pytest.raises(StaleRegistryHeadError, match="head changed"):
        ledger.reserve_proposal_attempt(
            research_study_definition.study_id,
            expected_head_hash=registered.event_hash,
            actor_label="stale-host",
        )
    with pytest.raises(ResearchRegistryError, match="regress"):
        ledger.complete_proposal_attempt(
            research_study_definition.study_id,
            reserved.payload.attempt_id,
            outcome="crashed",
            agent_run_id="run-1",
            detail="crash",
            expected_head_hash=reserved.event_hash,
            actor_label="host",
            event_ts=datetime(2026, 7, 16, 1, 59, tzinfo=UTC),
        )
    with pytest.raises(ResearchRegistryError, match="incomplete attempt"):
        ledger.close_study(
            research_study_definition.study_id,
            reason="too early",
            expected_head_hash=reserved.event_hash,
            actor_label="reviewer",
        )
    with pytest.raises(ResearchRegistryError, match="timezone-aware"):
        ledger.complete_proposal_attempt(
            research_study_definition.study_id,
            reserved.payload.attempt_id,
            outcome="crashed",
            agent_run_id="run-1",
            detail="crash",
            expected_head_hash=reserved.event_hash,
            actor_label="host",
            event_ts=datetime(2026, 7, 16, 2, 2),
        )


def test_registry_external_holdout_projection_detects_conservative_overlap(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    identity = _holdout(research_study_definition, assets=("asset-a", "asset-b"))
    event = ledger.register_external_holdout(
        identity,
        reason="Inspected in an external standard run.",
        expected_head_hash="0" * 64,
        actor_label="reviewer",
        event_ts=datetime(2026, 7, 16, 3, 0, tzinfo=UTC),
    )

    candidate = _holdout(research_study_definition, assets=("asset-b", "asset-c"))
    projection = ledger.project()
    overlaps = projection.holdout_use.overlapping(candidate)
    assert projection.head_hash == event.event_hash
    assert len(overlaps) == 1
    assert overlaps[0].holdout_identity == identity


def test_active_reveal_reservation_blocks_overlap_and_completion_keeps_one_use_record(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    first_study = research_study_definition
    second_study = _distinct_study(first_study, label="Independent second study.")
    first_identity = _freeze_study_for_holdout(
        ledger,
        first_study,
        assets=("asset-a", "asset-b"),
    )
    second_identity = _freeze_study_for_holdout(
        ledger,
        second_study,
        assets=("asset-b", "asset-c"),
    )

    reservation = ledger.reserve_holdout_reveal(
        first_study.study_id,
        candidate_hash="2" * 64,
        holdout_identity=first_identity,
        expected_head_hash=ledger.head_hash(),
        actor_label="reviewer",
    )

    active_records = ledger.project().holdout_use.overlapping(second_identity)
    assert len(active_records) == 1
    assert active_records[0].source == "reveal_reservation"
    assert active_records[0].registry_event_hash == reservation.event_hash
    with pytest.raises(ResearchRegistryError, match="overlaps prior global use"):
        ledger.reserve_holdout_reveal(
            second_study.study_id,
            candidate_hash="2" * 64,
            holdout_identity=second_identity,
            expected_head_hash=ledger.head_hash(),
            actor_label="reviewer",
        )

    ledger.complete_holdout_reveal(
        first_study.study_id,
        reservation.payload.reservation_id,
        candidate_hash="2" * 64,
        holdout_identity=first_identity,
        result_manifest_hash="9" * 64,
        expected_head_hash=ledger.head_hash(),
        actor_label="host",
    )
    completed_records = ledger.project().holdout_use.overlapping(first_identity)
    assert len(completed_records) == 1
    assert completed_records[0] == active_records[0]


def test_failed_reveal_reservation_permanently_blocks_overlapping_study(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    first_study = research_study_definition
    second_study = _distinct_study(first_study, label="Independent failed-overlap study.")
    first_identity = _freeze_study_for_holdout(
        ledger,
        first_study,
        assets=("asset-a", "asset-b"),
    )
    second_identity = _freeze_study_for_holdout(
        ledger,
        second_study,
        assets=("asset-b", "asset-c"),
    )
    reservation = ledger.reserve_holdout_reveal(
        first_study.study_id,
        candidate_hash="2" * 64,
        holdout_identity=first_identity,
        expected_head_hash=ledger.head_hash(),
        actor_label="reviewer",
    )
    ledger.fail_holdout_reveal(
        first_study.study_id,
        reservation.payload.reservation_id,
        failure_stage="prediction",
        detail="Synthetic failure after the reservation may have observed holdout outcomes.",
        expected_head_hash=ledger.head_hash(),
        actor_label="host",
    )

    failed_records = ledger.project().holdout_use.overlapping(second_identity)
    assert len(failed_records) == 1
    assert failed_records[0].source == "reveal_reservation"
    assert failed_records[0].registry_event_hash == reservation.event_hash
    with pytest.raises(ResearchRegistryError, match="overlaps prior global use"):
        ledger.reserve_holdout_reveal(
            second_study.study_id,
            candidate_hash="2" * 64,
            holdout_identity=second_identity,
            expected_head_hash=ledger.head_hash(),
            actor_label="reviewer",
        )


def test_multiprocess_last_slot_reservation_cannot_overspend_budget(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    root = (tmp_path / "agent-artifacts").resolve()
    ledger = ResearchRegistryLedger(root)
    registered = ledger.register_study(
        research_study_definition,
        expected_head_hash="0" * 64,
        actor_label="reviewer",
    )
    first = ledger.reserve_proposal_attempt(
        research_study_definition.study_id,
        expected_head_hash=registered.event_hash,
        actor_label="host",
    )
    completed = ledger.complete_proposal_attempt(
        research_study_definition.study_id,
        first.payload.attempt_id,
        outcome="crashed",
        agent_run_id="first-run",
        detail="consume the first slot",
        expected_head_hash=first.event_hash,
        actor_label="host",
    )
    expected_head = completed.event_hash
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_reserve_worker,
            args=(str(root), research_study_definition.study_id, expected_head, start, output),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    results = [output.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    assert [status for status, _ in results].count("ok") == 1
    assert {status for status, _ in results} <= {
        "ok",
        "ResearchRegistryLockError",
        "StaleRegistryHeadError",
    }
    projection = ledger.project().studies[research_study_definition.study_id]
    assert projection.proposal_budget_consumed == 2
    assert projection.proposal_budget_remaining == 0


def test_registry_replay_rejects_tamper_duplicate_partial_nonfinite_and_symlink(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    root = (tmp_path / "valid").resolve()
    ledger = ResearchRegistryLedger(root)
    event = ledger.register_study(
        research_study_definition,
        expected_head_hash="0" * 64,
        actor_label="reviewer",
    )
    record = json.loads(ledger.path.read_text(encoding="utf-8"))
    record["sequence"] = 2
    unhashed = {key: value for key, value in record.items() if key != "event_hash"}
    record["event_hash"] = hashlib.sha256(canonical_json(unhashed).encode()).hexdigest()
    ledger.path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    with pytest.raises(ResearchRegistryError, match="expected 1"):
        ledger.replay()

    duplicate = ResearchRegistryLedger((tmp_path / "duplicate").resolve())
    duplicate.register_study(
        research_study_definition,
        expected_head_hash="0" * 64,
        actor_label="reviewer",
    )
    duplicate.path.write_text(
        duplicate.path.read_text(encoding="utf-8").replace(
            "{", '{"event_id":"' + event.event_id + '",', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ResearchRegistryError, match="repeats JSON key"):
        duplicate.replay()

    partial = ResearchRegistryLedger((tmp_path / "partial").resolve())
    partial.path.parent.mkdir(parents=True)
    partial.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ResearchRegistryError, match="incomplete"):
        partial.replay()

    nonfinite = ResearchRegistryLedger((tmp_path / "nonfinite").resolve())
    nonfinite.path.parent.mkdir(parents=True)
    nonfinite.path.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ResearchRegistryError, match="strict JSON"):
        nonfinite.replay()

    if hasattr(os, "O_NOFOLLOW"):
        symlinked = ResearchRegistryLedger((tmp_path / "symlink").resolve())
        symlinked.path.parent.mkdir(parents=True, exist_ok=True)
        target = (tmp_path / "target.jsonl").resolve()
        target.write_text("", encoding="utf-8")
        symlinked.path.symlink_to(target)
        with pytest.raises(ResearchRegistryError, match="opened safely"):
            symlinked.replay()


def test_registry_replay_rejects_semantically_duplicate_completion(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    registered = ledger.register_study(
        research_study_definition,
        expected_head_hash="0" * 64,
        actor_label="reviewer",
    )
    reserved = ledger.reserve_proposal_attempt(
        research_study_definition.study_id,
        expected_head_hash=registered.event_hash,
        actor_label="host",
    )
    completed = ledger.complete_proposal_attempt(
        research_study_definition.study_id,
        reserved.payload.attempt_id,
        outcome="crashed",
        agent_run_id="run-1",
        detail="first completion",
        expected_head_hash=reserved.event_hash,
        actor_label="host",
    )
    duplicate = RegistryEvent.create(
        sequence=4,
        previous_event_hash=completed.event_hash,
        event_ts=completed.event_ts + timedelta(seconds=1),
        study_id=research_study_definition.study_id,
        actor_kind="host",
        actor_label="host",
        payload=ProposalAttemptCompletedPayload(
            attempt_id=reserved.payload.attempt_id,
            outcome="crashed",
            agent_run_id="run-1",
            detail="second completion",
        ),
    )
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write(duplicate.canonical_json() + "\n")

    with pytest.raises(ResearchRegistryError, match="twice"):
        ledger.replay()


def test_registry_lock_contention_fails_without_blocking(tmp_path: Path) -> None:
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    with (
        advisory_file_lock(ledger.lock_path),
        pytest.raises(ResearchRegistryLockError, match="unavailable"),
    ):
        ledger.replay()
