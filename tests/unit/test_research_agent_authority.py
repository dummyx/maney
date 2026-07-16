from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nlp_trader.experiment_execution import (
    DevelopmentResultManifest,
    FrozenDevelopmentModelManifest,
)
from nlp_trader.research_agents.approvals import CandidateFreezeRecord
from nlp_trader.research_agents.authority import load_authoritative_candidate_freeze
from nlp_trader.research_agents.contracts import HoldoutIdentity, StudyDefinition, canonical_json
from nlp_trader.research_agents.registry import ResearchRegistryLedger


def test_authoritative_freeze_loads_exact_completed_development_lineage(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger, freeze, development_root = _frozen_candidate(
        tmp_path,
        research_study_definition,
    )

    authority = load_authoritative_candidate_freeze(
        ledger,
        freeze,
        development_root=development_root,
    )

    assert authority.record == freeze
    assert authority.development_root == development_root
    assert authority.development_result.development_run_id == authority.development_run_id
    assert authority.frozen_model_manifest.development_run_id == authority.development_run_id


def test_authoritative_freeze_rejects_self_valid_model_manifest_with_wrong_lineage(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    ledger, freeze, development_root = _frozen_candidate(
        tmp_path,
        research_study_definition,
        model_definition_hash="9" * 64,
    )

    with pytest.raises(ValueError, match="frozen model lineage"):
        load_authoritative_candidate_freeze(
            ledger,
            freeze,
            development_root=development_root,
        )
    assert ledger.project().studies[freeze.study_id].state == "candidate_frozen"
    assert not ledger.project().holdout_use.overlapping(freeze.holdout_identity)


def _frozen_candidate(
    tmp_path: Path,
    study: StudyDefinition,
    *,
    model_definition_hash: str = "5" * 64,
) -> tuple[ResearchRegistryLedger, CandidateFreezeRecord, Path]:
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    registered = ledger.register_study(
        study,
        expected_head_hash=ledger.head_hash(),
        actor_label="reviewer",
    )
    reserved = ledger.reserve_proposal_attempt(
        study.study_id,
        expected_head_hash=registered.event_hash,
        actor_label="host",
    )
    completed = ledger.complete_proposal_attempt(
        study.study_id,
        reserved.payload.attempt_id,
        outcome="proposal",
        agent_run_id="authority-test-agent-run",
        terminal_artifact_hash="3" * 64,
        detail="Synthetic authority test proposal.",
        expected_head_hash=reserved.event_hash,
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
        reviewer_reason="Approve the synthetic authority test definition.",
        expected_head_hash=verified.event_hash,
        actor_label="reviewer",
    )
    started = ledger.start_development_run(
        study.study_id,
        execution_definition_hash="5" * 64,
        expected_head_hash=approved.event_hash,
        actor_label="host",
    )
    run_id = started.payload.development_run_id
    development_root = ledger.artifact_root / "development_runs" / run_id
    development_root.mkdir(mode=0o700, parents=True)
    model_manifest = FrozenDevelopmentModelManifest(
        study_id=study.study_id,
        development_run_id=run_id,
        execution_definition_hash=model_definition_hash,
        pipeline_result_manifest_hash="a" * 64,
        model_artifact_hash="b" * 64,
        model_artifact_bytes=1,
    )
    model_manifest_bytes = _encoded(model_manifest.model_dump(mode="json"))
    model_manifest_hash = hashlib.sha256(model_manifest_bytes).hexdigest()
    result = DevelopmentResultManifest(
        study_id=study.study_id,
        development_run_id=run_id,
        execution_definition_hash="5" * 64,
        approval_event_hash=approved.event_hash,
        base_config_hash="c" * 64,
        patched_config_hash="d" * 64,
        pipeline_run_id=run_id,
        pipeline_result_manifest_hash="a" * 64,
        frozen_model_manifest_hash=model_manifest_hash,
        backtest_comparison_hash="e" * 64,
        prediction_metrics_hash="f" * 64,
        required_evaluation_contract_hash="8" * 64,
        cost_assumptions_hash="1" * 64,
        constraint_assumptions_hash="2" * 64,
        limitations=("Synthetic authority fixture only.",),
    )
    result_bytes = _encoded(result.model_dump(mode="json"))
    result_hash = hashlib.sha256(result_bytes).hexdigest()
    (development_root / "frozen_model.manifest.json").write_bytes(model_manifest_bytes)
    (development_root / "result_manifest.json").write_bytes(result_bytes)
    development = ledger.complete_development_run(
        study.study_id,
        run_id,
        result_manifest_hash=result_hash,
        frozen_model_manifest_hash=model_manifest_hash,
        expected_head_hash=started.event_hash,
        actor_label="host",
    )
    identity = _holdout_identity(study)
    frozen = ledger.freeze_candidate(
        study.study_id,
        proposal_hash="3" * 64,
        execution_definition_hash="5" * 64,
        development_approval_event_hash=approved.event_hash,
        development_result_manifest_hash=result_hash,
        frozen_model_manifest_hash=model_manifest_hash,
        candidate_config_hash="6" * 64,
        required_evaluation_contract_hash="8" * 64,
        holdout_identity=identity,
        reviewer_reason="Freeze the synthetic authority test candidate.",
        expected_head_hash=development.event_hash,
        actor_label="reviewer",
    )
    payload = frozen.payload
    freeze = CandidateFreezeRecord(
        study_id=study.study_id,
        proposal_hash=payload.proposal_hash,
        execution_definition_hash=payload.execution_definition_hash,
        development_approval_event_hash=payload.development_approval_event_hash,
        development_result_manifest_hash=payload.development_result_manifest_hash,
        frozen_model_manifest_hash=payload.frozen_model_manifest_hash,
        candidate_config_hash=payload.candidate_config_hash,
        required_evaluation_contract_hash=payload.required_evaluation_contract_hash,
        holdout_identity=payload.holdout_identity,
        freeze_event_hash=frozen.event_hash,
        actor_label=frozen.actor_label,
        reviewer_reason=payload.reviewer_reason,
        frozen_at=frozen.event_ts,
    )
    return ledger, freeze, development_root


def _holdout_identity(study: StudyDefinition) -> HoldoutIdentity:
    return HoldoutIdentity(
        data_lineage_id=study.data_lineage_id,
        input_snapshot_hashes=("7" * 64,),
        universe_snapshot_id=study.universe_snapshot_id,
        universe_asset_ids=("asset_aaa",),
        calendar_contract=study.calendar_contract,
        market_data_contract=study.market_data_contract,
        label_contract=study.label_contract,
        target_family=study.target_family,
        horizon_sessions=study.horizon_sessions,
        return_adjustment_contract=study.return_adjustment_contract,
        decision_interval=study.reserved_holdout_decisions,
        outcome_interval=study.reserved_holdout_outcomes,
        study_id=study.study_id,
        candidate_hash="6" * 64,
    )


def _encoded(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")
