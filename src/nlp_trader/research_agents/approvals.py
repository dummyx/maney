from __future__ import annotations

import hashlib
import json
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from nlp_trader.immutable.append import (
    SafeFileError,
    read_bytes_no_follow,
    write_bytes_exclusive_durable,
)
from nlp_trader.research_agents.compiler import (
    ExperimentExecutionDefinition,
    build_required_evaluation_contract,
    compile_proposal_typed_patches,
    load_compiled_execution_definition,
    load_compiler_proposal,
)
from nlp_trader.research_agents.contracts import (
    DevelopmentExecutionApprovedPayload,
    DevelopmentRunCompletedPayload,
    DevelopmentRunStartedPayload,
    HoldoutIdentity,
    ProposalVerification,
    Sha256,
    StrictModel,
    StudyDefinition,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_agents.registry import ResearchRegistryLedger
from nlp_trader.timestamps import format_utc, parse_utc


class DevelopmentExecutionApproval(StrictModel):
    artifact_schema_version: Literal["development-execution-approval-v1"] = (
        "development-execution-approval-v1"
    )
    approval_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    attempt_id: Sha256
    proposal_verification_hash: Sha256
    execution_definition_hash: Sha256
    approval_event_hash: Sha256
    actor_label: str = Field(min_length=1, max_length=256)
    reviewer_reason: str = Field(min_length=1, max_length=16_384)
    approved_at: datetime

    @field_validator("approved_at", mode="before")
    @classmethod
    def normalize_time(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("approved_at must be timezone-aware")
            return value.astimezone(UTC)
        if isinstance(value, str):
            return parse_utc(value)
        raise ValueError("approved_at must be a timestamp")

    @field_serializer("approved_at")
    def serialize_time(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = content_sha256(self.model_dump(mode="json", exclude={"approval_id"}))
        if self.approval_id and self.approval_id != expected:
            raise ValueError("approval_id does not match canonical content")
        if not self.approval_id:
            object.__setattr__(self, "approval_id", expected)
        return self


class CandidateFreezeRecord(StrictModel):
    artifact_schema_version: Literal["candidate-freeze-record-v1"] = "candidate-freeze-record-v1"
    freeze_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    proposal_hash: Sha256
    execution_definition_hash: Sha256
    development_approval_event_hash: Sha256
    development_result_manifest_hash: Sha256
    frozen_model_manifest_hash: Sha256
    candidate_config_hash: Sha256
    required_evaluation_contract_hash: Sha256
    holdout_identity: HoldoutIdentity
    freeze_event_hash: Sha256
    actor_label: str = Field(min_length=1, max_length=256)
    reviewer_reason: str = Field(min_length=1, max_length=16_384)
    frozen_at: datetime

    @field_validator("frozen_at", mode="before")
    @classmethod
    def normalize_time(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("frozen_at must be timezone-aware")
            return value.astimezone(UTC)
        if isinstance(value, str):
            return parse_utc(value)
        raise ValueError("frozen_at must be a timestamp")

    @field_serializer("frozen_at")
    def serialize_time(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.holdout_identity.candidate_hash != self.candidate_config_hash:
            raise ValueError("candidate config and holdout identity hashes must match")
        expected = content_sha256(self.model_dump(mode="json", exclude={"freeze_id"}))
        if self.freeze_id and self.freeze_id != expected:
            raise ValueError("freeze_id does not match canonical content")
        if not self.freeze_id:
            object.__setattr__(self, "freeze_id", expected)
        return self


def approve_development_execution(
    ledger: ResearchRegistryLedger,
    *,
    study: StudyDefinition,
    definition: ExperimentExecutionDefinition,
    verification: ProposalVerification,
    verification_artifact_hash: str,
    actor_label: str,
    reviewer_reason: str,
    event_ts: datetime | None = None,
) -> DevelopmentExecutionApproval:
    if (
        not verification.passed
        or definition.study_id != study.study_id
        or verification.study_id != study.study_id
        or verification.verifier_contract != study.verifier_contract
        or definition.proposal_hash != verification.terminal_artifact_hash
        or definition.proposal_verification_hash != verification_artifact_hash
        or not _definition_matches_study(definition, study)
    ):
        raise ValueError("development approval inputs do not bind one passed proposal")
    if ledger.study_definition(study.study_id) != study:
        raise ValueError("development approval study does not match the registered study")
    state = ledger.project().studies[study.study_id]
    attempt = next(
        (value for value in state.attempts if value.attempt_id == verification.attempt_id),
        None,
    )
    if (
        attempt is None
        or attempt.status != "completed"
        or attempt.outcome != "proposal"
        or attempt.agent_run_id is None
        or attempt.terminal_artifact_hash != definition.proposal_hash
        or attempt.verification_hash != verification_artifact_hash
        or attempt.verification_passed is not True
    ):
        raise ValueError("development approval does not bind the registered verified proposal")
    proposal = load_compiler_proposal(
        ledger.artifact_root,
        agent_run_id=attempt.agent_run_id,
        expected_sha256=definition.proposal_hash,
    )
    if (
        proposal.study_id != study.study_id
        or proposal.attempt_id != attempt.attempt_id
        or proposal.experiment_template_id != definition.template_id
        or compile_proposal_typed_patches(study, proposal) != definition.typed_patches
    ):
        raise ValueError("development approval patches do not match the retained verified proposal")
    compiled = load_compiled_execution_definition(
        ledger.artifact_root,
        definition.definition_id,
    )
    if compiled != definition:
        raise ValueError("development approval definition is not the exact compiled artifact")
    event = ledger.approve_development_execution(
        study.study_id,
        verification.attempt_id,
        proposal_verification_hash=verification_artifact_hash,
        execution_definition_hash=definition.definition_id,
        reviewer_reason=reviewer_reason,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor_label,
        event_ts=event_ts,
    )
    approval = DevelopmentExecutionApproval(
        study_id=study.study_id,
        attempt_id=verification.attempt_id,
        proposal_verification_hash=verification_artifact_hash,
        execution_definition_hash=definition.definition_id,
        approval_event_hash=event.event_hash,
        actor_label=actor_label,
        reviewer_reason=reviewer_reason,
        approved_at=event.event_ts,
    )
    destination = (
        ledger.artifact_root
        / "studies"
        / study.study_id
        / "approvals"
        / f"{approval.approval_id}.json"
    )
    _write_record(destination, approval.model_dump(mode="json"))
    return approval


def load_authoritative_development_approval(
    ledger: ResearchRegistryLedger,
    supplied: DevelopmentExecutionApproval,
) -> DevelopmentExecutionApproval:
    """Validate an approval against its exact registry event and immutable record bytes."""

    matches = tuple(
        event for event in ledger.replay() if event.event_hash == supplied.approval_event_hash
    )
    if len(matches) != 1:
        raise ValueError("development approval event is missing from the authoritative registry")
    event = matches[0]
    payload = event.payload
    if (
        event.study_id != supplied.study_id
        or event.actor_kind != "human"
        or not isinstance(payload, DevelopmentExecutionApprovedPayload)
    ):
        raise ValueError("development approval event is not authoritative")
    authoritative = DevelopmentExecutionApproval(
        study_id=event.study_id,
        attempt_id=payload.attempt_id,
        proposal_verification_hash=payload.proposal_verification_hash,
        execution_definition_hash=payload.execution_definition_hash,
        approval_event_hash=event.event_hash,
        actor_label=event.actor_label,
        reviewer_reason=payload.reviewer_reason,
        approved_at=event.event_ts,
    )
    if supplied.model_dump(mode="json") != authoritative.model_dump(mode="json"):
        raise ValueError("development approval does not match its authoritative registry event")
    state = ledger.project().studies[supplied.study_id]
    if (
        state.state != "development_locked"
        or authoritative.approval_event_hash not in state.transition_event_hashes
    ):
        raise ValueError("development approval does not match authoritative study state")
    approval_path = (
        ledger.artifact_root
        / "studies"
        / authoritative.study_id
        / "approvals"
        / f"{authoritative.approval_id}.json"
    )
    encoded = _read_authority_record(approval_path, artifact_root=ledger.artifact_root)
    expected = (canonical_json(authoritative.model_dump(mode="json")) + "\n").encode("utf-8")
    if encoded != expected:
        raise ValueError("development approval artifact does not match authoritative approval")
    return authoritative


def freeze_candidate(
    ledger: ResearchRegistryLedger,
    *,
    study: StudyDefinition,
    proposal_hash: str,
    definition: ExperimentExecutionDefinition,
    approval: DevelopmentExecutionApproval,
    development_result_manifest_hash: str,
    frozen_model_manifest_hash: str,
    candidate_config: object,
    required_evaluation_contract_hash: str,
    input_snapshot_hashes: tuple[str, ...],
    universe_asset_ids: tuple[str, ...],
    actor_label: str,
    reviewer_reason: str,
    event_ts: datetime | None = None,
) -> CandidateFreezeRecord:
    candidate_config = _validate_candidate_freeze_inputs(
        ledger,
        study=study,
        proposal_hash=proposal_hash,
        definition=definition,
        approval=approval,
        development_result_manifest_hash=development_result_manifest_hash,
        frozen_model_manifest_hash=frozen_model_manifest_hash,
        candidate_config=candidate_config,
        required_evaluation_contract_hash=required_evaluation_contract_hash,
    )
    candidate_config_hash = content_sha256(candidate_config)
    holdout = HoldoutIdentity(
        data_lineage_id=study.data_lineage_id,
        input_snapshot_hashes=tuple(sorted(input_snapshot_hashes)),
        universe_snapshot_id=study.universe_snapshot_id,
        universe_asset_ids=tuple(sorted(universe_asset_ids)),
        calendar_contract=study.calendar_contract,
        market_data_contract=study.market_data_contract,
        label_contract=study.label_contract,
        target_family=study.target_family,
        horizon_sessions=study.horizon_sessions,
        return_adjustment_contract=study.return_adjustment_contract,
        decision_interval=study.reserved_holdout_decisions,
        outcome_interval=study.reserved_holdout_outcomes,
        study_id=study.study_id,
        candidate_hash=candidate_config_hash,
    )
    event = ledger.freeze_candidate(
        study.study_id,
        proposal_hash=proposal_hash,
        execution_definition_hash=definition.definition_id,
        development_approval_event_hash=approval.approval_event_hash,
        development_result_manifest_hash=development_result_manifest_hash,
        frozen_model_manifest_hash=frozen_model_manifest_hash,
        candidate_config_hash=candidate_config_hash,
        required_evaluation_contract_hash=required_evaluation_contract_hash,
        holdout_identity=holdout,
        reviewer_reason=reviewer_reason,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor_label,
        event_ts=event_ts,
    )
    record = CandidateFreezeRecord(
        study_id=study.study_id,
        proposal_hash=proposal_hash,
        execution_definition_hash=definition.definition_id,
        development_approval_event_hash=approval.approval_event_hash,
        development_result_manifest_hash=development_result_manifest_hash,
        frozen_model_manifest_hash=frozen_model_manifest_hash,
        candidate_config_hash=candidate_config_hash,
        required_evaluation_contract_hash=required_evaluation_contract_hash,
        holdout_identity=holdout,
        freeze_event_hash=event.event_hash,
        actor_label=actor_label,
        reviewer_reason=reviewer_reason,
        frozen_at=event.event_ts,
    )
    candidate_root = ledger.artifact_root / "studies" / study.study_id / "candidate"
    _write_record(candidate_root / "candidate_config.json", candidate_config)
    _write_record(
        candidate_root / "freeze.json",
        record.model_dump(mode="json"),
    )
    return record


def _validate_candidate_freeze_inputs(
    ledger: ResearchRegistryLedger,
    *,
    study: StudyDefinition,
    proposal_hash: str,
    definition: ExperimentExecutionDefinition,
    approval: DevelopmentExecutionApproval,
    development_result_manifest_hash: str,
    frozen_model_manifest_hash: str,
    candidate_config: object,
    required_evaluation_contract_hash: str,
) -> dict[str, object]:
    """Validate all promotion inputs before the irreversible candidate-freeze event."""

    from nlp_trader.experiment_execution import (
        DevelopmentResultManifest,
        FrozenDevelopmentModelManifest,
    )

    if ledger.study_definition(study.study_id) != study:
        raise ValueError("candidate freeze study does not match the registered study")
    compiled = load_compiled_execution_definition(ledger.artifact_root, definition.definition_id)
    if compiled != definition:
        raise ValueError("candidate freeze definition is not the authoritative compiled artifact")
    authoritative_approval = load_authoritative_development_approval(ledger, approval)
    if (
        proposal_hash != definition.proposal_hash
        or authoritative_approval.execution_definition_hash != definition.definition_id
    ):
        raise ValueError("candidate freeze proposal or approval does not match the definition")

    events = ledger.replay()
    completed_events = tuple(
        event
        for event in events
        if event.study_id == study.study_id
        and isinstance(event.payload, DevelopmentRunCompletedPayload)
    )
    if len(completed_events) != 1:
        raise ValueError("candidate freeze requires one authoritative completed development run")
    completed_event = completed_events[0]
    completed = completed_event.payload
    assert isinstance(completed, DevelopmentRunCompletedPayload)
    if (
        completed.result_manifest_hash != development_result_manifest_hash
        or completed.frozen_model_manifest_hash != frozen_model_manifest_hash
    ):
        raise ValueError("candidate freeze artifacts do not match the completed development run")
    started_events = tuple(
        event
        for event in events
        if event.sequence < completed_event.sequence
        and event.study_id == study.study_id
        and isinstance(event.payload, DevelopmentRunStartedPayload)
        and event.payload.development_run_id == completed.development_run_id
    )
    if len(started_events) != 1:
        raise ValueError("candidate freeze development start lineage is ambiguous")
    started = started_events[0].payload
    assert isinstance(started, DevelopmentRunStartedPayload)
    if started.execution_definition_hash != definition.definition_id:
        raise ValueError("candidate freeze completed a different execution definition")

    development_root = ledger.artifact_root / "development_runs" / completed.development_run_id
    result_bytes = _read_authority_record(
        development_root / "result_manifest.json",
        artifact_root=ledger.artifact_root,
    )
    if hashlib.sha256(result_bytes).hexdigest() != development_result_manifest_hash:
        raise ValueError("candidate freeze development result bytes changed after completion")
    try:
        result = DevelopmentResultManifest.model_validate_json(result_bytes)
    except ValueError as exc:
        raise ValueError(
            "candidate freeze development result violates its strict contract"
        ) from exc
    if result_bytes != (canonical_json(result.model_dump(mode="json")) + "\n").encode("utf-8"):
        raise ValueError("candidate freeze development result is not canonical typed JSON")

    model_manifest_bytes = _read_authority_record(
        development_root / "frozen_model.manifest.json",
        artifact_root=ledger.artifact_root,
    )
    if hashlib.sha256(model_manifest_bytes).hexdigest() != frozen_model_manifest_hash:
        raise ValueError("candidate freeze model manifest bytes changed after completion")
    try:
        model_manifest = FrozenDevelopmentModelManifest.model_validate_json(model_manifest_bytes)
    except ValueError as exc:
        raise ValueError("candidate freeze model manifest violates its strict contract") from exc
    if model_manifest_bytes != (
        canonical_json(model_manifest.model_dump(mode="json")) + "\n"
    ).encode("utf-8"):
        raise ValueError("candidate freeze model manifest is not canonical typed JSON")

    result_lineage = (
        result.study_id == study.study_id,
        result.development_run_id == completed.development_run_id,
        result.execution_definition_hash == definition.definition_id,
        result.approval_event_hash == authoritative_approval.approval_event_hash,
        result.frozen_model_manifest_hash == frozen_model_manifest_hash,
        result.pipeline_result_manifest_hash == model_manifest.pipeline_result_manifest_hash,
    )
    model_lineage = (
        model_manifest.study_id == study.study_id,
        model_manifest.development_run_id == completed.development_run_id,
        model_manifest.execution_definition_hash == definition.definition_id,
    )
    if not all(result_lineage) or not all(model_lineage):
        raise ValueError("candidate freeze development artifacts have inconsistent lineage")

    expected_evaluation_contract = build_required_evaluation_contract(definition)
    derived_evaluation_contract_hash = content_sha256(expected_evaluation_contract)
    evaluation_contract_bytes = _read_authority_record(
        development_root / "required_evaluation_contract.json",
        artifact_root=ledger.artifact_root,
    )
    expected_evaluation_contract_bytes = (
        canonical_json(expected_evaluation_contract) + "\n"
    ).encode("utf-8")
    if evaluation_contract_bytes != expected_evaluation_contract_bytes:
        raise ValueError("candidate freeze evaluation contract is not derived from the definition")
    if (
        required_evaluation_contract_hash != derived_evaluation_contract_hash
        or result.required_evaluation_contract_hash != derived_evaluation_contract_hash
    ):
        raise ValueError("candidate freeze evaluation contract hash is not authoritative")

    model_bytes = _read_authority_record(
        development_root / "frozen_model.json",
        artifact_root=ledger.artifact_root,
    )
    if (
        hashlib.sha256(model_bytes).hexdigest() != model_manifest.model_artifact_hash
        or len(model_bytes) != model_manifest.model_artifact_bytes
    ):
        raise ValueError("candidate freeze model bytes do not match their manifest")
    try:
        model = json.loads(model_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate freeze model is not valid JSON") from exc
    model_families = model.get("families") if isinstance(model, dict) else None
    if not isinstance(model_families, dict):
        raise ValueError("candidate freeze model does not declare learned families")

    if not isinstance(candidate_config, dict) or set(candidate_config) != {
        "template_id",
        "typed_patches",
        "selected_family",
    }:
        raise ValueError("candidate config must contain the exact derived selection fields")
    selected_family = candidate_config.get("selected_family")
    expected_candidate_config: dict[str, object] = {
        "template_id": definition.template_id,
        "typed_patches": [value.model_dump(mode="json") for value in definition.typed_patches],
        "selected_family": selected_family,
    }
    if candidate_config != expected_candidate_config:
        raise ValueError("candidate config template or patches differ from the compiled definition")
    if (
        not isinstance(selected_family, str)
        or selected_family not in definition.required_learned_families
        or selected_family not in model_families
    ):
        raise ValueError("candidate config selected_family is not an evaluated learned family")
    return expected_candidate_config


def _write_record(path: Path, payload: object) -> None:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    try:
        write_bytes_exclusive_durable(path, encoded)
    except (FileExistsError, SafeFileError, OSError, ValueError) as exc:
        raise ValueError("approval artifact cannot be written immutably") from exc


def _read_authority_record(path: Path, *, artifact_root: Path) -> bytes:
    relative = path.relative_to(artifact_root)
    cursor = artifact_root
    for part in relative.parts[:-1]:
        cursor /= part
        try:
            metadata = cursor.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError("development approval artifact directory is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("development approval artifact path must use fixed directories")
    try:
        encoded = read_bytes_no_follow(path)
    except (FileNotFoundError, OSError, SafeFileError, ValueError) as exc:
        raise ValueError("development approval artifact cannot be read safely") from exc
    if encoded is None:  # pragma: no cover - missing_ok is false
        raise ValueError("development approval artifact is missing")
    return encoded


def _definition_matches_study(
    definition: ExperimentExecutionDefinition,
    study: StudyDefinition,
) -> bool:
    permitted_templates = {
        (value.template_id, value.version) for value in study.permitted_templates
    }
    return (
        (definition.template_id, definition.template_version) in permitted_templates
        and definition.development_decisions == study.development_decisions
        and definition.reserved_decision_boundary == study.reserved_holdout_decisions
        and definition.reserved_outcome_boundary == study.reserved_holdout_outcomes
        and definition.required_learned_families == study.required_learned_families
        and definition.required_fixed_benchmarks == study.required_fixed_benchmarks
        and definition.required_negative_controls == study.required_negative_controls
        and definition.required_robustness_checks == study.required_robustness_checks
        and definition.required_metrics == study.required_metrics
        and definition.universe_snapshot_id == study.universe_snapshot_id
        and definition.seeds == study.seeds
    )
