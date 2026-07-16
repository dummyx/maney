from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from nlp_trader.experiment_execution import (
    DevelopmentResultManifest,
    FrozenDevelopmentModelManifest,
)
from nlp_trader.immutable.append import SafeFileError, read_bytes_no_follow
from nlp_trader.research_agents.approvals import CandidateFreezeRecord
from nlp_trader.research_agents.contracts import (
    CandidateFrozenPayload,
    DevelopmentRunCompletedPayload,
    DevelopmentRunStartedPayload,
    RegistryEvent,
)
from nlp_trader.research_agents.registry import ResearchRegistryLedger


@dataclass(frozen=True, slots=True)
class AuthoritativeCandidateFreeze:
    """Registry-derived candidate and development artifacts trusted by a reveal."""

    record: CandidateFreezeRecord
    freeze_event: RegistryEvent
    development_completion_event: RegistryEvent
    development_run_id: str
    development_root: Path
    development_result: DevelopmentResultManifest
    frozen_model_manifest: FrozenDevelopmentModelManifest


def load_authoritative_candidate_freeze(
    ledger: ResearchRegistryLedger,
    supplied: CandidateFreezeRecord,
    *,
    development_root: Path,
) -> AuthoritativeCandidateFreeze:
    """Reconstruct freeze authority from the ledger and validate its development lineage."""

    events = ledger.replay()
    freeze_event = _event_by_hash(events, supplied.freeze_event_hash)
    payload = freeze_event.payload
    if (
        freeze_event.study_id != supplied.study_id
        or freeze_event.actor_kind != "human"
        or not isinstance(payload, CandidateFrozenPayload)
    ):
        raise ValueError("candidate freeze event is not authoritative")
    authoritative = CandidateFreezeRecord(
        study_id=freeze_event.study_id,
        proposal_hash=payload.proposal_hash,
        execution_definition_hash=payload.execution_definition_hash,
        development_approval_event_hash=payload.development_approval_event_hash,
        development_result_manifest_hash=payload.development_result_manifest_hash,
        frozen_model_manifest_hash=payload.frozen_model_manifest_hash,
        candidate_config_hash=payload.candidate_config_hash,
        required_evaluation_contract_hash=payload.required_evaluation_contract_hash,
        holdout_identity=payload.holdout_identity,
        freeze_event_hash=freeze_event.event_hash,
        actor_label=freeze_event.actor_label,
        reviewer_reason=payload.reviewer_reason,
        frozen_at=freeze_event.event_ts,
    )
    if supplied.model_dump(mode="json") != authoritative.model_dump(mode="json"):
        raise ValueError("candidate freeze record does not match its authoritative registry event")

    completed_events = tuple(
        event
        for event in events
        if event.sequence < freeze_event.sequence
        and event.study_id == supplied.study_id
        and isinstance(event.payload, DevelopmentRunCompletedPayload)
        and event.payload.result_manifest_hash == payload.development_result_manifest_hash
        and event.payload.frozen_model_manifest_hash == payload.frozen_model_manifest_hash
    )
    if len(completed_events) != 1:
        raise ValueError("candidate freeze does not identify one completed development run")
    completed_event = completed_events[0]
    completed = completed_event.payload
    assert isinstance(completed, DevelopmentRunCompletedPayload)
    started_events = tuple(
        event
        for event in events
        if event.sequence < completed_event.sequence
        and event.study_id == supplied.study_id
        and isinstance(event.payload, DevelopmentRunStartedPayload)
        and event.payload.development_run_id == completed.development_run_id
    )
    if len(started_events) != 1:
        raise ValueError("candidate freeze development start lineage is ambiguous")
    started = started_events[0].payload
    assert isinstance(started, DevelopmentRunStartedPayload)
    if started.execution_definition_hash != payload.execution_definition_hash:
        raise ValueError(
            "candidate freeze development definition does not match its registry start"
        )

    expected_root = ledger.artifact_root / "development_runs" / completed.development_run_id
    _require_exact_development_root(development_root, expected=expected_root, ledger=ledger)
    result_bytes = _read_regular_bytes(expected_root / "result_manifest.json")
    if hashlib.sha256(result_bytes).hexdigest() != payload.development_result_manifest_hash:
        raise ValueError("development result bytes do not match the authoritative candidate freeze")
    model_manifest_bytes = _read_regular_bytes(expected_root / "frozen_model.manifest.json")
    if hashlib.sha256(model_manifest_bytes).hexdigest() != payload.frozen_model_manifest_hash:
        raise ValueError(
            "frozen model manifest bytes do not match the authoritative candidate freeze"
        )
    try:
        result = DevelopmentResultManifest.model_validate_json(result_bytes)
        model_manifest = FrozenDevelopmentModelManifest.model_validate_json(model_manifest_bytes)
    except ValidationError as exc:
        raise ValueError("authoritative development artifact violates its strict contract") from exc

    result_lineage = (
        result.study_id == supplied.study_id,
        result.development_run_id == completed.development_run_id,
        result.execution_definition_hash == payload.execution_definition_hash,
        result.approval_event_hash == payload.development_approval_event_hash,
        result.frozen_model_manifest_hash == payload.frozen_model_manifest_hash,
        result.required_evaluation_contract_hash == payload.required_evaluation_contract_hash,
    )
    model_lineage = (
        model_manifest.study_id == supplied.study_id,
        model_manifest.development_run_id == completed.development_run_id,
        model_manifest.execution_definition_hash == payload.execution_definition_hash,
        model_manifest.pipeline_result_manifest_hash == result.pipeline_result_manifest_hash,
    )
    if not all(result_lineage):
        raise ValueError(
            "development result lineage does not match the authoritative candidate freeze"
        )
    if not all(model_lineage):
        raise ValueError("frozen model lineage does not match the authoritative development result")
    return AuthoritativeCandidateFreeze(
        record=authoritative,
        freeze_event=freeze_event,
        development_completion_event=completed_event,
        development_run_id=completed.development_run_id,
        development_root=expected_root,
        development_result=result,
        frozen_model_manifest=model_manifest,
    )


def _event_by_hash(events: tuple[RegistryEvent, ...], event_hash: str) -> RegistryEvent:
    matches = tuple(event for event in events if event.event_hash == event_hash)
    if len(matches) != 1:
        raise ValueError("candidate freeze event is missing from the authoritative registry")
    return matches[0]


def _require_exact_development_root(
    supplied: Path,
    *,
    expected: Path,
    ledger: ResearchRegistryLedger,
) -> None:
    candidate = Path(os.path.normpath(os.fspath(Path(supplied).expanduser())))
    if not candidate.is_absolute() or candidate != expected:
        raise ValueError("development root is not the registry-authoritative run directory")
    relative = expected.relative_to(ledger.artifact_root)
    cursor = ledger.artifact_root
    for part in relative.parts:
        cursor /= part
        try:
            metadata = cursor.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError("registry-authoritative development root is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("registry-authoritative development root cannot contain symlinks")
    if not stat.S_ISDIR(expected.stat(follow_symlinks=False).st_mode):
        raise ValueError("registry-authoritative development root must be a directory")


def _read_regular_bytes(path: Path) -> bytes:
    try:
        value = read_bytes_no_follow(path)
    except (FileNotFoundError, OSError, SafeFileError, ValueError) as exc:
        raise ValueError("authoritative development artifact cannot be read safely") from exc
    if value is None:  # pragma: no cover - missing_ok is false
        raise ValueError("authoritative development artifact is missing")
    return value
