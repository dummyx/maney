from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal

from pydantic import ValidationError

from nlp_trader.immutable.append import SafeFileError, append_bytes_durable, read_bytes_no_follow
from nlp_trader.immutable.locking import (
    AdvisoryFileLockError,
    AdvisoryFileLockUnavailable,
    advisory_file_lock,
)
from nlp_trader.research_agents.artifacts import ensure_agent_artifact_root
from nlp_trader.research_agents.contracts import (
    GENESIS_HASH,
    CandidateFrozenPayload,
    DevelopmentExecutionApprovedPayload,
    DevelopmentRunCompletedPayload,
    DevelopmentRunFailedPayload,
    DevelopmentRunStartedPayload,
    ExternalHoldoutRegisteredPayload,
    HoldoutIdentity,
    HoldoutRevealedPayload,
    HoldoutRevealFailedPayload,
    HoldoutRevealReservedPayload,
    HoldoutUseIndex,
    HoldoutUseRecord,
    ProposalAttemptCompletedPayload,
    ProposalAttemptOutcome,
    ProposalAttemptReservedPayload,
    ProposalAttemptSnapshot,
    ProposalVerifiedPayload,
    RegistryEvent,
    RegistryPayload,
    StudyClosedPayload,
    StudyDefinition,
    StudyRegisteredPayload,
    StudyStateSnapshot,
    canonical_json,
    content_sha256,
)


class ResearchRegistryError(ValueError):
    """Raised when authoritative research-registry data or a transition is invalid."""


class ResearchRegistryLockError(RuntimeError):
    """Raised when another process owns the global research-registry lock."""


class StaleRegistryHeadError(ResearchRegistryError):
    """Raised when a mutation was authorized against an obsolete global head."""


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


@dataclass(frozen=True, slots=True)
class RegistryProjection:
    studies: dict[str, StudyStateSnapshot]
    holdout_use: HoldoutUseIndex
    head_hash: str


RegistryTransition = Callable[
    [tuple[RegistryEvent, ...], RegistryProjection],
    tuple[str | None, RegistryPayload],
]


@dataclass(slots=True)
class _MutableAttempt:
    attempt_id: str
    attempt_number: int
    reservation_event_hash: str
    reserved_study_state_hash: str
    status: Literal["reserved", "completed"] = "reserved"
    outcome: ProposalAttemptOutcome | None = None
    agent_run_id: str | None = None
    terminal_artifact_hash: str | None = None
    verification_hash: str | None = None
    verification_passed: bool | None = None


@dataclass(slots=True)
class _MutableStudy:
    definition: StudyDefinition
    state: Literal[
        "development_open",
        "development_locked",
        "candidate_frozen",
        "holdout_revealed",
        "closed",
    ] = "development_open"
    attempts: list[_MutableAttempt] = field(default_factory=list)
    transition_event_hashes: list[str] = field(default_factory=list)
    approved_attempt_id: str | None = None
    approval_event_hash: str | None = None
    approved_verification_hash: str | None = None
    execution_definition_hash: str | None = None
    development_run_id: str | None = None
    development_run_status: Literal["started", "failed", "completed"] | None = None
    development_result_manifest_hash: str | None = None
    frozen_model_manifest_hash: str | None = None
    candidate_config_hash: str | None = None
    holdout_reservation_id: str | None = None
    holdout_reservation_event_hash: str | None = None
    holdout_identity: HoldoutIdentity | None = None
    holdout_reveal_status: Literal["reserved", "failed", "revealed"] | None = None


class ResearchRegistryLedger:
    """The one locked, append-only authority for research-agent study state."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = ensure_agent_artifact_root(artifact_root)
        self.path = self.artifact_root / "registry" / "research_events.jsonl"
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def replay(self) -> tuple[RegistryEvent, ...]:
        with self._exclusive_lock():
            return self._replay_unlocked()

    def head_hash(self) -> str:
        events = self.replay()
        return events[-1].event_hash if events else GENESIS_HASH

    def project(self) -> RegistryProjection:
        events = self.replay()
        return _fold_registry(events)

    def study_definition(self, study_id: str) -> StudyDefinition:
        for event in self.replay():
            if isinstance(event.payload, StudyRegisteredPayload):
                definition = event.payload.study_definition
                if definition.study_id == study_id:
                    return definition
        raise ResearchRegistryError("study is not registered")

    def register_study(
        self,
        definition: StudyDefinition,
        *,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        if not isinstance(definition, StudyDefinition):
            raise TypeError("definition must be a StudyDefinition")

        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            if definition.study_id in projection.studies:
                raise ResearchRegistryError("study is already registered")
            return definition.study_id, StudyRegisteredPayload(study_definition=definition)

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="human",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def reserve_proposal_attempt(
        self,
        study_id: str,
        *,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            if study.state != "development_open":
                raise ResearchRegistryError("proposal attempts require development_open state")
            if study.proposal_budget_remaining == 0:
                raise ResearchRegistryError("proposal budget is exhausted")
            attempt_number = study.proposal_budget_consumed + 1
            attempt_id = content_sha256(
                {
                    "study_id": study_id,
                    "attempt_number": attempt_number,
                    "registry_head_hash": projection.head_hash,
                }
            )
            return study_id, ProposalAttemptReservedPayload(
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                reserved_study_state_hash=study.snapshot_hash(),
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="host",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def complete_proposal_attempt(
        self,
        study_id: str,
        attempt_id: str,
        *,
        outcome: ProposalAttemptOutcome,
        agent_run_id: str,
        detail: str,
        expected_head_hash: str,
        actor_label: str,
        terminal_artifact_hash: str | None = None,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            if study.state != "development_open":
                raise ResearchRegistryError("attempt completion requires development_open state")
            attempt = _require_attempt(study, attempt_id)
            if attempt.status != "reserved":
                raise ResearchRegistryError("proposal attempt is already complete")
            return study_id, ProposalAttemptCompletedPayload(
                attempt_id=attempt_id,
                outcome=outcome,
                agent_run_id=agent_run_id,
                terminal_artifact_hash=terminal_artifact_hash,
                detail=detail,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="host",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def record_proposal_verification(
        self,
        study_id: str,
        attempt_id: str,
        *,
        terminal_artifact_hash: str,
        verification_hash: str,
        passed: bool,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            attempt = _require_attempt(study, attempt_id)
            if attempt.status != "completed" or attempt.outcome not in {"proposal", "abstention"}:
                raise ResearchRegistryError(
                    "only terminal proposals or abstentions can be verified"
                )
            if attempt.terminal_artifact_hash != terminal_artifact_hash:
                raise ResearchRegistryError("verification terminal artifact hash does not match")
            if attempt.verification_hash is not None:
                raise ResearchRegistryError("proposal attempt already has a verification record")
            return study_id, ProposalVerifiedPayload(
                attempt_id=attempt_id,
                terminal_artifact_hash=terminal_artifact_hash,
                verification_hash=verification_hash,
                passed=passed,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="host",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def approve_development_execution(
        self,
        study_id: str,
        attempt_id: str,
        *,
        proposal_verification_hash: str,
        execution_definition_hash: str,
        reviewer_reason: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            if study.state != "development_open":
                raise ResearchRegistryError("development approval requires development_open state")
            if any(value.status == "reserved" for value in study.attempts):
                raise ResearchRegistryError("development approval requires all attempts complete")
            attempt = _require_attempt(study, attempt_id)
            if attempt.verification_passed is not True:
                raise ResearchRegistryError("development approval requires passed verification")
            if attempt.verification_hash != proposal_verification_hash:
                raise ResearchRegistryError("development approval verification hash does not match")
            return study_id, DevelopmentExecutionApprovedPayload(
                attempt_id=attempt_id,
                proposal_verification_hash=proposal_verification_hash,
                execution_definition_hash=execution_definition_hash,
                reviewer_reason=reviewer_reason,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="human",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def start_development_run(
        self,
        study_id: str,
        *,
        execution_definition_hash: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            if study.state != "development_locked":
                raise ResearchRegistryError("development runs require development_locked state")
            approval = next(
                (
                    event
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(event.payload, DevelopmentExecutionApprovedPayload)
                ),
                None,
            )
            if approval is None:
                raise ResearchRegistryError("development run definition is not approved")
            approval_payload = approval.payload
            if not isinstance(approval_payload, DevelopmentExecutionApprovedPayload):
                raise ResearchRegistryError("development approval payload is invalid")
            if approval_payload.execution_definition_hash != execution_definition_hash:
                raise ResearchRegistryError("development run definition is not approved")
            last_run_event = next(
                (
                    event
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(
                        event.payload,
                        (
                            DevelopmentRunStartedPayload,
                            DevelopmentRunFailedPayload,
                            DevelopmentRunCompletedPayload,
                        ),
                    )
                ),
                None,
            )
            if last_run_event is not None and isinstance(
                last_run_event.payload,
                (DevelopmentRunStartedPayload, DevelopmentRunCompletedPayload),
            ):
                raise ResearchRegistryError("development run is already active or complete")
            technical_attempt = 1 + sum(
                isinstance(event.payload, DevelopmentRunStartedPayload)
                and event.study_id == study_id
                for event in events
            )
            development_run_id = content_sha256(
                {
                    "study_id": study_id,
                    "execution_definition_hash": execution_definition_hash,
                    "technical_attempt_number": technical_attempt,
                }
            )
            return study_id, DevelopmentRunStartedPayload(
                development_run_id=development_run_id,
                execution_definition_hash=execution_definition_hash,
                technical_attempt_number=technical_attempt,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="host",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def fail_development_run(
        self,
        study_id: str,
        development_run_id: str,
        *,
        failure_type: str,
        detail: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            active = next(
                (
                    event.payload
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(
                        event.payload,
                        (
                            DevelopmentRunStartedPayload,
                            DevelopmentRunFailedPayload,
                            DevelopmentRunCompletedPayload,
                        ),
                    )
                ),
                None,
            )
            if study.state != "development_locked" or not isinstance(
                active, DevelopmentRunStartedPayload
            ):
                raise ResearchRegistryError("only an active development run can fail")
            if active.development_run_id != development_run_id:
                raise ResearchRegistryError("development failure run ID does not match")
            return study_id, DevelopmentRunFailedPayload(
                development_run_id=development_run_id,
                failure_type=failure_type,
                detail=detail,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="host",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def complete_development_run(
        self,
        study_id: str,
        development_run_id: str,
        *,
        result_manifest_hash: str,
        frozen_model_manifest_hash: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            active = next(
                (
                    event.payload
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(
                        event.payload,
                        (
                            DevelopmentRunStartedPayload,
                            DevelopmentRunFailedPayload,
                            DevelopmentRunCompletedPayload,
                        ),
                    )
                ),
                None,
            )
            if study.state != "development_locked" or not isinstance(
                active, DevelopmentRunStartedPayload
            ):
                raise ResearchRegistryError("only an active development run can complete")
            if active.development_run_id != development_run_id:
                raise ResearchRegistryError("development completion run ID does not match")
            return study_id, DevelopmentRunCompletedPayload(
                development_run_id=development_run_id,
                result_manifest_hash=result_manifest_hash,
                frozen_model_manifest_hash=frozen_model_manifest_hash,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="host",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def freeze_candidate(
        self,
        study_id: str,
        *,
        proposal_hash: str,
        execution_definition_hash: str,
        development_approval_event_hash: str,
        development_result_manifest_hash: str,
        frozen_model_manifest_hash: str,
        candidate_config_hash: str,
        required_evaluation_contract_hash: str,
        holdout_identity: HoldoutIdentity,
        reviewer_reason: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            state = _require_study(projection, study_id)
            completed = next(
                (
                    event.payload
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(event.payload, DevelopmentRunCompletedPayload)
                ),
                None,
            )
            approval = next(
                (
                    event
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(event.payload, DevelopmentExecutionApprovedPayload)
                ),
                None,
            )
            if state.state != "development_locked" or completed is None or approval is None:
                raise ResearchRegistryError("candidate freeze requires a completed development run")
            approval_payload = approval.payload
            if not isinstance(approval_payload, DevelopmentExecutionApprovedPayload):
                raise ResearchRegistryError("candidate approval payload is invalid")
            attempt = _require_attempt(state, approval_payload.attempt_id)
            if attempt.terminal_artifact_hash != proposal_hash:
                raise ResearchRegistryError("candidate proposal hash does not match approval")
            expected_values = (
                approval_payload.execution_definition_hash == execution_definition_hash,
                approval.event_hash == development_approval_event_hash,
                completed.result_manifest_hash == development_result_manifest_hash,
                completed.frozen_model_manifest_hash == frozen_model_manifest_hash,
                holdout_identity.study_id == study_id,
                holdout_identity.candidate_hash == candidate_config_hash,
            )
            if not all(expected_values):
                raise ResearchRegistryError("candidate freeze lineage does not match registry")
            return study_id, CandidateFrozenPayload(
                proposal_hash=proposal_hash,
                execution_definition_hash=execution_definition_hash,
                development_approval_event_hash=development_approval_event_hash,
                development_result_manifest_hash=development_result_manifest_hash,
                frozen_model_manifest_hash=frozen_model_manifest_hash,
                candidate_config_hash=candidate_config_hash,
                required_evaluation_contract_hash=required_evaluation_contract_hash,
                holdout_identity=holdout_identity,
                reviewer_reason=reviewer_reason,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="human",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def register_external_holdout(
        self,
        identity: HoldoutIdentity,
        *,
        reason: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        if not isinstance(identity, HoldoutIdentity):
            raise TypeError("identity must be a HoldoutIdentity")

        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str | None, RegistryPayload]:
            if projection.holdout_use.overlapping(identity):
                raise ResearchRegistryError("holdout identity overlaps prior global use")
            return None, ExternalHoldoutRegisteredPayload(
                holdout_identity=identity,
                reason=reason,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="human",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def reserve_holdout_reveal(
        self,
        study_id: str,
        *,
        candidate_hash: str,
        holdout_identity: HoldoutIdentity,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            if study.state != "candidate_frozen":
                raise ResearchRegistryError("holdout reveal requires candidate_frozen state")
            if any(
                event.study_id == study_id
                and isinstance(event.payload, HoldoutRevealReservedPayload)
                for event in events
            ):
                raise ResearchRegistryError("this study already consumed its reveal reservation")
            frozen = next(
                (
                    event.payload
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(event.payload, CandidateFrozenPayload)
                ),
                None,
            )
            if (
                frozen is None
                or frozen.candidate_config_hash != candidate_hash
                or frozen.holdout_identity != holdout_identity
            ):
                raise ResearchRegistryError("reveal candidate or holdout identity is not frozen")
            if projection.holdout_use.overlapping(holdout_identity):
                raise ResearchRegistryError("holdout overlaps prior global use")
            reservation_id = content_sha256(
                {
                    "study_id": study_id,
                    "candidate_hash": candidate_hash,
                    "holdout_id": holdout_identity.holdout_id,
                    "registry_head_hash": projection.head_hash,
                }
            )
            return study_id, HoldoutRevealReservedPayload(
                reservation_id=reservation_id,
                candidate_hash=candidate_hash,
                holdout_identity=holdout_identity,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="human",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def fail_holdout_reveal(
        self,
        study_id: str,
        reservation_id: str,
        *,
        failure_stage: str,
        detail: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            active = next(
                (
                    event.payload
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(
                        event.payload,
                        (
                            HoldoutRevealReservedPayload,
                            HoldoutRevealFailedPayload,
                            HoldoutRevealedPayload,
                        ),
                    )
                ),
                None,
            )
            if study.state != "candidate_frozen" or not isinstance(
                active, HoldoutRevealReservedPayload
            ):
                raise ResearchRegistryError("only an active reveal reservation can fail")
            if active.reservation_id != reservation_id:
                raise ResearchRegistryError("holdout failure reservation ID does not match")
            return study_id, HoldoutRevealFailedPayload(
                reservation_id=reservation_id,
                failure_stage=failure_stage,
                detail=detail,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="host",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def complete_holdout_reveal(
        self,
        study_id: str,
        reservation_id: str,
        *,
        candidate_hash: str,
        holdout_identity: HoldoutIdentity,
        result_manifest_hash: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            active = next(
                (
                    event.payload
                    for event in reversed(events)
                    if event.study_id == study_id
                    and isinstance(
                        event.payload,
                        (
                            HoldoutRevealReservedPayload,
                            HoldoutRevealFailedPayload,
                            HoldoutRevealedPayload,
                        ),
                    )
                ),
                None,
            )
            if study.state != "candidate_frozen" or not isinstance(
                active, HoldoutRevealReservedPayload
            ):
                raise ResearchRegistryError("only an active reveal reservation can complete")
            if (
                active.reservation_id != reservation_id
                or active.candidate_hash != candidate_hash
                or active.holdout_identity != holdout_identity
            ):
                raise ResearchRegistryError("holdout completion does not match its reservation")
            return study_id, HoldoutRevealedPayload(
                reservation_id=reservation_id,
                candidate_hash=candidate_hash,
                holdout_identity=holdout_identity,
                result_manifest_hash=result_manifest_hash,
            )

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="host",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def close_study(
        self,
        study_id: str,
        *,
        reason: str,
        expected_head_hash: str,
        actor_label: str,
        event_ts: datetime | None = None,
    ) -> RegistryEvent:
        def transition(
            events: tuple[RegistryEvent, ...], projection: RegistryProjection
        ) -> tuple[str, RegistryPayload]:
            study = _require_study(projection, study_id)
            if study.state == "closed":
                raise ResearchRegistryError("study is already closed")
            if any(attempt.status == "reserved" for attempt in study.attempts):
                raise ResearchRegistryError("cannot close a study with an incomplete attempt")
            return study_id, StudyClosedPayload(reason=reason)

        return self._mutate(
            expected_head_hash=expected_head_hash,
            actor_kind="human",
            actor_label=actor_label,
            event_ts=event_ts,
            transition=transition,
        )

    def _mutate(
        self,
        *,
        expected_head_hash: str,
        actor_kind: Literal["host", "human"],
        actor_label: str,
        event_ts: datetime | None,
        transition: RegistryTransition,
    ) -> RegistryEvent:
        if not _is_sha256(expected_head_hash):
            raise ResearchRegistryError("expected_head_hash must be a lowercase SHA-256")
        with self._exclusive_lock():
            events = self._replay_unlocked()
            projection = _fold_registry(events)
            if projection.head_hash != expected_head_hash:
                raise StaleRegistryHeadError(
                    "registry head changed before the requested transition"
                )
            study_id, payload = transition(events, projection)
            if event_ts is not None and event_ts.tzinfo is None:
                raise ResearchRegistryError("registry event_ts must be timezone-aware")
            timestamp = (event_ts or datetime.now(UTC)).astimezone(UTC)
            if events and timestamp < events[-1].event_ts:
                raise ResearchRegistryError("registry event_ts cannot regress")
            try:
                event = RegistryEvent.create(
                    sequence=len(events) + 1,
                    previous_event_hash=projection.head_hash,
                    event_ts=timestamp,
                    study_id=study_id,
                    actor_kind=actor_kind,
                    actor_label=actor_label,
                    payload=payload,
                )
            except (ValidationError, TypeError, ValueError) as exc:
                raise ResearchRegistryError(
                    "registry transition does not satisfy its contract"
                ) from exc
            _fold_registry((*events, event))
            try:
                append_bytes_durable(
                    self.path,
                    (event.canonical_json() + "\n").encode("utf-8"),
                )
            except (SafeFileError, OSError, ValueError) as exc:
                raise ResearchRegistryError("registry event cannot be appended safely") from exc
            return event

    def _replay_unlocked(self) -> tuple[RegistryEvent, ...]:
        try:
            encoded = read_bytes_no_follow(self.path, missing_ok=True)
        except (SafeFileError, OSError, ValueError) as exc:
            raise ResearchRegistryError("registry ledger cannot be opened safely") from exc
        if encoded is None:
            return ()
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResearchRegistryError("registry ledger is not valid UTF-8") from exc
        events: list[RegistryEvent] = []
        expected_previous_hash = GENESIS_HASH
        previous_ts: datetime | None = None
        seen_event_ids: set[str] = set()
        for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
            context = f"registry ledger line {line_number}"
            if not raw_line.strip():
                raise ResearchRegistryError(f"{context} is blank")
            if not raw_line.endswith("\n"):
                raise ResearchRegistryError(f"{context} is incomplete")
            parsed = _parse_json_line(raw_line, context=context)
            if raw_line != canonical_json(parsed) + "\n":
                raise ResearchRegistryError(f"{context} is not canonical JSON")
            try:
                event = RegistryEvent.model_validate_json(raw_line)
            except ValidationError as exc:
                raise ResearchRegistryError(
                    f"{context} violates the registry event contract"
                ) from exc
            if raw_line != event.canonical_json() + "\n":
                raise ResearchRegistryError(f"{context} is not canonical typed event JSON")
            if event.sequence != line_number:
                raise ResearchRegistryError(
                    f"{context} has sequence {event.sequence}; expected {line_number}"
                )
            if event.previous_event_hash != expected_previous_hash:
                raise ResearchRegistryError(f"{context} breaks the previous event hash link")
            if event.event_id in seen_event_ids:
                raise ResearchRegistryError(f"{context} repeats event_id {event.event_id!r}")
            if previous_ts is not None and event.event_ts < previous_ts:
                raise ResearchRegistryError(f"{context} regresses event_ts")
            seen_event_ids.add(event.event_id)
            previous_ts = event.event_ts
            expected_previous_hash = event.event_hash
            events.append(event)
        if text and not text.endswith("\n"):
            raise ResearchRegistryError("registry ledger has an incomplete trailing line")
        result = tuple(events)
        _fold_registry(result)
        return result

    def _exclusive_lock(self) -> _RegistryLockContext:
        return _RegistryLockContext(self.lock_path)


class _RegistryLockContext:
    def __init__(self, path: Path) -> None:
        self._context = advisory_file_lock(path)

    def __enter__(self) -> None:
        try:
            self._context.__enter__()
        except (AdvisoryFileLockError, AdvisoryFileLockUnavailable) as exc:
            raise ResearchRegistryLockError("research registry lock is unavailable") from exc

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            return self._context.__exit__(exc_type, exc, traceback)
        except (AdvisoryFileLockError, AdvisoryFileLockUnavailable) as error:
            raise ResearchRegistryLockError("research registry lock is unavailable") from error


def _fold_registry(events: tuple[RegistryEvent, ...]) -> RegistryProjection:
    studies: dict[str, _MutableStudy] = {}
    holdouts: list[HoldoutUseRecord] = []
    for event in events:
        payload = event.payload
        if isinstance(payload, StudyRegisteredPayload):
            definition = payload.study_definition
            if definition.study_id in studies:
                raise ResearchRegistryError("registry re-registers an existing study")
            studies[definition.study_id] = _MutableStudy(
                definition=definition,
                transition_event_hashes=[event.event_hash],
            )
            continue
        if isinstance(payload, ExternalHoldoutRegisteredPayload):
            existing_index = HoldoutUseIndex(
                records=tuple(holdouts),
                last_registry_head_hash=event.previous_event_hash,
            )
            if existing_index.overlapping(payload.holdout_identity):
                raise ResearchRegistryError("registry repeats or overlaps a used holdout")
            holdouts.append(
                HoldoutUseRecord(
                    holdout_identity=payload.holdout_identity,
                    registry_event_hash=event.event_hash,
                    source="external",
                )
            )
            continue
        if event.study_id is None or event.study_id not in studies:
            raise ResearchRegistryError("registry event references an unknown study")
        study = studies[event.study_id]
        if isinstance(payload, ProposalAttemptReservedPayload):
            if study.state != "development_open":
                raise ResearchRegistryError("registry reserves an attempt outside development_open")
            if len(study.attempts) >= study.definition.proposal_budget:
                raise ResearchRegistryError("registry overspends the proposal budget")
            if payload.attempt_number != len(study.attempts) + 1:
                raise ResearchRegistryError("registry attempt numbers are not contiguous")
            expected_attempt_id = content_sha256(
                {
                    "study_id": event.study_id,
                    "attempt_number": payload.attempt_number,
                    "registry_head_hash": event.previous_event_hash,
                }
            )
            if payload.attempt_id != expected_attempt_id:
                raise ResearchRegistryError("registry attempt_id is not canonical")
            pre_snapshot = _snapshot(study, event.previous_event_hash)
            if payload.reserved_study_state_hash != pre_snapshot.snapshot_hash():
                raise ResearchRegistryError("reserved study-state hash does not match")
            study.attempts.append(
                _MutableAttempt(
                    attempt_id=payload.attempt_id,
                    attempt_number=payload.attempt_number,
                    reservation_event_hash=event.event_hash,
                    reserved_study_state_hash=payload.reserved_study_state_hash,
                )
            )
        elif isinstance(payload, ProposalAttemptCompletedPayload):
            if study.state != "development_open":
                raise ResearchRegistryError(
                    "registry completes an attempt outside development_open"
                )
            attempt = _mutable_attempt(study, payload.attempt_id)
            if attempt.status != "reserved":
                raise ResearchRegistryError("registry completes one proposal attempt twice")
            attempt.status = "completed"
            attempt.outcome = payload.outcome
            attempt.agent_run_id = payload.agent_run_id
            attempt.terminal_artifact_hash = payload.terminal_artifact_hash
        elif isinstance(payload, ProposalVerifiedPayload):
            attempt = _mutable_attempt(study, payload.attempt_id)
            if attempt.status != "completed" or attempt.outcome not in {"proposal", "abstention"}:
                raise ResearchRegistryError("registry verifies a non-terminal proposal attempt")
            if attempt.terminal_artifact_hash != payload.terminal_artifact_hash:
                raise ResearchRegistryError("registry verification artifact hash does not match")
            if attempt.verification_hash is not None:
                raise ResearchRegistryError("registry verifies one attempt twice")
            attempt.verification_hash = payload.verification_hash
            attempt.verification_passed = payload.passed
        elif isinstance(payload, DevelopmentExecutionApprovedPayload):
            if study.state != "development_open":
                raise ResearchRegistryError(
                    "registry approves development outside development_open"
                )
            if any(attempt.status == "reserved" for attempt in study.attempts):
                raise ResearchRegistryError("registry approves development with an active attempt")
            attempt = _mutable_attempt(study, payload.attempt_id)
            if (
                attempt.verification_passed is not True
                or attempt.verification_hash != payload.proposal_verification_hash
            ):
                raise ResearchRegistryError("registry approval does not bind a passed verification")
            study.state = "development_locked"
            study.approved_attempt_id = payload.attempt_id
            study.approval_event_hash = event.event_hash
            study.approved_verification_hash = payload.proposal_verification_hash
            study.execution_definition_hash = payload.execution_definition_hash
        elif isinstance(payload, DevelopmentRunStartedPayload):
            if study.state != "development_locked":
                raise ResearchRegistryError("registry starts development outside locked state")
            if study.execution_definition_hash != payload.execution_definition_hash:
                raise ResearchRegistryError("registry starts an unapproved development definition")
            expected_number = 1 + sum(
                isinstance(previous.payload, DevelopmentRunStartedPayload)
                and previous.study_id == event.study_id
                for previous in events[: event.sequence - 1]
            )
            if payload.technical_attempt_number != expected_number:
                raise ResearchRegistryError("registry development attempt number is not contiguous")
            if study.development_run_status in {"started", "completed"}:
                raise ResearchRegistryError("registry starts a duplicate development run")
            expected_run_id = content_sha256(
                {
                    "study_id": event.study_id,
                    "execution_definition_hash": payload.execution_definition_hash,
                    "technical_attempt_number": payload.technical_attempt_number,
                }
            )
            if payload.development_run_id != expected_run_id:
                raise ResearchRegistryError("registry development_run_id is not canonical")
            study.development_run_id = payload.development_run_id
            study.development_run_status = "started"
        elif isinstance(payload, DevelopmentRunFailedPayload):
            if (
                study.state != "development_locked"
                or study.development_run_status != "started"
                or study.development_run_id != payload.development_run_id
            ):
                raise ResearchRegistryError("registry fails a non-active development run")
            study.development_run_status = "failed"
        elif isinstance(payload, DevelopmentRunCompletedPayload):
            if (
                study.state != "development_locked"
                or study.development_run_status != "started"
                or study.development_run_id != payload.development_run_id
            ):
                raise ResearchRegistryError("registry completes a non-active development run")
            study.development_run_status = "completed"
            study.development_result_manifest_hash = payload.result_manifest_hash
            study.frozen_model_manifest_hash = payload.frozen_model_manifest_hash
        elif isinstance(payload, CandidateFrozenPayload):
            if study.state != "development_locked" or study.development_run_status != "completed":
                raise ResearchRegistryError("registry freezes before development completion")
            attempt = _mutable_attempt(study, study.approved_attempt_id or "")
            if (
                attempt.terminal_artifact_hash != payload.proposal_hash
                or study.execution_definition_hash != payload.execution_definition_hash
                or study.approval_event_hash != payload.development_approval_event_hash
                or study.development_result_manifest_hash
                != payload.development_result_manifest_hash
                or study.frozen_model_manifest_hash != payload.frozen_model_manifest_hash
                or payload.holdout_identity.study_id != event.study_id
                or payload.holdout_identity.candidate_hash != payload.candidate_config_hash
            ):
                raise ResearchRegistryError("registry candidate lineage is inconsistent")
            study.state = "candidate_frozen"
            study.candidate_config_hash = payload.candidate_config_hash
        elif isinstance(payload, HoldoutRevealReservedPayload):
            if study.state != "candidate_frozen" or study.holdout_reveal_status is not None:
                raise ResearchRegistryError("registry repeats or mistimes a reveal reservation")
            if (
                study.candidate_config_hash != payload.candidate_hash
                or payload.holdout_identity.study_id != event.study_id
                or payload.holdout_identity.candidate_hash != payload.candidate_hash
            ):
                raise ResearchRegistryError(
                    "registry reveal reservation is not the frozen candidate"
                )
            existing_index = HoldoutUseIndex(
                records=tuple(holdouts),
                last_registry_head_hash=event.previous_event_hash,
            )
            if existing_index.overlapping(payload.holdout_identity):
                raise ResearchRegistryError("registry reveal overlaps a used holdout")
            expected_reservation_id = content_sha256(
                {
                    "study_id": event.study_id,
                    "candidate_hash": payload.candidate_hash,
                    "holdout_id": payload.holdout_identity.holdout_id,
                    "registry_head_hash": event.previous_event_hash,
                }
            )
            if payload.reservation_id != expected_reservation_id:
                raise ResearchRegistryError("registry reveal reservation_id is not canonical")
            holdouts.append(
                HoldoutUseRecord(
                    holdout_identity=payload.holdout_identity,
                    registry_event_hash=event.event_hash,
                    source="reveal_reservation",
                )
            )
            study.holdout_reservation_id = payload.reservation_id
            study.holdout_reservation_event_hash = event.event_hash
            study.holdout_identity = payload.holdout_identity
            study.holdout_reveal_status = "reserved"
        elif isinstance(payload, HoldoutRevealFailedPayload):
            if (
                study.state != "candidate_frozen"
                or study.holdout_reveal_status != "reserved"
                or study.holdout_reservation_id != payload.reservation_id
            ):
                raise ResearchRegistryError("registry fails a non-active reveal")
            study.holdout_reveal_status = "failed"
        elif isinstance(payload, HoldoutRevealedPayload):
            if (
                study.state != "candidate_frozen"
                or study.holdout_reveal_status != "reserved"
                or study.holdout_reservation_id != payload.reservation_id
                or study.candidate_config_hash != payload.candidate_hash
                or study.holdout_identity != payload.holdout_identity
            ):
                raise ResearchRegistryError("registry completes a non-active reveal")
            overlaps = HoldoutUseIndex(
                records=tuple(holdouts),
                last_registry_head_hash=event.previous_event_hash,
            ).overlapping(payload.holdout_identity)
            if (
                len(overlaps) != 1
                or overlaps[0].source != "reveal_reservation"
                or overlaps[0].registry_event_hash != study.holdout_reservation_event_hash
            ):
                raise ResearchRegistryError(
                    "registry reveal reservation contamination record is inconsistent"
                )
            study.holdout_reveal_status = "revealed"
            study.state = "holdout_revealed"
        elif isinstance(payload, StudyClosedPayload):
            if study.state == "closed":
                raise ResearchRegistryError("registry closes one study twice")
            if any(attempt.status == "reserved" for attempt in study.attempts):
                raise ResearchRegistryError("registry closes a study with an incomplete attempt")
            if study.development_run_status == "started":
                raise ResearchRegistryError(
                    "registry closes a study with an active development run"
                )
            if study.holdout_reveal_status == "reserved":
                raise ResearchRegistryError("registry closes a study with an active reveal")
            study.state = "closed"
        else:  # pragma: no cover - discriminated union is exhaustive
            raise ResearchRegistryError("registry payload is unsupported")
        study.transition_event_hashes.append(event.event_hash)

    head = events[-1].event_hash if events else GENESIS_HASH
    snapshots = {study_id: _snapshot(study, head) for study_id, study in studies.items()}
    return RegistryProjection(
        studies=snapshots,
        holdout_use=HoldoutUseIndex(records=tuple(holdouts), last_registry_head_hash=head),
        head_hash=head,
    )


def _snapshot(study: _MutableStudy, head_hash: str) -> StudyStateSnapshot:
    attempts = tuple(
        ProposalAttemptSnapshot(
            attempt_id=value.attempt_id,
            attempt_number=value.attempt_number,
            reservation_event_hash=value.reservation_event_hash,
            reserved_study_state_hash=value.reserved_study_state_hash,
            status=value.status,
            outcome=value.outcome,
            agent_run_id=value.agent_run_id,
            terminal_artifact_hash=value.terminal_artifact_hash,
            verification_hash=value.verification_hash,
            verification_passed=value.verification_passed,
        )
        for value in study.attempts
    )
    consumed = len(attempts)
    return StudyStateSnapshot(
        study_id=study.definition.study_id,
        definition_hash=study.definition.study_id,
        state=study.state,
        proposal_budget=study.definition.proposal_budget,
        proposal_budget_consumed=consumed,
        proposal_budget_remaining=study.definition.proposal_budget - consumed,
        attempts=attempts,
        transition_event_hashes=tuple(study.transition_event_hashes),
        last_registry_head_hash=head_hash,
    )


def _require_study(projection: RegistryProjection, study_id: str) -> StudyStateSnapshot:
    try:
        return projection.studies[study_id]
    except KeyError as exc:
        raise ResearchRegistryError("study is not registered") from exc


def _require_attempt(study: StudyStateSnapshot, attempt_id: str) -> ProposalAttemptSnapshot:
    for attempt in study.attempts:
        if attempt.attempt_id == attempt_id:
            return attempt
    raise ResearchRegistryError("proposal attempt is not reserved")


def _mutable_attempt(study: _MutableStudy, attempt_id: str) -> _MutableAttempt:
    for attempt in study.attempts:
        if attempt.attempt_id == attempt_id:
            return attempt
    raise ResearchRegistryError("registry references an unknown proposal attempt")


def _parse_json_line(raw_line: str, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw_line,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _DuplicateJsonKeyError as exc:
        raise ResearchRegistryError(f"{context} repeats JSON key {exc.key!r}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ResearchRegistryError(f"{context} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ResearchRegistryError(f"{context} must contain an object")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
