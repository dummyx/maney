from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from nlp_trader.research_agents.approvals import CandidateFreezeRecord
from nlp_trader.research_agents.contracts import (
    HoldoutRevealReservedPayload,
    Sha256,
    StrictModel,
    content_sha256,
)
from nlp_trader.research_agents.registry import ResearchRegistryLedger
from nlp_trader.timestamps import format_utc, parse_utc


class HoldoutRevealReservation(StrictModel):
    artifact_schema_version: Literal["holdout-reveal-reservation-v1"] = (
        "holdout-reveal-reservation-v1"
    )
    record_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    reservation_id: Sha256
    candidate_hash: Sha256
    holdout_id: Sha256
    registry_event_hash: Sha256
    reserved_at: datetime

    @field_validator("reserved_at", mode="before")
    @classmethod
    def normalize_time(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("reserved_at must be timezone-aware")
            return value.astimezone(UTC)
        if isinstance(value, str):
            return parse_utc(value)
        raise ValueError("reserved_at must be a timestamp")

    @field_serializer("reserved_at")
    def serialize_time(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = content_sha256(self.model_dump(mode="json", exclude={"record_id"}))
        if self.record_id and self.record_id != expected:
            raise ValueError("holdout reservation record_id does not match canonical content")
        if not self.record_id:
            object.__setattr__(self, "record_id", expected)
        return self


def reserve_frozen_holdout(
    ledger: ResearchRegistryLedger,
    freeze: CandidateFreezeRecord,
    *,
    actor_label: str,
    event_ts: datetime | None = None,
) -> HoldoutRevealReservation:
    event = ledger.reserve_holdout_reveal(
        freeze.study_id,
        candidate_hash=freeze.candidate_config_hash,
        holdout_identity=freeze.holdout_identity,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor_label,
        event_ts=event_ts,
    )
    payload = event.payload
    if not isinstance(payload, HoldoutRevealReservedPayload):  # pragma: no cover - invariant
        raise RuntimeError("registry returned an unexpected holdout reservation")
    return HoldoutRevealReservation(
        study_id=freeze.study_id,
        reservation_id=payload.reservation_id,
        candidate_hash=payload.candidate_hash,
        holdout_id=payload.holdout_identity.holdout_id,
        registry_event_hash=event.event_hash,
        reserved_at=event.event_ts,
    )


def register_external_holdout_use(
    ledger: ResearchRegistryLedger,
    freeze: CandidateFreezeRecord,
    *,
    reason: str,
    actor_label: str,
    event_ts: datetime | None = None,
) -> str:
    event = ledger.register_external_holdout(
        freeze.holdout_identity,
        reason=reason,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor_label,
        event_ts=event_ts,
    )
    return event.event_hash
