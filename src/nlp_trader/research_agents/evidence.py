from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from nlp_trader.research_agents.contracts import (
    EvidenceRecord,
    Identifier,
    NonBlankText,
    Sha256,
    StrictModel,
)
from nlp_trader.timestamps import format_utc, parse_utc

MAX_SOURCE_TEXT_CHARS = 1_000_000
DEFAULT_SPAN_CHARS = 512


class EvidenceSourceRecord(StrictModel):
    """Trusted pre-index input with no raw path or un-hashed author identity."""

    item_id: Identifier
    source_type: Identifier
    language: Identifier
    title: str | None = Field(default=None, max_length=MAX_SOURCE_TEXT_CHARS)
    body: str | None = Field(default=None, max_length=MAX_SOURCE_TEXT_CHARS)
    source_text_hash: Sha256
    content_status: Literal["active", "deleted", "private", "protected", "unknown"]
    relationship_type: Literal["original", "repost", "quote", "reply", "unknown"]
    license_or_terms_ref: NonBlankText
    retention_permitted: Literal[True]
    asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    active_period_valid: Literal[True]
    published_at: datetime
    available_at: datetime
    source_artifact_id: Identifier
    source_artifact_hash: Sha256
    author_hash: Sha256 | None = None
    url_hash: Sha256 | None = None

    @field_validator("published_at", "available_at", mode="before")
    @classmethod
    def normalize_timestamp(cls, value: object, info: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError(f"{info.field_name} must be timezone-aware")
            return value.astimezone(UTC)
        if isinstance(value, str):
            return parse_utc(value)
        raise ValueError(f"{info.field_name} must be a datetime or ISO timestamp")

    @field_serializer("published_at", "available_at")
    def serialize_timestamp(self, value: datetime) -> str:
        return format_utc(value)

    @field_validator("asset_ids")
    @classmethod
    def validate_asset_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)) or values != tuple(sorted(values)):
            raise ValueError("evidence source asset_ids must be unique and sorted")
        return values

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.content_status in {"deleted", "private", "protected"}:
            raise ValueError("evidence source content status is not exportable")
        if self.available_at < self.published_at:
            raise ValueError("evidence available_at cannot precede published_at")
        normalized = normalized_source_text(self.title, self.body)
        if not normalized:
            raise ValueError("evidence source requires title or body")
        if len(normalized) > MAX_SOURCE_TEXT_CHARS:
            raise ValueError("evidence source exceeds the maximum retained character bound")
        if hashlib.sha256(normalized.encode("utf-8")).hexdigest() != self.source_text_hash:
            raise ValueError("evidence source_text_hash does not match normalized source text")
        return self


def normalized_source_text(title: str | None, body: str | None) -> str:
    """Normalize line endings only and join the exact retained title/body values."""

    parts = [_normalize_lines(value) for value in (title, body) if value]
    return "\n".join(parts)


def build_evidence_snapshot(
    sources: tuple[EvidenceSourceRecord, ...],
    *,
    analysis_cutoff: datetime,
    span_chars: int = DEFAULT_SPAN_CHARS,
) -> tuple[EvidenceRecord, ...]:
    if analysis_cutoff.tzinfo is None:
        raise ValueError("analysis_cutoff must be timezone-aware")
    cutoff = analysis_cutoff.astimezone(UTC)
    if span_chars < 64 or span_chars > 4096:
        raise ValueError("span_chars must be between 64 and 4096")
    seen_items: set[str] = set()
    records: list[EvidenceRecord] = []
    for source in sorted(sources, key=lambda value: value.item_id):
        if source.item_id in seen_items:
            raise ValueError(f"duplicate evidence source item_id: {source.item_id}")
        seen_items.add(source.item_id)
        if source.available_at > cutoff:
            raise ValueError("evidence source is not available by the analysis cutoff")
        text = normalized_source_text(source.title, source.body)
        title_length = len(_normalize_lines(source.title)) if source.title else 0
        for start in range(0, len(text), span_chars):
            end = min(start + span_chars, len(text))
            quoted = text[start:end]
            parts: list[Literal["title", "body"]] = []
            if source.title and start < title_length:
                parts.append("title")
            body_start = title_length + (1 if source.title and source.body else 0)
            if source.body and end > body_start:
                parts.append("body")
            span_hash = hashlib.sha256(quoted.encode("utf-8")).hexdigest()
            records.append(
                EvidenceRecord(
                    source_item_id=source.item_id,
                    span_id=f"{source.item_id}:chars:{start}-{end}",
                    source_text_hash=source.source_text_hash,
                    span_hash=span_hash,
                    source_type=source.source_type,
                    content_status=source.content_status,
                    relationship_type=source.relationship_type,
                    license_or_terms_ref=source.license_or_terms_ref,
                    retention_permitted=True,
                    asset_ids=source.asset_ids,
                    active_period_valid=True,
                    published_at=source.published_at,
                    available_at=source.available_at,
                    snapshot_cutoff=cutoff,
                    quoted_span=quoted,
                    start_offset=start,
                    end_offset=end,
                    text_parts=tuple(parts),
                    source_artifact_id=source.source_artifact_id,
                    source_artifact_hash=source.source_artifact_hash,
                )
            )
    return tuple(records)


def evidence_snapshot_bytes(records: tuple[EvidenceRecord, ...]) -> bytes:
    if not records:
        raise ValueError("evidence snapshot cannot be empty")
    ids = tuple(record.evidence_id for record in records)
    if len(ids) != len(set(ids)):
        raise ValueError("evidence snapshot IDs must be unique")
    return "".join(record.canonical_json() + "\n" for record in records).encode("utf-8")


def _normalize_lines(value: str | None) -> str:
    return "" if value is None else value.replace("\r\n", "\n").replace("\r", "\n")
