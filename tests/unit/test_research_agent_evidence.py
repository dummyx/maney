from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nlp_trader.research_agents.evidence import (
    EvidenceSourceRecord,
    build_evidence_snapshot,
    normalized_source_text,
)


def _source(
    *, available_at: datetime | None = None, body: str = "根拠は需要の改善です。"
) -> EvidenceSourceRecord:
    title = "Synthetic update"
    text = normalized_source_text(title, body)
    return EvidenceSourceRecord(
        item_id="item-1",
        source_type="licensed-news",
        language="ja",
        title=title,
        body=body,
        source_text_hash=hashlib.sha256(text.encode()).hexdigest(),
        content_status="active",
        relationship_type="original",
        license_or_terms_ref="synthetic-fixture-terms",
        retention_permitted=True,
        asset_ids=("asset-a",),
        active_period_valid=True,
        published_at=datetime(2025, 6, 1, tzinfo=UTC),
        available_at=available_at or datetime(2025, 6, 1, 0, 1, tzinfo=UTC),
        source_artifact_id="silver-text-v1",
        source_artifact_hash="1" * 64,
    )


def test_evidence_snapshot_is_deterministic_bounded_and_point_in_time() -> None:
    source = _source(body="A" * 80 + "\r\n" + "B" * 80)
    cutoff = datetime(2025, 12, 31, tzinfo=UTC)

    first = build_evidence_snapshot((source,), analysis_cutoff=cutoff, span_chars=64)
    second = build_evidence_snapshot((source,), analysis_cutoff=cutoff, span_chars=64)

    assert first == second
    assert len(first) >= 2
    assert all(record.available_at <= record.snapshot_cutoff for record in first)
    assert all(
        record.end_offset == record.start_offset + len(record.quoted_span) for record in first
    )
    assert "\r" not in "".join(record.quoted_span for record in first)


def test_evidence_snapshot_rejects_future_rights_and_hash_failures() -> None:
    with pytest.raises(ValueError, match="analysis cutoff"):
        build_evidence_snapshot(
            (_source(available_at=datetime(2026, 1, 1, tzinfo=UTC)),),
            analysis_cutoff=datetime(2025, 12, 31, tzinfo=UTC),
        )

    payload = _source().model_dump(mode="python")
    payload["retention_permitted"] = False
    with pytest.raises(ValidationError):
        EvidenceSourceRecord.model_validate(payload)

    payload = _source().model_dump(mode="python")
    payload["source_text_hash"] = "2" * 64
    with pytest.raises(ValidationError, match="source_text_hash"):
        EvidenceSourceRecord.model_validate(payload)
