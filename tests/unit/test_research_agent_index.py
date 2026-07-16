from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from nlp_trader.research_agents.contracts import SearchEvidenceRequest
from nlp_trader.research_agents.evidence import (
    EvidenceSourceRecord,
    build_evidence_snapshot,
    evidence_snapshot_bytes,
    normalized_source_text,
)
from nlp_trader.research_agents.index import build_lexical_index, search_evidence


def _evidence() -> tuple:
    cutoff = datetime(2025, 12, 31, tzinfo=UTC)
    sources = []
    for item_id, body, asset in (
        ("item-a", "需要が改善し受注が増加した。", "asset-a"),
        ("item-b", "需要は弱く受注が減少した。", "asset-b"),
        ("item-c", "Demand improved while supply risk remained.", "asset-a"),
    ):
        text = normalized_source_text(None, body)
        sources.append(
            EvidenceSourceRecord(
                item_id=item_id,
                source_type="licensed-news",
                language="ja",
                body=body,
                source_text_hash=hashlib.sha256(text.encode()).hexdigest(),
                content_status="active",
                relationship_type="original",
                license_or_terms_ref="fixture-terms",
                retention_permitted=True,
                asset_ids=(asset,),
                active_period_valid=True,
                published_at=datetime(2025, 6, 1, tzinfo=UTC),
                available_at=datetime(2025, 6, 1, 0, 1, tzinfo=UTC),
                source_artifact_id="silver-text-v1",
                source_artifact_hash="1" * 64,
            )
        )
    return build_evidence_snapshot(tuple(sources), analysis_cutoff=cutoff)


def test_lexical_index_is_deterministic_multilingual_and_paginated() -> None:
    evidence = _evidence()
    snapshot_hash = hashlib.sha256(evidence_snapshot_bytes(evidence)).hexdigest()
    index = build_lexical_index(evidence, evidence_snapshot_hash=snapshot_hash)
    request = SearchEvidenceRequest(
        query="需要 改善",
        purpose="support",
        result_limit=1,
    )

    first = search_evidence(index, evidence, request)
    replay = search_evidence(index, evidence, request)

    assert first == replay
    assert first.results[0].evidence.source_item_id == "item-a"
    assert first.next_cursor is not None
    second = search_evidence(
        index, evidence, request.model_copy(update={"cursor": first.next_cursor})
    )
    assert second.results
    assert second.results[0].evidence.evidence_id != first.results[0].evidence.evidence_id


def test_lexical_search_filters_before_scoring_and_rejects_unsafe_or_stale_cursors() -> None:
    evidence = _evidence()
    snapshot_hash = hashlib.sha256(evidence_snapshot_bytes(evidence)).hexdigest()
    index = build_lexical_index(evidence, evidence_snapshot_hash=snapshot_hash)
    filtered = search_evidence(
        index,
        evidence,
        SearchEvidenceRequest(
            query="需要",
            purpose="challenge",
            asset_ids=("asset-b",),
            result_limit=10,
        ),
    )
    assert {ref.evidence.asset_ids for ref in filtered.results} == {("asset-b",)}
    assert all(ref.citation_role == "counterevidence" for ref in filtered.results)

    with pytest.raises(ValueError, match="forbidden"):
        search_evidence(
            index,
            evidence,
            SearchEvidenceRequest(query="file:///tmp/secret", purpose="support", result_limit=1),
        )

    page = search_evidence(
        index,
        evidence,
        SearchEvidenceRequest(query="需要", purpose="support", result_limit=1),
    )
    assert page.next_cursor is not None
    with pytest.raises(ValueError, match="invalid|hash|stale"):
        search_evidence(
            index,
            evidence,
            SearchEvidenceRequest(
                query="different",
                purpose="support",
                result_limit=1,
                cursor=page.next_cursor,
            ),
        )
