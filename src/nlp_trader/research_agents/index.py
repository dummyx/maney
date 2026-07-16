from __future__ import annotations

import base64
import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from nlp_trader.research_agents.contracts import (
    EvidenceRecord,
    EvidenceReference,
    SearchEvidenceRequest,
    Sha256,
    StrictModel,
    canonical_json,
    content_sha256,
)

INDEX_VERSION: Literal["lexical_char_ngram_v1"] = "lexical_char_ngram_v1"
_FORBIDDEN_QUERY = re.compile(
    r"(?:https?://|file://|\\|(?:^|\s)(?:/|~/|\.\.?/)|`|\$\(|\b(?:select|insert|update|delete|drop)\b)",
    re.IGNORECASE,
)


class TermCount(StrictModel):
    term: str = Field(min_length=1, max_length=8)
    count: int = Field(ge=1)


class IndexedEvidenceDocument(StrictModel):
    evidence_id: Sha256
    terms: tuple[TermCount, ...]
    term_total: int = Field(ge=1)

    @field_validator("terms")
    @classmethod
    def validate_terms(cls, values: tuple[TermCount, ...]) -> tuple[TermCount, ...]:
        terms = tuple(value.term for value in values)
        if len(terms) != len(set(terms)) or terms != tuple(sorted(terms)):
            raise ValueError("indexed terms must be unique and sorted")
        return values


class DocumentFrequency(StrictModel):
    term: str = Field(min_length=1, max_length=8)
    documents: int = Field(ge=1)


class LexicalCharNgramIndex(StrictModel):
    artifact_schema_version: Literal["lexical-char-ngram-index-v1"] = "lexical-char-ngram-index-v1"
    index_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    version: Literal["lexical_char_ngram_v1"] = INDEX_VERSION
    evidence_snapshot_hash: Sha256
    ngram_min: int = Field(default=2, ge=1, le=8)
    ngram_max: int = Field(default=5, ge=1, le=8)
    score_round_digits: int = Field(default=12, ge=6, le=15)
    documents: tuple[IndexedEvidenceDocument, ...]
    document_frequencies: tuple[DocumentFrequency, ...]

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        if self.ngram_min > self.ngram_max:
            raise ValueError("ngram_min cannot exceed ngram_max")
        document_ids = tuple(value.evidence_id for value in self.documents)
        if not document_ids or document_ids != tuple(sorted(document_ids)):
            raise ValueError("index documents must be nonempty and sorted by evidence_id")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("index document evidence IDs must be unique")
        terms = tuple(value.term for value in self.document_frequencies)
        if terms != tuple(sorted(terms)) or len(terms) != len(set(terms)):
            raise ValueError("document frequencies must be unique and sorted")
        expected = self.computed_index_id()
        if self.index_id and self.index_id != expected:
            raise ValueError("index_id does not match canonical index content")
        if not self.index_id:
            object.__setattr__(self, "index_id", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"index_id"})

    def computed_index_id(self) -> str:
        return content_sha256(self.content_payload())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class EvidenceSearchPage(StrictModel):
    query_id: Sha256
    normalized_query_hash: Sha256
    filters_hash: Sha256
    results: tuple[EvidenceReference, ...]
    total_result_count: int = Field(ge=0)
    next_cursor: str | None = Field(default=None, max_length=4096)


def normalize_search_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def character_ngrams(value: str, *, minimum: int, maximum: int) -> Counter[str]:
    normalized = normalize_search_text(value)
    counts: Counter[str] = Counter()
    for width in range(minimum, maximum + 1):
        if len(normalized) < width:
            continue
        counts.update(
            normalized[index : index + width] for index in range(len(normalized) - width + 1)
        )
    return counts


def build_lexical_index(
    evidence: tuple[EvidenceRecord, ...],
    *,
    evidence_snapshot_hash: str,
    ngram_min: int = 2,
    ngram_max: int = 5,
) -> LexicalCharNgramIndex:
    if ngram_min < 1 or ngram_max > 8 or ngram_min > ngram_max:
        raise ValueError("invalid lexical n-gram bounds")
    documents: list[IndexedEvidenceDocument] = []
    frequencies: Counter[str] = Counter()
    for record in sorted(evidence, key=lambda value: value.evidence_id):
        counts = character_ngrams(record.quoted_span, minimum=ngram_min, maximum=ngram_max)
        if not counts:
            raise ValueError("evidence span is too short for the configured lexical index")
        frequencies.update(counts.keys())
        terms = tuple(TermCount(term=term, count=count) for term, count in sorted(counts.items()))
        documents.append(
            IndexedEvidenceDocument(
                evidence_id=record.evidence_id,
                terms=terms,
                term_total=sum(counts.values()),
            )
        )
    document_frequencies = tuple(
        DocumentFrequency(term=term, documents=count) for term, count in sorted(frequencies.items())
    )
    return LexicalCharNgramIndex(
        evidence_snapshot_hash=evidence_snapshot_hash,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        documents=tuple(documents),
        document_frequencies=document_frequencies,
    )


def search_evidence(
    index: LexicalCharNgramIndex,
    evidence: tuple[EvidenceRecord, ...],
    request: SearchEvidenceRequest,
) -> EvidenceSearchPage:
    if _FORBIDDEN_QUERY.search(request.query):
        raise ValueError("evidence query contains a forbidden path, URL, code, or SQL marker")
    normalized_query = normalize_search_text(request.query)
    if not normalized_query:
        raise ValueError("evidence query is empty after normalization")
    query_hash = content_sha256(normalized_query)
    filters = {
        "purpose": request.purpose,
        "asset_ids": sorted(request.asset_ids),
        "source_types": sorted(request.source_types),
        "available_range": (
            request.available_range.model_dump(mode="json") if request.available_range else None
        ),
        "result_limit": request.result_limit,
    }
    filters_hash = content_sha256(filters)
    query_id = content_sha256(
        {"index_id": index.index_id, "query_hash": query_hash, "filters_hash": filters_hash}
    )
    record_by_id = {record.evidence_id: record for record in evidence}
    if set(record_by_id) != {document.evidence_id for document in index.documents}:
        raise ValueError("evidence snapshot does not match the lexical index membership")
    query_terms = character_ngrams(
        normalized_query,
        minimum=index.ngram_min,
        maximum=index.ngram_max,
    )
    frequencies = {value.term: value.documents for value in index.document_frequencies}
    eligible: list[tuple[float, str]] = []
    requested_assets = set(request.asset_ids)
    requested_sources = set(request.source_types)
    for document in index.documents:
        record = record_by_id[document.evidence_id]
        if requested_assets and not requested_assets.intersection(record.asset_ids):
            continue
        if requested_sources and record.source_type not in requested_sources:
            continue
        if request.available_range and not (
            request.available_range.start <= record.available_at <= request.available_range.end
        ):
            continue
        document_terms = {value.term: value.count for value in document.terms}
        score = 0.0
        for term, query_count in query_terms.items():
            count = document_terms.get(term, 0)
            if not count:
                continue
            inverse = math.log((1 + len(index.documents)) / (1 + frequencies[term])) + 1.0
            score += query_count * (count / document.term_total) * inverse
        rounded = round(score, index.score_round_digits)
        if rounded > 0:
            eligible.append((rounded, document.evidence_id))
    eligible.sort(key=lambda value: (-value[0], value[1]))
    offset = _cursor_offset(
        request.cursor,
        index_id=index.index_id,
        query_id=query_id,
        ordered_ids=tuple(value[1] for value in eligible),
    )
    page = eligible[offset : offset + request.result_limit]
    role: Literal["supporting", "counterevidence"] = (
        "supporting" if request.purpose == "support" else "counterevidence"
    )
    references = tuple(
        EvidenceReference(
            evidence=record_by_id[evidence_id],
            query_id=query_id,
            rank=offset + rank,
            score=score,
            citation_role=role,
        )
        for rank, (score, evidence_id) in enumerate(page, start=1)
    )
    next_offset = offset + len(page)
    next_cursor = None
    if next_offset < len(eligible):
        next_cursor = _encode_cursor(
            {
                "index_id": index.index_id,
                "query_id": query_id,
                "offset": next_offset,
                "last_evidence_id": eligible[next_offset - 1][1],
            }
        )
    return EvidenceSearchPage(
        query_id=query_id,
        normalized_query_hash=query_hash,
        filters_hash=filters_hash,
        results=references,
        total_result_count=len(eligible),
        next_cursor=next_cursor,
    )


def _encode_cursor(payload: dict[str, object]) -> str:
    body = {**payload, "cursor_hash": content_sha256(payload)}
    return base64.urlsafe_b64encode(canonical_json(body).encode("utf-8")).decode("ascii")


def _cursor_offset(
    cursor: str | None,
    *,
    index_id: str,
    query_id: str,
    ordered_ids: tuple[str, ...],
) -> int:
    if cursor is None:
        return 0
    try:
        raw = base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True).decode(
            "utf-8"
        )
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("evidence cursor is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "index_id",
        "query_id",
        "offset",
        "last_evidence_id",
        "cursor_hash",
    }:
        raise ValueError("evidence cursor has an invalid schema")
    semantic = {key: value for key, value in payload.items() if key != "cursor_hash"}
    if payload["cursor_hash"] != content_sha256(semantic):
        raise ValueError("evidence cursor hash does not match")
    if payload["index_id"] != index_id or payload["query_id"] != query_id:
        raise ValueError("evidence cursor is stale for this index or query")
    offset = payload["offset"]
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 1
        or offset > len(ordered_ids)
    ):
        raise ValueError("evidence cursor offset is invalid")
    if ordered_ids[offset - 1] != payload["last_evidence_id"]:
        raise ValueError("evidence cursor does not match deterministic ranking")
    return offset
