from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nlp_trader.schemas import TextItem
from nlp_trader.utils.device import get_torch_device

type StanceLabel = Literal["positive", "negative", "neutral", "abstain"]
type EventType = Literal[
    "bankruptcy",
    "merger_acquisition",
    "guidance",
    "earnings",
    "dividend",
    "litigation",
    "regulatory",
    "capital_raise",
]

_SPAN_ID = re.compile(r"^S[1-9][0-9]*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_TOKEN = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?(?![\w.])")

VERIFIER_VERSION = "semantic-evidence-verifier-v1"

PROMPT_TEXT = """You are a source-grounded financial semantic-signal engine.

For every candidate asset, infer only what the supplied evidence spans explicitly support.
Do not use outside knowledge, prices, returns, dates, later documents, or investment advice.
Treat source quality as a noisy host-supplied feature, not proof that a claim is true.
Return exactly one JSON object and no Markdown or explanatory text.

The object must contain an `annotations` array with exactly one entry per candidate asset.
Each entry must contain exactly these fields:
- asset_id
- stance_label: positive, negative, neutral, or abstain
- semantic_signal: integer -2, -1, 0, 1, or 2; sign must match stance_label
- raw_confidence: number from 0 to 1; this is an uncalibrated feature, not a probability
- uncertainty: number from 0 to 1
- horizon_days: exactly the supplied target_horizon_days
- primary_event_type: bankruptcy, merger_acquisition, guidance, earnings, dividend,
  litigation, regulatory, capital_raise, or null
- event_confidence: number from 0 to 1
- supporting_evidence_span_ids: unique supplied span IDs
- counterevidence_span_ids: unique supplied span IDs, disjoint from supporting evidence
- mechanism: concise source-grounded causal mechanism, or null for abstain
- invalidation_conditions: one or more concise source-grounded conditions, empty for abstain
- abstain_reason: string or null

For abstain, use semantic_signal 0, raw_confidence 0, uncertainty 1, null event,
event_confidence 0, no evidence IDs, null mechanism, no invalidation conditions, and a concise
nonempty abstain_reason. For every other stance, cite at least one supporting evidence span,
provide a mechanism and invalidation condition, and set abstain_reason to null. Never quote or
invent a span ID. Do not invent numbers: every number in a mechanism or invalidation condition
must occur in a cited evidence span. Use null event with event_confidence 0 when no listed event
is explicitly supported. Do not generate orders, position sizes, return forecasts, or advice.
"""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        strict=True,
    )


class EntityAnnotation(_StrictModel):
    asset_id: str = Field(min_length=1)
    stance_label: StanceLabel
    semantic_signal: int = Field(ge=-2, le=2)
    raw_confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    horizon_days: int = Field(ge=1, le=252)
    primary_event_type: EventType | None
    event_confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence_span_ids: tuple[str, ...]
    counterevidence_span_ids: tuple[str, ...]
    mechanism: str | None
    invalidation_conditions: tuple[str, ...]
    abstain_reason: str | None

    @model_validator(mode="after")
    def validate_semantics(self) -> EntityAnnotation:
        all_evidence_ids = self.supporting_evidence_span_ids + self.counterevidence_span_ids
        if len(set(self.supporting_evidence_span_ids)) != len(self.supporting_evidence_span_ids):
            raise ValueError("supporting_evidence_span_ids must be unique")
        if len(set(self.counterevidence_span_ids)) != len(self.counterevidence_span_ids):
            raise ValueError("counterevidence_span_ids must be unique")
        overlap = set(self.supporting_evidence_span_ids) & set(self.counterevidence_span_ids)
        if overlap:
            raise ValueError("supporting and counterevidence span IDs must be disjoint")
        invalid_ids = [value for value in all_evidence_ids if not _SPAN_ID.fullmatch(value)]
        if invalid_ids:
            raise ValueError("evidence span IDs must use the supplied S<number> format")
        if len(set(self.invalidation_conditions)) != len(self.invalidation_conditions):
            raise ValueError("invalidation_conditions must be unique")
        if any(not value.strip() for value in self.invalidation_conditions):
            raise ValueError("invalidation_conditions must not contain empty values")
        if self.stance_label == "abstain":
            if self.semantic_signal != 0:
                raise ValueError("abstain semantic_signal must be 0")
            if self.raw_confidence != 0.0:
                raise ValueError("abstain raw_confidence must be 0")
            if self.uncertainty != 1.0:
                raise ValueError("abstain uncertainty must be 1")
            if self.primary_event_type is not None or self.event_confidence != 0.0:
                raise ValueError("abstain annotations cannot contain an event")
            if all_evidence_ids:
                raise ValueError("abstain annotations cannot cite evidence spans")
            if self.mechanism is not None:
                raise ValueError("abstain annotations must use a null mechanism")
            if self.invalidation_conditions:
                raise ValueError("abstain annotations cannot contain invalidation conditions")
            if self.abstain_reason is None or not self.abstain_reason.strip():
                raise ValueError("abstain annotations require a nonempty abstain_reason")
            return self
        expected_signals = {
            "positive": {1, 2},
            "negative": {-2, -1},
            "neutral": {0},
        }
        if self.semantic_signal not in expected_signals[self.stance_label]:
            raise ValueError("semantic_signal sign must match stance_label")
        if not self.supporting_evidence_span_ids:
            raise ValueError("non-abstained annotations require supporting evidence")
        if self.mechanism is None or not self.mechanism.strip():
            raise ValueError("non-abstained annotations require a nonempty mechanism")
        if not self.invalidation_conditions:
            raise ValueError("non-abstained annotations require invalidation conditions")
        if self.abstain_reason is not None:
            raise ValueError("non-abstained annotations must use a null abstain_reason")
        if self.primary_event_type is None and self.event_confidence != 0.0:
            raise ValueError("a null primary_event_type requires event_confidence 0")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class _GeneratedPayload(_StrictModel):
    annotations: tuple[EntityAnnotation, ...]


class AnnotationResponse(_StrictModel):
    item_id: str = Field(min_length=1)
    annotations: tuple[EntityAnnotation, ...]

    @model_validator(mode="after")
    def validate_assets_are_unique(self) -> AnnotationResponse:
        asset_ids = [annotation.asset_id for annotation in self.annotations]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("annotation asset_id values must be unique")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class VerificationCheck(_StrictModel):
    name: str = Field(min_length=1)
    passed: bool
    detail: str = Field(min_length=1)


class AnnotationVerification(_StrictModel):
    verifier_version: str = Field(min_length=1)
    valid: bool
    checks: tuple[VerificationCheck, ...]

    @model_validator(mode="after")
    def validate_summary(self) -> AnnotationVerification:
        names = [check.name for check in self.checks]
        if not self.checks or len(names) != len(set(names)):
            raise ValueError("verification checks must be nonempty and uniquely named")
        if self.valid != all(check.passed for check in self.checks):
            raise ValueError("verification valid flag must equal the conjunction of checks")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class AssetCandidate:
    asset_id: str
    symbol: str
    name: str

    def __post_init__(self) -> None:
        for field_name in ("asset_id", "symbol", "name"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"candidate {field_name} must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {"asset_id": self.asset_id, "symbol": self.symbol, "name": self.name}


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    span_id: str
    text: str

    def __post_init__(self) -> None:
        if not _SPAN_ID.fullmatch(self.span_id):
            raise ValueError("evidence span IDs must use S<number>")
        if not self.text.strip():
            raise ValueError("evidence span text must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {"span_id": self.span_id, "text": self.text}


@dataclass(frozen=True, slots=True)
class AnnotationRequest:
    item_id: str
    source_text_hash: str
    source_available_at: datetime
    decision_time: datetime
    target_horizon_days: int
    source_type: str
    source_quality: float
    candidates: tuple[AssetCandidate, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    prompt: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_text_hash):
            raise ValueError("source_text_hash must be a lowercase SHA-256 digest")
        if self.source_available_at.tzinfo is None or self.decision_time.tzinfo is None:
            raise ValueError("annotation source and decision timestamps must be timezone-aware")
        if self.source_available_at.astimezone(UTC) > self.decision_time.astimezone(UTC):
            raise ValueError("annotation source_available_at must not exceed decision_time")
        if not 1 <= self.target_horizon_days <= 252:
            raise ValueError("target_horizon_days must be between 1 and 252")
        if not self.source_type.strip():
            raise ValueError("source_type must not be empty")
        if not 0.0 <= self.source_quality <= 1.0:
            raise ValueError("source_quality must be between 0 and 1")
        if not self.candidates:
            raise ValueError("annotation requests require at least one asset candidate")
        asset_ids = [candidate.asset_id for candidate in self.candidates]
        if asset_ids != sorted(asset_ids) or len(asset_ids) != len(set(asset_ids)):
            raise ValueError("annotation candidates must be unique and sorted by asset_id")
        if not self.prompt.strip():
            raise ValueError("annotation prompt must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    prompt: str

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.prompt.strip():
            raise ValueError("generation request_id and prompt must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    request_id: str
    generated_text: str | None = None
    input_too_long: bool = False
    output_truncated: bool = False
    input_token_count: int | None = None
    output_token_count: int | None = None
    generation_latency_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("generation response request_id must not be empty")
        if self.generated_text is not None and not isinstance(self.generated_text, str):
            raise ValueError("generation response text must be a string when supplied")
        if type(self.input_too_long) is not bool or type(self.output_truncated) is not bool:
            raise ValueError("generation response state flags must be booleans")
        if self.input_too_long:
            if self.generated_text is not None or self.output_truncated:
                raise ValueError("input-too-long responses cannot contain generated output")
        elif self.generated_text is None or not self.generated_text.strip():
            raise ValueError("generation response text must not be empty")
        for field_name in ("input_token_count", "output_token_count"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.generation_latency_seconds is not None and (
            isinstance(self.generation_latency_seconds, bool)
            or not isinstance(self.generation_latency_seconds, (int, float))
            or not math.isfinite(self.generation_latency_seconds)
            or self.generation_latency_seconds < 0.0
        ):
            raise ValueError("generation_latency_seconds must be finite and non-negative")


RawGenerator = Callable[[list[GenerationRequest]], list[GenerationResponse]]


@dataclass(frozen=True, slots=True)
class LLMAnnotationConfig:
    model_path: Path
    model_id: str
    model_revision: str
    model_license_or_terms_ref: str
    prompt_version: str
    schema_version: str
    verifier_version: str
    cache_dir: Path
    attempt_dir: Path | None = None
    batch_size: int = 1
    max_input_tokens: int = 2048
    max_new_tokens: int = 384
    decoding: Literal["greedy"] = "greedy"
    seed: int = 7
    input_cost_per_million_tokens_usd: float | None = None
    output_cost_per_million_tokens_usd: float | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "model_id",
            "model_revision",
            "model_license_or_terms_ref",
            "prompt_version",
            "schema_version",
            "verifier_version",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if not self.model_path.is_dir():
            raise ValueError(f"local LLM model directory does not exist: {self.model_path}")
        if not any(path.is_file() for path in self.model_path.rglob("*")):
            raise ValueError(f"local LLM model directory contains no files: {self.model_path}")
        if self.batch_size < 1 or self.max_input_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("LLM batch and token limits must be positive")
        if self.seed < 0:
            raise ValueError("LLM seed must be non-negative")
        rates = (
            self.input_cost_per_million_tokens_usd,
            self.output_cost_per_million_tokens_usd,
        )
        if (rates[0] is None) != (rates[1] is None):
            raise ValueError("LLM input and output token cost rates must be configured together")
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            )
            for value in rates
        ):
            raise ValueError("LLM token cost rates must be finite and non-negative")
        if self.decoding != "greedy":
            raise ValueError("only deterministic greedy LLM decoding is supported")
        if not self.local_files_only:
            raise ValueError("LLM annotations require local_files_only=True")
        if self.trust_remote_code:
            raise ValueError("LLM annotations require trust_remote_code=False")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strict_json_loads(value: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        value,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite_constant,
    )


def _effective_generated_tokens(
    token_ids: list[int],
    *,
    eos_ids: set[int],
    pad_token_id: int | None,
) -> tuple[list[int], bool]:
    """Remove batch padding and report whether generation ended normally."""

    effective: list[int] = []
    for token_id in token_ids:
        if pad_token_id is not None and token_id == pad_token_id and token_id not in eos_ids:
            return effective, True
        effective.append(token_id)
        if token_id in eos_ids:
            return effective, True
    return effective, False


def _model_directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        file_digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def _source_hash(item: TextItem) -> str:
    if item.canonical_text_hash is not None:
        return item.canonical_text_hash
    return _sha256_text(f"{item.title or ''}\0{item.body or ''}")


def _evidence_spans(item: TextItem) -> tuple[EvidenceSpan, ...]:
    values: list[str] = []
    if item.title and item.title.strip():
        values.append(item.title.strip())
    if item.body and item.body.strip():
        values.extend(
            part.strip() for part in _SENTENCE_BOUNDARY.split(item.body.strip()) if part.strip()
        )
    return tuple(
        EvidenceSpan(span_id=f"S{index}", text=value) for index, value in enumerate(values, start=1)
    )


def build_annotation_request(
    item: TextItem,
    candidates: Iterable[AssetCandidate],
    *,
    decision_time: datetime | None = None,
    target_horizon_days: int = 1,
    source_quality: float = 0.5,
) -> AnnotationRequest:
    """Build one point-in-time request without prices, labels, or other documents."""

    indexed: dict[str, AssetCandidate] = {}
    for candidate in candidates:
        previous = indexed.get(candidate.asset_id)
        if previous is not None and previous != candidate:
            raise ValueError(f"conflicting duplicate asset candidate: {candidate.asset_id}")
        indexed[candidate.asset_id] = candidate
    ordered = tuple(indexed[asset_id] for asset_id in sorted(indexed))
    if not ordered:
        raise ValueError("annotation requests require at least one asset candidate")
    spans = _evidence_spans(item)
    request_payload = {
        "target_horizon_days": target_horizon_days,
        "source": {
            "source_type": item.source_type,
            "quality_score": source_quality,
        },
        "candidates": [candidate.to_dict() for candidate in ordered],
        "evidence_spans": [span.to_dict() for span in spans],
    }
    prompt = (
        PROMPT_TEXT
        + "\nREQUEST_JSON:\n"
        + json.dumps(request_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\nRESPONSE_JSON:\n"
    )
    return AnnotationRequest(
        item_id=item.item_id,
        source_text_hash=_source_hash(item),
        source_available_at=item.available_at,
        decision_time=decision_time or item.available_at,
        target_horizon_days=target_horizon_days,
        source_type=item.source_type,
        source_quality=source_quality,
        candidates=ordered,
        evidence_spans=spans,
        prompt=prompt,
    )


def _verification_check(name: str, passed: bool, detail: str) -> VerificationCheck:
    return VerificationCheck(name=name, passed=passed, detail=detail)


def verify_annotation_response(
    request: AnnotationRequest,
    response: AnnotationResponse,
    *,
    verifier_version: str = VERIFIER_VERSION,
) -> AnnotationVerification:
    """Run deterministic identity, temporal, evidence, horizon, and numeric-claim checks."""

    expected_assets = {candidate.asset_id for candidate in request.candidates}
    observed_assets = {annotation.asset_id for annotation in response.annotations}
    valid_spans = {span.span_id: span.text for span in request.evidence_spans}
    cited_ids = {
        span_id
        for annotation in response.annotations
        for span_id in (
            annotation.supporting_evidence_span_ids + annotation.counterevidence_span_ids
        )
    }
    unknown_spans = cited_ids - set(valid_spans)
    horizon_mismatches = sorted(
        {
            annotation.horizon_days
            for annotation in response.annotations
            if annotation.horizon_days != request.target_horizon_days
        }
    )
    ungrounded_numbers: list[str] = []
    for annotation in response.annotations:
        cited_text = " ".join(
            valid_spans.get(span_id, "")
            for span_id in (
                annotation.supporting_evidence_span_ids + annotation.counterevidence_span_ids
            )
        )
        cited_numbers = set(_NUMBER_TOKEN.findall(cited_text))
        claims = (annotation.mechanism or "", *annotation.invalidation_conditions)
        for claim in claims:
            for number in _NUMBER_TOKEN.findall(claim):
                if number not in cited_numbers:
                    ungrounded_numbers.append(f"{annotation.asset_id}:{number}")

    checks = (
        _verification_check(
            "item_identity",
            response.item_id == request.item_id,
            "response item_id must match the request item_id",
        ),
        _verification_check(
            "candidate_coverage",
            observed_assets == expected_assets
            and len(response.annotations) == len(expected_assets),
            "response assets must exactly cover the unique request candidates",
        ),
        _verification_check(
            "temporal_validity",
            request.source_available_at.astimezone(UTC) <= request.decision_time.astimezone(UTC),
            "source_available_at must be no later than the feature decision_time",
        ),
        _verification_check(
            "horizon_alignment",
            not horizon_mismatches,
            (
                "every annotation horizon must match target_horizon_days"
                if not horizon_mismatches
                else f"mismatched horizons: {horizon_mismatches}"
            ),
        ),
        _verification_check(
            "evidence_reference_validity",
            not unknown_spans,
            (
                "all cited evidence spans must exist in the request"
                if not unknown_spans
                else "unknown evidence spans: " + ", ".join(sorted(unknown_spans))
            ),
        ),
        _verification_check(
            "numeric_claim_grounding",
            not ungrounded_numbers,
            (
                "numbers in mechanisms and invalidation conditions must occur in cited evidence"
                if not ungrounded_numbers
                else "ungrounded numeric claims: " + ", ".join(sorted(ungrounded_numbers))
            ),
        ),
    )
    return AnnotationVerification(
        verifier_version=verifier_version,
        valid=all(check.passed for check in checks),
        checks=checks,
    )


class CachedLocalLLMAnnotator:
    """Strict, batched, content-addressed local causal-LM annotation engine."""

    def __init__(
        self,
        config: LLMAnnotationConfig,
        *,
        generator: RawGenerator | None = None,
    ) -> None:
        self.config = config
        self._injected_generator = generator
        self._loaded_generator: RawGenerator | None = None
        self._model_directory_sha256 = _model_directory_hash(config.model_path)
        self._records: dict[str, dict[str, Any]] = {}
        self._device_used = "injected_generator" if generator is not None else "not_loaded"
        self.cache_hit_count = 0
        self.generation_request_count = 0
        self.deduplicated_request_count = 0
        self.generated_input_token_count: int | None = 0
        self.generated_output_token_count: int | None = 0
        self.generation_latency_seconds = 0.0
        self.estimated_inference_cost_usd: float | None = None
        self._inference_sources: dict[str, Literal["generated", "cache", "deduplicated"]] = {}
        self.attempt_paths: list[Path] = []

    @property
    def prompt_text(self) -> str:
        return PROMPT_TEXT

    @property
    def schema_payload(self) -> dict[str, Any]:
        return _GeneratedPayload.model_json_schema()

    @property
    def provenance_payload(self) -> dict[str, Any]:
        return {
            "backend": "transformers_causal_lm",
            "model_id": self.config.model_id,
            "model_revision": self.config.model_revision,
            "model_license_or_terms_ref": self.config.model_license_or_terms_ref,
            "model_directory_sha256": self._model_directory_sha256,
            "prompt_version": self.config.prompt_version,
            "prompt_sha256": _sha256_text(PROMPT_TEXT),
            "schema_version": self.config.schema_version,
            "schema_sha256": _sha256_text(
                json.dumps(self.schema_payload, sort_keys=True, separators=(",", ":"))
            ),
            "batch_size": self.config.batch_size,
            "max_input_tokens": self.config.max_input_tokens,
            "max_new_tokens": self.config.max_new_tokens,
            "decoding": self.config.decoding,
            "seed": self.config.seed,
            "verifier_version": self.config.verifier_version,
            "input_cost_per_million_tokens_usd": (self.config.input_cost_per_million_tokens_usd),
            "output_cost_per_million_tokens_usd": (self.config.output_cost_per_million_tokens_usd),
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
            "device": self._device_used,
            "device_policy": "MPS when available, otherwise CPU",
        }

    def _identity_payload(self, request: AnnotationRequest) -> dict[str, Any]:
        return {
            "source_text_hash": request.source_text_hash,
            "request_prompt_sha256": _sha256_text(request.prompt),
            "candidates": [candidate.to_dict() for candidate in request.candidates],
            "model_directory_sha256": self._model_directory_sha256,
            "model_id": self.config.model_id,
            "model_revision": self.config.model_revision,
            "prompt_version": self.config.prompt_version,
            "schema_version": self.config.schema_version,
            "verifier_version": self.config.verifier_version,
            "schema_sha256": _sha256_text(
                json.dumps(self.schema_payload, sort_keys=True, separators=(",", ":"))
            ),
            "backend": "transformers_causal_lm",
            "decoding": self.config.decoding,
            "seed": self.config.seed,
            "batch_size": self.config.batch_size,
            "max_input_tokens": self.config.max_input_tokens,
            "max_new_tokens": self.config.max_new_tokens,
        }

    def _estimated_cost(self, response: GenerationResponse) -> float | None:
        input_rate = self.config.input_cost_per_million_tokens_usd
        output_rate = self.config.output_cost_per_million_tokens_usd
        if input_rate is None or output_rate is None:
            return None
        if response.input_token_count is None or response.output_token_count is None:
            return None
        return (
            response.input_token_count * input_rate + response.output_token_count * output_rate
        ) / 1_000_000.0

    def inference_source_for(
        self, request: AnnotationRequest
    ) -> Literal["generated", "cache", "deduplicated"]:
        try:
            return self._inference_sources[request.item_id]
        except KeyError as exc:
            raise ValueError("annotation request has not been processed in this run") from exc

    def cache_key_for(self, request: AnnotationRequest) -> str:
        encoded = json.dumps(
            self._identity_payload(request),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir / f"{key}.json"

    def _write_generation_attempt(
        self,
        request: AnnotationRequest,
        key: str,
        generated: GenerationResponse,
    ) -> None:
        if self.config.attempt_dir is None:
            return
        record = {
            "artifact_schema_version": "llm-generation-attempt-v1",
            "cache_key": key,
            "identity": self._identity_payload(request),
            "request": {
                "source_text_hash": request.source_text_hash,
                "prompt": request.prompt,
                "evidence_spans": [span.to_dict() for span in request.evidence_spans],
            },
            "generation": {
                "request_id": generated.request_id,
                "generated_text": generated.generated_text,
                "input_too_long": generated.input_too_long,
                "output_truncated": generated.output_truncated,
                "input_token_count": generated.input_token_count,
                "output_token_count": generated.output_token_count,
                "generation_latency_seconds": generated.generation_latency_seconds,
                "estimated_cost_usd": self._estimated_cost(generated),
            },
            "provenance": self.provenance_payload,
        }
        self.config.attempt_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.attempt_dir / f"{key}.json"
        if path.exists():
            if _strict_json_loads(path.read_text(encoding="utf-8")) != record:
                raise ValueError(f"LLM generation attempt identity collision: {path}")
        else:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        self.attempt_paths.append(path)

    def _validate_response(
        self,
        request: AnnotationRequest,
        response: AnnotationResponse,
    ) -> AnnotationResponse:
        verification = verify_annotation_response(
            request,
            response,
            verifier_version=self.config.verifier_version,
        )
        if not verification.valid:
            failures = [check.name for check in verification.checks if not check.passed]
            raise ValueError("annotation verification failed: " + ", ".join(failures))
        return response

    def verification_for(
        self,
        request: AnnotationRequest,
        response: AnnotationResponse,
    ) -> AnnotationVerification:
        verification = verify_annotation_response(
            request,
            response,
            verifier_version=self.config.verifier_version,
        )
        if not verification.valid:
            raise ValueError("cannot return a failed annotation verification as valid output")
        return verification

    def _read_cache(
        self,
        request: AnnotationRequest,
        key: str,
    ) -> AnnotationResponse | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            record = _strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid LLM annotation cache record: {path}") from exc
        if not isinstance(record, dict) or record.get("cache_key") != key:
            raise ValueError(f"LLM annotation cache identity mismatch: {path}")
        expected_record_fields = {
            "cache_schema_version",
            "cache_key",
            "identity",
            "request",
            "generation",
            "annotation_payload",
            "verification",
            "provenance",
        }
        if set(record) != expected_record_fields:
            raise ValueError(f"LLM annotation cache fields mismatch: {path}")
        if record.get("cache_schema_version") != "llm-semantic-signal-cache-v2":
            raise ValueError(f"LLM annotation cache schema mismatch: {path}")
        if record.get("identity") != self._identity_payload(request):
            raise ValueError(f"LLM annotation cache request mismatch: {path}")
        expected_request = {
            "source_text_hash": request.source_text_hash,
            "prompt": request.prompt,
            "evidence_spans": [span.to_dict() for span in request.evidence_spans],
        }
        if record.get("request") != expected_request:
            raise ValueError(f"LLM annotation cache source mismatch: {path}")
        generation_record = record.get("generation")
        if not isinstance(generation_record, dict):
            raise ValueError(f"invalid LLM annotation cache generation: {path}")
        expected_generation_fields = {
            "request_id",
            "generated_text",
            "input_too_long",
            "output_truncated",
            "input_token_count",
            "output_token_count",
            "generation_latency_seconds",
            "estimated_cost_usd",
        }
        if set(generation_record) != expected_generation_fields:
            raise ValueError(f"LLM annotation cache generation fields mismatch: {path}")
        try:
            generated = GenerationResponse(
                request_id=generation_record["request_id"],
                generated_text=generation_record.get("generated_text"),
                input_too_long=generation_record.get("input_too_long", False),
                output_truncated=generation_record.get("output_truncated", False),
                input_token_count=generation_record.get("input_token_count"),
                output_token_count=generation_record.get("output_token_count"),
                generation_latency_seconds=generation_record.get("generation_latency_seconds"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid LLM annotation cache generation: {path}") from exc
        if generated.request_id != key:
            raise ValueError(f"LLM annotation cache generation identity mismatch: {path}")
        provenance_record = record.get("provenance")
        if not isinstance(provenance_record, dict):
            raise ValueError(f"invalid LLM annotation cache provenance: {path}")
        current_provenance = self.provenance_payload
        stable_provenance_fields = (
            "backend",
            "model_id",
            "model_revision",
            "model_directory_sha256",
            "prompt_version",
            "prompt_sha256",
            "schema_version",
            "schema_sha256",
            "batch_size",
            "max_input_tokens",
            "max_new_tokens",
            "decoding",
            "seed",
            "verifier_version",
            "local_files_only",
            "trust_remote_code",
        )
        if any(
            provenance_record.get(field_name) != current_provenance[field_name]
            for field_name in stable_provenance_fields
        ):
            raise ValueError(f"LLM annotation cache provenance mismatch: {path}")
        input_rate = provenance_record.get("input_cost_per_million_tokens_usd")
        output_rate = provenance_record.get("output_cost_per_million_tokens_usd")
        rates = (input_rate, output_rate)
        if (input_rate is None) != (output_rate is None) or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            )
            for value in rates
        ):
            raise ValueError(f"invalid LLM annotation cache cost provenance: {path}")
        expected_cost = (
            None
            if input_rate is None
            or output_rate is None
            or generated.input_token_count is None
            or generated.output_token_count is None
            else (
                generated.input_token_count * input_rate
                + generated.output_token_count * output_rate
            )
            / 1_000_000.0
        )
        stored_cost = generation_record.get("estimated_cost_usd")
        if expected_cost is None:
            cost_matches = stored_cost is None
        else:
            cost_matches = (
                not isinstance(stored_cost, bool)
                and isinstance(stored_cost, (int, float))
                and math.isfinite(stored_cost)
                and math.isclose(stored_cost, expected_cost, rel_tol=0.0, abs_tol=1e-15)
            )
        if not cost_matches:
            raise ValueError(f"LLM annotation cache cost mismatch: {path}")
        try:
            payload = _GeneratedPayload.model_validate_json(
                json.dumps(record["annotation_payload"], ensure_ascii=False)
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid LLM annotation cache payload: {path}") from exc
        stored = self._validate_response(
            request,
            AnnotationResponse(item_id=request.item_id, annotations=payload.annotations),
        )
        try:
            reparsed = self._parse_generation(request, generated)
        except ValueError as exc:
            raise ValueError(f"LLM annotation cache raw generation is invalid: {path}") from exc
        if reparsed != stored:
            raise ValueError(
                f"LLM annotation cache payload does not match its raw generation: {path}"
            )
        validated = reparsed
        expected_verification = self.verification_for(request, validated).to_dict()
        if record.get("verification") != expected_verification:
            raise ValueError(f"LLM annotation cache verifier mismatch: {path}")
        self._records[key] = record
        return validated

    def _parse_generation(
        self,
        request: AnnotationRequest,
        generated: GenerationResponse,
    ) -> AnnotationResponse:
        if generated.input_too_long:
            return AnnotationResponse(
                item_id=request.item_id,
                annotations=tuple(
                    EntityAnnotation(
                        asset_id=candidate.asset_id,
                        stance_label="abstain",
                        semantic_signal=0,
                        raw_confidence=0.0,
                        uncertainty=1.0,
                        horizon_days=request.target_horizon_days,
                        primary_event_type=None,
                        event_confidence=0.0,
                        supporting_evidence_span_ids=(),
                        counterevidence_span_ids=(),
                        mechanism=None,
                        invalidation_conditions=(),
                        abstain_reason="input_too_long",
                    )
                    for candidate in request.candidates
                ),
            )
        if generated.output_truncated:
            raise ValueError("LLM generation reached max_new_tokens before a complete response")
        try:
            raw = _strict_json_loads(generated.generated_text or "")
            payload = _GeneratedPayload.model_validate_json(json.dumps(raw, ensure_ascii=False))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"LLM response for item {request.item_id} is not valid strict annotation JSON"
            ) from exc
        return self._validate_response(
            request,
            AnnotationResponse(item_id=request.item_id, annotations=payload.annotations),
        )

    def _cache_record(
        self,
        request: AnnotationRequest,
        key: str,
        generated: GenerationResponse,
        response: AnnotationResponse,
    ) -> dict[str, Any]:
        verification = self.verification_for(request, response)
        return {
            "cache_schema_version": "llm-semantic-signal-cache-v2",
            "cache_key": key,
            "identity": self._identity_payload(request),
            "request": {
                "source_text_hash": request.source_text_hash,
                "prompt": request.prompt,
                "evidence_spans": [span.to_dict() for span in request.evidence_spans],
            },
            "generation": {
                "request_id": generated.request_id,
                "generated_text": generated.generated_text,
                "input_too_long": generated.input_too_long,
                "output_truncated": generated.output_truncated,
                "input_token_count": generated.input_token_count,
                "output_token_count": generated.output_token_count,
                "generation_latency_seconds": generated.generation_latency_seconds,
                "estimated_cost_usd": self._estimated_cost(generated),
            },
            "annotation_payload": {
                "annotations": [annotation.to_dict() for annotation in response.annotations]
            },
            "verification": verification.to_dict(),
            "provenance": self.provenance_payload,
        }

    def _write_cache(self, key: str, record: dict[str, Any]) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        if path.exists():
            existing = _strict_json_loads(path.read_text(encoding="utf-8"))
            if existing != record:
                raise ValueError(f"LLM annotation cache key collision: {path}")
            return
        with path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    def cache_record_for(self, request: AnnotationRequest) -> dict[str, Any]:
        key = self.cache_key_for(request)
        record = self._records.get(key)
        if record is None:
            cached = self._read_cache(request, key)
            if cached is None:
                raise ValueError("annotation request has not been generated or cached")
            record = self._records[key]
        copied = cast(
            dict[str, Any],
            json.loads(json.dumps(record, ensure_ascii=False)),
        )
        copied["run_request"] = {"item_id": request.item_id}
        return copied

    def _default_generator(self) -> RawGenerator:
        try:
            transformers = import_module("transformers")
            torch = import_module("torch")
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Generative LLM annotations are optional; run `uv sync --extra nlp` first"
            ) from exc

        if _model_directory_hash(self.config.model_path) != self._model_directory_sha256:
            raise ValueError("local LLM model changed before model loading")
        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            str(self.config.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        model: Any = transformers.AutoModelForCausalLM.from_pretrained(
            str(self.config.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        if _model_directory_hash(self.config.model_path) != self._model_directory_sha256:
            raise ValueError("local LLM model changed while model files were loading")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("local causal-LM tokenizer requires a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        device = get_torch_device()
        self._device_used = str(device)
        model.to(device)
        model.eval()
        torch.manual_seed(self.config.seed)
        context_limits: list[int] = []
        for raw_limit in (
            getattr(tokenizer, "model_max_length", None),
            getattr(model.config, "max_position_embeddings", None),
        ):
            if isinstance(raw_limit, int) and 0 < raw_limit < 1_000_000:
                context_limits.append(raw_limit)
        model_context_tokens = min(context_limits) if context_limits else None

        def generate(batch: list[GenerationRequest]) -> list[GenerationResponse]:
            results: dict[str, GenerationResponse] = {}
            eligible: list[GenerationRequest] = []
            input_token_counts: dict[str, int] = {}
            for request in batch:
                tokenized = tokenizer(
                    request.prompt,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False,
                )
                input_tokens = len(tokenized["input_ids"])
                input_token_counts[request.request_id] = input_tokens
                exceeds_model_context = (
                    model_context_tokens is not None
                    and input_tokens + self.config.max_new_tokens > model_context_tokens
                )
                if input_tokens > self.config.max_input_tokens or exceeds_model_context:
                    results[request.request_id] = GenerationResponse(
                        request_id=request.request_id,
                        input_too_long=True,
                        input_token_count=input_tokens,
                        output_token_count=0,
                    )
                else:
                    eligible.append(request)
            if eligible:
                encoded = tokenizer(
                    [request.prompt for request in eligible],
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                ).to(device)
                input_width = int(encoded["input_ids"].shape[1])
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **encoded,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=self.config.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                eos_ids = tokenizer.eos_token_id
                eos_set = (
                    set(eos_ids)
                    if isinstance(eos_ids, list)
                    else {eos_ids}
                    if eos_ids is not None
                    else set()
                )
                for request, sequence in zip(eligible, generated_ids, strict=True):
                    continuation = sequence[input_width:]
                    padded_token_ids = continuation.detach().cpu().tolist()
                    token_ids, terminated = _effective_generated_tokens(
                        padded_token_ids,
                        eos_ids=eos_set,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    output_truncated = (
                        not terminated and len(token_ids) >= self.config.max_new_tokens
                    )
                    results[request.request_id] = GenerationResponse(
                        request_id=request.request_id,
                        generated_text=tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                        ),
                        output_truncated=output_truncated,
                        input_token_count=input_token_counts[request.request_id],
                        output_token_count=len(token_ids),
                    )
            return [results[request.request_id] for request in batch]

        return generate

    def _generator(self) -> RawGenerator:
        if self._injected_generator is not None:
            return self._injected_generator
        if self._loaded_generator is None:
            self._loaded_generator = self._default_generator()
        return self._loaded_generator

    def annotate(self, requests: Iterable[AnnotationRequest]) -> list[AnnotationResponse]:
        values = list(requests)
        self.cache_hit_count = 0
        self.generation_request_count = 0
        self.deduplicated_request_count = 0
        self.generated_input_token_count = 0
        self.generated_output_token_count = 0
        self.generation_latency_seconds = 0.0
        self.estimated_inference_cost_usd = (
            0.0
            if self.config.input_cost_per_million_tokens_usd is not None
            and self.config.output_cost_per_million_tokens_usd is not None
            else None
        )
        self._inference_sources = {}
        self.attempt_paths = []
        item_ids = [request.item_id for request in values]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("annotation request item_id values must be unique")

        results: list[AnnotationResponse | None] = [None] * len(values)
        missing_by_key: dict[str, tuple[AnnotationRequest, list[int]]] = {}
        for index, request in enumerate(values):
            key = self.cache_key_for(request)
            cached = self._read_cache(request, key)
            if cached is None:
                existing = missing_by_key.get(key)
                if existing is None:
                    missing_by_key[key] = (request, [index])
                else:
                    representative, indices = existing
                    if self._identity_payload(representative) != self._identity_payload(request):
                        raise ValueError("LLM annotation cache-key collision within request batch")
                    indices.append(index)
            else:
                results[index] = cached
                self.cache_hit_count += 1
                self._inference_sources[request.item_id] = "cache"

        missing = [(request, key, indices) for key, (request, indices) in missing_by_key.items()]
        self.generation_request_count = len(missing)
        self.deduplicated_request_count = sum(max(0, len(indices) - 1) for _, _, indices in missing)
        pending_records: list[tuple[str, AnnotationResponse, dict[str, Any], list[int]]] = []
        generator = self._generator() if missing else None
        for start in range(0, len(missing), self.config.batch_size):
            chunk = missing[start : start + self.config.batch_size]
            generation_requests = [
                GenerationRequest(request_id=key, prompt=request.prompt)
                for request, key, _ in chunk
            ]
            assert generator is not None
            started_at = time.perf_counter()
            generated_values = generator(generation_requests)
            elapsed = time.perf_counter() - started_at
            generated_values = [
                generated
                if generated.generation_latency_seconds is not None
                else replace(
                    generated,
                    generation_latency_seconds=elapsed,
                )
                for generated in generated_values
            ]
            generated_by_id: dict[str, GenerationResponse] = {}
            expected_ids = {request.request_id for request in generation_requests}
            for generated in generated_values:
                if generated.request_id in generated_by_id:
                    raise ValueError(
                        f"generator returned duplicate request_id: {generated.request_id}"
                    )
                if generated.request_id not in expected_ids:
                    raise ValueError(
                        f"generator returned unknown request_id: {generated.request_id}"
                    )
                generated_by_id[generated.request_id] = generated
            for request, key, _indices in chunk:
                generated_attempt = generated_by_id.get(key)
                if generated_attempt is not None:
                    self._write_generation_attempt(request, key, generated_attempt)
            if set(generated_by_id) != expected_ids:
                raise ValueError("generator did not return every requested annotation")
            self.generation_latency_seconds += max(
                elapsed,
                max(
                    (generated.generation_latency_seconds or 0.0 for generated in generated_values),
                    default=0.0,
                ),
            )
            for request, key, indices in chunk:
                generated = generated_by_id[key]
                response = self._parse_generation(request, generated)
                record = self._cache_record(request, key, generated, response)
                pending_records.append((key, response, record, indices))
                if self.generated_input_token_count is not None:
                    if generated.input_token_count is None:
                        self.generated_input_token_count = None
                    else:
                        self.generated_input_token_count += generated.input_token_count
                if self.generated_output_token_count is not None:
                    if generated.output_token_count is None:
                        self.generated_output_token_count = None
                    else:
                        self.generated_output_token_count += generated.output_token_count
                estimated_cost = self._estimated_cost(generated)
                if self.estimated_inference_cost_usd is not None:
                    if estimated_cost is None:
                        self.estimated_inference_cost_usd = None
                    else:
                        self.estimated_inference_cost_usd += estimated_cost

        # Validate every generated batch before making any new cache write.
        for key, response, record, indices in pending_records:
            self._write_cache(key, record)
            self._records[key] = record
            for offset, index in enumerate(indices):
                item_response = AnnotationResponse(
                    item_id=values[index].item_id,
                    annotations=response.annotations,
                )
                self._validate_response(values[index], item_response)
                results[index] = item_response
                self._inference_sources[values[index].item_id] = (
                    "generated" if offset == 0 else "deduplicated"
                )
        if any(result is None for result in results):
            raise RuntimeError("LLM annotation inference did not produce every response")
        return [result for result in results if result is not None]
