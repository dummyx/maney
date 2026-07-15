from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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

PROMPT_TEXT = """You are a source-grounded financial text annotation engine.

For every candidate asset, infer only what the supplied evidence spans explicitly support.
Do not use outside knowledge, prices, returns, dates, later documents, or investment advice.
Return exactly one JSON object and no Markdown or explanatory text.

The object must contain an `annotations` array with exactly one entry per candidate asset.
Each entry must contain exactly these fields:
- asset_id
- stance_label: positive, negative, neutral, or abstain
- stance_confidence: number from 0 to 1
- uncertainty: number from 0 to 1
- primary_event_type: bankruptcy, merger_acquisition, guidance, earnings, dividend,
  litigation, regulatory, capital_raise, or null
- event_confidence: number from 0 to 1
- evidence_span_ids: unique supplied span IDs
- abstain_reason: string or null

For abstain, use stance_confidence 0, uncertainty 1, null event, event_confidence 0,
no evidence IDs, and a concise nonempty abstain_reason. For every other stance, cite at
least one supplied evidence span and set abstain_reason to null. Never quote or invent a
span ID. Use null event with event_confidence 0 when no listed event is explicitly supported.
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
    stance_confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    primary_event_type: EventType | None
    event_confidence: float = Field(ge=0.0, le=1.0)
    evidence_span_ids: tuple[str, ...]
    abstain_reason: str | None

    @model_validator(mode="after")
    def validate_semantics(self) -> EntityAnnotation:
        if len(set(self.evidence_span_ids)) != len(self.evidence_span_ids):
            raise ValueError("evidence_span_ids must be unique")
        invalid_ids = [value for value in self.evidence_span_ids if not _SPAN_ID.fullmatch(value)]
        if invalid_ids:
            raise ValueError("evidence_span_ids must use the supplied S<number> format")
        if self.stance_label == "abstain":
            if self.stance_confidence != 0.0:
                raise ValueError("abstain stance_confidence must be 0")
            if self.uncertainty != 1.0:
                raise ValueError("abstain uncertainty must be 1")
            if self.primary_event_type is not None or self.event_confidence != 0.0:
                raise ValueError("abstain annotations cannot contain an event")
            if self.evidence_span_ids:
                raise ValueError("abstain annotations cannot cite evidence spans")
            if self.abstain_reason is None or not self.abstain_reason.strip():
                raise ValueError("abstain annotations require a nonempty abstain_reason")
            return self
        if not self.evidence_span_ids:
            raise ValueError("non-abstained annotations require at least one evidence span")
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
    candidates: tuple[AssetCandidate, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    prompt: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_text_hash):
            raise ValueError("source_text_hash must be a lowercase SHA-256 digest")
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

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("generation response request_id must not be empty")
        if self.input_too_long:
            if self.generated_text is not None or self.output_truncated:
                raise ValueError("input-too-long responses cannot contain generated output")
        elif self.generated_text is None or not self.generated_text.strip():
            raise ValueError("generation response text must not be empty")


RawGenerator = Callable[[list[GenerationRequest]], list[GenerationResponse]]


@dataclass(frozen=True, slots=True)
class LLMAnnotationConfig:
    model_path: Path
    model_id: str
    model_revision: str
    model_license_or_terms_ref: str
    prompt_version: str
    schema_version: str
    cache_dir: Path
    batch_size: int = 1
    max_input_tokens: int = 2048
    max_new_tokens: int = 384
    decoding: Literal["greedy"] = "greedy"
    seed: int = 7
    local_files_only: bool = True
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "model_id",
            "model_revision",
            "model_license_or_terms_ref",
            "prompt_version",
            "schema_version",
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

    return json.loads(value, object_pairs_hook=reject_duplicate_keys)


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
) -> AnnotationRequest:
    """Build one evidence-only request without timestamps, prices, labels, or other documents."""

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
        candidates=ordered,
        evidence_spans=spans,
        prompt=prompt,
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

    def cache_key_for(self, request: AnnotationRequest) -> str:
        encoded = json.dumps(
            self._identity_payload(request),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir / f"{key}.json"

    def _validate_response(
        self,
        request: AnnotationRequest,
        response: AnnotationResponse,
    ) -> AnnotationResponse:
        expected_assets = {candidate.asset_id for candidate in request.candidates}
        observed_assets = {annotation.asset_id for annotation in response.annotations}
        if response.item_id != request.item_id:
            raise ValueError("cached or generated annotation response has the wrong item_id")
        if observed_assets != expected_assets or len(response.annotations) != len(expected_assets):
            raise ValueError("annotation response assets must exactly match the request candidates")
        valid_spans = {span.span_id for span in request.evidence_spans}
        for annotation in response.annotations:
            unknown_spans = set(annotation.evidence_span_ids) - valid_spans
            if unknown_spans:
                raise ValueError(
                    "annotation cites unknown evidence spans: " + ", ".join(sorted(unknown_spans))
                )
        return response

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
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid LLM annotation cache record: {path}") from exc
        if not isinstance(record, dict) or record.get("cache_key") != key:
            raise ValueError(f"LLM annotation cache identity mismatch: {path}")
        if record.get("identity") != self._identity_payload(request):
            raise ValueError(f"LLM annotation cache request mismatch: {path}")
        try:
            payload = _GeneratedPayload.model_validate_json(
                json.dumps(record["annotation_payload"], ensure_ascii=False)
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid LLM annotation cache payload: {path}") from exc
        validated = self._validate_response(
            request,
            AnnotationResponse(item_id=request.item_id, annotations=payload.annotations),
        )
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
                        stance_confidence=0.0,
                        uncertainty=1.0,
                        primary_event_type=None,
                        event_confidence=0.0,
                        evidence_span_ids=(),
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
        return {
            "cache_schema_version": "llm-annotation-cache-v1",
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
            },
            "annotation_payload": {
                "annotations": [annotation.to_dict() for annotation in response.annotations]
            },
            "provenance": self.provenance_payload,
        }

    def _write_cache(self, key: str, record: dict[str, Any]) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
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
            for request in batch:
                tokenized = tokenizer(
                    request.prompt,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False,
                )
                input_tokens = len(tokenized["input_ids"])
                exceeds_model_context = (
                    model_context_tokens is not None
                    and input_tokens + self.config.max_new_tokens > model_context_tokens
                )
                if input_tokens > self.config.max_input_tokens or exceeds_model_context:
                    results[request.request_id] = GenerationResponse(
                        request_id=request.request_id,
                        input_too_long=True,
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
                    token_ids = continuation.detach().cpu().tolist()
                    output_truncated = len(token_ids) >= self.config.max_new_tokens and (
                        not token_ids or token_ids[-1] not in eos_set
                    )
                    results[request.request_id] = GenerationResponse(
                        request_id=request.request_id,
                        generated_text=tokenizer.decode(
                            continuation,
                            skip_special_tokens=True,
                        ),
                        output_truncated=output_truncated,
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
            generated_values = generator(generation_requests)
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
            if set(generated_by_id) != expected_ids:
                raise ValueError("generator did not return every requested annotation")
            for request, key, indices in chunk:
                generated = generated_by_id[key]
                response = self._parse_generation(request, generated)
                record = self._cache_record(request, key, generated, response)
                pending_records.append((key, response, record, indices))

        # Validate every generated batch before making any new cache write.
        for key, response, record, indices in pending_records:
            self._write_cache(key, record)
            self._records[key] = record
            for index in indices:
                results[index] = AnnotationResponse(
                    item_id=values[index].item_id,
                    annotations=response.annotations,
                )
        if any(result is None for result in results):
            raise RuntimeError("LLM annotation inference did not produce every response")
        return [result for result in results if result is not None]
