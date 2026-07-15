from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from nlp_trader.timestamps import format_utc, parse_utc

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$", strict=True)]
Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        strict=True,
    ),
]


class DecisionRoundLedgerError(ValueError):
    """Raised when a decision-round ledger cannot be verified exactly."""


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class ModelIdentity(_StrictModel):
    provider: Identifier
    model_id: Identifier
    revision: Identifier
    sha256: Sha256


class VersionedContract(_StrictModel):
    version: Identifier
    sha256: Sha256


class SamplingSettings(_StrictModel):
    decoding: Literal["greedy", "sampling"]
    seed: int = Field(ge=0)
    max_input_tokens: int = Field(ge=1)
    max_new_tokens: int = Field(ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


class CurrentSourceRetrieval(_StrictModel):
    source_scope: Literal["current_source_only"] = "current_source_only"
    evidence_ids: tuple[Identifier, ...]

    @field_validator("evidence_ids")
    @classmethod
    def validate_unique_evidence_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("retrieval evidence_ids must be unique")
        return values


class RawGeneration(_StrictModel):
    request_id: Identifier
    generated_text: str | None
    input_too_long: bool = False
    output_truncated: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_generation_state(self) -> Self:
        if self.input_too_long:
            if self.generated_text is not None or self.output_truncated:
                raise ValueError(
                    "input-too-long generation cannot contain text or be output-truncated"
                )
        elif self.generated_text is None:
            raise ValueError("generation text is required unless the input was too long")
        _canonical_json(self.metadata)
        return self


class VerifierCheck(_StrictModel):
    check_id: Identifier
    passed: bool
    detail: str | None = None


class VerifierResult(_StrictModel):
    version: Identifier
    passed: bool
    checks: tuple[VerifierCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        check_ids = [check.check_id for check in self.checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("verifier check_id values must be unique")
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("verifier passed must equal the conjunction of its checks")
        return self


class InferenceUsage(_StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    estimated_usd_cost: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_cost_inputs(self) -> Self:
        if self.estimated_usd_cost is not None and (
            self.input_tokens is None or self.output_tokens is None
        ):
            raise ValueError("estimated usage cost requires input and output token counts")
        return self


class DecisionRound(_StrictModel):
    """One immutable, replayable trace of an LLM semantic-research decision."""

    artifact_schema_version: Literal["llm-decision-round-v1"] = "llm-decision-round-v1"
    round_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    run_id: Identifier
    config_hash: Sha256
    input_snapshot_hash: Sha256
    item_id: Identifier
    source_text_hash: Sha256
    source_available_at: datetime
    decision_time: datetime
    horizon_days: int = Field(ge=1)
    model: ModelIdentity
    prompt: VersionedContract
    schema_contract: VersionedContract
    sampling: SamplingSettings
    retrieval: CurrentSourceRetrieval
    raw_generation: RawGeneration
    structured_output: dict[str, JsonValue] | None
    verifier: VerifierResult
    inference_source: Literal["generated", "cache", "deduplicated"]
    usage: InferenceUsage
    application_mode: Literal["sidecar", "augment"]
    retrospective_parser: Literal[True] = True
    tool_calls: tuple[dict[str, JsonValue], ...] = ()
    calibration: dict[str, JsonValue] = Field(default_factory=dict)
    portfolio: dict[str, JsonValue] = Field(default_factory=dict)
    risk: dict[str, JsonValue] = Field(default_factory=dict)
    orders: tuple[dict[str, JsonValue], ...] = ()

    @field_validator("source_available_at", "decision_time", mode="before")
    @classmethod
    def normalize_timestamp(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("decision-round timestamps must be timezone-aware")
            return value.astimezone(UTC)
        if isinstance(value, str):
            try:
                return parse_utc(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("decision-round timestamps must be timezone-aware") from exc
        raise ValueError("decision-round timestamps must be datetime values or ISO timestamps")

    @field_serializer("source_available_at", "decision_time")
    def serialize_timestamp(self, value: datetime) -> str:
        return format_utc(value)

    @field_validator("tool_calls", "orders")
    @classmethod
    def validate_empty_sequences(
        cls,
        values: tuple[dict[str, JsonValue], ...],
    ) -> tuple[dict[str, JsonValue], ...]:
        if values:
            raise ValueError("decision-round downstream sequence fields must remain empty")
        return values

    @field_validator("calibration", "portfolio", "risk")
    @classmethod
    def validate_empty_mappings(
        cls,
        value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        if value:
            raise ValueError("decision-round downstream mapping fields must remain empty")
        return value

    @model_validator(mode="after")
    def validate_temporal_and_identity_invariants(self) -> Self:
        if self.source_available_at > self.decision_time:
            raise ValueError("source_available_at must be no later than decision_time")
        _canonical_json(self.structured_output)
        if self.inference_source != "generated":
            usage_values = (
                self.usage.input_tokens,
                self.usage.output_tokens,
                self.usage.latency_ms,
                self.usage.estimated_usd_cost,
            )
            if any(value is not None and value != 0 for value in usage_values):
                raise ValueError("cache and deduplicated rounds cannot report new inference usage")
        self._validate_generation_binding()
        expected = self.computed_round_id()
        if self.round_id and self.round_id != expected:
            raise ValueError("decision round_id does not match canonical content")
        if not self.round_id:
            object.__setattr__(self, "round_id", expected)
        return self

    def _validate_generation_binding(self) -> None:
        if self.raw_generation.output_truncated:
            if self.structured_output is not None:
                raise ValueError("a truncated generation cannot have structured output")
            if self.verifier.passed:
                raise ValueError("a truncated generation cannot have a passing verifier result")
            return
        if self.structured_output is None:
            if self.verifier.passed:
                raise ValueError("a passing verifier result requires structured output")
            return
        if self.structured_output.get("item_id") != self.item_id:
            raise ValueError("structured output item_id must match the decision round")
        annotations = self.structured_output.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("structured output requires an annotations array")
        if self.raw_generation.input_too_long:
            if not annotations:
                raise ValueError("input-too-long structured output requires canonical abstentions")
            for annotation in annotations:
                if not isinstance(annotation, dict) or not _is_input_too_long_abstention(
                    annotation,
                    horizon_days=self.horizon_days,
                ):
                    raise ValueError(
                        "input-too-long structured output must contain only canonical abstentions"
                    )
            return
        if self.verifier.passed and not annotations:
            raise ValueError("verified structured output requires a nonempty annotations array")
        raw_text = self.raw_generation.generated_text
        if raw_text is None:  # pragma: no cover - RawGeneration already enforces this
            raise ValueError("verified generation text is missing")
        try:
            raw_payload = json.loads(
                raw_text,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except (_DuplicateJsonKeyError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("verified raw generation must be strict JSON") from exc
        if not isinstance(raw_payload, dict) or set(raw_payload) != {"annotations"}:
            raise ValueError("verified raw generation must contain only annotations")
        expected_output = {"item_id": self.item_id, "annotations": raw_payload["annotations"]}
        if _canonical_json(self.structured_output) != _canonical_json(expected_output):
            raise ValueError("verified structured output must match the raw generation exactly")

    def content_payload(self) -> dict[str, Any]:
        """Return the canonical hash input, excluding the self-authenticating ID."""

        return self.model_dump(mode="json", exclude={"round_id"})

    def computed_round_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.content_payload()).encode("utf-8")).hexdigest()

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


class DecisionRoundLedger:
    """Exclusive canonical-JSONL writer and verifying replay reader."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write_exclusive(self, rounds: Iterable[DecisionRound]) -> tuple[DecisionRound, ...]:
        """Create a ledger once, after validating every row and duplicate ID."""

        validated: list[DecisionRound] = []
        seen_ids: set[str] = set()
        encoded_rows: list[str] = []
        for index, value in enumerate(rounds, start=1):
            if not isinstance(value, DecisionRound):
                raise DecisionRoundLedgerError(
                    f"decision round {index} must be a DecisionRound instance"
                )
            try:
                replayed = DecisionRound.model_validate_json(value.canonical_json())
            except ValidationError as exc:
                raise DecisionRoundLedgerError(
                    f"decision round {index} is not internally valid: "
                    f"{_first_validation_message(exc)}"
                ) from exc
            if replayed.round_id in seen_ids:
                raise DecisionRoundLedgerError(
                    f"decision round {index} repeats round_id {replayed.round_id!r}"
                )
            seen_ids.add(replayed.round_id)
            validated.append(replayed)
            encoded_rows.append(replayed.canonical_json() + "\n")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("xb") as handle:
                handle.write("".join(encoded_rows).encode("utf-8"))
        except FileExistsError as exc:
            raise DecisionRoundLedgerError(
                f"decision-round ledger already exists: {self.path}"
            ) from exc
        return tuple(validated)

    def replay_and_verify(self) -> tuple[DecisionRound, ...]:
        """Replay every row after canonical, schema, temporal, and hash verification."""

        if not self.path.exists():
            raise DecisionRoundLedgerError(f"decision-round ledger does not exist: {self.path}")

        rounds: list[DecisionRound] = []
        seen_ids: set[str] = set()
        try:
            handle = self.path.open("r", encoding="utf-8", newline="")
        except OSError as exc:
            raise DecisionRoundLedgerError(
                f"cannot open decision-round ledger: {self.path}"
            ) from exc
        with handle:
            try:
                for line_number, raw_line in enumerate(handle, start=1):
                    context = f"decision-round ledger line {line_number}"
                    if not raw_line.strip():
                        raise DecisionRoundLedgerError(f"{context} is blank")
                    if not raw_line.endswith("\n"):
                        raise DecisionRoundLedgerError(f"{context} is incomplete")
                    try:
                        parsed = json.loads(
                            raw_line,
                            object_pairs_hook=_object_without_duplicate_keys,
                            parse_constant=_reject_nonfinite_constant,
                        )
                    except _DuplicateJsonKeyError as exc:
                        raise DecisionRoundLedgerError(
                            f"{context} repeats JSON key {exc.key!r}"
                        ) from exc
                    except json.JSONDecodeError as exc:
                        raise DecisionRoundLedgerError(f"{context} is not valid JSON") from exc
                    except ValueError as exc:
                        raise DecisionRoundLedgerError(f"{context} is not valid JSON") from exc
                    if not isinstance(parsed, dict):
                        raise DecisionRoundLedgerError(f"{context} must contain an object")
                    if raw_line != _canonical_json(parsed) + "\n":
                        raise DecisionRoundLedgerError(f"{context} is not canonical JSON")
                    try:
                        decision_round = DecisionRound.model_validate_json(raw_line)
                    except ValidationError as exc:
                        raise DecisionRoundLedgerError(
                            f"{context} contains an invalid DecisionRound: "
                            f"{_first_validation_message(exc)}"
                        ) from exc
                    if raw_line != decision_round.canonical_json() + "\n":
                        raise DecisionRoundLedgerError(
                            f"{context} is not canonical DecisionRound JSON"
                        )
                    if decision_round.round_id in seen_ids:
                        raise DecisionRoundLedgerError(
                            f"{context} repeats round_id {decision_round.round_id!r}"
                        )
                    seen_ids.add(decision_round.round_id)
                    rounds.append(decision_round)
            except UnicodeDecodeError as exc:
                raise DecisionRoundLedgerError("decision-round ledger is not valid UTF-8") from exc
        return tuple(rounds)

    def replay(self) -> tuple[DecisionRound, ...]:
        return self.replay_and_verify()


def write_decision_rounds_jsonl(
    path: str | Path,
    rounds: Iterable[DecisionRound],
) -> tuple[DecisionRound, ...]:
    return DecisionRoundLedger(path).write_exclusive(rounds)


def replay_decision_rounds_jsonl(path: str | Path) -> tuple[DecisionRound, ...]:
    return DecisionRoundLedger(path).replay_and_verify()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _first_validation_message(error: ValidationError) -> str:
    errors = error.errors(include_url=False)
    return str(errors[0]["msg"]) if errors else "validation failed"


def _is_input_too_long_abstention(
    annotation: dict[str, Any],
    *,
    horizon_days: int,
) -> bool:
    return (
        annotation.get("stance_label") == "abstain"
        and annotation.get("semantic_signal") == 0
        and annotation.get("raw_confidence") == 0.0
        and annotation.get("uncertainty") == 1.0
        and annotation.get("horizon_days") == horizon_days
        and annotation.get("primary_event_type") is None
        and annotation.get("event_confidence") == 0.0
        and annotation.get("supporting_evidence_span_ids") == []
        and annotation.get("counterevidence_span_ids") == []
        and annotation.get("mechanism") is None
        and annotation.get("invalidation_conditions") == []
        and annotation.get("abstain_reason") == "input_too_long"
    )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DecisionRoundLedgerError(
            "decision-round values must be finite canonical JSON data"
        ) from exc
