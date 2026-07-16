from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from nlp_trader.timestamps import format_utc, parse_utc

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$", strict=True)]
GitCommit = Annotated[
    str,
    StringConstraints(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", strict=True),
]
Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$",
        strict=True,
    ),
]
NonBlankText = Annotated[str, StringConstraints(min_length=1, max_length=16_384, strict=True)]
ScalarValue = str | int | float | bool
ProposalAttemptOutcome = Literal[
    "proposal",
    "abstention",
    "malformed",
    "rejected",
    "duplicate",
    "exhausted",
    "crashed",
]
GENESIS_HASH = "0" * 64
_NUMERIC_TOKEN = re.compile(r"\d")


class StrictModel(BaseModel):
    """Strict immutable base for every research-agent boundary contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


def canonical_json(value: object) -> str:
    """Encode finite typed content using the repository's canonical JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _utc(value: object, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"{field_name} must be timezone-aware")
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            return parse_utc(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be timezone-aware") from exc
    raise ValueError(f"{field_name} must be a datetime or ISO timestamp")


def _require_unique(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")
    return values


class ContractIdentity(StrictModel):
    contract_id: Identifier
    version: Identifier
    sha256: Sha256


class LocalModelIdentity(StrictModel):
    logical_id: Identifier
    revision: Identifier
    file_sha256: Sha256
    license_or_terms_ref: NonBlankText


class TimeRange(StrictModel):
    start: datetime
    end: datetime

    @field_validator("start", "end", mode="before")
    @classmethod
    def normalize_timestamp(cls, value: object, info: Any) -> datetime:
        return _utc(value, field_name=str(info.field_name))

    @field_serializer("start", "end")
    def serialize_timestamp(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.end < self.start:
            raise ValueError("time range end must be on or after start")
        return self


class ParameterRange(StrictModel):
    parameter_id: Identifier
    value_type: Literal["integer", "number", "string", "boolean"]
    minimum: int | float | None = None
    maximum: int | float | None = None
    allowed_values: tuple[ScalarValue, ...] = ()

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("parameter minimum cannot exceed maximum")
        if self.value_type in {"string", "boolean"} and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError("string and boolean parameters cannot have numeric bounds")
        if not self.allowed_values and self.minimum is None and self.maximum is None:
            raise ValueError("parameter range requires bounds or allowed_values")
        expected_types: dict[str, tuple[type[Any], ...]] = {
            "integer": (int,),
            "number": (int, float),
            "string": (str,),
            "boolean": (bool,),
        }
        for value in self.allowed_values:
            if self.value_type != "boolean" and isinstance(value, bool):
                raise ValueError("parameter allowed_values do not match value_type")
            if not isinstance(value, expected_types[self.value_type]):
                raise ValueError("parameter allowed_values do not match value_type")
        if len(self.allowed_values) != len(set(self.allowed_values)):
            raise ValueError("parameter allowed_values must be unique")
        return self


class ExperimentTemplateSpace(StrictModel):
    template_id: Identifier
    version: Identifier
    parameters: tuple[ParameterRange, ...] = ()

    @field_validator("parameters")
    @classmethod
    def validate_unique_parameters(
        cls, values: tuple[ParameterRange, ...]
    ) -> tuple[ParameterRange, ...]:
        names = tuple(value.parameter_id for value in values)
        _require_unique(names, field_name="template parameter_id")
        return values


class StudyDefinition(StrictModel):
    artifact_schema_version: Literal["research-study-v1"] = "research-study-v1"
    study_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    created_at: datetime
    research_question: NonBlankText
    analysis_cutoff: datetime
    intent: Literal["exploratory", "confirmatory"]
    data_lineage_id: Identifier
    parent_identity_requirements: tuple[ContractIdentity, ...] = ()
    development_decisions: TimeRange
    reserved_holdout_decisions: TimeRange
    reserved_holdout_outcomes: TimeRange
    universe_snapshot_id: Identifier
    calendar_contract: ContractIdentity
    market_data_contract: ContractIdentity
    feature_contract: ContractIdentity
    label_contract: ContractIdentity
    target_contract: ContractIdentity
    target_family: Identifier
    horizon_sessions: int = Field(ge=1)
    return_adjustment_contract: ContractIdentity
    permitted_templates: tuple[ExperimentTemplateSpace, ...] = Field(min_length=1)
    proposal_budget: int = Field(ge=1, le=10_000)
    required_learned_families: tuple[Identifier, ...]
    required_fixed_benchmarks: tuple[Identifier, ...]
    required_negative_controls: tuple[Identifier, ...] = Field(min_length=1)
    required_robustness_checks: tuple[Identifier, ...] = Field(min_length=1)
    required_metrics: tuple[Identifier, ...] = Field(min_length=1)
    model: LocalModelIdentity
    prompt_contract: ContractIdentity
    action_schema_contract: ContractIdentity
    proposal_schema_contract: ContractIdentity
    tool_catalog_contract: ContractIdentity
    verifier_contract: ContractIdentity
    view_contract: ContractIdentity
    evidence_index_contract: ContractIdentity
    registry_contract: ContractIdentity
    seeds: tuple[int, ...] = Field(min_length=1)
    known_limitations: tuple[NonBlankText, ...] = Field(min_length=1)

    @field_validator("created_at", "analysis_cutoff", mode="before")
    @classmethod
    def normalize_timestamp(cls, value: object, info: Any) -> datetime:
        return _utc(value, field_name=str(info.field_name))

    @field_serializer("created_at", "analysis_cutoff")
    def serialize_timestamp(self, value: datetime) -> str:
        return format_utc(value)

    @field_validator(
        "required_learned_families",
        "required_fixed_benchmarks",
        "required_negative_controls",
        "required_robustness_checks",
        "required_metrics",
    )
    @classmethod
    def validate_unique_identifiers(cls, values: tuple[str, ...], info: Any) -> tuple[str, ...]:
        return _require_unique(values, field_name=str(info.field_name))

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 for value in values):
            raise ValueError("seeds must be non-negative")
        if len(values) != len(set(values)):
            raise ValueError("seeds must be unique")
        return values

    @model_validator(mode="after")
    def validate_study(self) -> Self:
        if self.analysis_cutoff > self.development_decisions.end:
            raise ValueError("analysis_cutoff cannot exceed the development decision boundary")
        if self.analysis_cutoff >= self.reserved_holdout_decisions.start:
            raise ValueError("analysis_cutoff must precede the reserved holdout")
        if self.development_decisions.end >= self.reserved_holdout_decisions.start:
            raise ValueError("development decisions must end before reserved holdout decisions")
        if self.reserved_holdout_outcomes.start < self.reserved_holdout_decisions.start:
            raise ValueError("holdout outcomes cannot begin before holdout decisions")
        template_ids = tuple(value.template_id for value in self.permitted_templates)
        _require_unique(template_ids, field_name="permitted template_id")
        if not {"traditional", "text", "combined"}.issubset(self.required_learned_families):
            raise ValueError(
                "required learned families must retain traditional, text, and combined"
            )
        if not {"equal_weight", "momentum_only", "no_trade"}.issubset(
            self.required_fixed_benchmarks
        ):
            raise ValueError(
                "required fixed benchmarks must retain equal_weight, momentum_only, and no_trade"
            )
        expected = self.computed_study_id()
        if self.study_id and self.study_id != expected:
            raise ValueError("study_id does not match canonical study content")
        if not self.study_id:
            object.__setattr__(self, "study_id", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"study_id"})

    def computed_study_id(self) -> str:
        return content_sha256(self.content_payload())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class HoldoutIdentity(StrictModel):
    artifact_schema_version: Literal["holdout-identity-v1"] = "holdout-identity-v1"
    holdout_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    data_lineage_id: Identifier
    input_snapshot_hashes: tuple[Sha256, ...] = Field(min_length=1)
    universe_snapshot_id: Identifier
    universe_asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    calendar_contract: ContractIdentity
    market_data_contract: ContractIdentity
    label_contract: ContractIdentity
    target_family: Identifier
    horizon_sessions: int = Field(ge=1)
    return_adjustment_contract: ContractIdentity
    decision_interval: TimeRange
    outcome_interval: TimeRange
    selection_identity: ContractIdentity | None = None
    study_id: Sha256
    candidate_hash: Sha256

    @field_validator("input_snapshot_hashes", "universe_asset_ids")
    @classmethod
    def validate_unique_sorted(cls, values: tuple[str, ...], info: Any) -> tuple[str, ...]:
        _require_unique(values, field_name=str(info.field_name))
        if values != tuple(sorted(values)):
            raise ValueError(f"{info.field_name} must be sorted")
        return values

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.outcome_interval.start < self.decision_interval.start:
            raise ValueError("holdout outcome interval cannot precede the decision interval")
        expected = self.computed_holdout_id()
        if self.holdout_id and self.holdout_id != expected:
            raise ValueError("holdout_id does not match canonical holdout content")
        if not self.holdout_id:
            object.__setattr__(self, "holdout_id", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"holdout_id"})

    def computed_holdout_id(self) -> str:
        return content_sha256(self.content_payload())


class ArtifactEntry(StrictModel):
    role: Identifier
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: Sha256
    bytes: int = Field(ge=0)
    schema_version: Identifier

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path == PurePosixPath(".")
            or path.is_absolute()
            or ".." in path.parts
            or value.startswith("./")
            or "\\" in value
        ):
            raise ValueError("artifact relative_path must be normalized and remain below its root")
        if path.as_posix() != value:
            raise ValueError("artifact relative_path must use normalized POSIX separators")
        return value


class AgentArtifactManifest(StrictModel):
    artifact_schema_version: Literal["research-agent-manifest-v1"] = "research-agent-manifest-v1"
    manifest_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    artifact_root_id: Identifier
    created_at: datetime
    git_commit: GitCommit | None
    dirty_worktree: bool | None
    parent_hashes: tuple[Sha256, ...] = ()
    artifacts: tuple[ArtifactEntry, ...]
    limitations: tuple[NonBlankText, ...]
    next_questions: tuple[NonBlankText, ...]

    @field_validator("created_at", mode="before")
    @classmethod
    def normalize_created_at(cls, value: object) -> datetime:
        return _utc(value, field_name="created_at")

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        paths = tuple(value.relative_path for value in self.artifacts)
        _require_unique(paths, field_name="artifact relative_path")
        expected = self.computed_manifest_id()
        if self.manifest_id and self.manifest_id != expected:
            raise ValueError("manifest_id does not match canonical manifest content")
        if not self.manifest_id:
            object.__setattr__(self, "manifest_id", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"manifest_id"})

    def computed_manifest_id(self) -> str:
        return content_sha256(self.content_payload())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class DevelopmentViewBundleManifest(StrictModel):
    artifact_schema_version: Literal["development-view-bundle-v1"] = "development-view-bundle-v1"
    bundle_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    source_mode: Literal["exploratory_standard_run", "development_only_run"]
    confirmatory_eligible: bool
    parent_run_id: Identifier
    parent_manifest_hash: Sha256
    study_id: Sha256
    analysis_cutoff: datetime
    development_decisions: TimeRange
    reserved_holdout_decisions: TimeRange
    reserved_holdout_outcomes: TimeRange
    files: tuple[ArtifactEntry, ...] = Field(min_length=1)
    exporter_contract: ContractIdentity
    git_commit: GitCommit | None
    dirty_worktree: bool | None
    included_metric_groups: tuple[Identifier, ...]
    excluded_roles: tuple[Identifier, ...]
    excluded_field_families: tuple[Identifier, ...]
    evidence_snapshot_hash: Sha256
    evidence_index_identity: ContractIdentity
    feature_catalog_identity: ContractIdentity
    limitations: tuple[NonBlankText, ...] = Field(min_length=1)

    @field_validator("analysis_cutoff", mode="before")
    @classmethod
    def normalize_analysis_cutoff(cls, value: object) -> datetime:
        return _utc(value, field_name="analysis_cutoff")

    @field_serializer("analysis_cutoff")
    def serialize_analysis_cutoff(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        if self.source_mode == "exploratory_standard_run" and self.confirmatory_eligible:
            raise ValueError("ordinary completed runs cannot be confirmatory-eligible")
        if self.analysis_cutoff > self.development_decisions.end:
            raise ValueError("bundle analysis cutoff cannot exceed development decisions")
        if self.analysis_cutoff >= self.reserved_holdout_decisions.start:
            raise ValueError("bundle analysis cutoff must precede the reserved holdout")
        paths = tuple(value.relative_path for value in self.files)
        _require_unique(paths, field_name="bundle file relative_path")
        expected = self.computed_bundle_id()
        if self.bundle_id and self.bundle_id != expected:
            raise ValueError("bundle_id does not match canonical bundle manifest content")
        if not self.bundle_id:
            object.__setattr__(self, "bundle_id", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"bundle_id"})

    def computed_bundle_id(self) -> str:
        return content_sha256(self.content_payload())


class EvidenceRecord(StrictModel):
    artifact_schema_version: Literal["evidence-record-v1"] = "evidence-record-v1"
    evidence_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    source_item_id: Identifier
    span_id: Identifier
    source_text_hash: Sha256
    span_hash: Sha256
    source_type: Identifier
    content_status: Literal["active", "deleted", "private", "protected", "unknown"]
    relationship_type: Literal["original", "repost", "quote", "reply", "unknown"]
    license_or_terms_ref: NonBlankText
    retention_permitted: Literal[True]
    asset_ids: tuple[Identifier, ...]
    active_period_valid: Literal[True]
    published_at: datetime
    available_at: datetime
    snapshot_cutoff: datetime
    quoted_span: NonBlankText
    start_offset: int = Field(default=0, ge=0)
    end_offset: int | None = Field(default=None, ge=1)
    text_parts: tuple[Literal["title", "body"], ...] = ()
    source_artifact_id: Identifier
    source_artifact_hash: Sha256

    @field_validator("published_at", "available_at", "snapshot_cutoff", mode="before")
    @classmethod
    def normalize_timestamp(cls, value: object, info: Any) -> datetime:
        return _utc(value, field_name=str(info.field_name))

    @field_serializer("published_at", "available_at", "snapshot_cutoff")
    def serialize_timestamp(self, value: datetime) -> str:
        return format_utc(value)

    @field_validator("asset_ids")
    @classmethod
    def validate_asset_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(values, field_name="evidence asset_ids")
        if values != tuple(sorted(values)):
            raise ValueError("evidence asset_ids must be sorted")
        return values

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.available_at > self.snapshot_cutoff:
            raise ValueError("evidence available_at cannot exceed its snapshot cutoff")
        if self.end_offset is not None:
            if self.end_offset <= self.start_offset:
                raise ValueError("evidence end_offset must follow start_offset")
            if self.end_offset - self.start_offset != len(self.quoted_span):
                raise ValueError("evidence offsets must match quoted_span length")
        if len(self.text_parts) != len(set(self.text_parts)):
            raise ValueError("evidence text_parts must be unique")
        expected = self.computed_evidence_id()
        if self.evidence_id and self.evidence_id != expected:
            raise ValueError("evidence_id does not match canonical evidence content")
        if not self.evidence_id:
            object.__setattr__(self, "evidence_id", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"evidence_id"})

    def computed_evidence_id(self) -> str:
        return content_sha256(self.content_payload())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class EvidenceReference(StrictModel):
    evidence: EvidenceRecord
    query_id: Sha256
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0)
    citation_role: Literal["supporting", "counterevidence"]


class CounterevidenceSearchRecord(StrictModel):
    query_id: Sha256
    normalized_query_hash: Sha256
    filters_hash: Sha256
    result_count: int = Field(ge=0)
    pages_inspected: int = Field(ge=1)
    cited_result_ids: tuple[Sha256, ...] = ()
    reason: Literal["no_result", "insufficient_result"] | None = None

    @model_validator(mode="after")
    def validate_search(self) -> Self:
        _require_unique(self.cited_result_ids, field_name="counterevidence cited_result_ids")
        if len(self.cited_result_ids) > self.result_count:
            raise ValueError("counterevidence citations cannot exceed the result count")
        if not self.cited_result_ids and self.reason is None:
            raise ValueError("a search without retained counterevidence requires a reason")
        if self.cited_result_ids and self.result_count == 0:
            raise ValueError("a no-result search cannot cite result IDs")
        return self


class MetricReference(StrictModel):
    metric_reference_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    metric_group: Identifier
    family: Identifier | None = None
    segment: Identifier | None = None
    metric_id: Identifier
    value: float
    unit: Identifier
    scope: Identifier
    window: TimeRange
    source_artifact_id: Identifier
    source_artifact_hash: Sha256
    tool_result_hash: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = content_sha256(self.model_dump(mode="json", exclude={"metric_reference_id"}))
        if self.metric_reference_id and self.metric_reference_id != expected:
            raise ValueError("metric_reference_id does not match canonical content")
        if not self.metric_reference_id:
            object.__setattr__(self, "metric_reference_id", expected)
        return self


class CalculationInput(StrictModel):
    source_id: Sha256
    value: float


class CalculationReference(StrictModel):
    calculation_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    operation: Literal[
        "sum",
        "mean",
        "difference",
        "ratio",
        "percent_change",
        "population_stddev",
        "sample_stddev",
        "pearson_correlation",
    ]
    inputs: tuple[CalculationInput, ...] = Field(min_length=1)
    output: float
    rounding_policy: Identifier
    tool_result_hash: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = content_sha256(self.model_dump(mode="json", exclude={"calculation_id"}))
        if self.calculation_id and self.calculation_id != expected:
            raise ValueError("calculation_id does not match canonical content")
        if not self.calculation_id:
            object.__setattr__(self, "calculation_id", expected)
        return self


class QuantitativeClaim(StrictModel):
    claim_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    metric_id: Identifier
    comparator: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    value: float
    unit: Identifier
    scope: Identifier
    window: TimeRange
    classification: Literal["hypothesis", "observation"]
    reference_id: Sha256
    reference_kind: Literal["evidence", "metric", "calculation"]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = content_sha256(self.model_dump(mode="json", exclude={"claim_id"}))
        if self.claim_id and self.claim_id != expected:
            raise ValueError("claim_id does not match canonical content")
        if not self.claim_id:
            object.__setattr__(self, "claim_id", expected)
        return self


class ParameterChoice(StrictModel):
    parameter_id: Identifier
    value: ScalarValue


class ResearchProposal(StrictModel):
    artifact_schema_version: Literal["research-proposal-v1"] = "research-proposal-v1"
    proposal_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    attempt_id: Sha256
    bundle_id: Sha256
    input_snapshot_hash: Sha256
    hypothesis: NonBlankText
    mechanism: NonBlankText
    affected_universe_id: Identifier
    horizon_sessions: int = Field(ge=1)
    target_family: Identifier
    expected_direction: Literal["positive", "negative", "non_monotonic"]
    direction_is_hypothesis: Literal[True]
    supporting_evidence_ids: tuple[Sha256, ...] = Field(min_length=1)
    counterevidence_ids: tuple[Sha256, ...] = ()
    counterevidence_searches: tuple[CounterevidenceSearchRecord, ...] = ()
    quantitative_claims: tuple[QuantitativeClaim, ...] = ()
    falsification_conditions: tuple[NonBlankText, ...] = Field(min_length=1)
    invalidation_conditions: tuple[NonBlankText, ...] = Field(min_length=1)
    required_input_ids: tuple[Identifier, ...] = Field(min_length=1)
    availability_requirements: tuple[NonBlankText, ...] = Field(min_length=1)
    experiment_template_id: Identifier
    parameter_choices: tuple[ParameterChoice, ...]
    required_learned_families: tuple[Identifier, ...]
    required_fixed_benchmarks: tuple[Identifier, ...]
    negative_controls: tuple[Identifier, ...] = Field(min_length=1)
    sensitivity_checks: tuple[Identifier, ...] = Field(min_length=1)
    expected_failure_modes: tuple[NonBlankText, ...] = Field(min_length=1)
    known_limitations: tuple[NonBlankText, ...] = Field(min_length=1)
    expected_output_metrics: tuple[Identifier, ...] = Field(min_length=1)
    acceptance_interpretation: NonBlankText

    @field_validator(
        "hypothesis",
        "mechanism",
        "falsification_conditions",
        "invalidation_conditions",
        "availability_requirements",
        "expected_failure_modes",
        "known_limitations",
        "acceptance_interpretation",
    )
    @classmethod
    def reject_unstructured_numeric_claims(
        cls, value: str | tuple[str, ...]
    ) -> str | tuple[str, ...]:
        texts = (value,) if isinstance(value, str) else value
        if any(_NUMERIC_TOKEN.search(text) for text in texts):
            raise ValueError(
                "proposal free text cannot contain numeric tokens; use typed quantitative fields"
            )
        return value

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        for name in (
            "supporting_evidence_ids",
            "counterevidence_ids",
            "required_input_ids",
            "required_learned_families",
            "required_fixed_benchmarks",
            "negative_controls",
            "sensitivity_checks",
            "expected_output_metrics",
        ):
            _require_unique(getattr(self, name), field_name=name)
        if set(self.supporting_evidence_ids).intersection(self.counterevidence_ids):
            raise ValueError("supporting and counterevidence IDs must be disjoint")
        if not self.counterevidence_ids and not self.counterevidence_searches:
            raise ValueError("proposal requires counterevidence or a bounded challenge search")
        parameter_ids = tuple(value.parameter_id for value in self.parameter_choices)
        _require_unique(parameter_ids, field_name="proposal parameter_id")
        if not {"traditional", "text", "combined"}.issubset(self.required_learned_families):
            raise ValueError("proposal must retain traditional, text, and combined baselines")
        if not {"equal_weight", "momentum_only", "no_trade"}.issubset(
            self.required_fixed_benchmarks
        ):
            raise ValueError("proposal must retain all fixed benchmark paths")
        expected = content_sha256(self.model_dump(mode="json", exclude={"proposal_id"}))
        if self.proposal_id and self.proposal_id != expected:
            raise ValueError("proposal_id does not match canonical proposal content")
        if not self.proposal_id:
            object.__setattr__(self, "proposal_id", expected)
        return self

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class ResearchAbstention(StrictModel):
    artifact_schema_version: Literal["research-abstention-v1"] = "research-abstention-v1"
    abstention_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    attempt_id: Sha256
    bundle_id: Sha256
    input_snapshot_hash: Sha256
    reason: Literal[
        "insufficient_evidence",
        "conflicting_evidence",
        "missing_input",
        "unsupported_template",
        "limit_exhausted",
    ]
    explanation: NonBlankText
    missing_input_ids: tuple[Identifier, ...] = ()
    tool_query_ids: tuple[Sha256, ...] = ()
    resolvable_in_new_study: bool

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _require_unique(self.missing_input_ids, field_name="abstention missing_input_ids")
        _require_unique(self.tool_query_ids, field_name="abstention tool_query_ids")
        expected = content_sha256(self.model_dump(mode="json", exclude={"abstention_id"}))
        if self.abstention_id and self.abstention_id != expected:
            raise ValueError("abstention_id does not match canonical abstention content")
        if not self.abstention_id:
            object.__setattr__(self, "abstention_id", expected)
        return self


class SearchEvidenceRequest(StrictModel):
    query: NonBlankText
    purpose: Literal["support", "challenge"]
    asset_ids: tuple[Identifier, ...] = ()
    source_types: tuple[Identifier, ...] = ()
    available_range: TimeRange | None = None
    result_limit: int = Field(ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=2048)


class ReadDevelopmentMetricsRequest(StrictModel):
    metric_group: Identifier
    family: Identifier | None = None
    segment: Identifier | None = None


class ReadFeatureCatalogRequest(StrictModel):
    section: Literal[
        "features",
        "models",
        "benchmarks",
        "selectors",
        "metrics",
        "controls",
        "templates",
    ]


class CalculateRequest(StrictModel):
    operation: Literal[
        "sum",
        "mean",
        "difference",
        "ratio",
        "percent_change",
        "population_stddev",
        "sample_stddev",
        "pearson_correlation",
    ]
    inputs: tuple[CalculationInput, ...] = Field(min_length=1)
    rounding_policy: Identifier


class SearchEvidenceToolCall(StrictModel):
    tool_name: Literal["search_evidence"] = "search_evidence"
    request: SearchEvidenceRequest


class ReadDevelopmentMetricsToolCall(StrictModel):
    tool_name: Literal["read_development_metrics"] = "read_development_metrics"
    request: ReadDevelopmentMetricsRequest


class ReadFeatureCatalogToolCall(StrictModel):
    tool_name: Literal["read_feature_catalog"] = "read_feature_catalog"
    request: ReadFeatureCatalogRequest


class CalculateToolCall(StrictModel):
    tool_name: Literal["calculate"] = "calculate"
    request: CalculateRequest


ResearchAgentToolCall = Annotated[
    SearchEvidenceToolCall
    | ReadDevelopmentMetricsToolCall
    | ReadFeatureCatalogToolCall
    | CalculateToolCall,
    Field(discriminator="tool_name"),
]


class ToolCallAction(StrictModel):
    action_type: Literal["tool_call"] = "tool_call"
    tool_call: ResearchAgentToolCall


class ProposalAction(StrictModel):
    action_type: Literal["proposal"] = "proposal"
    proposal: ResearchProposal


class AbstentionAction(StrictModel):
    action_type: Literal["abstention"] = "abstention"
    abstention: ResearchAbstention


ResearchAgentAction = Annotated[
    ToolCallAction | ProposalAction | AbstentionAction,
    Field(discriminator="action_type"),
]


class ProposalCheck(StrictModel):
    check_id: Identifier
    passed: bool
    detail: str | None = Field(default=None, max_length=2048)


class ProposalVerification(StrictModel):
    artifact_schema_version: Literal["proposal-verification-v1"] = "proposal-verification-v1"
    verification_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    attempt_id: Sha256
    terminal_artifact_hash: Sha256
    registry_head_hash: Sha256
    bundle_id: Sha256
    verifier_contract: ContractIdentity
    passed: bool
    checks: tuple[ProposalCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_verification(self) -> Self:
        check_ids = tuple(value.check_id for value in self.checks)
        _require_unique(check_ids, field_name="proposal verification check_id")
        if self.passed != all(value.passed for value in self.checks):
            raise ValueError("verification passed must equal the conjunction of all checks")
        expected = content_sha256(self.model_dump(mode="json", exclude={"verification_id"}))
        if self.verification_id and self.verification_id != expected:
            raise ValueError("verification_id does not match canonical verification content")
        if not self.verification_id:
            object.__setattr__(self, "verification_id", expected)
        return self


class RoundCheck(StrictModel):
    check_id: Identifier
    passed: bool
    detail: str | None = Field(default=None, max_length=2048)


class GenerationDiagnostics(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    throughput_tokens_per_second: float | None = Field(default=None, ge=0.0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    requested_gpu_layers: int | None = Field(default=None, ge=0)
    effective_gpu_layers: int | None = Field(default=None, ge=0)
    device_path: Literal["cpu", "metal", "injected"]


class ResearchAgentRound(StrictModel):
    artifact_schema_version: Literal["research-agent-round-v1"] = "research-agent-round-v1"
    round_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    agent_run_id: Identifier
    study_id: Sha256
    attempt_id: Sha256
    step: int = Field(ge=1)
    previous_round_hash: Sha256
    model: LocalModelIdentity
    prompt_contract: ContractIdentity
    action_schema_contract: ContractIdentity
    proposal_schema_contract: ContractIdentity
    verifier_contract: ContractIdentity
    tool_catalog_contract: ContractIdentity
    bundle_id: Sha256
    input_snapshot_hash: Sha256
    attempt_reservation_event_hash: Sha256
    reserved_study_state_hash: Sha256
    context_hash: Sha256
    raw_generation: str
    parse_status: Literal["passed", "failed"]
    parsed_action: ResearchAgentAction | None
    tool_request_hash: Sha256 | None = None
    tool_result_hash: Sha256 | None = None
    origin: Literal["generated", "replay"]
    checks: tuple[RoundCheck, ...] = ()
    diagnostics: GenerationDiagnostics
    termination_reason: Identifier | None = None

    @model_validator(mode="after")
    def validate_round(self) -> Self:
        if self.parse_status == "passed" and self.parsed_action is None:
            raise ValueError("a passed parse requires parsed_action")
        if self.parse_status == "failed" and self.parsed_action is not None:
            raise ValueError("a failed parse cannot contain parsed_action")
        if (self.tool_request_hash is None) != (self.tool_result_hash is None):
            raise ValueError("tool request and result hashes must be present together")
        check_ids = tuple(check.check_id for check in self.checks)
        _require_unique(check_ids, field_name="round check_id")
        expected = self.computed_round_id()
        if self.round_id and self.round_id != expected:
            raise ValueError("round_id does not match canonical round content")
        if not self.round_id:
            object.__setattr__(self, "round_id", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"round_id"})

    def computed_round_id(self) -> str:
        return content_sha256(self.content_payload())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class StudyRegisteredPayload(StrictModel):
    kind: Literal["study_registered"] = "study_registered"
    study_definition: StudyDefinition


class ProposalAttemptReservedPayload(StrictModel):
    kind: Literal["proposal_attempt_reserved"] = "proposal_attempt_reserved"
    attempt_id: Sha256
    attempt_number: int = Field(ge=1)
    reserved_study_state_hash: Sha256


class ProposalAttemptCompletedPayload(StrictModel):
    kind: Literal["proposal_attempt_completed"] = "proposal_attempt_completed"
    attempt_id: Sha256
    outcome: ProposalAttemptOutcome
    agent_run_id: Identifier
    terminal_artifact_hash: Sha256 | None = None
    detail: NonBlankText

    @model_validator(mode="after")
    def validate_terminal_hash(self) -> Self:
        if self.outcome in {"proposal", "abstention"} and self.terminal_artifact_hash is None:
            raise ValueError("proposal and abstention outcomes require a terminal artifact hash")
        return self


class ProposalVerifiedPayload(StrictModel):
    kind: Literal["proposal_verified"] = "proposal_verified"
    attempt_id: Sha256
    terminal_artifact_hash: Sha256
    verification_hash: Sha256
    passed: bool


class DevelopmentExecutionApprovedPayload(StrictModel):
    kind: Literal["development_execution_approved"] = "development_execution_approved"
    attempt_id: Sha256
    proposal_verification_hash: Sha256
    execution_definition_hash: Sha256
    reviewer_reason: NonBlankText


class DevelopmentRunStartedPayload(StrictModel):
    kind: Literal["development_run_started"] = "development_run_started"
    development_run_id: Identifier
    execution_definition_hash: Sha256
    technical_attempt_number: int = Field(ge=1)


class DevelopmentRunFailedPayload(StrictModel):
    kind: Literal["development_run_failed"] = "development_run_failed"
    development_run_id: Identifier
    failure_type: Identifier
    detail: NonBlankText


class DevelopmentRunCompletedPayload(StrictModel):
    kind: Literal["development_run_completed"] = "development_run_completed"
    development_run_id: Identifier
    result_manifest_hash: Sha256
    frozen_model_manifest_hash: Sha256


class CandidateFrozenPayload(StrictModel):
    kind: Literal["candidate_frozen"] = "candidate_frozen"
    proposal_hash: Sha256
    execution_definition_hash: Sha256
    development_approval_event_hash: Sha256
    development_result_manifest_hash: Sha256
    frozen_model_manifest_hash: Sha256
    candidate_config_hash: Sha256
    required_evaluation_contract_hash: Sha256
    holdout_identity: HoldoutIdentity
    reviewer_reason: NonBlankText


class HoldoutRevealReservedPayload(StrictModel):
    kind: Literal["holdout_reveal_reserved"] = "holdout_reveal_reserved"
    reservation_id: Sha256
    candidate_hash: Sha256
    holdout_identity: HoldoutIdentity


class HoldoutRevealFailedPayload(StrictModel):
    kind: Literal["holdout_reveal_failed"] = "holdout_reveal_failed"
    reservation_id: Sha256
    failure_stage: Identifier
    detail: NonBlankText


class HoldoutRevealedPayload(StrictModel):
    kind: Literal["holdout_revealed"] = "holdout_revealed"
    reservation_id: Sha256
    candidate_hash: Sha256
    holdout_identity: HoldoutIdentity
    result_manifest_hash: Sha256


class ExternalHoldoutRegisteredPayload(StrictModel):
    kind: Literal["external_holdout_registered"] = "external_holdout_registered"
    holdout_identity: HoldoutIdentity
    reason: NonBlankText


class StudyClosedPayload(StrictModel):
    kind: Literal["study_closed"] = "study_closed"
    reason: NonBlankText


RegistryPayload = Annotated[
    StudyRegisteredPayload
    | ProposalAttemptReservedPayload
    | ProposalAttemptCompletedPayload
    | ProposalVerifiedPayload
    | DevelopmentExecutionApprovedPayload
    | DevelopmentRunStartedPayload
    | DevelopmentRunFailedPayload
    | DevelopmentRunCompletedPayload
    | CandidateFrozenPayload
    | HoldoutRevealReservedPayload
    | HoldoutRevealFailedPayload
    | HoldoutRevealedPayload
    | ExternalHoldoutRegisteredPayload
    | StudyClosedPayload,
    Field(discriminator="kind"),
]
RegistryEventType = Literal[
    "study_registered",
    "proposal_attempt_reserved",
    "proposal_attempt_completed",
    "proposal_verified",
    "development_execution_approved",
    "development_run_started",
    "development_run_failed",
    "development_run_completed",
    "candidate_frozen",
    "holdout_reveal_reserved",
    "holdout_reveal_failed",
    "holdout_revealed",
    "external_holdout_registered",
    "study_closed",
]


class RegistryEvent(StrictModel):
    artifact_schema_version: Literal["research-registry-event-v1"] = "research-registry-event-v1"
    event_id: Sha256
    sequence: int = Field(ge=1)
    previous_event_hash: Sha256
    event_hash: Sha256
    event_type: RegistryEventType
    event_ts: datetime
    study_id: Sha256 | None
    actor_kind: Literal["host", "human"]
    actor_label: Identifier
    payload: RegistryPayload

    @field_validator("event_ts", mode="before")
    @classmethod
    def normalize_event_ts(cls, value: object) -> datetime:
        return _utc(value, field_name="event_ts")

    @field_serializer("event_ts")
    def serialize_event_ts(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.event_type != self.payload.kind:
            raise ValueError("registry event_type must match payload kind")
        if isinstance(self.payload, ExternalHoldoutRegisteredPayload):
            if (
                self.study_id is not None
                and self.study_id != self.payload.holdout_identity.study_id
            ):
                raise ValueError("external holdout study_id must match its identity when present")
        elif self.study_id is None:
            raise ValueError("study-scoped registry events require study_id")
        if (
            isinstance(self.payload, StudyRegisteredPayload)
            and self.study_id != self.payload.study_definition.study_id
        ):
            raise ValueError("registered study_id must match the study definition")
        if self.event_id != self.computed_event_id():
            raise ValueError("registry event_id does not match canonical event body")
        if self.event_hash != self.computed_event_hash():
            raise ValueError("registry event_hash does not match canonical event content")
        return self

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        previous_event_hash: str,
        event_ts: datetime,
        study_id: str | None,
        actor_kind: Literal["host", "human"],
        actor_label: str,
        payload: RegistryPayload,
    ) -> RegistryEvent:
        body = {
            "artifact_schema_version": "research-registry-event-v1",
            "event_type": payload.kind,
            "event_ts": format_utc(_utc(event_ts, field_name="event_ts")),
            "study_id": study_id,
            "actor_kind": actor_kind,
            "actor_label": actor_label,
            "payload": payload.model_dump(mode="json"),
        }
        event_id = content_sha256(body)
        unhashed = {
            **body,
            "event_id": event_id,
            "sequence": sequence,
            "previous_event_hash": previous_event_hash,
        }
        encoded = canonical_json({**unhashed, "event_hash": content_sha256(unhashed)})
        return cls.model_validate_json(encoded)

    def event_body(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            include={
                "artifact_schema_version",
                "event_type",
                "event_ts",
                "study_id",
                "actor_kind",
                "actor_label",
                "payload",
            },
        )

    def computed_event_id(self) -> str:
        return content_sha256(self.event_body())

    def computed_event_hash(self) -> str:
        return content_sha256(self.model_dump(mode="json", exclude={"event_hash"}))

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class ProposalAttemptSnapshot(StrictModel):
    attempt_id: Sha256
    attempt_number: int = Field(ge=1)
    reservation_event_hash: Sha256
    reserved_study_state_hash: Sha256
    status: Literal["reserved", "completed"]
    outcome: ProposalAttemptOutcome | None = None
    agent_run_id: Identifier | None = None
    terminal_artifact_hash: Sha256 | None = None
    verification_hash: Sha256 | None = None
    verification_passed: bool | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "reserved" and any(
            value is not None
            for value in (
                self.outcome,
                self.agent_run_id,
                self.terminal_artifact_hash,
                self.verification_hash,
                self.verification_passed,
            )
        ):
            raise ValueError("a reserved attempt cannot contain terminal or verification fields")
        if self.status == "completed" and (self.outcome is None or self.agent_run_id is None):
            raise ValueError("a completed attempt requires outcome and agent_run_id")
        if (self.verification_hash is None) != (self.verification_passed is None):
            raise ValueError("verification hash and result must be present together")
        if self.verification_hash is not None and self.outcome not in {"proposal", "abstention"}:
            raise ValueError("only proposals and abstentions can have verification fields")
        return self


class StudyStateSnapshot(StrictModel):
    artifact_schema_version: Literal["research-study-state-v1"] = "research-study-state-v1"
    study_id: Sha256
    definition_hash: Sha256
    state: Literal[
        "development_open",
        "development_locked",
        "candidate_frozen",
        "holdout_revealed",
        "closed",
    ]
    proposal_budget: int = Field(ge=1)
    proposal_budget_consumed: int = Field(ge=0)
    proposal_budget_remaining: int = Field(ge=0)
    attempts: tuple[ProposalAttemptSnapshot, ...]
    transition_event_hashes: tuple[Sha256, ...]
    last_registry_head_hash: Sha256

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.proposal_budget_consumed + self.proposal_budget_remaining != self.proposal_budget:
            raise ValueError("study proposal budget projection is inconsistent")
        if self.proposal_budget_consumed != len(self.attempts):
            raise ValueError("every reserved attempt must consume exactly one proposal budget slot")
        attempt_ids = tuple(value.attempt_id for value in self.attempts)
        _require_unique(attempt_ids, field_name="study attempt_id")
        if tuple(value.attempt_number for value in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("study attempt numbers must be contiguous")
        _require_unique(self.transition_event_hashes, field_name="study transition_event_hashes")
        return self

    def snapshot_hash(self) -> str:
        return content_sha256(self.model_dump(mode="json"))


class HoldoutUseRecord(StrictModel):
    holdout_identity: HoldoutIdentity
    registry_event_hash: Sha256
    source: Literal["external", "reveal_reservation"]


class HoldoutUseIndex(StrictModel):
    artifact_schema_version: Literal["holdout-use-index-v1"] = "holdout-use-index-v1"
    records: tuple[HoldoutUseRecord, ...]
    last_registry_head_hash: Sha256

    def overlapping(self, candidate: HoldoutIdentity) -> tuple[HoldoutUseRecord, ...]:
        assets = set(candidate.universe_asset_ids)
        return tuple(
            record
            for record in self.records
            if record.holdout_identity.data_lineage_id == candidate.data_lineage_id
            and record.holdout_identity.target_family == candidate.target_family
            and record.holdout_identity.label_contract.contract_id
            == candidate.label_contract.contract_id
            and assets.intersection(record.holdout_identity.universe_asset_ids)
            and _ranges_overlap(
                candidate.outcome_interval,
                record.holdout_identity.outcome_interval,
            )
        )


def _ranges_overlap(left: TimeRange, right: TimeRange) -> bool:
    return left.start <= right.end and right.start <= left.end
