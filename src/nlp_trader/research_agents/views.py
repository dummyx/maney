from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, ValidationError, field_serializer, field_validator, model_validator

from nlp_trader.immutable.append import (
    SafeFileError,
    read_bytes_no_follow,
    write_bytes_exclusive_durable,
)
from nlp_trader.research_agents.artifacts import AgentArtifactError, ensure_agent_artifact_root
from nlp_trader.research_agents.catalog import FeatureCatalog
from nlp_trader.research_agents.contracts import (
    ArtifactEntry,
    ContractIdentity,
    DevelopmentViewBundleManifest,
    EvidenceRecord,
    Identifier,
    Sha256,
    StrictModel,
    StudyDefinition,
    TimeRange,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_agents.evidence import (
    EvidenceSourceRecord,
    build_evidence_snapshot,
    evidence_snapshot_bytes,
)
from nlp_trader.research_agents.index import LexicalCharNgramIndex, build_lexical_index
from nlp_trader.timestamps import format_utc, parse_utc

_BUNDLE_FILES = {
    "development_view.json",
    "feature_catalog.json",
    "evidence_snapshot.jsonl",
    "evidence_index.json",
}
_FORBIDDEN_VIEW_NAMES = (
    "final_holdout",
    "holdout_result",
    "paper",
    "broker",
    "account",
    "order",
    "secret",
    "target_weight",
)


class DevelopmentMetric(StrictModel):
    metric_value_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    metric_group: Identifier
    family: Identifier | None = None
    segment: Identifier | None = None
    metric_id: Identifier
    value: float
    unit: Identifier
    scope: Literal["development"] = "development"
    window: TimeRange
    source_artifact_id: Identifier
    source_artifact_hash: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        for value in (self.metric_group, self.family, self.segment, self.metric_id):
            if value is None:
                continue
            lowered = value.casefold()
            if any(marker in lowered for marker in _FORBIDDEN_VIEW_NAMES):
                raise ValueError("development metric identity contains a forbidden field family")
        expected = content_sha256(self.model_dump(mode="json", exclude={"metric_value_id"}))
        if self.metric_value_id and self.metric_value_id != expected:
            raise ValueError("metric_value_id does not match canonical metric content")
        if not self.metric_value_id:
            object.__setattr__(self, "metric_value_id", expected)
        return self


class DevelopmentRunView(StrictModel):
    artifact_schema_version: Literal["development-run-view-v1"] = "development-run-view-v1"
    view_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    parent_run_id: Identifier
    parent_manifest_hash: Sha256
    source_mode: Literal["exploratory_standard_run", "development_only_run"]
    confirmatory_eligible: bool
    analysis_cutoff: datetime
    development_decisions: TimeRange
    universe_snapshot_id: Identifier
    universe_asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    horizon_sessions: int = Field(ge=1)
    rebalance_frequency: Identifier
    calendar_contract: ContractIdentity
    cost_assumptions_hash: Sha256
    constraint_assumptions_hash: Sha256
    metrics: tuple[DevelopmentMetric, ...] = Field(min_length=1)

    @field_validator("analysis_cutoff", mode="before")
    @classmethod
    def normalize_analysis_cutoff(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("analysis_cutoff must be timezone-aware")
            return value.astimezone(UTC)
        if isinstance(value, str):
            return parse_utc(value)
        raise ValueError("analysis_cutoff must be a datetime or ISO timestamp")

    @field_serializer("analysis_cutoff")
    def serialize_analysis_cutoff(self, value: datetime) -> str:
        return format_utc(value)

    @field_validator("universe_asset_ids")
    @classmethod
    def validate_universe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise ValueError("development-view universe_asset_ids must be unique and sorted")
        return values

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls, values: tuple[DevelopmentMetric, ...]
    ) -> tuple[DevelopmentMetric, ...]:
        keys = tuple(
            (
                value.metric_group,
                value.family or "",
                value.segment or "",
                value.metric_id,
            )
            for value in values
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("development metrics must be unique and canonically sorted")
        return values

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        if self.source_mode == "exploratory_standard_run" and self.confirmatory_eligible:
            raise ValueError("ordinary completed runs cannot produce confirmatory-eligible views")
        if self.analysis_cutoff > self.development_decisions.end:
            raise ValueError("view analysis cutoff cannot exceed development decisions")
        if any(value.window.end > self.development_decisions.end for value in self.metrics):
            raise ValueError("development metric window exceeds the development boundary")
        expected = self.computed_view_id()
        if self.view_id and self.view_id != expected:
            raise ValueError("view_id does not match canonical development-view content")
        if not self.view_id:
            object.__setattr__(self, "view_id", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"view_id"})

    def computed_view_id(self) -> str:
        return content_sha256(self.content_payload())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class LoadedDevelopmentViewBundle:
    manifest: DevelopmentViewBundleManifest
    development_view: DevelopmentRunView
    feature_catalog: FeatureCatalog
    evidence: tuple[EvidenceRecord, ...]
    evidence_index: LexicalCharNgramIndex


def export_development_view_bundle(
    artifact_root: str | Path,
    *,
    study: StudyDefinition,
    development_view: DevelopmentRunView,
    feature_catalog: FeatureCatalog,
    evidence_sources: tuple[EvidenceSourceRecord, ...],
    exporter_contract: ContractIdentity,
    git_commit: str | None,
    dirty_worktree: bool | None,
    limitations: tuple[str, ...],
) -> tuple[Path, DevelopmentViewBundleManifest]:
    """Write one content-named sealed bundle without exposing its source path in content."""

    if development_view.study_id != study.study_id:
        raise ValueError("development view does not belong to the study")
    if development_view.analysis_cutoff != study.analysis_cutoff:
        raise ValueError("development view analysis cutoff does not match the study")
    if development_view.development_decisions != study.development_decisions:
        raise ValueError("development view decision boundary does not match the study")
    if development_view.horizon_sessions != study.horizon_sessions:
        raise ValueError("development view horizon does not match the study")
    if development_view.calendar_contract != study.calendar_contract:
        raise ValueError("development view calendar contract does not match the study")
    root = ensure_agent_artifact_root(artifact_root)
    evidence = build_evidence_snapshot(
        evidence_sources,
        analysis_cutoff=study.analysis_cutoff,
    )
    snapshot_encoded = evidence_snapshot_bytes(evidence)
    snapshot_hash = hashlib.sha256(snapshot_encoded).hexdigest()
    index = build_lexical_index(evidence, evidence_snapshot_hash=snapshot_hash)
    encoded_files = {
        "development_view.json": (development_view.canonical_json() + "\n").encode("utf-8"),
        "feature_catalog.json": (feature_catalog.canonical_json() + "\n").encode("utf-8"),
        "evidence_snapshot.jsonl": snapshot_encoded,
        "evidence_index.json": (index.canonical_json() + "\n").encode("utf-8"),
    }
    schema_versions = {
        "development_view.json": development_view.artifact_schema_version,
        "feature_catalog.json": feature_catalog.artifact_schema_version,
        "evidence_snapshot.jsonl": "evidence-record-v1",
        "evidence_index.json": index.artifact_schema_version,
    }
    roles = {
        "development_view.json": "development-view",
        "feature_catalog.json": "feature-catalog",
        "evidence_snapshot.jsonl": "evidence-snapshot",
        "evidence_index.json": "evidence-index",
    }
    files = tuple(
        ArtifactEntry(
            role=roles[name],
            relative_path=name,
            sha256=hashlib.sha256(encoded).hexdigest(),
            bytes=len(encoded),
            schema_version=schema_versions[name],
        )
        for name, encoded in sorted(encoded_files.items())
    )
    manifest = DevelopmentViewBundleManifest(
        source_mode=development_view.source_mode,
        confirmatory_eligible=development_view.confirmatory_eligible,
        parent_run_id=development_view.parent_run_id,
        parent_manifest_hash=development_view.parent_manifest_hash,
        study_id=study.study_id,
        analysis_cutoff=study.analysis_cutoff,
        development_decisions=study.development_decisions,
        reserved_holdout_decisions=study.reserved_holdout_decisions,
        reserved_holdout_outcomes=study.reserved_holdout_outcomes,
        files=files,
        exporter_contract=exporter_contract,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
        included_metric_groups=tuple(
            sorted({metric.metric_group for metric in development_view.metrics})
        ),
        excluded_roles=(
            "accounts",
            "broker",
            "final-holdout",
            "orders",
            "paper",
            "positions",
            "secrets",
        ),
        excluded_field_families=(
            "environment",
            "final_holdout",
            "paths",
            "raw_private_payloads",
            "target_weights",
        ),
        evidence_snapshot_hash=snapshot_hash,
        evidence_index_identity=ContractIdentity(
            contract_id=INDEX_CONTRACT_ID,
            version=index.version,
            sha256=index.index_id,
        ),
        feature_catalog_identity=ContractIdentity(
            contract_id="research-feature-catalog",
            version=feature_catalog.artifact_schema_version,
            sha256=feature_catalog.catalog_id,
        ),
        limitations=limitations,
    )
    bundle_root = root / "views" / manifest.bundle_id
    if bundle_root.exists():
        raise AgentArtifactError("sealed development-view bundle already exists")
    bundle_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(bundle_root, 0o700)
    try:
        for name, encoded in encoded_files.items():
            write_bytes_exclusive_durable(bundle_root / name, encoded)
        write_bytes_exclusive_durable(
            bundle_root / "bundle.manifest.json",
            (canonical_json(manifest.model_dump(mode="json")) + "\n").encode("utf-8"),
        )
    except (OSError, SafeFileError, ValueError) as exc:
        raise AgentArtifactError("sealed development-view bundle could not be written") from exc
    return bundle_root, manifest


INDEX_CONTRACT_ID = "lexical-char-ngram-index"


def load_development_view_bundle(path: str | Path) -> LoadedDevelopmentViewBundle:
    """Load only fixed bundle filenames, validate exact hashes, and return no source path."""

    root = Path(path).expanduser()
    if not root.is_absolute():
        raise AgentArtifactError("development-view bundle root must be absolute")
    if root.is_symlink():
        raise AgentArtifactError("development-view bundle root cannot be a symlink")
    try:
        metadata = root.stat(follow_symlinks=False)
    except OSError as exc:
        raise AgentArtifactError("development-view bundle root is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise AgentArtifactError("development-view bundle root must be a directory")
    manifest_raw = _read_complete_file(root / "bundle.manifest.json")
    try:
        manifest = DevelopmentViewBundleManifest.model_validate_json(manifest_raw)
    except ValidationError as exc:
        raise AgentArtifactError("bundle manifest violates its strict contract") from exc
    if manifest_raw != (canonical_json(manifest.model_dump(mode="json")) + "\n").encode("utf-8"):
        raise AgentArtifactError("bundle manifest is not canonical typed JSON")
    if root.name != manifest.bundle_id:
        raise AgentArtifactError("bundle directory name does not match bundle_id")
    declared = {entry.relative_path: entry for entry in manifest.files}
    if set(declared) != _BUNDLE_FILES:
        raise AgentArtifactError("bundle manifest does not declare exactly the fixed bundle files")
    raw_files: dict[str, bytes] = {}
    for name in sorted(_BUNDLE_FILES):
        raw = _read_complete_file(root / name, allow_multiple_lines=name.endswith(".jsonl"))
        entry = declared[name]
        if len(raw) != entry.bytes or hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise AgentArtifactError(f"bundle file hash or size does not match: {name}")
        raw_files[name] = raw
    try:
        development_view = DevelopmentRunView.model_validate_json(
            raw_files["development_view.json"]
        )
        feature_catalog = FeatureCatalog.model_validate_json(raw_files["feature_catalog.json"])
        index = LexicalCharNgramIndex.model_validate_json(raw_files["evidence_index.json"])
        evidence = tuple(
            EvidenceRecord.model_validate_json(line)
            for line in raw_files["evidence_snapshot.jsonl"].splitlines()
        )
    except ValidationError as exc:
        raise AgentArtifactError("bundle file violates its strict typed contract") from exc
    _require_canonical_single_json(raw_files["development_view.json"], development_view)
    _require_canonical_single_json(raw_files["feature_catalog.json"], feature_catalog)
    _require_canonical_single_json(raw_files["evidence_index.json"], index)
    if evidence_snapshot_bytes(evidence) != raw_files["evidence_snapshot.jsonl"]:
        raise AgentArtifactError("evidence snapshot is not canonical typed JSONL")
    if (
        hashlib.sha256(raw_files["evidence_snapshot.jsonl"]).hexdigest()
        != manifest.evidence_snapshot_hash
    ):
        raise AgentArtifactError("evidence snapshot hash does not match the bundle manifest")
    if index.evidence_snapshot_hash != manifest.evidence_snapshot_hash:
        raise AgentArtifactError("evidence index references a different snapshot")
    if index.index_id != manifest.evidence_index_identity.sha256:
        raise AgentArtifactError("evidence index identity does not match the bundle manifest")
    if feature_catalog.catalog_id != manifest.feature_catalog_identity.sha256:
        raise AgentArtifactError("feature catalog identity does not match the bundle manifest")
    if development_view.study_id != manifest.study_id:
        raise AgentArtifactError("development view study identity does not match the manifest")
    return LoadedDevelopmentViewBundle(
        manifest=manifest,
        development_view=development_view,
        feature_catalog=feature_catalog,
        evidence=evidence,
        evidence_index=index,
    )


def _read_complete_file(path: Path, *, allow_multiple_lines: bool = False) -> bytes:
    try:
        raw = read_bytes_no_follow(path)
    except (FileNotFoundError, OSError, SafeFileError, ValueError) as exc:
        raise AgentArtifactError("bundle fixed file cannot be read safely") from exc
    if raw is None:  # pragma: no cover - missing_ok is false
        raise AgentArtifactError("bundle fixed file does not exist")
    if not raw.endswith(b"\n") or (not allow_multiple_lines and raw.count(b"\n") != 1):
        raise AgentArtifactError("bundle fixed file is incomplete")
    for line in raw.splitlines():
        _strict_json_object(line)
    return raw


def _strict_json_object(raw: bytes) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AgentArtifactError("bundle fixed file is not strict JSON") from exc
    if not isinstance(value, dict):
        raise AgentArtifactError("bundle fixed file must contain JSON objects")
    if canonical_json(value).encode("utf-8") != raw:
        raise AgentArtifactError("bundle fixed file contains noncanonical JSON")
    return value


def _require_canonical_single_json(raw: bytes, value: StrictModel) -> None:
    if raw != (canonical_json(value.model_dump(mode="json")) + "\n").encode("utf-8"):
        raise AgentArtifactError("bundle fixed file is not canonical typed JSON")
