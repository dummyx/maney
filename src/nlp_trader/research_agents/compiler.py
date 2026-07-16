from __future__ import annotations

import hashlib
import json
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from nlp_trader.config import ResearchConfig
from nlp_trader.immutable.append import (
    SafeFileError,
    read_bytes_no_follow,
    write_bytes_exclusive_durable,
)
from nlp_trader.research_agents.artifacts import ensure_agent_artifact_root
from nlp_trader.research_agents.contracts import (
    ArtifactEntry,
    ContractIdentity,
    ProposalVerification,
    ResearchProposal,
    Sha256,
    StrictModel,
    StudyDefinition,
    TimeRange,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_templates.matched_feature_ablation import compile_template_patch


class TypedConfigPatch(StrictModel):
    field_path: Literal["features.text_decay_half_life_days"]
    value: int | float | str | bool


class CompiledExperimentManifest(StrictModel):
    artifact_schema_version: Literal["compiled-experiment-manifest-v1"] = (
        "compiled-experiment-manifest-v1"
    )
    definition_id: Sha256
    files: tuple[ArtifactEntry, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        expected_paths = (
            "base_config.snapshot.json",
            "compiler_provenance.json",
            "execution_definition.json",
        )
        if tuple(value.relative_path for value in self.files) != expected_paths:
            raise ValueError("compiled manifest must bind the complete fixed file set")
        return self


class ExperimentExecutionDefinition(StrictModel):
    artifact_schema_version: Literal["experiment-execution-definition-v1"] = (
        "experiment-execution-definition-v1"
    )
    definition_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    definition_version: Literal["1"] = "1"
    study_id: Sha256
    proposal_hash: Sha256
    proposal_verification_hash: Sha256
    template_id: Literal["matched_feature_ablation_v1"]
    template_version: Literal["1"]
    base_config_hash: Sha256
    base_config_snapshot_hash: Sha256
    typed_patches: tuple[TypedConfigPatch, ...]
    evaluation_scope: Literal["development_only"] = "development_only"
    development_decisions: TimeRange
    reserved_decision_boundary: TimeRange
    reserved_outcome_boundary: TimeRange
    required_learned_families: tuple[str, ...]
    required_fixed_benchmarks: tuple[str, ...]
    required_negative_controls: tuple[str, ...]
    required_robustness_checks: tuple[str, ...]
    required_metrics: tuple[str, ...]
    cost_assumptions_hash: Sha256
    constraint_assumptions_hash: Sha256
    universe_snapshot_id: str
    seeds: tuple[int, ...]
    expected_artifact_schemas: tuple[str, ...]
    compiler_contract: ContractIdentity

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if not {"traditional", "text", "combined"}.issubset(self.required_learned_families):
            raise ValueError("compiled definition must retain all learned baselines")
        if not {"equal_weight", "momentum_only", "no_trade"}.issubset(
            self.required_fixed_benchmarks
        ):
            raise ValueError("compiled definition must retain all fixed benchmarks")
        if self.expected_artifact_schemas != (
            "development-backtest-comparison-v1",
            "development-result-manifest-v1",
            "frozen-development-model-v1",
        ):
            raise ValueError("compiled definition must retain the fixed artifact schemas")
        expected = content_sha256(self.model_dump(mode="json", exclude={"definition_id"}))
        if self.definition_id and self.definition_id != expected:
            raise ValueError("definition_id does not match canonical content")
        if not self.definition_id:
            object.__setattr__(self, "definition_id", expected)
        return self

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


def compile_proposal_typed_patches(
    study: StudyDefinition,
    proposal: ResearchProposal,
) -> tuple[TypedConfigPatch, ...]:
    """Deterministically map the retained proposal through its frozen template."""

    if proposal.study_id != study.study_id:
        raise ValueError("proposal does not belong to the compiled study")
    templates = {value.template_id: value for value in study.permitted_templates}
    template = templates.get(proposal.experiment_template_id)
    if template is None:
        raise ValueError("proposal template is not frozen in the study")
    mapped = compile_template_patch(template, proposal.parameter_choices)
    return tuple(
        TypedConfigPatch(field_path=field_path, value=value) for field_path, value in mapped
    )


def build_required_evaluation_contract(
    definition: ExperimentExecutionDefinition,
) -> dict[str, object]:
    """Derive the immutable downstream-evaluation contract from a compiled definition."""

    return {
        "artifact_schema_version": "required-evaluation-contract-v1",
        "required_learned_families": tuple(sorted(definition.required_learned_families)),
        "required_fixed_benchmarks": tuple(sorted(definition.required_fixed_benchmarks)),
        "required_negative_controls": tuple(sorted(definition.required_negative_controls)),
        "required_robustness_checks": tuple(sorted(definition.required_robustness_checks)),
        "development_execution_coverage": {
            "learned_and_fixed_families": "verified_in_backtest_comparison",
            "negative_controls": "predeclared_not_executed",
            "robustness_checks": "predeclared_not_executed",
        },
    }


def compile_verified_proposal(
    *,
    artifact_root: str | Path,
    study: StudyDefinition,
    proposal: ResearchProposal,
    proposal_artifact_hash: str,
    verification: ProposalVerification,
    verification_artifact_hash: str,
    base_config: ResearchConfig,
    compiler_contract: ContractIdentity,
) -> tuple[Path, ExperimentExecutionDefinition]:
    if not verification.passed:
        raise ValueError("only a deterministically verified proposal can be compiled")
    if (
        proposal.study_id != study.study_id
        or verification.study_id != study.study_id
        or verification.attempt_id != proposal.attempt_id
        or verification.terminal_artifact_hash != proposal_artifact_hash
        or proposal.proposal_id
        != content_sha256(proposal.model_dump(mode="json", exclude={"proposal_id"}))
    ):
        raise ValueError("proposal, verification, and study identities do not match")
    typed_patches = compile_proposal_typed_patches(study, proposal)
    if not set(study.required_learned_families).issubset(base_config.models.families):
        raise ValueError("base config does not retain the study's learned families")
    snapshot = base_config.model_dump(mode="json", exclude={"path"})
    snapshot_hash = content_sha256(snapshot)
    if snapshot_hash != base_config.content_hash():
        raise ValueError("base config hash is not canonical")
    backtest_payload = base_config.backtest.model_dump(mode="json")
    definition = ExperimentExecutionDefinition(
        study_id=study.study_id,
        proposal_hash=proposal_artifact_hash,
        proposal_verification_hash=verification_artifact_hash,
        template_id="matched_feature_ablation_v1",
        template_version="1",
        base_config_hash=base_config.content_hash(),
        base_config_snapshot_hash=snapshot_hash,
        typed_patches=typed_patches,
        development_decisions=study.development_decisions,
        reserved_decision_boundary=study.reserved_holdout_decisions,
        reserved_outcome_boundary=study.reserved_holdout_outcomes,
        required_learned_families=study.required_learned_families,
        required_fixed_benchmarks=study.required_fixed_benchmarks,
        required_negative_controls=study.required_negative_controls,
        required_robustness_checks=study.required_robustness_checks,
        required_metrics=study.required_metrics,
        cost_assumptions_hash=content_sha256(backtest_payload),
        constraint_assumptions_hash=content_sha256(backtest_payload),
        universe_snapshot_id=study.universe_snapshot_id,
        seeds=study.seeds,
        expected_artifact_schemas=(
            "development-backtest-comparison-v1",
            "development-result-manifest-v1",
            "frozen-development-model-v1",
        ),
        compiler_contract=compiler_contract,
    )
    root = ensure_agent_artifact_root(artifact_root)
    destination = root / "compiled" / definition.definition_id
    try:
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise ValueError("compiled definition directory cannot be created exclusively") from exc
    encoded_definition = (definition.canonical_json() + "\n").encode("utf-8")
    encoded_config = (canonical_json(snapshot) + "\n").encode("utf-8")
    provenance = {
        "artifact_schema_version": "research-agent-compiler-provenance-v1",
        "compiler_contract": compiler_contract.model_dump(mode="json"),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "behavior": "inert_typed_mapping_only",
    }
    encoded_provenance = (canonical_json(provenance) + "\n").encode("utf-8")
    files = {
        "execution_definition.json": encoded_definition,
        "base_config.snapshot.json": encoded_config,
        "compiler_provenance.json": encoded_provenance,
    }
    manifest_payload = {
        "artifact_schema_version": "compiled-experiment-manifest-v1",
        "definition_id": definition.definition_id,
        "files": [
            ArtifactEntry(
                role=name.removesuffix(".json"),
                relative_path=name,
                sha256=hashlib.sha256(encoded).hexdigest(),
                bytes=len(encoded),
                schema_version=(
                    definition.artifact_schema_version
                    if name == "execution_definition.json"
                    else "canonical-json-v1"
                ),
            ).model_dump(mode="json")
            for name, encoded in sorted(files.items())
        ],
    }
    files["manifest.json"] = (canonical_json(manifest_payload) + "\n").encode("utf-8")
    try:
        for name, encoded in files.items():
            write_bytes_exclusive_durable(destination / name, encoded)
    except (FileExistsError, SafeFileError, OSError, ValueError) as exc:
        raise ValueError("compiled definition cannot be persisted immutably") from exc
    return destination, definition


def load_compiled_execution_definition(
    artifact_root: str | Path,
    definition_id: str,
) -> ExperimentExecutionDefinition:
    """Load one complete, canonical compiler output from its fixed content-addressed path."""

    if len(definition_id) != 64 or any(value not in "0123456789abcdef" for value in definition_id):
        raise ValueError("compiled definition ID must be a lowercase SHA-256 digest")
    root = ensure_agent_artifact_root(artifact_root)
    compiled_root = root / "compiled"
    definition_root = compiled_root / definition_id
    _require_nonsymlink_directory(compiled_root, label="compiled artifact root")
    _require_nonsymlink_directory(definition_root, label="compiled definition directory")
    expected_names = {
        "base_config.snapshot.json",
        "compiler_provenance.json",
        "execution_definition.json",
        "manifest.json",
    }
    try:
        actual_names = {value.name for value in definition_root.iterdir()}
    except OSError as exc:
        raise ValueError("compiled definition directory cannot be inspected") from exc
    if actual_names != expected_names:
        raise ValueError("compiled definition directory does not contain the exact fixed file set")

    manifest_raw, _ = _read_canonical_artifact_object(
        definition_root / "manifest.json", label="compiled manifest"
    )
    try:
        manifest = CompiledExperimentManifest.model_validate_json(manifest_raw)
    except ValueError as exc:
        raise ValueError("compiled manifest violates its strict contract") from exc
    if manifest_raw != (canonical_json(manifest.model_dump(mode="json")) + "\n").encode("utf-8"):
        raise ValueError("compiled manifest does not match canonical typed content")
    if manifest.definition_id != definition_id:
        raise ValueError("compiled manifest references a different execution definition")

    encoded_files: dict[str, bytes] = {}
    entries = {value.relative_path: value for value in manifest.files}
    for relative_path in sorted(entries):
        raw, _ = _read_canonical_artifact_object(
            definition_root / relative_path,
            label=f"compiled {entries[relative_path].role}",
        )
        entry = entries[relative_path]
        if entry.bytes != len(raw) or entry.sha256 != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"compiled manifest hash mismatch for {relative_path}")
        encoded_files[relative_path] = raw

    try:
        definition = ExperimentExecutionDefinition.model_validate_json(
            encoded_files["execution_definition.json"]
        )
    except ValueError as exc:
        raise ValueError("compiled execution definition violates its strict contract") from exc
    expected_definition_bytes = (definition.canonical_json() + "\n").encode("utf-8")
    if encoded_files["execution_definition.json"] != expected_definition_bytes:
        raise ValueError("compiled execution definition is not canonical typed JSON")
    if definition.definition_id != definition_id:
        raise ValueError("compiled execution definition identity does not match its fixed path")
    execution_entry = entries["execution_definition.json"]
    if (
        execution_entry.role != "execution_definition"
        or execution_entry.schema_version != definition.artifact_schema_version
    ):
        raise ValueError("compiled manifest execution-definition role is invalid")

    base_snapshot = _canonical_payload(
        encoded_files["base_config.snapshot.json"], label="compiled base config snapshot"
    )
    if (
        content_sha256(base_snapshot) != definition.base_config_snapshot_hash
        or definition.base_config_snapshot_hash != definition.base_config_hash
    ):
        raise ValueError("compiled base config snapshot does not match the execution definition")
    try:
        snapshot_config = ResearchConfig.model_validate(
            {"path": definition_root / "base_config.snapshot.json", **base_snapshot}
        )
    except ValueError as exc:
        raise ValueError("compiled base config snapshot violates its strict contract") from exc
    if snapshot_config.model_dump(mode="json", exclude={"path"}) != base_snapshot:
        raise ValueError("compiled base config snapshot is not canonical typed config content")
    patched_snapshot = apply_definition_patch(definition, snapshot_config)
    backtest_contract_hash = content_sha256(patched_snapshot.backtest.model_dump(mode="json"))
    if (
        definition.cost_assumptions_hash != backtest_contract_hash
        or definition.constraint_assumptions_hash != backtest_contract_hash
    ):
        raise ValueError("compiled cost or constraint hashes do not match the base snapshot")
    base_entry = entries["base_config.snapshot.json"]
    if (
        base_entry.role != "base_config.snapshot"
        or base_entry.schema_version != "canonical-json-v1"
    ):
        raise ValueError("compiled manifest base-config role is invalid")

    provenance = _canonical_payload(
        encoded_files["compiler_provenance.json"], label="compiled provenance"
    )
    if (
        set(provenance)
        != {"artifact_schema_version", "behavior", "compiler_contract", "created_at"}
        or provenance.get("artifact_schema_version") != "research-agent-compiler-provenance-v1"
        or provenance.get("behavior") != "inert_typed_mapping_only"
        or provenance.get("compiler_contract")
        != definition.compiler_contract.model_dump(mode="json")
    ):
        raise ValueError("compiled provenance does not match the execution definition")
    provenance_entry = entries["compiler_provenance.json"]
    if (
        provenance_entry.role != "compiler_provenance"
        or provenance_entry.schema_version != "canonical-json-v1"
    ):
        raise ValueError("compiled manifest provenance role is invalid")
    return definition


def load_compiler_proposal(
    artifact_root: str | Path,
    *,
    agent_run_id: str,
    expected_sha256: str,
) -> ResearchProposal:
    """Load the exact retained terminal proposal used as deterministic compiler input."""

    if (
        not agent_run_id
        or agent_run_id in {".", ".."}
        or "/" in agent_run_id
        or "\\" in agent_run_id
    ):
        raise ValueError("proposal agent run ID is not a safe fixed path component")
    if len(expected_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in expected_sha256
    ):
        raise ValueError("proposal artifact hash must be a lowercase SHA-256 digest")
    root = ensure_agent_artifact_root(artifact_root)
    runs_root = root / "runs"
    run_root = runs_root / agent_run_id
    reports_root = run_root / "reports"
    _require_nonsymlink_directory(runs_root, label="proposal runs root")
    _require_nonsymlink_directory(run_root, label="proposal run directory")
    _require_nonsymlink_directory(reports_root, label="proposal reports directory")
    raw, _ = _read_canonical_artifact_object(
        reports_root / "proposal.json",
        label="retained terminal proposal",
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("retained terminal proposal bytes do not match the registry hash")
    try:
        proposal = ResearchProposal.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("retained terminal proposal violates its strict contract") from exc
    if raw != (proposal.canonical_json() + "\n").encode("utf-8"):
        raise ValueError("retained terminal proposal is not canonical typed JSON")
    return proposal


def apply_definition_patch(
    definition: ExperimentExecutionDefinition,
    base_config: ResearchConfig,
) -> ResearchConfig:
    if base_config.content_hash() != definition.base_config_hash:
        raise ValueError("execution base config hash does not match the compiled definition")
    payload: dict[str, Any] = base_config.model_dump(mode="python")
    for patch in definition.typed_patches:
        if patch.field_path != "features.text_decay_half_life_days":
            raise ValueError("compiled definition contains an unknown patch path")
        feature_payload = dict(payload["features"])
        feature_payload["text_decay_half_life_days"] = patch.value
        payload["features"] = feature_payload
    return ResearchConfig.model_validate(payload)


def _require_nonsymlink_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory")


def _read_canonical_artifact_object(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    try:
        raw = read_bytes_no_follow(path)
    except (FileNotFoundError, OSError, SafeFileError, ValueError) as exc:
        raise ValueError(f"{label} cannot be read safely") from exc
    if raw is None:  # pragma: no cover - missing_ok is false
        raise ValueError(f"{label} does not exist")
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{label} must contain one complete canonical JSON line")
    payload = _canonical_payload(raw, label=label)
    if raw != (canonical_json(payload) + "\n").encode("utf-8"):
        raise ValueError(f"{label} is not canonical JSON")
    return raw, payload


def _canonical_payload(raw: bytes, *, label: str) -> dict[str, object]:
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
        payload = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object")
    return {str(key): value for key, value in payload.items()}
