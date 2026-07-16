from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from nlp_trader.config import ResearchConfig
from nlp_trader.immutable.append import SafeFileError, write_bytes_exclusive_durable
from nlp_trader.pipeline import PipelineExecutionScope, run_to_stage
from nlp_trader.research import sha256_file
from nlp_trader.research_agents.approvals import (
    DevelopmentExecutionApproval,
    load_authoritative_development_approval,
)
from nlp_trader.research_agents.compiler import (
    ExperimentExecutionDefinition,
    apply_definition_patch,
    build_required_evaluation_contract,
    load_compiled_execution_definition,
)
from nlp_trader.research_agents.contracts import (
    Sha256,
    StrictModel,
    StudyDefinition,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_agents.registry import ResearchRegistryLedger


class FrozenDevelopmentModelManifest(StrictModel):
    artifact_schema_version: Literal["frozen-development-model-v1"] = "frozen-development-model-v1"
    manifest_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    development_run_id: Sha256
    execution_definition_hash: Sha256
    pipeline_result_manifest_hash: Sha256
    model_artifact_hash: Sha256
    model_artifact_bytes: int = Field(ge=1)
    training_rule: Literal["no_updates_after_candidate_freeze"] = (
        "no_updates_after_candidate_freeze"
    )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = content_sha256(self.model_dump(mode="json", exclude={"manifest_id"}))
        if self.manifest_id and self.manifest_id != expected:
            raise ValueError("frozen model manifest_id does not match canonical content")
        if not self.manifest_id:
            object.__setattr__(self, "manifest_id", expected)
        return self


class DevelopmentResultManifest(StrictModel):
    artifact_schema_version: Literal["development-result-manifest-v1"] = (
        "development-result-manifest-v1"
    )
    manifest_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    development_run_id: Sha256
    execution_definition_hash: Sha256
    approval_event_hash: Sha256
    base_config_hash: Sha256
    patched_config_hash: Sha256
    execution_scope: Literal["development_only"] = "development_only"
    pipeline_run_id: Sha256
    pipeline_result_manifest_hash: Sha256
    frozen_model_manifest_hash: Sha256
    backtest_comparison_hash: Sha256
    prediction_metrics_hash: Sha256
    required_evaluation_contract_hash: Sha256
    cost_assumptions_hash: Sha256
    constraint_assumptions_hash: Sha256
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = content_sha256(self.model_dump(mode="json", exclude={"manifest_id"}))
        if self.manifest_id and self.manifest_id != expected:
            raise ValueError("development result manifest_id does not match canonical content")
        if not self.manifest_id:
            object.__setattr__(self, "manifest_id", expected)
        return self


def run_approved_development(
    *,
    ledger: ResearchRegistryLedger,
    study: StudyDefinition,
    definition: ExperimentExecutionDefinition,
    approval: DevelopmentExecutionApproval,
    base_config: ResearchConfig,
    actor_label: str = "development-runner",
) -> tuple[Path, DevelopmentResultManifest, FrozenDevelopmentModelManifest]:
    authoritative_approval = load_authoritative_development_approval(ledger, approval)
    compiled = load_compiled_execution_definition(
        ledger.artifact_root,
        definition.definition_id,
    )
    if (
        authoritative_approval != approval
        or compiled != definition
        or approval.study_id != study.study_id
        or definition.study_id != study.study_id
        or approval.execution_definition_hash != definition.definition_id
        or approval.proposal_verification_hash != definition.proposal_verification_hash
        or base_config.content_hash() != definition.base_config_hash
    ):
        raise ValueError("development execution inputs do not match the exact approval")
    start_event = ledger.start_development_run(
        study.study_id,
        execution_definition_hash=definition.definition_id,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor_label,
    )
    payload = start_event.payload
    from nlp_trader.research_agents.contracts import DevelopmentRunStartedPayload

    if not isinstance(payload, DevelopmentRunStartedPayload):  # pragma: no cover - ledger invariant
        raise RuntimeError("registry returned an unexpected development-run event")
    development_run_id = payload.development_run_id
    destination = ledger.artifact_root / "development_runs" / development_run_id
    try:
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        ledger.fail_development_run(
            study.study_id,
            development_run_id,
            failure_type="artifact_initialization",
            detail="development artifact root could not be initialized",
            expected_head_hash=ledger.head_hash(),
            actor_label=actor_label,
        )
        raise ValueError("development artifact root cannot be initialized") from exc
    try:
        patched_config = apply_definition_patch(definition, base_config)
        outputs = run_to_stage(
            patched_config,
            "backtest",
            execution_scope=PipelineExecutionScope(
                mode="development_only",
                decision_start=definition.development_decisions.start,
                decision_end=definition.development_decisions.end,
                development_information_cutoff=definition.reserved_decision_boundary.start,
            ),
            run_id=development_run_id,
        )
        final_path = Path(str(outputs["final_manifest"]))
        model_path = Path(str(outputs["model"]))
        comparison_path = Path(str(outputs["backtest_comparison"]))
        metrics_path = Path(str(outputs["model_evaluation"]))
        _reject_forbidden_development_outputs(final_path, patched_config, development_run_id)
        pipeline_manifest_hash = sha256_file(final_path)
        model_bytes = model_path.read_bytes()
        frozen_model_path = destination / "frozen_model.json"
        _write_bytes(frozen_model_path, model_bytes)
        frozen_manifest = FrozenDevelopmentModelManifest(
            study_id=study.study_id,
            development_run_id=development_run_id,
            execution_definition_hash=definition.definition_id,
            pipeline_result_manifest_hash=pipeline_manifest_hash,
            model_artifact_hash=hashlib.sha256(model_bytes).hexdigest(),
            model_artifact_bytes=len(model_bytes),
        )
        frozen_manifest_bytes = _write_json(
            destination / "frozen_model.manifest.json",
            frozen_manifest.model_dump(mode="json"),
        )
        frozen_manifest_hash = hashlib.sha256(frozen_manifest_bytes).hexdigest()
        comparison = _read_object(comparison_path)
        family_ids = tuple(sorted(str(value) for value in comparison["families"]))
        required_evaluation_contract = build_required_evaluation_contract(definition)
        required_family_ids = set(definition.required_learned_families).union(
            definition.required_fixed_benchmarks
        )
        if not required_family_ids.issubset(family_ids):
            raise ValueError("development result omitted a required family or benchmark")
        required_evaluation_contract_hash = content_sha256(required_evaluation_contract)
        _write_json(
            destination / "required_evaluation_contract.json",
            required_evaluation_contract,
        )
        result = DevelopmentResultManifest(
            study_id=study.study_id,
            development_run_id=development_run_id,
            execution_definition_hash=definition.definition_id,
            approval_event_hash=approval.approval_event_hash,
            base_config_hash=definition.base_config_hash,
            patched_config_hash=patched_config.content_hash(),
            pipeline_run_id=development_run_id,
            pipeline_result_manifest_hash=pipeline_manifest_hash,
            frozen_model_manifest_hash=frozen_manifest_hash,
            backtest_comparison_hash=sha256_file(comparison_path),
            prediction_metrics_hash=sha256_file(metrics_path),
            required_evaluation_contract_hash=required_evaluation_contract_hash,
            cost_assumptions_hash=definition.cost_assumptions_hash,
            constraint_assumptions_hash=definition.constraint_assumptions_hash,
            limitations=(
                "Development-only output is hypothetical research and contains no reserved result.",
                "Predeclared negative-control and robustness scenarios are not executed here.",
                "Candidate promotion still requires explicit human freeze and one-time evaluation.",
            ),
        )
        result_bytes = _write_json(
            destination / "result_manifest.json", result.model_dump(mode="json")
        )
        result_hash = hashlib.sha256(result_bytes).hexdigest()
        _write_json(
            destination / "linkage.json",
            {
                "artifact_schema_version": "development-run-linkage-v1",
                "development_run_id": development_run_id,
                "start_event_hash": start_event.event_hash,
                "approval_id": approval.approval_id,
                "execution_definition_hash": definition.definition_id,
                "pipeline_result_manifest_hash": pipeline_manifest_hash,
            },
        )
        ledger.complete_development_run(
            study.study_id,
            development_run_id,
            result_manifest_hash=result_hash,
            frozen_model_manifest_hash=frozen_manifest_hash,
            expected_head_hash=ledger.head_hash(),
            actor_label=actor_label,
        )
        return destination, result, frozen_manifest
    except Exception as exc:
        state = ledger.project().studies[study.study_id]
        last_started = any(
            value == start_event.event_hash for value in state.transition_event_hashes
        )
        if last_started:
            with suppress(ValueError):
                ledger.fail_development_run(
                    study.study_id,
                    development_run_id,
                    failure_type=type(exc).__name__,
                    detail="approved development execution failed before immutable completion",
                    expected_head_hash=ledger.head_hash(),
                    actor_label=actor_label,
                )
        raise


def _reject_forbidden_development_outputs(
    final_path: Path,
    config: ResearchConfig,
    run_id: str,
) -> None:
    roots = (
        config.paths.interim_dir / run_id,
        config.paths.processed_dir / run_id,
        config.paths.models_dir / run_id,
        config.paths.reports_dir / run_id,
    )
    if final_path != roots[-1] / "run.final.json":
        raise ValueError("development pipeline final manifest is outside its exact run root")
    snapshot_path = roots[-1] / "config.snapshot.json"
    snapshot = _read_object(snapshot_path)
    snapshot_hash = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if snapshot_hash != config.content_hash():
        raise ValueError("development config snapshot does not match its reported config hash")
    for root in roots:
        for path in root.rglob("*"):
            if "final_holdout" in path.name.casefold():
                raise ValueError("development-only execution emitted a reserved-result artifact")
            if path.suffix == ".json" and path.is_file() and path != snapshot_path:
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("development JSON artifact cannot be inspected") from exc
                if _contains_forbidden_key(value):
                    raise ValueError("development-only JSON contains a reserved-result key")


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "final_holdout" in str(key).casefold() or _contains_forbidden_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(nested) for nested in value)
    return False


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("development JSON artifact cannot be read strictly") from exc
    if not isinstance(value, dict):
        raise ValueError("development JSON artifact must contain an object")
    return {str(key): nested for key, nested in value.items()}


def _write_json(path: Path, value: object) -> bytes:
    return _write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def _write_bytes(path: Path, encoded: bytes) -> bytes:
    try:
        write_bytes_exclusive_durable(path, encoded)
    except (FileExistsError, SafeFileError, OSError, ValueError) as exc:
        raise ValueError("development artifact cannot be written immutably") from exc
    return encoded
