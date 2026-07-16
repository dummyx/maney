from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from nlp_trader.config import ResearchConfig
from nlp_trader.research import RunPaths, artifact_manifest, sha256_file
from nlp_trader.research_agents.approvals import CandidateFreezeRecord
from nlp_trader.research_agents.authority import load_authoritative_candidate_freeze
from nlp_trader.research_agents.contracts import Sha256, StrictModel, content_sha256
from nlp_trader.research_agents.registry import ResearchRegistryLedger


class AuditFinding(StrictModel):
    check_id: str = Field(min_length=1, max_length=256)
    severity: Literal["critical", "high", "medium", "low", "info"]
    passed: bool
    detail: str = Field(min_length=1, max_length=4096)
    artifact_hashes: tuple[Sha256, ...] = ()


class DeterministicAuditReport(StrictModel):
    artifact_schema_version: Literal["research-deterministic-audit-v1"] = (
        "research-deterministic-audit-v1"
    )
    report_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    registry_head_hash: Sha256
    candidate_freeze_id: Sha256
    findings: tuple[AuditFinding, ...] = Field(min_length=1)
    passed: bool

    @field_validator("findings")
    @classmethod
    def unique_checks(cls, values: tuple[AuditFinding, ...]) -> tuple[AuditFinding, ...]:
        ids = tuple(value.check_id for value in values)
        if len(ids) != len(set(ids)):
            raise ValueError("audit finding check IDs must be unique")
        return values

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.passed != all(value.passed for value in self.findings):
            raise ValueError("audit passed must equal all deterministic findings")
        expected = content_sha256(self.model_dump(mode="json", exclude={"report_id"}))
        if self.report_id and self.report_id != expected:
            raise ValueError("audit report_id does not match canonical content")
        if not self.report_id:
            object.__setattr__(self, "report_id", expected)
        return self


class ProposalQualityScore(StrictModel):
    artifact_schema_version: Literal["proposal-quality-score-v1"] = "proposal-quality-score-v1"
    blinded_label: str = Field(min_length=1, max_length=64)
    rubric_version: Literal["blinded-proposal-rubric-v1"] = "blinded-proposal-rubric-v1"
    hypothesis_testability: int = Field(ge=0, le=4)
    evidence_grounding: int = Field(ge=0, le=4)
    counterevidence_quality: int = Field(ge=0, le=4)
    falsification_quality: int = Field(ge=0, le=4)
    control_completeness: int = Field(ge=0, le=4)
    point_in_time_safety: int = Field(ge=0, le=4)
    total: int = Field(default=0, ge=0, le=24)

    @model_validator(mode="after")
    def calculate_total(self) -> Self:
        expected = sum(
            (
                self.hypothesis_testability,
                self.evidence_grounding,
                self.counterevidence_quality,
                self.falsification_quality,
                self.control_completeness,
                self.point_in_time_safety,
            )
        )
        if self.total not in {0, expected}:
            raise ValueError("proposal quality total does not match the frozen rubric")
        if self.total == 0 and expected:
            object.__setattr__(self, "total", expected)
        return self


class ProposalQualityComparison(StrictModel):
    artifact_schema_version: Literal["proposal-quality-comparison-v1"] = (
        "proposal-quality-comparison-v1"
    )
    analyst: ProposalQualityScore
    fixed_template_control: ProposalQualityScore
    total_difference: int
    interpretation: Literal[
        "blinded rubric difference only; acceptance rate is not an evaluation metric"
    ] = "blinded rubric difference only; acceptance rate is not an evaluation metric"

    @model_validator(mode="after")
    def validate_difference(self) -> Self:
        if self.total_difference != self.analyst.total - self.fixed_template_control.total:
            raise ValueError("proposal quality difference is inconsistent")
        return self


def compare_proposal_quality(
    analyst: ProposalQualityScore,
    fixed_template_control: ProposalQualityScore,
) -> ProposalQualityComparison:
    if analyst.blinded_label == fixed_template_control.blinded_label:
        raise ValueError("blinded proposal labels must be distinct")
    return ProposalQualityComparison(
        analyst=analyst,
        fixed_template_control=fixed_template_control,
        total_difference=analyst.total - fixed_template_control.total,
    )


def audit_completed_holdout(
    *,
    ledger: ResearchRegistryLedger,
    freeze: CandidateFreezeRecord,
    development_root: Path,
    holdout_root: Path,
    holdout_pipeline_roots: tuple[Path, Path, Path, Path],
    development_pipeline_roots: tuple[Path, Path, Path, Path],
) -> DeterministicAuditReport:
    from nlp_trader.holdout_execution import HoldoutResultManifest
    from nlp_trader.research_agents.compiler import load_compiled_execution_definition

    authority = load_authoritative_candidate_freeze(
        ledger,
        freeze,
        development_root=development_root,
    )
    freeze = authority.record
    development_root = authority.development_root

    holdout_manifest_path = holdout_root / "result_manifest.json"
    holdout = HoldoutResultManifest.model_validate_json(
        holdout_manifest_path.read_text(encoding="utf-8")
    )
    prediction_results_hash = _sha256_artifact(
        holdout_root / "predictions.json", label="holdout predictions"
    )
    backtest_results_hash = _sha256_artifact(
        holdout_root / "backtests.json", label="holdout backtests"
    )
    metrics_hash = _sha256_artifact(holdout_root / "metrics.json", label="holdout metrics")
    holdout_pipeline_paths = RunPaths(
        interim=holdout_pipeline_roots[0],
        processed=holdout_pipeline_roots[1],
        models=holdout_pipeline_roots[2],
        reports=holdout_pipeline_roots[3],
    )
    pipeline_final_path = holdout_pipeline_paths.reports / "run.final.json"
    pipeline_final_hash = _sha256_artifact(
        pipeline_final_path,
        label="holdout pipeline final manifest",
    )
    pipeline_final = _read_json_object(
        pipeline_final_path,
        label="holdout pipeline final manifest",
    )
    current_pipeline_artifacts = artifact_manifest(holdout_pipeline_paths)
    development_pipeline_paths = RunPaths(
        interim=development_pipeline_roots[0],
        processed=development_pipeline_roots[1],
        models=development_pipeline_roots[2],
        reports=development_pipeline_roots[3],
    )
    development_pipeline_final_path = development_pipeline_paths.reports / "run.final.json"
    development_pipeline_final_hash = _sha256_artifact(
        development_pipeline_final_path,
        label="development pipeline final manifest",
    )
    development_pipeline_final = _read_json_object(
        development_pipeline_final_path,
        label="development pipeline final manifest",
    )
    current_development_pipeline_artifacts = artifact_manifest(development_pipeline_paths)
    development_config = _load_pipeline_config(
        development_pipeline_paths.reports / "config.snapshot.json"
    )
    authoritative_development_roots = (
        development_config.paths.interim_dir / authority.development_run_id,
        development_config.paths.processed_dir / authority.development_run_id,
        development_config.paths.models_dir / authority.development_run_id,
        development_config.paths.reports_dir / authority.development_run_id,
    )
    events = ledger.replay()
    reserved_sequence = next(
        (event.sequence for event in events if event.event_hash == holdout.reservation_event_hash),
        None,
    )
    revealed_event = next(
        (
            event
            for event in events
            if event.study_id == freeze.study_id and event.payload.kind == "holdout_revealed"
        ),
        None,
    )
    state = ledger.project().studies[freeze.study_id]
    used = ledger.project().holdout_use.overlapping(freeze.holdout_identity)
    model_path = development_root / "frozen_model.json"
    model_hash = sha256_file(model_path)
    development_forbidden = _find_forbidden_development_artifact(development_pipeline_roots)
    analyst_leak = _find_analyst_holdout_leak(
        ledger.artifact_root / "runs",
        holdout.holdout_identity.holdout_id,
        holdout.manifest_id,
    )
    definition = load_compiled_execution_definition(
        ledger.artifact_root,
        freeze.execution_definition_hash,
    )
    attempted_proposals_hash = content_sha256(
        [value.model_dump(mode="json") for value in state.attempts]
    )
    attempts_complete = all(value.status == "completed" for value in state.attempts)
    proposals_retained = all(
        value.outcome != "proposal" or value.terminal_artifact_hash is not None
        for value in state.attempts
    )
    backtests = _read_json_object(holdout_root / "backtests.json", label="holdout backtests")
    families = backtests.get("families")
    family_ids = set(families) if isinstance(families, dict) else set()
    required_families = set(definition.required_learned_families).union(
        definition.required_fixed_benchmarks
    )
    study = ledger.study_definition(freeze.study_id)
    findings = (
        AuditFinding(
            check_id="registry_state_revealed",
            severity="critical",
            passed=state.state == "holdout_revealed",
            detail="Registry state must be holdout_revealed after immutable completion.",
        ),
        AuditFinding(
            check_id="reservation_precedes_result",
            severity="critical",
            passed=reserved_sequence is not None
            and revealed_event is not None
            and reserved_sequence < revealed_event.sequence,
            detail="Reveal reservation must precede the result event.",
        ),
        AuditFinding(
            check_id="global_holdout_use_recorded",
            severity="critical",
            passed=len(used) == 1
            and used[0].source == "reveal_reservation"
            and used[0].registry_event_hash == holdout.reservation_event_hash,
            detail=(
                "Exactly one overlapping global use record must bind the reveal reservation "
                "that first contaminated the holdout."
            ),
        ),
        AuditFinding(
            check_id="frozen_model_unchanged",
            severity="critical",
            passed=holdout.training_updates == 0
            and holdout.frozen_model_hash_before == holdout.frozen_model_hash_after == model_hash,
            detail="Frozen model bytes must remain unchanged and training updates must be zero.",
            artifact_hashes=(model_hash,),
        ),
        AuditFinding(
            check_id="candidate_lineage_exact",
            severity="high",
            passed=holdout.candidate_hash == freeze.candidate_config_hash
            and holdout.frozen_model_manifest_hash == freeze.frozen_model_manifest_hash,
            detail="Holdout result must bind the exact frozen candidate and model manifest.",
        ),
        AuditFinding(
            check_id="execution_contracts_unchanged",
            severity="critical",
            passed=holdout.execution_definition_hash
            == definition.definition_id
            == freeze.execution_definition_hash
            and holdout.required_evaluation_contract_hash
            == freeze.required_evaluation_contract_hash
            and holdout.cost_assumptions_hash == definition.cost_assumptions_hash
            and holdout.constraint_assumptions_hash == definition.constraint_assumptions_hash,
            detail=(
                "The exact compiled evaluation declarations, costs, fills, and constraints must "
                "remain bound."
            ),
        ),
        AuditFinding(
            check_id="required_baselines_present",
            severity="high",
            passed=required_families.issubset(family_ids),
            detail="Every required learned family and fixed benchmark must be evaluated.",
        ),
        AuditFinding(
            check_id="holdout_result_artifact_hashes_bound",
            severity="critical",
            passed=prediction_results_hash == holdout.prediction_results_hash
            and backtest_results_hash == holdout.backtest_results_hash
            and metrics_hash == holdout.metrics_hash,
            detail=(
                "Predictions, backtests, and metrics bytes must match the immutable holdout "
                "result manifest."
            ),
            artifact_hashes=(prediction_results_hash, backtest_results_hash, metrics_hash),
        ),
        AuditFinding(
            check_id="holdout_pipeline_run_bound",
            severity="critical",
            passed=pipeline_final_hash == holdout.pipeline_run_final_manifest_hash
            and pipeline_final.get("run_id") == holdout.reservation_id
            and pipeline_final.get("status") == "complete"
            and pipeline_final.get("completed_stage") == "holdout_evaluation"
            and pipeline_final.get("config_hash") == holdout.patched_config_hash
            and pipeline_final.get("artifact_manifest") == current_pipeline_artifacts,
            detail=(
                "The canonical completed pipeline manifest and every listed run artifact must "
                "remain bound to the holdout result."
            ),
            artifact_hashes=(pipeline_final_hash,),
        ),
        AuditFinding(
            check_id="development_pipeline_run_bound",
            severity="critical",
            passed=development_pipeline_roots == authoritative_development_roots
            and development_pipeline_final_hash
            == authority.development_result.pipeline_result_manifest_hash
            and development_pipeline_final.get("run_id") == authority.development_run_id
            and development_pipeline_final.get("status") == "complete"
            and development_pipeline_final.get("completed_stage") == "backtest"
            and development_pipeline_final.get("config_hash")
            == authority.development_result.patched_config_hash
            == development_config.content_hash()
            and development_pipeline_final.get("artifact_manifest")
            == current_development_pipeline_artifacts,
            detail=(
                "The canonical completed development pipeline manifest and every listed run "
                "artifact must remain bound to the frozen development result."
            ),
            artifact_hashes=(development_pipeline_final_hash,),
        ),
        AuditFinding(
            check_id="all_attempts_retained",
            severity="high",
            passed=attempts_complete
            and proposals_retained
            and holdout.attempted_proposal_count == len(state.attempts)
            and holdout.attempted_proposals_hash == attempted_proposals_hash,
            detail=(
                "Every consumed proposal attempt and terminal proposal artifact must be retained."
            ),
        ),
        AuditFinding(
            check_id="holdout_contract_exact",
            severity="critical",
            passed=holdout.holdout_identity == freeze.holdout_identity
            and holdout.holdout_identity.decision_interval == study.reserved_holdout_decisions
            and holdout.holdout_identity.outcome_interval == study.reserved_holdout_outcomes
            and holdout.holdout_identity.label_contract == study.label_contract
            and holdout.holdout_identity.target_family == study.target_family
            and holdout.holdout_identity.horizon_sessions == study.horizon_sessions
            and holdout.holdout_identity.universe_snapshot_id == study.universe_snapshot_id,
            detail="Universe, label, horizon, and reserved intervals must match the frozen study.",
        ),
        AuditFinding(
            check_id="development_reserved_result_absent",
            severity="critical",
            passed=development_forbidden is None,
            detail=(
                "Development pipeline contains no reserved-result key or artifact."
                if development_forbidden is None
                else f"Forbidden development artifact detected: {development_forbidden}"
            ),
        ),
        AuditFinding(
            check_id="analyst_holdout_result_unreachable",
            severity="critical",
            passed=analyst_leak is None,
            detail=(
                "No analyst artifact contains the revealed holdout identity or result."
                if analyst_leak is None
                else f"Analyst artifact contains revealed output: {analyst_leak}"
            ),
        ),
        AuditFinding(
            check_id="holdout_manifest_file_hash_bound",
            severity="high",
            passed=revealed_event is not None
            and getattr(revealed_event.payload, "result_manifest_hash", None)
            == sha256_file(holdout_manifest_path),
            detail="Registry completion must bind the exact holdout result-manifest bytes.",
            artifact_hashes=(sha256_file(holdout_manifest_path),),
        ),
    )
    return DeterministicAuditReport(
        study_id=freeze.study_id,
        registry_head_hash=ledger.head_hash(),
        candidate_freeze_id=freeze.freeze_id,
        findings=findings,
        passed=all(value.passed for value in findings),
    )


def _find_forbidden_development_artifact(roots: tuple[Path, ...]) -> str | None:
    for root in roots:
        for path in root.rglob("*"):
            if "final_holdout" in path.name.casefold():
                return str(path)
            if path.suffix == ".json" and path.is_file():
                text = path.read_text(encoding="utf-8")
                if '"final_holdout"' in text:
                    return str(path)
    return None


def _load_pipeline_config(path: Path) -> ResearchConfig:
    snapshot = _read_json_object(path, label="development pipeline config snapshot")
    try:
        return ResearchConfig.model_validate({"path": path, **snapshot})
    except ValidationError as exc:
        raise ValueError("development pipeline config snapshot violates its contract") from exc


def _find_analyst_holdout_leak(
    root: Path,
    holdout_id: str,
    result_manifest_id: str,
) -> str | None:
    if not root.exists():
        return None
    markers = (holdout_id, result_manifest_id)
    for path in root.rglob("*"):
        if path.is_file():
            try:
                encoded = path.read_bytes()
            except OSError:
                return str(path)
            if any(marker.encode() in encoded for marker in markers):
                return str(path)
    return None


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return {str(key): nested for key, nested in value.items()}


def _sha256_artifact(path: Path, *, label: str) -> str:
    try:
        return sha256_file(path)
    except OSError as exc:
        raise ValueError(f"{label} cannot be read") from exc
