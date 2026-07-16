from __future__ import annotations

import hashlib
import json
import math
import stat
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ValidationError

from nlp_trader.immutable.append import (
    SafeFileError,
    read_bytes_no_follow,
    write_bytes_exclusive_durable,
)
from nlp_trader.research import code_version, sha256_file
from nlp_trader.research_agents.artifacts import (
    AgentArtifactError,
    ensure_agent_artifact_root,
    load_agent_manifest,
    write_agent_manifest_exclusive,
)
from nlp_trader.research_agents.catalog import CatalogEntry, FeatureCatalog
from nlp_trader.research_agents.config import ResearchAgentConfig, load_research_agent_config
from nlp_trader.research_agents.contracts import (
    AgentArtifactManifest,
    ArtifactEntry,
    ContractIdentity,
    ProposalAttemptReservedPayload,
    StudyDefinition,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_agents.evidence import EvidenceSourceRecord, normalized_source_text
from nlp_trader.research_agents.registry import ResearchRegistryLedger
from nlp_trader.research_agents.views import (
    DevelopmentMetric,
    DevelopmentRunView,
    export_development_view_bundle,
)
from nlp_trader.timestamps import parse_utc

DEFAULT_AGENT_CONFIG = Path("configs/research_agent.disabled.yaml")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

AgentConfigOption = Annotated[
    Path,
    typer.Option(
        "--agent-config",
        dir_okay=False,
        readable=True,
        help="Independent local research-agent configuration.",
    ),
]
ActorOption = Annotated[
    str,
    typer.Option("--actor", help="Local human/host attestation label; not authenticated identity."),
]

agent_study_app = typer.Typer(
    name="agent-study",
    help="Trusted deterministic research-study authority and export commands.",
    add_completion=False,
    no_args_is_help=True,
)


@agent_study_app.command("create")
def create_study_command(
    definition: Annotated[
        Path,
        typer.Option(
            "--definition", dir_okay=False, readable=True, help="Strict StudyDefinition JSON."
        ),
    ],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
    actor: ActorOption = "local-reviewer",
) -> None:
    """Register one immutable study and materialize a derived definition copy."""

    config = load_research_agent_config(agent_config)
    study = _load_study_definition(definition)
    ledger = ResearchRegistryLedger(config.artifact_root)
    event = ledger.register_study(
        study,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor,
    )
    _write_study_copy(config, study, event.event_ts)
    _print(
        study_id=study.study_id,
        registry_event_hash=event.event_hash,
        state="development_open",
    )


@agent_study_app.command("reserve-attempt")
def reserve_attempt_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
    actor: ActorOption = "agent-host",
) -> None:
    """Atomically reserve and consume the next proposal-budget slot."""

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    event = ledger.reserve_proposal_attempt(
        study_id,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor,
    )
    payload = event.payload
    if not isinstance(payload, ProposalAttemptReservedPayload):  # pragma: no cover - API invariant
        raise RuntimeError("registry returned an unexpected reservation event")
    _print(
        study_id=study_id,
        attempt_id=payload.attempt_id,
        attempt_number=payload.attempt_number,
        reservation_event_hash=event.event_hash,
        reserved_study_state_hash=payload.reserved_study_state_hash,
    )


@agent_study_app.command("export-view")
def export_view_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    source_run_final: Annotated[
        Path,
        typer.Option(
            "--source-run-final",
            dir_okay=False,
            readable=True,
            help="Trusted ordinary run.final.json to sanitize; never copied into the bundle.",
        ),
    ],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
) -> None:
    """Export a sealed exploratory development view from one completed ordinary run."""

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    projection = ledger.project()
    state = projection.studies.get(study_id)
    if state is None or state.state != "development_open":
        raise typer.BadParameter("view export requires a registered development_open study")
    study = ledger.study_definition(study_id)
    bundle_root, manifest = export_exploratory_standard_run(
        config,
        study=study,
        source_run_final=source_run_final,
    )
    _print(bundle_id=manifest.bundle_id, bundle_root=bundle_root)


@agent_study_app.command("close")
def close_study_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    reason: Annotated[str, typer.Option("--reason", help="Nonempty local closure rationale.")],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
    actor: ActorOption = "local-reviewer",
) -> None:
    """Close a study that has no incomplete proposal attempt."""

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    event = ledger.close_study(
        study_id,
        reason=reason,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor,
    )
    _print(study_id=study_id, state="closed", registry_event_hash=event.event_hash)


@agent_study_app.command("compile")
def compile_study_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    proposal_path: Annotated[
        Path,
        typer.Option("--proposal", dir_okay=False, readable=True, help="Verified proposal JSON."),
    ],
    verification_path: Annotated[
        Path,
        typer.Option(
            "--verification", dir_okay=False, readable=True, help="Passed verification JSON."
        ),
    ],
    base_config_path: Annotated[
        Path,
        typer.Option("--base-config", dir_okay=False, readable=True, help="Base research YAML."),
    ],
    compiler_contract_path: Annotated[
        Path,
        typer.Option(
            "--compiler-contract",
            dir_okay=False,
            readable=True,
            help="Frozen ContractIdentity JSON for the deterministic compiler.",
        ),
    ],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
) -> None:
    """Compile one verified proposal to an inert development-only definition."""

    from nlp_trader.config import load_config
    from nlp_trader.research_agents.compiler import compile_verified_proposal
    from nlp_trader.research_agents.contracts import ProposalVerification, ResearchProposal

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    study = ledger.study_definition(study_id)
    proposal = _load_contract(proposal_path, ResearchProposal, label="proposal")
    verification = _load_contract(
        verification_path, ProposalVerification, label="proposal verification"
    )
    compiler_contract = _load_contract(
        compiler_contract_path, ContractIdentity, label="compiler contract"
    )
    destination, definition = compile_verified_proposal(
        artifact_root=config.artifact_root,
        study=study,
        proposal=proposal,
        proposal_artifact_hash=sha256_file(_absolute_regular_file(proposal_path, label="proposal")),
        verification=verification,
        verification_artifact_hash=sha256_file(
            _absolute_regular_file(verification_path, label="proposal verification")
        ),
        base_config=load_config(base_config_path),
        compiler_contract=compiler_contract,
    )
    _print(definition_id=definition.definition_id, compiled_root=destination)


@agent_study_app.command("approve-development")
def approve_development_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    definition_path: Annotated[
        Path,
        typer.Option(
            "--definition", dir_okay=False, readable=True, help="Compiled execution definition."
        ),
    ],
    verification_path: Annotated[
        Path,
        typer.Option("--verification", dir_okay=False, readable=True, help="Passed verification."),
    ],
    reason: Annotated[str, typer.Option("--reason", help="Human approval rationale.")],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
    actor: ActorOption = "local-reviewer",
) -> None:
    """Lock development and approve one exact compiled definition."""

    from nlp_trader.research_agents.approvals import approve_development_execution
    from nlp_trader.research_agents.compiler import ExperimentExecutionDefinition
    from nlp_trader.research_agents.contracts import ProposalVerification

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    study = ledger.study_definition(study_id)
    definition = _load_contract(
        definition_path, ExperimentExecutionDefinition, label="execution definition"
    )
    verification = _load_contract(
        verification_path, ProposalVerification, label="proposal verification"
    )
    approval = approve_development_execution(
        ledger,
        study=study,
        definition=definition,
        verification=verification,
        verification_artifact_hash=sha256_file(
            _absolute_regular_file(verification_path, label="proposal verification")
        ),
        actor_label=actor,
        reviewer_reason=reason,
    )
    _print(
        study_id=study_id,
        approval_id=approval.approval_id,
        approval_event_hash=approval.approval_event_hash,
        state="development_locked",
    )


@agent_study_app.command("run-development")
def run_development_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    definition_path: Annotated[
        Path,
        typer.Option("--definition", dir_okay=False, readable=True),
    ],
    approval_path: Annotated[
        Path,
        typer.Option("--approval", dir_okay=False, readable=True),
    ],
    base_config_path: Annotated[
        Path,
        typer.Option("--base-config", dir_okay=False, readable=True),
    ],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
    actor: ActorOption = "development-runner",
) -> None:
    """Run one exactly approved development-only experiment."""

    from nlp_trader.config import load_config
    from nlp_trader.experiment_execution import run_approved_development
    from nlp_trader.research_agents.approvals import DevelopmentExecutionApproval
    from nlp_trader.research_agents.compiler import ExperimentExecutionDefinition

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    study = ledger.study_definition(study_id)
    definition = _load_contract(
        definition_path, ExperimentExecutionDefinition, label="execution definition"
    )
    approval = _load_contract(
        approval_path, DevelopmentExecutionApproval, label="development approval"
    )
    destination, result, frozen = run_approved_development(
        ledger=ledger,
        study=study,
        definition=definition,
        approval=approval,
        base_config=load_config(base_config_path),
        actor_label=actor,
    )
    _print(
        development_run_id=result.development_run_id,
        development_root=destination,
        result_manifest_id=result.manifest_id,
        frozen_model_manifest_id=frozen.manifest_id,
    )


@agent_study_app.command("freeze-candidate")
def freeze_candidate_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    definition_path: Annotated[
        Path,
        typer.Option("--definition", dir_okay=False, readable=True),
    ],
    approval_path: Annotated[
        Path,
        typer.Option("--approval", dir_okay=False, readable=True),
    ],
    development_root: Annotated[
        Path,
        typer.Option("--development-root", file_okay=False, readable=True),
    ],
    candidate_config_path: Annotated[
        Path,
        typer.Option("--candidate-config", dir_okay=False, readable=True),
    ],
    holdout_inputs_path: Annotated[
        Path,
        typer.Option(
            "--holdout-inputs",
            dir_okay=False,
            readable=True,
            help="JSON object containing input_snapshot_hashes and universe_asset_ids.",
        ),
    ],
    reason: Annotated[str, typer.Option("--reason", help="Human freeze rationale.")],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
    actor: ActorOption = "local-reviewer",
) -> None:
    """Freeze one exact post-development candidate; replacement is impossible."""

    from nlp_trader.experiment_execution import DevelopmentResultManifest
    from nlp_trader.research_agents.approvals import (
        DevelopmentExecutionApproval,
        freeze_candidate,
    )
    from nlp_trader.research_agents.compiler import ExperimentExecutionDefinition

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    study = ledger.study_definition(study_id)
    definition = _load_contract(
        definition_path, ExperimentExecutionDefinition, label="execution definition"
    )
    approval = _load_contract(
        approval_path, DevelopmentExecutionApproval, label="development approval"
    )
    root = _absolute_directory(development_root, label="development root")
    result_path = root / "result_manifest.json"
    result = _load_contract(result_path, DevelopmentResultManifest, label="development result")
    candidate_config = _read_json(
        _absolute_regular_file(candidate_config_path, label="candidate config")
    )
    holdout_inputs = _read_json(_absolute_regular_file(holdout_inputs_path, label="holdout inputs"))
    _require_exact_keys(
        holdout_inputs,
        {"input_snapshot_hashes", "universe_asset_ids"},
        label="holdout inputs",
    )
    input_hashes = _string_tuple(
        holdout_inputs["input_snapshot_hashes"], label="input_snapshot_hashes", sha256=True
    )
    universe = _string_tuple(
        holdout_inputs["universe_asset_ids"], label="universe_asset_ids", sha256=False
    )
    freeze = freeze_candidate(
        ledger,
        study=study,
        proposal_hash=definition.proposal_hash,
        definition=definition,
        approval=approval,
        development_result_manifest_hash=sha256_file(result_path),
        frozen_model_manifest_hash=sha256_file(root / "frozen_model.manifest.json"),
        candidate_config=candidate_config,
        required_evaluation_contract_hash=result.required_evaluation_contract_hash,
        input_snapshot_hashes=input_hashes,
        universe_asset_ids=universe,
        actor_label=actor,
        reviewer_reason=reason,
    )
    _print(
        study_id=study_id,
        freeze_id=freeze.freeze_id,
        candidate_config_hash=freeze.candidate_config_hash,
        holdout_id=freeze.holdout_identity.holdout_id,
        state="candidate_frozen",
    )


@agent_study_app.command("reveal-holdout")
def reveal_holdout_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    development_root: Annotated[
        Path,
        typer.Option("--development-root", file_okay=False, readable=True),
    ],
    base_config_path: Annotated[
        Path,
        typer.Option("--base-config", dir_okay=False, readable=True),
    ],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
    actor: ActorOption = "local-reviewer",
) -> None:
    """Reserve and execute the one-time frozen-candidate holdout reveal."""

    from nlp_trader.config import load_config
    from nlp_trader.holdout_execution import reveal_frozen_holdout
    from nlp_trader.research_agents.approvals import CandidateFreezeRecord

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    freeze_path = config.artifact_root / "studies" / study_id / "candidate" / "freeze.json"
    freeze = _load_contract(freeze_path, CandidateFreezeRecord, label="candidate freeze")
    destination, result = reveal_frozen_holdout(
        ledger=ledger,
        freeze=freeze,
        development_root=_absolute_directory(development_root, label="development root"),
        base_config=load_config(base_config_path),
        actor_label=actor,
    )
    _print(
        study_id=study_id,
        reservation_id=result.reservation_id,
        holdout_root=destination,
        result_manifest_id=result.manifest_id,
        state="holdout_revealed",
    )


@agent_study_app.command("register-external-holdout")
def register_external_holdout_command(
    holdout_identity_path: Annotated[
        Path,
        typer.Option(
            "--holdout-identity", dir_okay=False, readable=True, help="HoldoutIdentity JSON."
        ),
    ],
    reason: Annotated[str, typer.Option("--reason", help="External-use rationale.")],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
    actor: ActorOption = "local-reviewer",
) -> None:
    """Register out-of-band holdout use in the global contamination projection."""

    from nlp_trader.research_agents.contracts import HoldoutIdentity

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    identity = _load_contract(
        holdout_identity_path, HoldoutIdentity, label="external holdout identity"
    )
    event = ledger.register_external_holdout(
        identity,
        reason=reason,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor,
    )
    _print(
        holdout_id=identity.holdout_id,
        registry_event_hash=event.event_hash,
        contamination_status="recorded",
    )


@agent_study_app.command("audit")
def audit_study_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    development_root: Annotated[
        Path,
        typer.Option("--development-root", file_okay=False, readable=True),
    ],
    holdout_root: Annotated[
        Path,
        typer.Option("--holdout-root", file_okay=False, readable=True),
    ],
    base_config_path: Annotated[
        Path,
        typer.Option("--base-config", dir_okay=False, readable=True),
    ],
    agent_config: AgentConfigOption = DEFAULT_AGENT_CONFIG,
) -> None:
    """Run and persist the deterministic post-reveal lineage and isolation audit."""

    from nlp_trader.config import load_config
    from nlp_trader.experiment_execution import DevelopmentResultManifest
    from nlp_trader.holdout_execution import HoldoutResultManifest
    from nlp_trader.research_agents.approvals import CandidateFreezeRecord
    from nlp_trader.research_agents.audit import audit_completed_holdout

    config = load_research_agent_config(agent_config)
    ledger = ResearchRegistryLedger(config.artifact_root)
    freeze = _load_contract(
        config.artifact_root / "studies" / study_id / "candidate" / "freeze.json",
        CandidateFreezeRecord,
        label="candidate freeze",
    )
    development = _absolute_directory(development_root, label="development root")
    holdout = _absolute_directory(holdout_root, label="holdout root")
    holdout_result = _load_contract(
        holdout / "result_manifest.json",
        HoldoutResultManifest,
        label="holdout result",
    )
    result = _load_contract(
        development / "result_manifest.json",
        DevelopmentResultManifest,
        label="development result",
    )
    base_config = load_config(base_config_path)
    pipeline_roots = (
        base_config.paths.interim_dir / result.pipeline_run_id,
        base_config.paths.processed_dir / result.pipeline_run_id,
        base_config.paths.models_dir / result.pipeline_run_id,
        base_config.paths.reports_dir / result.pipeline_run_id,
    )
    holdout_pipeline_roots = (
        base_config.paths.interim_dir / holdout_result.reservation_id,
        base_config.paths.processed_dir / holdout_result.reservation_id,
        base_config.paths.models_dir / holdout_result.reservation_id,
        base_config.paths.reports_dir / holdout_result.reservation_id,
    )
    report = audit_completed_holdout(
        ledger=ledger,
        freeze=freeze,
        development_root=development,
        holdout_root=holdout,
        holdout_pipeline_roots=holdout_pipeline_roots,
        development_pipeline_roots=pipeline_roots,
    )
    destination = config.artifact_root / "audits" / f"{report.report_id}.json"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_bytes_exclusive_durable(
        destination,
        (canonical_json(report.model_dump(mode="json")) + "\n").encode("utf-8"),
    )
    _print(report_id=report.report_id, report_path=destination, passed=report.passed)
    if not report.passed:
        raise typer.Exit(code=2)


def export_exploratory_standard_run(
    config: ResearchAgentConfig,
    *,
    study: StudyDefinition,
    source_run_final: str | Path,
) -> tuple[Path, Any]:
    """Trusted field-level adapter for one ordinary completed research run."""

    final_path = _absolute_regular_file(source_run_final, label="source run.final.json")
    if final_path.name != "run.final.json":
        raise ValueError("source run manifest must be named run.final.json")
    final = _read_json(final_path)
    expected_final_fields = {
        "artifact_manifest",
        "code_version",
        "completed_at",
        "completed_stage",
        "config_hash",
        "constraints",
        "cost_model",
        "created_at",
        "data_manifest",
        "feature_set_version",
        "known_limitations",
        "label_version",
        "metrics",
        "model_version",
        "next_questions",
        "period",
        "rebalance_frequency",
        "run_id",
        "status",
        "universe",
    }
    _require_exact_keys(final, expected_final_fields, label="source run manifest")
    if final["status"] != "complete" or final["completed_stage"] != "report":
        raise ValueError("source run must be a completed report run")
    run_id = _identifier(final["run_id"], label="source run_id")
    if final_path.parent.name != run_id:
        raise ValueError("source run directory does not match run_id")
    artifacts = _artifact_entries(final["artifact_manifest"], run_id=run_id)
    config_entry = _single_artifact(artifacts, suffix="/config.snapshot.json")
    if config_entry["area"] != "reports":
        raise ValueError("source config snapshot has an invalid artifact area")
    config_path = final_path.parent / "config.snapshot.json"
    _verify_artifact_file(config_path, config_entry)
    source_config = _read_json(config_path)
    _require_exact_keys(
        source_config,
        {
            "backtest",
            "data",
            "features",
            "llm_annotations",
            "mode",
            "models",
            "paths",
            "runtime",
            "transformer",
        },
        label="source config snapshot",
    )
    from nlp_trader.config import ResearchConfig

    try:
        source_config_contract = ResearchConfig.model_validate(
            {"path": config_path, **source_config}
        )
    except ValidationError as exc:
        raise ValueError("source config snapshot violates its typed contract") from exc
    if source_config_contract.content_hash() != final["config_hash"]:
        raise ValueError("source config snapshot does not match the run config hash")
    paths = source_config["paths"]
    if not isinstance(paths, dict):
        raise ValueError("source config paths must be a mapping")
    resolved_artifacts = {
        entry["path"]: _resolve_and_verify_artifact(entry, paths=paths, run_id=run_id)
        for entry in artifacts
    }
    prediction_entry = _single_artifact(artifacts, suffix="/evaluation/prediction_metrics.json")
    backtest_entry = _single_artifact(artifacts, suffix="/evaluation/backtest_comparison.json")
    prediction = _read_json(resolved_artifacts[prediction_entry["path"]])
    backtest = _read_json(resolved_artifacts[backtest_entry["path"]])
    _validate_source_period(study, final.get("period"), backtest=backtest)
    metrics = _development_metrics(
        study,
        prediction=prediction,
        prediction_hash=prediction_entry["sha256"],
        backtest=backtest,
        backtest_hash=backtest_entry["sha256"],
    )
    universe_assets, asset_rows = _universe_assets(
        final,
        artifacts=artifacts,
        resolved=resolved_artifacts,
    )
    evidence_sources = _evidence_sources(
        study,
        universe_assets=universe_assets,
        asset_rows=asset_rows,
        artifacts=artifacts,
        resolved=resolved_artifacts,
    )
    assumptions = backtest["assumptions"]
    if not isinstance(assumptions, dict):
        raise ValueError("source backtest assumptions must be a mapping")
    costs_constraints = assumptions.get("costs_and_constraints")
    if not isinstance(costs_constraints, dict):
        raise ValueError("source backtest costs_and_constraints must be a mapping")
    if assumptions.get("horizon_steps") != study.horizon_sessions:
        raise ValueError("source backtest horizon does not match the study")
    cost_keys = {
        "borrow_bps_per_year",
        "commission_bps",
        "half_spread_bps",
        "market_impact_multiplier",
        "participation_slippage_bps",
        "slippage_bps",
        "volatility_slippage_multiplier",
    }
    cost_assumptions = {key: costs_constraints[key] for key in sorted(cost_keys)}
    view = DevelopmentRunView(
        study_id=study.study_id,
        parent_run_id=run_id,
        parent_manifest_hash=sha256_file(final_path),
        source_mode="exploratory_standard_run",
        confirmatory_eligible=False,
        analysis_cutoff=study.analysis_cutoff,
        development_decisions=study.development_decisions,
        universe_snapshot_id=study.universe_snapshot_id,
        universe_asset_ids=universe_assets,
        horizon_sessions=study.horizon_sessions,
        rebalance_frequency=_identifier(
            assumptions.get("rebalance_frequency"), label="source rebalance_frequency"
        ),
        calendar_contract=study.calendar_contract,
        cost_assumptions_hash=content_sha256(cost_assumptions),
        constraint_assumptions_hash=content_sha256(costs_constraints),
        metrics=metrics,
    )
    catalog = _feature_catalog(study, source_config=source_config, metrics=metrics)
    source_code = final["code_version"]
    git_commit = source_code.get("git_commit") if isinstance(source_code, dict) else None
    dirty = source_code.get("dirty") if isinstance(source_code, dict) else None
    exporter_contract = ContractIdentity(
        contract_id="exploratory-standard-run-exporter",
        version="v1",
        sha256=content_sha256(
            {
                "version": "v1",
                "source_manifest_fields": sorted(expected_final_fields),
                "bundle_files": [
                    "development_view.json",
                    "feature_catalog.json",
                    "evidence_snapshot.jsonl",
                    "evidence_index.json",
                ],
            }
        ),
    )
    return export_development_view_bundle(
        config.artifact_root,
        study=study,
        development_view=view,
        feature_catalog=catalog,
        evidence_sources=evidence_sources,
        exporter_contract=exporter_contract,
        git_commit=git_commit if isinstance(git_commit, str) else None,
        dirty_worktree=dirty if isinstance(dirty, bool) else None,
        limitations=(
            "Ordinary completed source runs are exploratory because their holdout already existed.",
            "Rights and active-membership fields are assertions from the source artifacts.",
        ),
    )


def _development_metrics(
    study: StudyDefinition,
    *,
    prediction: dict[str, object],
    prediction_hash: str,
    backtest: dict[str, object],
    backtest_hash: str,
) -> tuple[DevelopmentMetric, ...]:
    _require_exact_keys(
        prediction,
        {
            "evaluation_protocol",
            "families",
            "final_holdout",
            "metric_definitions",
            "segment_definitions",
            "segments",
        },
        label="prediction metrics",
    )
    _require_exact_keys(
        backtest,
        {
            "artifact_schema_version",
            "assumptions",
            "evaluation_protocol",
            "evaluation_window",
            "families",
            "provenance",
        },
        label="backtest comparison",
    )
    evaluation_window = backtest["evaluation_window"]
    if not isinstance(evaluation_window, dict) or set(evaluation_window) != {
        "end_exclusive",
        "name",
    }:
        raise ValueError("backtest development evaluation window has an invalid schema")
    if evaluation_window["name"] != "development":
        raise ValueError("backtest comparison is not development-scoped")
    end_exclusive = parse_utc(evaluation_window["end_exclusive"])
    if end_exclusive > study.reserved_holdout_decisions.start:
        raise ValueError("source development metrics cross the reserved holdout boundary")
    expected_families = set(study.required_learned_families).union(study.required_fixed_benchmarks)
    values: list[DevelopmentMetric] = []
    specs = (
        (
            "prediction",
            prediction["families"],
            _PREDICTION_METRIC_IDS,
            "prediction-metrics",
            prediction_hash,
        ),
        (
            "backtest",
            backtest["families"],
            _BACKTEST_METRIC_IDS,
            "backtest-comparison",
            backtest_hash,
        ),
    )
    for group, family_payload, metric_ids, artifact_id, artifact_hash in specs:
        if not isinstance(family_payload, dict) or not expected_families.issubset(family_payload):
            raise ValueError(f"{group} metrics omit a required family or benchmark")
        for family, metric_payload in sorted(family_payload.items()):
            family_id = _identifier(family, label=f"{group} family")
            if not isinstance(metric_payload, dict) or set(metric_payload) != metric_ids:
                raise ValueError(f"{group} metric schema changed for family {family_id}")
            for metric_id, value in sorted(metric_payload.items()):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{group} metric values must be numeric")
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"{group} metric values must be finite")
                values.append(
                    DevelopmentMetric(
                        metric_group=group,
                        family=family_id,
                        metric_id=metric_id,
                        value=numeric,
                        unit=_metric_unit(metric_id),
                        window=study.development_decisions,
                        source_artifact_id=artifact_id,
                        source_artifact_hash=artifact_hash,
                    )
                )
    present_metric_ids = {value.metric_id for value in values}
    if not set(study.required_metrics).issubset(present_metric_ids):
        raise ValueError("source metrics omit a study-required metric")
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value.metric_group,
                value.family or "",
                value.segment or "",
                value.metric_id,
            ),
        )
    )


def _validate_source_period(
    study: StudyDefinition,
    value: object,
    *,
    backtest: dict[str, object],
) -> None:
    if not isinstance(value, dict) or set(value) != {"start", "end"}:
        raise ValueError("source run period has an invalid schema")
    start_value = value.get("start")
    end_value = value.get("end")
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        raise ValueError("source run period must contain exact timestamps")
    source_start = parse_utc(start_value)
    source_end = parse_utc(end_value)
    if (
        source_start > study.development_decisions.start
        or source_end < study.development_decisions.end
    ):
        raise ValueError("source run period does not cover the study development window")
    evaluation_window = backtest.get("evaluation_window")
    if not isinstance(evaluation_window, dict):
        raise ValueError("source backtest evaluation window is unavailable")
    end_exclusive_value = evaluation_window.get("end_exclusive")
    if not isinstance(end_exclusive_value, str):
        raise ValueError("source backtest development end is unavailable")
    if parse_utc(end_exclusive_value) <= study.development_decisions.end:
        raise ValueError("source development metrics do not cover the study development window")


_PREDICTION_METRIC_IDS = {
    "dates",
    "hit_rate",
    "ic_dates",
    "mean_daily_pearson_ic",
    "mean_daily_precision_at_k",
    "mean_daily_spearman_ic",
    "pearson_ic",
    "precision_at_k",
    "rows",
    "spearman_ic",
}
_BACKTEST_METRIC_IDS = {
    "annualized_return",
    "annualized_volatility",
    "average_beta_exposure",
    "average_cost_return",
    "average_gross_exposure",
    "average_holding_period_days",
    "average_net_exposure",
    "average_turnover",
    "cost_adjusted_return",
    "final_equity",
    "gross_total_return",
    "hit_rate",
    "max_drawdown",
    "max_participation_rate",
    "minimum_capacity_proxy_equity",
    "periods",
    "sharpe",
    "sortino",
    "tail_loss_5pct",
    "total_borrow_return",
    "total_commission_return",
    "total_cost_return",
    "total_market_impact_return",
    "total_return",
    "total_slippage_return",
    "total_spread_return",
    "trades",
}


def _universe_assets(
    final: dict[str, object],
    *,
    artifacts: tuple[dict[str, Any], ...],
    resolved: dict[str, Path],
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    import polars as pl

    universe = final["universe"]
    if (
        not isinstance(universe, list)
        or not universe
        or not all(isinstance(value, str) for value in universe)
    ):
        raise ValueError("source run universe must be a nonempty string list")
    rows: list[dict[str, object]] = []
    for entry in _artifacts_with_fragment(artifacts, "/silver/assets/"):
        rows.extend(pl.read_parquet(resolved[entry["path"]]).to_dicts())
    expected = {
        "active_from",
        "active_to",
        "asset_id",
        "cik",
        "currency",
        "exchange",
        "figi",
        "hard_to_borrow",
        "industry",
        "isin",
        "name",
        "sector",
        "short_available",
        "symbol",
        "trading_unit",
    }
    by_symbol: dict[str, dict[str, object]] = {}
    by_asset: dict[str, dict[str, object]] = {}
    for row in rows:
        _require_exact_keys(row, expected, label="silver asset row")
        symbol = _identifier(row["symbol"], label="asset symbol")
        asset_id = _identifier(row["asset_id"], label="asset_id")
        if symbol in by_symbol or asset_id in by_asset:
            raise ValueError("silver asset rows contain duplicate identity")
        by_symbol[symbol] = row
        by_asset[asset_id] = row
    selected: list[str] = []
    for symbol in universe:
        if symbol not in by_symbol:
            raise ValueError("source universe symbol is absent from the asset snapshot")
        selected.append(_identifier(by_symbol[symbol]["asset_id"], label="asset_id"))
    return tuple(sorted(selected)), by_asset


def _evidence_sources(
    study: StudyDefinition,
    *,
    universe_assets: tuple[str, ...],
    asset_rows: dict[str, dict[str, object]],
    artifacts: tuple[dict[str, Any], ...],
    resolved: dict[str, Path],
) -> tuple[EvidenceSourceRecord, ...]:
    import polars as pl

    expected = {
        "author_hash",
        "available_at",
        "body",
        "canonical_text_hash",
        "content_status",
        "date",
        "entities",
        "event_ts",
        "event_type",
        "ingested_at",
        "item_id",
        "language",
        "license_or_terms_ref",
        "parent_item_id_hash",
        "processed_at",
        "published_at",
        "raw_text_hash",
        "raw_text_path",
        "relationship_type",
        "retention_permitted",
        "source",
        "source_type",
        "title",
        "url_hash",
        "vendor_received_at",
    }
    selected_assets = set(universe_assets)
    sources: list[EvidenceSourceRecord] = []
    seen_items: set[str] = set()
    for entry in _artifacts_with_fragment(artifacts, "/silver/text/"):
        for row in pl.read_parquet(resolved[entry["path"]]).to_dicts():
            _require_exact_keys(row, expected, label="silver text row")
            available_at = parse_utc(row["available_at"])
            if available_at > study.analysis_cutoff:
                continue
            item_id = _identifier(row["item_id"], label="text item_id")
            if item_id in seen_items:
                raise ValueError("silver text rows contain duplicate item_id")
            entities = _strict_entities(row["entities"])
            asset_ids = tuple(
                sorted(
                    {
                        _identifier(value["asset_id"], label="entity asset_id")
                        for value in entities
                        if value.get("asset_id") in selected_assets
                    }
                )
            )
            if not asset_ids:
                continue
            published_at = parse_utc(row["published_at"])
            for asset_id in asset_ids:
                _validate_active_membership(asset_rows[asset_id], published_at.date())
            if row["raw_text_path"] is not None:
                raise ValueError("silver text with raw_text_path cannot enter a sealed bundle")
            title = row["title"] if isinstance(row["title"], str) else None
            body = row["body"] if isinstance(row["body"], str) else None
            source_text = normalized_source_text(title, body)
            sources.append(
                EvidenceSourceRecord(
                    item_id=item_id,
                    source_type=_identifier(row["source_type"], label="source_type"),
                    language=_identifier(row["language"], label="language"),
                    title=title,
                    body=body,
                    source_text_hash=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                    content_status=row["content_status"],
                    relationship_type=row["relationship_type"],
                    license_or_terms_ref=row["license_or_terms_ref"],
                    retention_permitted=row["retention_permitted"],
                    asset_ids=asset_ids,
                    active_period_valid=True,
                    published_at=published_at,
                    available_at=available_at,
                    source_artifact_id=f"silver-text:{entry['sha256'][:16]}",
                    source_artifact_hash=entry["sha256"],
                    author_hash=row["author_hash"],
                    url_hash=row["url_hash"],
                )
            )
            seen_items.add(item_id)
    if not sources:
        raise ValueError("source run has no permitted point-in-time evidence for this study")
    return tuple(sorted(sources, key=lambda value: value.item_id))


def _feature_catalog(
    study: StudyDefinition,
    *,
    source_config: dict[str, object],
    metrics: tuple[DevelopmentMetric, ...],
) -> FeatureCatalog:
    features = source_config["features"]
    if not isinstance(features, dict):
        raise ValueError("source features config must be a mapping")

    def entries(values: set[str] | tuple[str, ...], definition: str) -> tuple[CatalogEntry, ...]:
        return tuple(
            CatalogEntry(entry_id=value, version="v1", definition=definition)
            for value in sorted(values)
        )

    return FeatureCatalog(
        features=entries(
            {_identifier(features.get("feature_set_version"), label="feature_set_version")},
            "Frozen point-in-time feature set available in the source run.",
        ),
        models=entries(
            set(study.required_learned_families),
            "Required learned model-family comparison path.",
        ),
        benchmarks=entries(
            set(study.required_fixed_benchmarks),
            "Required fixed benchmark comparison path.",
        ),
        selectors=entries({"full_eligible_universe"}, "Full eligible source-run universe."),
        metrics=entries(
            {value.metric_id for value in metrics},
            "Allowlisted development-only diagnostic with an explicit source hash.",
        ),
        controls=entries(
            set(study.required_negative_controls).union(study.required_robustness_checks),
            "Predeclared negative control or robustness check.",
        ),
        templates=tuple(
            CatalogEntry(
                entry_id=value.template_id,
                version=value.version,
                definition="Allowlisted inert experiment template with frozen typed parameters.",
            )
            for value in sorted(study.permitted_templates, key=lambda item: item.template_id)
        ),
    )


def _write_study_copy(
    config: ResearchAgentConfig,
    study: StudyDefinition,
    created_at: datetime,
) -> None:
    root = ensure_agent_artifact_root(config.artifact_root)
    study_root = root / "studies" / study.study_id
    definition_path = study_root / "definition.json"
    encoded = (study.canonical_json() + "\n").encode("utf-8")
    try:
        write_bytes_exclusive_durable(definition_path, encoded)
    except FileExistsError:
        existing = read_bytes_no_follow(definition_path)
        if existing != encoded:
            raise ValueError("materialized study definition conflicts with the registry") from None
    provenance = code_version(_REPOSITORY_ROOT)
    manifest = AgentArtifactManifest(
        artifact_root_id=config.content_hash(),
        created_at=created_at.astimezone(UTC),
        git_commit=(
            provenance["git_commit"] if isinstance(provenance.get("git_commit"), str) else None
        ),
        dirty_worktree=(provenance["dirty"] if isinstance(provenance.get("dirty"), bool) else None),
        parent_hashes=(study.study_id,),
        artifacts=(
            ArtifactEntry(
                role="study-definition",
                relative_path="definition.json",
                sha256=hashlib.sha256(encoded).hexdigest(),
                bytes=len(encoded),
                schema_version=study.artifact_schema_version,
            ),
        ),
        limitations=("Registry events, not this convenience copy, are authoritative.",),
        next_questions=("Reserve a bounded proposal attempt or close the study.",),
    )
    manifest_path = study_root / "definition.manifest.json"
    try:
        write_agent_manifest_exclusive(manifest_path, manifest)
    except AgentArtifactError as exc:
        if not manifest_path.exists() or load_agent_manifest(manifest_path) != manifest:
            raise ValueError("materialized study manifest conflicts with the registry") from exc


def _load_study_definition(path: Path) -> StudyDefinition:
    candidate = _absolute_regular_file(path, label="study definition")
    raw = _read_exact_bytes(candidate)
    try:
        return StudyDefinition.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError("study definition violates its strict contract") from exc


def _load_contract[ContractT: BaseModel](
    path: Path, contract: type[ContractT], *, label: str
) -> ContractT:
    candidate = _absolute_regular_file(path, label=label)
    try:
        return contract.model_validate_json(_read_exact_bytes(candidate))
    except ValidationError as exc:
        raise ValueError(f"{label} violates its strict contract") from exc


def _absolute_directory(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ValueError(f"{label} must be an absolute non-symlink directory")
    try:
        metadata = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")
    return candidate


def _string_tuple(value: object, *, label: str, sha256: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty JSON list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} values must be nonempty strings")
    result = tuple(value)
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{label} values must be unique and sorted")
    if sha256 and any(
        len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item) for item in result
    ):
        raise ValueError(f"{label} values must be lowercase SHA-256 digests")
    return result


def _artifact_entries(value: object, *, run_id: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError("source artifact_manifest must be a list")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("source artifact manifest entries must be mappings")
        _require_exact_keys(raw, {"area", "bytes", "path", "sha256"}, label="artifact entry")
        area = raw["area"]
        path = raw["path"]
        digest = raw["sha256"]
        size = raw["bytes"]
        if area not in {"interim", "processed", "models", "reports"}:
            raise ValueError("source artifact has an unknown area")
        if not isinstance(path, str) or PurePosixPath(path).parts[0] != run_id:
            raise ValueError("source artifact path is not bound to run_id")
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\\" in path:
            raise ValueError("source artifact path is not normalized relative POSIX content")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("source artifact SHA-256 is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("source artifact byte count is invalid")
        key = (area, path)
        if key in seen:
            raise ValueError("source artifact manifest repeats an entry")
        seen.add(key)
        result.append(dict(raw))
    return tuple(result)


def _resolve_and_verify_artifact(
    entry: dict[str, Any], *, paths: dict[str, object], run_id: str
) -> Path:
    field = {
        "interim": "interim_dir",
        "processed": "processed_dir",
        "models": "models_dir",
        "reports": "reports_dir",
    }[entry["area"]]
    root_value = paths.get(field)
    if not isinstance(root_value, str):
        raise ValueError(f"source config {field} must be a path string")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        raise ValueError(f"source config {field} must be absolute")
    relative = PurePosixPath(entry["path"])
    if relative.parts[0] != run_id:
        raise ValueError("source artifact is not below its run ID")
    candidate = root.joinpath(*relative.parts)
    _verify_artifact_file(candidate, entry)
    return candidate


def _verify_artifact_file(path: Path, entry: dict[str, Any]) -> None:
    candidate = _absolute_regular_file(path, label="source artifact")
    if candidate.stat().st_size != entry["bytes"] or sha256_file(candidate) != entry["sha256"]:
        raise ValueError("source artifact changed after its run manifest was written")


def _absolute_regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if candidate.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    try:
        metadata = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return candidate


def _read_exact_bytes(path: Path) -> bytes:
    try:
        raw = read_bytes_no_follow(path)
    except (FileNotFoundError, OSError, SafeFileError, ValueError) as exc:
        raise ValueError("trusted JSON input cannot be read safely") from exc
    if raw is None:  # pragma: no cover - missing_ok is false
        raise ValueError("trusted JSON input is missing")
    return raw


def _read_json(path: Path) -> dict[str, object]:
    raw = _read_exact_bytes(path)

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
        raise ValueError("trusted input is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("trusted JSON input must contain an object")
    return value


def _strict_entities(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, str):
        raise ValueError("silver text entities must be canonical JSON text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("silver text entities are invalid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("silver text entities must be a list")
    expected = {"asset_id", "confidence", "mention_type", "name", "relevance", "symbol"}
    result: list[dict[str, object]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("silver text entity must be a mapping")
        _require_exact_keys(item, expected, label="silver text entity")
        result.append(item)
    return tuple(result)


def _validate_active_membership(asset: dict[str, object], published: date) -> None:
    active_from = date.fromisoformat(str(asset["active_from"]))
    active_to_value = asset["active_to"]
    active_to = date.fromisoformat(str(active_to_value)) if active_to_value is not None else None
    if published < active_from or (active_to is not None and published > active_to):
        raise ValueError("evidence entity is outside the asset active period")


def _single_artifact(artifacts: tuple[dict[str, Any], ...], *, suffix: str) -> dict[str, Any]:
    matches = [entry for entry in artifacts if entry["path"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"source run must contain exactly one artifact ending in {suffix}")
    return matches[0]


def _artifacts_with_fragment(
    artifacts: tuple[dict[str, Any], ...], fragment: str
) -> tuple[dict[str, Any], ...]:
    matches = tuple(entry for entry in artifacts if fragment in entry["path"])
    if not matches:
        raise ValueError(f"source run contains no artifacts for {fragment}")
    return matches


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    try:
        ContractIdentity(contract_id=value, version="v1", sha256="0" * 64)
    except ValidationError as exc:
        raise ValueError(f"{label} is not a valid identifier") from exc
    return value


def _metric_unit(metric_id: str) -> str:
    if metric_id in {"dates", "ic_dates", "rows", "periods", "trades"}:
        return "count"
    if "return" in metric_id or metric_id in {"hit_rate", "precision_at_k", "max_drawdown"}:
        return "ratio"
    if metric_id == "final_equity" or metric_id == "minimum_capacity_proxy_equity":
        return "equity_multiple"
    if metric_id == "average_holding_period_days":
        return "days"
    return "coefficient"


def _require_exact_keys(value: dict[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema changed; unknown or missing fields are not exportable")


def _print(**values: object) -> None:
    for key, value in values.items():
        typer.echo(f"{key}: {value}")
