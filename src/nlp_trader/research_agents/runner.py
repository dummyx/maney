from __future__ import annotations

import hashlib
import os
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from nlp_trader.immutable.append import (
    SafeFileError,
    read_bytes_no_follow,
    write_bytes_exclusive_durable,
)
from nlp_trader.nlp.local_generation import RawGenerator
from nlp_trader.research_agents.analyst import BoundedResearchAnalyst
from nlp_trader.research_agents.config import ResearchAgentConfig
from nlp_trader.research_agents.contracts import (
    GENESIS_HASH,
    AbstentionAction,
    GenerationDiagnostics,
    ProposalAction,
    ProposalAttemptOutcome,
    ProposalAttemptSnapshot,
    ProposalVerification,
    ResearchAbstention,
    ResearchAgentAction,
    ResearchAgentRound,
    ResearchProposal,
    RoundCheck,
    Sha256,
    StrictModel,
    StudyDefinition,
    ToolCallAction,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_agents.ledger import (
    ResearchAgentRoundLedger,
    ResearchAgentRoundLedgerError,
    ResearchAgentRoundLedgerLockError,
)
from nlp_trader.research_agents.prompts import (
    abstention_schema,
    action_schema,
    continuation_prompt,
    initial_prompt,
    input_snapshot_hash,
    proposal_schema,
    tool_catalog,
)
from nlp_trader.research_agents.registry import ResearchRegistryLedger
from nlp_trader.research_agents.runtime import (
    AgentGenerationRecord,
    AgentGenerationRequest,
    ResearchAgentGenerationRuntime,
    build_agent_generation_request,
)
from nlp_trader.research_agents.tools import (
    ResearchToolGateway,
    ToolLimits,
    ToolResultEnvelope,
)
from nlp_trader.research_agents.verifier import verify_terminal_action
from nlp_trader.research_agents.views import LoadedDevelopmentViewBundle


class AgentRunError(ValueError):
    """Raised when an agent run or stored trace violates its immutable contract."""


class AgentRunInitial(StrictModel):
    artifact_schema_version: Literal["research-agent-run-initial-v1"] = (
        "research-agent-run-initial-v1"
    )
    agent_run_id: Sha256
    study_id: Sha256
    attempt: ProposalAttemptSnapshot
    bundle_id: Sha256
    input_snapshot_hash: Sha256
    config_hash: Sha256
    registry_head_at_start: Sha256


class AgentRunFinal(StrictModel):
    artifact_schema_version: Literal["research-agent-run-final-v1"] = "research-agent-run-final-v1"
    agent_run_id: Sha256
    outcome: Literal[
        "proposal_verified",
        "proposal_rejected",
        "abstention_verified",
        "abstention_rejected",
    ]
    terminal_artifact_hash: Sha256
    verification_hash: Sha256
    completion_event_hash: Sha256
    verification_event_hash: Sha256
    final_round_hash: Sha256
    tool_result_hashes: tuple[Sha256, ...]


class AgentRunFailed(StrictModel):
    artifact_schema_version: Literal["research-agent-run-failed-v1"] = (
        "research-agent-run-failed-v1"
    )
    agent_run_id: Sha256
    outcome: ProposalAttemptOutcome
    detail: str = Field(min_length=1, max_length=2048)
    completion_event_hash: Sha256
    final_round_hash: Sha256 | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    run_dir: Path
    initial: AgentRunInitial
    final: AgentRunFinal | AgentRunFailed
    terminal: ResearchProposal | ResearchAbstention | None
    verification: ProposalVerification | None
    rounds: tuple[ResearchAgentRound, ...]
    tool_results: tuple[ToolResultEnvelope, ...]


def scrub_agent_environment(
    environment: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Remove inherited secret-bearing and nonessential values without recording values."""

    allow_exact = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TMPDIR",
        "USER",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "VECLIB_MAXIMUM_THREADS",
    }
    values = os.environ if environment is None else environment
    removed: list[str] = []
    for name in tuple(values):
        if name in allow_exact:
            continue
        removed.append(name)
        del values[name]
    return tuple(sorted(removed))


def _validate_inputs(
    config: ResearchAgentConfig,
    ledger: ResearchRegistryLedger,
    study: StudyDefinition,
    bundle: LoadedDevelopmentViewBundle,
    attempt_id: str,
) -> ProposalAttemptSnapshot:
    if config.artifact_root.resolve() != ledger.artifact_root.resolve():
        raise AgentRunError("agent config and registry must use the same artifact root")
    if bundle.manifest.study_id != study.study_id:
        raise AgentRunError("development bundle does not belong to the study")
    if (
        config.model_logical_id != study.model.logical_id
        or config.model_revision != study.model.revision
        or config.model_expected_sha256 != study.model.file_sha256
        or config.model_license_or_terms_ref != study.model.license_or_terms_ref
    ):
        raise AgentRunError("agent model identity does not match the frozen study")
    versions = {
        "prompt_version": study.prompt_contract.version,
        "action_schema_version": study.action_schema_contract.version,
        "proposal_schema_version": study.proposal_schema_contract.version,
        "tool_catalog_version": study.tool_catalog_contract.version,
        "verifier_version": study.verifier_contract.version,
    }
    for name, expected in versions.items():
        if getattr(config, name) != expected:
            raise AgentRunError(f"agent {name} does not match the frozen study contract")
    projection = ledger.project()
    state = projection.studies.get(study.study_id)
    if state is None or state.state != "development_open":
        raise AgentRunError("agent attempts require a development_open registered study")
    for attempt in state.attempts:
        if attempt.attempt_id == attempt_id:
            if attempt.status != "reserved":
                raise AgentRunError("proposal attempt is not reserved")
            return attempt
    raise AgentRunError("proposal attempt is not registered")


def run_research_agent(
    *,
    config: ResearchAgentConfig,
    ledger: ResearchRegistryLedger,
    study: StudyDefinition,
    bundle: LoadedDevelopmentViewBundle,
    attempt_id: str,
    generator: RawGenerator,
    actor_label: str = "research-agent-host",
    device_path: Literal["cpu", "metal", "injected"] = "injected",
    effective_gpu_layers: int | None = None,
) -> AgentRunResult:
    attempt = _validate_inputs(config, ledger, study, bundle, attempt_id)
    snapshot_hash = input_snapshot_hash(
        study,
        bundle,
        attempt_id=attempt_id,
        reserved_study_state_hash=attempt.reserved_study_state_hash,
    )
    config_hash = content_sha256(config.model_dump(mode="json"))
    run_id = content_sha256(
        {
            "study_id": study.study_id,
            "attempt_id": attempt_id,
            "bundle_id": bundle.manifest.bundle_id,
            "input_snapshot_hash": snapshot_hash,
            "config_hash": config_hash,
        }
    )
    run_dir = ledger.artifact_root / "runs" / run_id
    try:
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        for relative in (
            "reports",
            "model/generation_attempts",
            "model/responses",
            "tools/requests",
            "tools/results",
        ):
            (run_dir / relative).mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise AgentRunError("agent run directory cannot be created exclusively") from exc

    initial = AgentRunInitial(
        agent_run_id=run_id,
        study_id=study.study_id,
        attempt=attempt,
        bundle_id=bundle.manifest.bundle_id,
        input_snapshot_hash=snapshot_hash,
        config_hash=config_hash,
        registry_head_at_start=ledger.head_hash(),
    )
    _write_model_inputs(run_dir, config, initial, study, bundle, snapshot_hash)
    initial_text = initial_prompt(
        study,
        bundle,
        attempt_id=attempt_id,
        snapshot_hash=snapshot_hash,
    )
    _write_text_exclusive(run_dir / "model" / "prompt.txt", initial_text)
    runtime = ResearchAgentGenerationRuntime(generator)
    analyst = BoundedResearchAnalyst(runtime)
    gateway = ResearchToolGateway(
        bundle,
        ToolLimits(
            max_evidence_results=config.max_evidence_results,
            max_result_bytes=config.max_tool_result_bytes_per_step,
        ),
    )
    rounds: list[ResearchAgentRound] = []
    results: list[ToolResultEnvelope] = []
    transcript: list[dict[str, Any]] = []
    previous_round_hash = GENESIS_HASH
    total_tool_bytes = 0
    tool_calls = 0
    evidence_pages = 0
    metric_reads = 0
    started = time.monotonic()

    try:
        for step in range(1, config.max_steps + 1):
            if time.monotonic() - started > config.max_wall_time_seconds:
                return _fail_attempt(
                    ledger=ledger,
                    run_dir=run_dir,
                    initial=initial,
                    outcome="exhausted",
                    detail="agent wall-time limit exhausted",
                    actor_label=actor_label,
                    rounds=rounds,
                    results=results,
                )
            prompt = continuation_prompt(initial_text, transcript) if transcript else initial_text
            request = build_agent_generation_request(
                study_id=study.study_id,
                attempt_id=attempt_id,
                bundle_id=bundle.manifest.bundle_id,
                round_index=step - 1,
                prompt=prompt,
                action_schema=action_schema(),
                transcript=transcript,
            )
            _write_json_exclusive(
                run_dir / "model" / "generation_attempts" / f"{step:04d}.json",
                request.model_dump(mode="json"),
            )
            step_result = analyst.generate_step(request)
            _write_json_exclusive(
                run_dir / "model" / "responses" / f"{step:04d}.json",
                step_result.generation.model_dump(mode="json"),
            )
            diagnostics = _round_diagnostics(
                step_result.generation,
                config=config,
                device_path=device_path,
                effective_gpu_layers=effective_gpu_layers,
            )
            if step_result.action is None:
                round_record = _make_round(
                    initial=initial,
                    study=study,
                    step=step,
                    previous_round_hash=previous_round_hash,
                    context_hash=request.request_id,
                    generation=step_result.generation,
                    action=None,
                    diagnostics=diagnostics,
                    termination_reason="malformed_action",
                )
                _append_round(run_dir, round_record)
                rounds.append(round_record)
                return _fail_attempt(
                    ledger=ledger,
                    run_dir=run_dir,
                    initial=initial,
                    outcome="malformed",
                    detail=step_result.parse_error or "model output could not be parsed",
                    actor_label=actor_label,
                    rounds=rounds,
                    results=results,
                )
            action = step_result.action
            if isinstance(action, ToolCallAction):
                if tool_calls >= config.max_tool_calls:
                    round_record = _make_round(
                        initial=initial,
                        study=study,
                        step=step,
                        previous_round_hash=previous_round_hash,
                        context_hash=request.request_id,
                        generation=step_result.generation,
                        action=action,
                        diagnostics=diagnostics,
                        termination_reason="tool_call_limit",
                    )
                    _append_round(run_dir, round_record)
                    rounds.append(round_record)
                    return _fail_attempt(
                        ledger=ledger,
                        run_dir=run_dir,
                        initial=initial,
                        outcome="exhausted",
                        detail="agent tool-call limit exhausted",
                        actor_label=actor_label,
                        rounds=rounds,
                        results=results,
                    )
                tool_name = action.tool_call.tool_name
                if tool_name == "search_evidence":
                    evidence_pages += 1
                    if evidence_pages > config.max_evidence_pages:
                        raise ValueError("evidence-page limit exhausted")
                if tool_name == "read_development_metrics":
                    metric_reads += 1
                    if metric_reads > config.max_metric_reads:
                        raise ValueError("development-metric read limit exhausted")
                _write_json_exclusive(
                    run_dir / "tools" / "requests" / f"{step:04d}.json",
                    action.tool_call.model_dump(mode="json"),
                )
                result = gateway.execute(action)
                total_tool_bytes += result.result_bytes
                if total_tool_bytes > config.max_tool_result_bytes_per_run:
                    raise ValueError("per-run tool-result byte limit exhausted")
                _write_json_exclusive(
                    run_dir / "tools" / "results" / f"{step:04d}.json",
                    result.model_dump(mode="json"),
                )
                round_record = _make_round(
                    initial=initial,
                    study=study,
                    step=step,
                    previous_round_hash=previous_round_hash,
                    context_hash=request.request_id,
                    generation=step_result.generation,
                    action=action,
                    diagnostics=diagnostics,
                    tool_request_hash=result.request_hash,
                    tool_result_hash=result.result_hash,
                )
                _append_round(run_dir, round_record)
                rounds.append(round_record)
                results.append(result)
                previous_round_hash = round_record.round_id
                tool_calls += 1
                transcript.append(
                    {
                        "step": step,
                        "action": action.model_dump(mode="json"),
                        "tool_result_untrusted_data": result.model_visible_payload(),
                    }
                )
                if step == config.max_steps:
                    return _fail_attempt(
                        ledger=ledger,
                        run_dir=run_dir,
                        initial=initial,
                        outcome="exhausted",
                        detail="agent step limit exhausted before a terminal action",
                        actor_label=actor_label,
                        rounds=rounds,
                        results=results,
                    )
                continue

            assert isinstance(action, (ProposalAction, AbstentionAction))
            round_record = _make_round(
                initial=initial,
                study=study,
                step=step,
                previous_round_hash=previous_round_hash,
                context_hash=request.request_id,
                generation=step_result.generation,
                action=action,
                diagnostics=diagnostics,
                termination_reason=action.action_type,
            )
            _append_round(run_dir, round_record)
            rounds.append(round_record)
            return _finish_terminal(
                ledger=ledger,
                run_dir=run_dir,
                initial=initial,
                study=study,
                bundle=bundle,
                action=action,
                actor_label=actor_label,
                rounds=rounds,
                results=results,
                max_retained_bytes=config.max_retained_artifact_bytes,
            )
    except Exception as exc:
        state = ledger.project().studies[study.study_id]
        stored_attempt = next(
            value for value in state.attempts if value.attempt_id == initial.attempt.attempt_id
        )
        if stored_attempt.status == "completed":
            raise AgentRunError("agent run failed after immutable attempt completion") from exc
        return _fail_attempt(
            ledger=ledger,
            run_dir=run_dir,
            initial=initial,
            outcome="rejected" if isinstance(exc, ValueError) else "crashed",
            detail="agent host rejected a step"
            if isinstance(exc, ValueError)
            else "agent host crashed",
            actor_label=actor_label,
            rounds=rounds,
            results=results,
        )
    raise AssertionError("bounded agent loop must terminate")


def _write_model_inputs(
    run_dir: Path,
    config: ResearchAgentConfig,
    initial: AgentRunInitial,
    study: StudyDefinition,
    bundle: LoadedDevelopmentViewBundle,
    snapshot_hash: str,
) -> None:
    _write_json_exclusive(run_dir / "reports" / "run.initial.json", initial.model_dump(mode="json"))
    _write_json_exclusive(
        run_dir / "reports" / "config.snapshot.json", config.model_dump(mode="json")
    )
    _write_json_exclusive(
        run_dir / "reports" / "registry.snapshot.json",
        {"study_id": study.study_id, "attempt": initial.attempt.model_dump(mode="json")},
    )
    _write_json_exclusive(
        run_dir / "reports" / "bundle.manifest.json", bundle.manifest.model_dump(mode="json")
    )
    _write_json_exclusive(run_dir / "model" / "action.schema.json", action_schema())
    _write_json_exclusive(run_dir / "model" / "proposal.schema.json", proposal_schema())
    _write_json_exclusive(run_dir / "model" / "abstention.schema.json", abstention_schema())
    _write_json_exclusive(run_dir / "model" / "tool_catalog.json", tool_catalog())
    _write_json_exclusive(
        run_dir / "model" / "provenance.json",
        {
            "artifact_schema_version": "research-agent-model-provenance-v1",
            "model": {
                "logical_id": config.model_logical_id,
                "revision": config.model_revision,
                "file_sha256": config.model_expected_sha256,
                "license_or_terms_ref": config.model_license_or_terms_ref,
            },
            "runtime_version": config.runtime_version,
            "decoding": config.decoding,
            "seed": config.seed,
            "requested_gpu_layers": config.gpu_layers,
            "environment_scrub_policy_version": config.environment_scrub_policy_version,
            "input_snapshot_hash": snapshot_hash,
        },
    )


def _make_round(
    *,
    initial: AgentRunInitial,
    study: StudyDefinition,
    step: int,
    previous_round_hash: str,
    context_hash: str,
    generation: AgentGenerationRecord,
    action: ResearchAgentAction | None,
    diagnostics: GenerationDiagnostics,
    termination_reason: str | None = None,
    tool_request_hash: str | None = None,
    tool_result_hash: str | None = None,
) -> ResearchAgentRound:
    return ResearchAgentRound(
        agent_run_id=initial.agent_run_id,
        study_id=study.study_id,
        attempt_id=initial.attempt.attempt_id,
        step=step,
        previous_round_hash=previous_round_hash,
        model=study.model,
        prompt_contract=study.prompt_contract,
        action_schema_contract=study.action_schema_contract,
        proposal_schema_contract=study.proposal_schema_contract,
        verifier_contract=study.verifier_contract,
        tool_catalog_contract=study.tool_catalog_contract,
        bundle_id=initial.bundle_id,
        input_snapshot_hash=initial.input_snapshot_hash,
        attempt_reservation_event_hash=initial.attempt.reservation_event_hash,
        reserved_study_state_hash=initial.attempt.reserved_study_state_hash,
        context_hash=context_hash,
        raw_generation=generation.generated_text or "",
        parse_status="passed" if action is not None else "failed",
        parsed_action=action,
        tool_request_hash=tool_request_hash,
        tool_result_hash=tool_result_hash,
        origin="generated",
        checks=(
            RoundCheck(check_id="request_identity_bound", passed=True),
            RoundCheck(check_id="one_action_only", passed=action is not None),
        ),
        diagnostics=diagnostics,
        termination_reason=termination_reason,
    )


def _round_diagnostics(
    generation: AgentGenerationRecord,
    *,
    config: ResearchAgentConfig,
    device_path: Literal["cpu", "metal", "injected"],
    effective_gpu_layers: int | None,
) -> GenerationDiagnostics:
    return GenerationDiagnostics(
        input_tokens=generation.input_token_count,
        output_tokens=generation.output_token_count,
        latency_ms=(
            generation.generation_latency_seconds * 1000.0
            if generation.generation_latency_seconds is not None
            else None
        ),
        throughput_tokens_per_second=generation.output_tokens_per_second,
        requested_gpu_layers=config.gpu_layers if config.gpu_layers >= 0 else None,
        effective_gpu_layers=effective_gpu_layers,
        device_path=device_path,
    )


def _finish_terminal(
    *,
    ledger: ResearchRegistryLedger,
    run_dir: Path,
    initial: AgentRunInitial,
    study: StudyDefinition,
    bundle: LoadedDevelopmentViewBundle,
    action: ProposalAction | AbstentionAction,
    actor_label: str,
    rounds: list[ResearchAgentRound],
    results: list[ToolResultEnvelope],
    max_retained_bytes: int,
) -> AgentRunResult:
    terminal = action.proposal if isinstance(action, ProposalAction) else action.abstention
    terminal_name = "proposal.json" if isinstance(action, ProposalAction) else "abstention.json"
    terminal_bytes = _write_json_exclusive(
        run_dir / "reports" / terminal_name, terminal.model_dump(mode="json")
    )
    terminal_hash = hashlib.sha256(terminal_bytes).hexdigest()
    completion = ledger.complete_proposal_attempt(
        study.study_id,
        initial.attempt.attempt_id,
        outcome="proposal" if isinstance(action, ProposalAction) else "abstention",
        agent_run_id=initial.agent_run_id,
        terminal_artifact_hash=terminal_hash,
        detail="terminal action persisted for deterministic verification",
        expected_head_hash=ledger.head_hash(),
        actor_label=actor_label,
    )
    verification = verify_terminal_action(
        action,
        study=study,
        attempt=initial.attempt,
        bundle=bundle,
        input_snapshot_hash=initial.input_snapshot_hash,
        registry_head_hash=completion.event_hash,
        terminal_artifact_hash=terminal_hash,
        tool_results=tuple(results),
        rounds=tuple(rounds),
    )
    verification_bytes = _write_json_exclusive(
        run_dir / "reports" / "proposal_verification.json",
        verification.model_dump(mode="json"),
    )
    verification_hash = hashlib.sha256(verification_bytes).hexdigest()
    verification_event = ledger.record_proposal_verification(
        study.study_id,
        initial.attempt.attempt_id,
        terminal_artifact_hash=terminal_hash,
        verification_hash=verification_hash,
        passed=verification.passed,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor_label,
    )
    outcome: Literal[
        "proposal_verified",
        "proposal_rejected",
        "abstention_verified",
        "abstention_rejected",
    ]
    if isinstance(action, ProposalAction):
        outcome = "proposal_verified" if verification.passed else "proposal_rejected"
    else:
        outcome = "abstention_verified" if verification.passed else "abstention_rejected"
    final = AgentRunFinal(
        agent_run_id=initial.agent_run_id,
        outcome=outcome,
        terminal_artifact_hash=terminal_hash,
        verification_hash=verification_hash,
        completion_event_hash=completion.event_hash,
        verification_event_hash=verification_event.event_hash,
        final_round_hash=rounds[-1].round_id,
        tool_result_hashes=tuple(value.result_hash for value in results),
    )
    _write_json_exclusive(run_dir / "reports" / "run.final.json", final.model_dump(mode="json"))
    _write_tool_summary(run_dir, results)
    _check_retained_bytes(run_dir, max_retained_bytes=max_retained_bytes)
    return AgentRunResult(
        run_dir=run_dir,
        initial=initial,
        final=final,
        terminal=terminal,
        verification=verification,
        rounds=tuple(rounds),
        tool_results=tuple(results),
    )


def _fail_attempt(
    *,
    ledger: ResearchRegistryLedger,
    run_dir: Path,
    initial: AgentRunInitial,
    outcome: ProposalAttemptOutcome,
    detail: str,
    actor_label: str,
    rounds: list[ResearchAgentRound],
    results: list[ToolResultEnvelope],
) -> AgentRunResult:
    completion = ledger.complete_proposal_attempt(
        initial.study_id,
        initial.attempt.attempt_id,
        outcome=outcome,
        agent_run_id=initial.agent_run_id,
        detail=detail,
        expected_head_hash=ledger.head_hash(),
        actor_label=actor_label,
    )
    failed = AgentRunFailed(
        agent_run_id=initial.agent_run_id,
        outcome=outcome,
        detail=detail,
        completion_event_hash=completion.event_hash,
        final_round_hash=rounds[-1].round_id if rounds else None,
    )
    _write_json_exclusive(run_dir / "reports" / "run.failed.json", failed.model_dump(mode="json"))
    _write_tool_summary(run_dir, results)
    return AgentRunResult(
        run_dir=run_dir,
        initial=initial,
        final=failed,
        terminal=None,
        verification=None,
        rounds=tuple(rounds),
        tool_results=tuple(results),
    )


def _write_tool_summary(run_dir: Path, results: list[ToolResultEnvelope]) -> None:
    _write_json_exclusive(
        run_dir / "tools" / "summary.json",
        {
            "artifact_schema_version": "research-agent-tool-summary-v1",
            "result_count": len(results),
            "result_hashes": [value.result_hash for value in results],
            "total_model_visible_bytes": sum(value.result_bytes for value in results),
        },
    )


def _check_retained_bytes(run_dir: Path, *, max_retained_bytes: int) -> None:
    if any(path.is_symlink() for path in run_dir.rglob("*")):
        raise AgentRunError("agent run contains an unsafe symlink")
    total = sum(
        path.stat(follow_symlinks=False).st_size for path in run_dir.rglob("*") if path.is_file()
    )
    if total > max_retained_bytes:
        raise AgentRunError("agent run exceeds the retained-artifact byte bound")


def _append_round(run_dir: Path, round_record: ResearchAgentRound) -> None:
    try:
        ResearchAgentRoundLedger(run_dir / "model" / "rounds.jsonl").append(round_record)
    except (ResearchAgentRoundLedgerError, ResearchAgentRoundLedgerLockError) as exc:
        raise AgentRunError("agent round cannot be persisted durably") from exc


def _write_text_exclusive(path: Path, text: str) -> bytes:
    encoded = text.encode("utf-8")
    try:
        write_bytes_exclusive_durable(path, encoded)
    except (FileExistsError, SafeFileError, OSError, ValueError) as exc:
        raise AgentRunError("agent artifact cannot be written exclusively") from exc
    return encoded


def _write_json_exclusive(path: Path, value: object) -> bytes:
    return _write_text_exclusive(path, canonical_json(value) + "\n")


def _read_typed[T: StrictModel](path: Path, model_type: type[T]) -> T:
    try:
        encoded = read_bytes_no_follow(path)
    except (FileNotFoundError, SafeFileError, OSError, ValueError) as exc:
        raise AgentRunError("stored agent artifact cannot be read safely") from exc
    assert encoded is not None
    try:
        raw = encoded.decode("utf-8")
        value = model_type.model_validate_json(raw)
    except (UnicodeDecodeError, ValidationError) as exc:
        raise AgentRunError("stored agent artifact violates its strict contract") from exc
    if raw != canonical_json(value.model_dump(mode="json")) + "\n":
        raise AgentRunError("stored agent artifact is not canonical typed JSON")
    return value


def load_stored_rounds(run_dir: Path) -> tuple[ResearchAgentRound, ...]:
    try:
        rounds = ResearchAgentRoundLedger(run_dir / "model" / "rounds.jsonl").replay()
    except (ResearchAgentRoundLedgerError, ResearchAgentRoundLedgerLockError) as exc:
        raise AgentRunError("stored round ledger cannot be read safely") from exc
    if not rounds:
        raise AgentRunError("stored round ledger is empty")
    return rounds


def replay_agent_run(run_dir: str | Path) -> tuple[ResearchAgentRound, ...]:
    directory = Path(run_dir).expanduser().resolve()
    initial = _read_typed(directory / "reports" / "run.initial.json", AgentRunInitial)
    rounds = load_stored_rounds(directory)
    if any(value.agent_run_id != initial.agent_run_id for value in rounds):
        raise AgentRunError("stored rounds do not belong to the initial run")
    for round_record in rounds:
        response_path = directory / "model" / "responses" / f"{round_record.step:04d}.json"
        request_path = directory / "model" / "generation_attempts" / f"{round_record.step:04d}.json"
        request = _read_typed(request_path, AgentGenerationRequest)
        generation = _read_typed(response_path, AgentGenerationRecord)
        if generation.request != request:
            raise AgentRunError("stored generation attempt and response request differ")
        if generation.request.request_id != round_record.context_hash:
            raise AgentRunError("stored generation request does not match its round")
        if (generation.generated_text or "") != round_record.raw_generation:
            raise AgentRunError("stored raw generation does not match its round")
        ResearchAgentGenerationRuntime.replay(generation.request, generation)
    return rounds


def verify_stored_run(
    run_dir: str | Path,
    *,
    ledger: ResearchRegistryLedger,
    study: StudyDefinition,
    bundle: LoadedDevelopmentViewBundle,
) -> ProposalVerification:
    directory = Path(run_dir).expanduser().resolve()
    replay_agent_run(directory)
    initial = _read_typed(directory / "reports" / "run.initial.json", AgentRunInitial)
    final = _read_typed(directory / "reports" / "run.final.json", AgentRunFinal)
    rounds = load_stored_rounds(directory)
    results = tuple(
        _read_typed(path, ToolResultEnvelope)
        for path in sorted((directory / "tools" / "results").glob("*.json"))
    )
    proposal_path = directory / "reports" / "proposal.json"
    if proposal_path.exists():
        proposal = _read_typed(proposal_path, ResearchProposal)
        action: ProposalAction | AbstentionAction = ProposalAction(proposal=proposal)
    else:
        abstention = _read_typed(directory / "reports" / "abstention.json", ResearchAbstention)
        action = AbstentionAction(abstention=abstention)
    recomputed = verify_terminal_action(
        action,
        study=study,
        attempt=initial.attempt,
        bundle=bundle,
        input_snapshot_hash=initial.input_snapshot_hash,
        registry_head_hash=final.completion_event_hash,
        terminal_artifact_hash=final.terminal_artifact_hash,
        tool_results=results,
        rounds=rounds,
    )
    stored = _read_typed(directory / "reports" / "proposal_verification.json", ProposalVerification)
    if recomputed != stored:
        raise AgentRunError("stored deterministic verification does not recompute")
    projection = ledger.project()
    state = projection.studies.get(study.study_id)
    if state is None or not any(
        value.attempt_id == initial.attempt.attempt_id
        and value.verification_hash == final.verification_hash
        and value.verification_passed == stored.passed
        for value in state.attempts
    ):
        raise AgentRunError("registry does not bind the stored verification")
    return stored
