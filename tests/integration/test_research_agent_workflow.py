from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nlp_trader.nlp.local_generation import GenerationRequest, GenerationResponse
from nlp_trader.research_agents.config import ResearchAgentConfig
from nlp_trader.research_agents.contracts import (
    CounterevidenceSearchRecord,
    ParameterChoice,
    ProposalAction,
    ResearchProposal,
    SearchEvidenceRequest,
    SearchEvidenceToolCall,
    StudyDefinition,
    ToolCallAction,
    canonical_json,
)
from nlp_trader.research_agents.registry import ResearchRegistryLedger
from nlp_trader.research_agents.runner import (
    AgentRunFinal,
    replay_agent_run,
    run_research_agent,
    verify_stored_run,
)
from nlp_trader.research_agents.tools import ResearchToolGateway, ToolLimits
from nlp_trader.research_agents.verifier import verify_terminal_action
from nlp_trader.research_agents.views import load_development_view_bundle


def _config(tmp_path: Path, study: StudyDefinition) -> ResearchAgentConfig:
    return ResearchAgentConfig(
        enabled=False,
        model_path=None,
        model_logical_id=study.model.logical_id,
        model_revision=study.model.revision,
        model_expected_sha256=study.model.file_sha256,
        model_license_or_terms_ref=study.model.license_or_terms_ref,
        prompt_version=study.prompt_contract.version,
        action_schema_version=study.action_schema_contract.version,
        proposal_schema_version=study.proposal_schema_contract.version,
        tool_catalog_version=study.tool_catalog_contract.version,
        verifier_version=study.verifier_contract.version,
        runtime_version="research-agent-runtime-v1",
        artifact_root=(tmp_path / "agent-artifacts").resolve(),
        environment_scrub_policy_version="agent-env-scrub-v1",
    )


def test_bounded_analyst_proposal_is_verified_persisted_and_replayable(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
    research_agent_bundle_factory: Callable[..., tuple[Path, object]],
) -> None:
    study = research_study_definition
    config = _config(tmp_path, study)
    ledger = ResearchRegistryLedger(config.artifact_root)
    ledger.register_study(
        study,
        expected_head_hash=ledger.head_hash(),
        actor_label="test-reviewer",
    )
    reservation = ledger.reserve_proposal_attempt(
        study.study_id,
        expected_head_hash=ledger.head_hash(),
        actor_label="test-host",
    )
    attempt = ledger.project().studies[study.study_id].attempts[0]
    assert reservation.event_hash == attempt.reservation_event_hash
    bundle_path, _ = research_agent_bundle_factory(
        tmp_path,
        study,
        adversarial_evidence=True,
    )
    bundle = load_development_view_bundle(bundle_path)
    support_call = ToolCallAction(
        tool_call=SearchEvidenceToolCall(
            request=SearchEvidenceRequest(
                query="Demand commentary weakened",
                purpose="support",
                asset_ids=("asset-a",),
                result_limit=5,
            )
        )
    )
    challenge_call = ToolCallAction(
        tool_call=SearchEvidenceToolCall(
            request=SearchEvidenceRequest(
                query="Alternative demand explanation",
                purpose="challenge",
                source_types=("unlisted-source",),
                result_limit=5,
            )
        )
    )
    gateway = ResearchToolGateway(
        bundle,
        ToolLimits(max_evidence_results=10, max_result_bytes=64_000),
    )
    support_result = gateway.execute(support_call)
    challenge_result = gateway.execute(challenge_call)
    support_page = support_result.payload.page  # type: ignore[union-attr]
    challenge_page = challenge_result.payload.page  # type: ignore[union-attr]
    assert support_page.results and challenge_page.total_result_count == 0

    from nlp_trader.research_agents.prompts import input_snapshot_hash

    snapshot_hash = input_snapshot_hash(
        study,
        bundle,
        attempt_id=attempt.attempt_id,
        reserved_study_state_hash=attempt.reserved_study_state_hash,
    )
    proposal = ResearchProposal(
        study_id=study.study_id,
        attempt_id=attempt.attempt_id,
        bundle_id=bundle.manifest.bundle_id,
        input_snapshot_hash=snapshot_hash,
        hypothesis="Causal source tone may add ranked predictive information.",
        mechanism="Source updates may reach expectations before slower market measures.",
        affected_universe_id=study.universe_snapshot_id,
        horizon_sessions=study.horizon_sessions,
        target_family=study.target_family,
        expected_direction="positive",
        direction_is_hypothesis=True,
        supporting_evidence_ids=(support_page.results[0].evidence.evidence_id,),
        counterevidence_searches=(
            CounterevidenceSearchRecord(
                query_id=challenge_page.query_id,
                normalized_query_hash=challenge_page.normalized_query_hash,
                filters_hash=challenge_page.filters_hash,
                result_count=0,
                pages_inspected=1,
                reason="no_result",
            ),
        ),
        falsification_conditions=("Matched diagnostics do not support the hypothesized relation.",),
        invalidation_conditions=("Historical source availability cannot be established.",),
        required_input_ids=("causal-text",),
        availability_requirements=("Every source must satisfy available_at <= asof_ts per row.",),
        experiment_template_id="matched_feature_ablation_v1",
        parameter_choices=(ParameterChoice(parameter_id="text_decay_days", value=5),),
        required_learned_families=study.required_learned_families,
        required_fixed_benchmarks=study.required_fixed_benchmarks,
        negative_controls=study.required_negative_controls,
        sensitivity_checks=study.required_robustness_checks,
        expected_failure_modes=("Sparse retained evidence may weaken stability.",),
        known_limitations=("The evidence snapshot is intentionally bounded.",),
        expected_output_metrics=study.required_metrics,
        acceptance_interpretation="Results remain hypothetical and require human review.",
    )
    outputs = [
        support_call.model_dump(mode="json"),
        challenge_call.model_dump(mode="json"),
        ProposalAction(proposal=proposal).model_dump(mode="json"),
    ]

    def generator(requests: list[GenerationRequest]) -> list[GenerationResponse]:
        payload = outputs.pop(0)
        return [
            GenerationResponse(
                request_id=requests[0].request_id,
                generated_text=canonical_json(payload),
                input_token_count=20,
                output_token_count=10,
                generation_latency_seconds=0.1,
            )
        ]

    result = run_research_agent(
        config=config,
        ledger=ledger,
        study=study,
        bundle=bundle,
        attempt_id=attempt.attempt_id,
        generator=generator,
    )

    assert isinstance(result.final, AgentRunFinal)
    assert result.final.outcome == "proposal_verified"
    assert result.verification is not None and result.verification.passed
    assert len(replay_agent_run(result.run_dir)) == 3
    assert verify_stored_run(
        result.run_dir,
        ledger=ledger,
        study=study,
        bundle=bundle,
    ).passed

    def rebuilt_proposal(**updates: object) -> ResearchProposal:
        payload = proposal.model_dump(mode="python", exclude={"proposal_id"})
        payload.update(updates)
        return ResearchProposal.model_validate(payload)

    unknown_citation = rebuilt_proposal(supporting_evidence_ids=("f" * 64,))
    unknown_verification = verify_terminal_action(
        ProposalAction(proposal=unknown_citation),
        study=study,
        attempt=attempt,
        bundle=bundle,
        input_snapshot_hash=snapshot_hash,
        registry_head_hash=ledger.head_hash(),
        terminal_artifact_hash="e" * 64,
        tool_results=result.tool_results,
        rounds=result.rounds,
    )
    unknown_checks = {value.check_id: value.passed for value in unknown_verification.checks}
    assert unknown_checks["evidence_citations_valid"] is False
    assert unknown_checks["evidence_roles_disjoint"] is False

    invented_search = rebuilt_proposal(
        counterevidence_searches=(
            CounterevidenceSearchRecord(
                query_id=challenge_page.query_id,
                normalized_query_hash=challenge_page.normalized_query_hash,
                filters_hash=challenge_page.filters_hash,
                result_count=1,
                pages_inspected=1,
                reason="insufficient_result",
            ),
        )
    )
    invented_verification = verify_terminal_action(
        ProposalAction(proposal=invented_search),
        study=study,
        attempt=attempt,
        bundle=bundle,
        input_snapshot_hash=snapshot_hash,
        registry_head_hash=ledger.head_hash(),
        terminal_artifact_hash="d" * 64,
        tool_results=result.tool_results,
        rounds=result.rounds,
    )
    invented_checks = {value.check_id: value.passed for value in invented_verification.checks}
    assert invented_checks["counterevidence_challenged"] is False
    assert not outputs
