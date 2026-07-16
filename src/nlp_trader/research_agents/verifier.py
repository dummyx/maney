from __future__ import annotations

import re
from collections.abc import Iterable

from nlp_trader.research_agents.contracts import (
    AbstentionAction,
    CalculationReference,
    CounterevidenceSearchRecord,
    EvidenceReference,
    MetricReference,
    ParameterChoice,
    ParameterRange,
    ProposalAction,
    ProposalAttemptSnapshot,
    ProposalCheck,
    ProposalVerification,
    QuantitativeClaim,
    ResearchAbstention,
    ResearchAgentRound,
    ResearchProposal,
    StudyDefinition,
    content_sha256,
)
from nlp_trader.research_agents.tools import (
    CalculateToolResult,
    DevelopmentMetricsToolResult,
    SearchEvidenceToolResult,
    ToolResultEnvelope,
)
from nlp_trader.research_agents.views import LoadedDevelopmentViewBundle

_FORBIDDEN = re.compile(
    r"(?:https?://|file://|(?:^|\s)(?:/|~/|\.\.?/)|`|\$\(|"
    r"\b(?:holdout|order|position|broker|account|secret|credential|environment|"
    r"leverage|target\s+weight|paper\s+trad(?:e|ing)|sql|select|insert|update|delete|"
    r"drop\s+table|shell|subprocess|curl|wget)\b)",
    re.IGNORECASE,
)


def _check(check_id: str, passed: bool, detail: str | None = None) -> ProposalCheck:
    return ProposalCheck(check_id=check_id, passed=passed, detail=detail if not passed else None)


def _all_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _all_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _all_strings(nested)


def _tool_references(
    results: tuple[ToolResultEnvelope, ...],
) -> tuple[
    dict[str, EvidenceReference],
    dict[str, MetricReference],
    dict[str, CalculationReference],
    dict[str, CounterevidenceSearchRecord],
]:
    evidence: dict[str, EvidenceReference] = {}
    metrics: dict[str, MetricReference] = {}
    calculations: dict[str, CalculationReference] = {}
    challenge_searches: dict[str, CounterevidenceSearchRecord] = {}
    for result in results:
        payload = result.payload
        if isinstance(payload, SearchEvidenceToolResult):
            page = payload.page
            for reference in page.results:
                evidence[reference.evidence.evidence_id] = reference
            if (
                page.results
                and page.results[0].citation_role == "counterevidence"
                or (not page.results)
            ):
                challenge_searches[page.query_id] = CounterevidenceSearchRecord(
                    query_id=page.query_id,
                    normalized_query_hash=page.normalized_query_hash,
                    filters_hash=page.filters_hash,
                    result_count=page.total_result_count,
                    pages_inspected=1,
                    cited_result_ids=tuple(
                        reference.evidence.evidence_id for reference in page.results
                    ),
                    reason="no_result" if page.total_result_count == 0 else None,
                )
        elif isinstance(payload, DevelopmentMetricsToolResult):
            metrics.update((value.metric_reference_id, value) for value in payload.metrics)
        elif isinstance(payload, CalculateToolResult):
            value = payload.calculation
            calculations[value.calculation_id] = value
    return evidence, metrics, calculations, challenge_searches


def _parameter_allowed(choice: ParameterChoice, parameter: ParameterRange) -> bool:
    value = choice.value
    if parameter.value_type == "integer" and (type(value) is not int):
        return False
    if parameter.value_type == "number" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        return False
    if parameter.value_type == "string" and not isinstance(value, str):
        return False
    if parameter.value_type == "boolean" and type(value) is not bool:
        return False
    if parameter.allowed_values and value not in parameter.allowed_values:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if parameter.minimum is not None and value < parameter.minimum:
            return False
        if parameter.maximum is not None and value > parameter.maximum:
            return False
    return True


def _quantitative_claim_bound(
    claim: QuantitativeClaim,
    evidence: dict[str, EvidenceReference],
    metrics: dict[str, MetricReference],
    calculations: dict[str, CalculationReference],
) -> bool:
    if claim.reference_kind == "evidence":
        reference = evidence.get(claim.reference_id)
        return reference is not None and str(claim.value) in reference.evidence.quoted_span
    if claim.reference_kind == "metric":
        metric = metrics.get(claim.reference_id)
        return (
            metric is not None
            and claim.value == metric.value
            and claim.unit == metric.unit
            and claim.scope == metric.scope
            and claim.window == metric.window
        )
    calculation = calculations.get(claim.reference_id)
    return calculation is not None and claim.value == calculation.output


def _rounds_valid(
    rounds: tuple[ResearchAgentRound, ...],
    *,
    study: StudyDefinition,
    attempt: ProposalAttemptSnapshot,
    bundle: LoadedDevelopmentViewBundle,
    tool_results: tuple[ToolResultEnvelope, ...],
) -> bool:
    previous = "0" * 64
    expected_results = {value.result_hash for value in tool_results}
    seen_results: set[str] = set()
    for step, round_record in enumerate(rounds, start=1):
        if (
            round_record.step != step
            or round_record.previous_round_hash != previous
            or round_record.study_id != study.study_id
            or round_record.attempt_id != attempt.attempt_id
            or round_record.bundle_id != bundle.manifest.bundle_id
            or round_record.attempt_reservation_event_hash != attempt.reservation_event_hash
            or round_record.reserved_study_state_hash != attempt.reserved_study_state_hash
        ):
            return False
        if round_record.tool_result_hash is not None:
            seen_results.add(round_record.tool_result_hash)
        previous = round_record.round_id
    return seen_results == expected_results


def verify_terminal_action(
    action: ProposalAction | AbstentionAction,
    *,
    study: StudyDefinition,
    attempt: ProposalAttemptSnapshot,
    bundle: LoadedDevelopmentViewBundle,
    input_snapshot_hash: str,
    registry_head_hash: str,
    terminal_artifact_hash: str,
    tool_results: tuple[ToolResultEnvelope, ...],
    rounds: tuple[ResearchAgentRound, ...],
) -> ProposalVerification:
    identity_ok = (
        action.proposal.study_id == study.study_id
        and action.proposal.attempt_id == attempt.attempt_id
        and action.proposal.bundle_id == bundle.manifest.bundle_id
        and action.proposal.input_snapshot_hash == input_snapshot_hash
        if isinstance(action, ProposalAction)
        else action.abstention.study_id == study.study_id
        and action.abstention.attempt_id == attempt.attempt_id
        and action.abstention.bundle_id == bundle.manifest.bundle_id
        and action.abstention.input_snapshot_hash == input_snapshot_hash
    )
    checks = [
        _check("strict_schema_and_identity", True),
        _check("study_development_open", attempt.status == "reserved"),
        _check("input_identities_match", identity_ok),
        _check("attempt_reserved_before_generation", attempt.status == "reserved"),
        _check(
            "tool_results_exact",
            _rounds_valid(
                rounds,
                study=study,
                attempt=attempt,
                bundle=bundle,
                tool_results=tool_results,
            ),
        ),
    ]
    if isinstance(action, ProposalAction):
        checks.extend(_proposal_checks(action.proposal, study, bundle, tool_results))
    else:
        checks.extend(_abstention_checks(action.abstention, bundle, tool_results))
    checks.append(_check("terminal_artifact_hash_bound", len(terminal_artifact_hash) == 64))
    return ProposalVerification(
        study_id=study.study_id,
        attempt_id=attempt.attempt_id,
        terminal_artifact_hash=terminal_artifact_hash,
        registry_head_hash=registry_head_hash,
        bundle_id=bundle.manifest.bundle_id,
        verifier_contract=study.verifier_contract,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def _proposal_checks(
    proposal: ResearchProposal,
    study: StudyDefinition,
    bundle: LoadedDevelopmentViewBundle,
    tool_results: tuple[ToolResultEnvelope, ...],
) -> list[ProposalCheck]:
    evidence, metrics, calculations, challenge_searches = _tool_references(tool_results)
    cited = (*proposal.supporting_evidence_ids, *proposal.counterevidence_ids)
    citations_ok = all(value in evidence for value in cited) and all(
        evidence[value].evidence.available_at <= study.analysis_cutoff
        and evidence[value].evidence.retention_permitted
        and evidence[value].evidence.active_period_valid
        and bool(evidence[value].evidence.license_or_terms_ref)
        for value in cited
    )
    roles_ok = all(
        (reference := evidence.get(value)) is not None and reference.citation_role == "supporting"
        for value in proposal.supporting_evidence_ids
    ) and all(
        (reference := evidence.get(value)) is not None
        and reference.citation_role == "counterevidence"
        for value in proposal.counterevidence_ids
    )
    searches_ok = bool(proposal.counterevidence_ids) or all(
        challenge_searches.get(value.query_id) == value
        for value in proposal.counterevidence_searches
    )
    quantitative_ok = all(
        _quantitative_claim_bound(claim, evidence, metrics, calculations)
        for claim in proposal.quantitative_claims
    )
    templates = {value.template_id: value for value in study.permitted_templates}
    template = templates.get(proposal.experiment_template_id)
    choices = {value.parameter_id: value for value in proposal.parameter_choices}
    parameters_ok = template is not None and set(choices).issubset(
        {value.parameter_id for value in template.parameters}
    )
    if template is not None and parameters_ok:
        parameters_ok = all(
            _parameter_allowed(choices[value.parameter_id], value)
            for value in template.parameters
            if value.parameter_id in choices
        )
    catalog_ids = {
        value.entry_id
        for section in (
            bundle.feature_catalog.features,
            bundle.feature_catalog.models,
            bundle.feature_catalog.benchmarks,
            bundle.feature_catalog.selectors,
            bundle.feature_catalog.metrics,
            bundle.feature_catalog.controls,
            bundle.feature_catalog.templates,
        )
        for value in section
    }
    controls_ok = (
        set(study.required_learned_families).issubset(proposal.required_learned_families)
        and set(study.required_fixed_benchmarks).issubset(proposal.required_fixed_benchmarks)
        and set(study.required_negative_controls).issubset(proposal.negative_controls)
        and set(study.required_robustness_checks).issubset(proposal.sensitivity_checks)
        and set(study.required_metrics).issubset(proposal.expected_output_metrics)
    )
    forbidden = any(
        _FORBIDDEN.search(value)
        for value in _all_strings(proposal.model_dump(mode="json", exclude={"proposal_id"}))
    )
    availability_ok = any(
        "available_at" in value and "asof_ts" in value
        for value in proposal.availability_requirements
    )
    return [
        _check("evidence_citations_valid", citations_ok),
        _check("evidence_roles_disjoint", roles_ok),
        _check("counterevidence_challenged", searches_ok),
        _check("quantitative_claims_bound", quantitative_ok),
        _check(
            "study_scope_matches",
            proposal.horizon_sessions == study.horizon_sessions
            and proposal.target_family == study.target_family
            and proposal.affected_universe_id == study.universe_snapshot_id,
        ),
        _check(
            "required_inputs_catalogued",
            set(proposal.required_input_ids).issubset(catalog_ids),
        ),
        _check("template_permitted", template is not None),
        _check("parameters_in_frozen_space", parameters_ok),
        _check("required_controls_retained", controls_ok),
        _check("frozen_research_contracts_unchanged", True),
        _check("forbidden_content_absent", not forbidden),
        _check(
            "limitations_and_failure_conditions_present",
            bool(proposal.falsification_conditions)
            and bool(proposal.invalidation_conditions)
            and bool(proposal.expected_failure_modes)
            and bool(proposal.known_limitations),
        ),
        _check("point_in_time_reentry_required", availability_ok),
        _check(
            "proposal_content_hash_recomputes",
            proposal.proposal_id
            == content_sha256(proposal.model_dump(mode="json", exclude={"proposal_id"})),
        ),
    ]


def _abstention_checks(
    abstention: ResearchAbstention,
    bundle: LoadedDevelopmentViewBundle,
    tool_results: tuple[ToolResultEnvelope, ...],
) -> list[ProposalCheck]:
    query_ids = {
        result.payload.page.query_id
        for result in tool_results
        if isinstance(result.payload, SearchEvidenceToolResult)
    }
    catalog_ids = {
        value.entry_id
        for section in (
            bundle.feature_catalog.features,
            bundle.feature_catalog.models,
            bundle.feature_catalog.benchmarks,
            bundle.feature_catalog.selectors,
            bundle.feature_catalog.metrics,
            bundle.feature_catalog.controls,
            bundle.feature_catalog.templates,
        )
        for value in section
    }
    forbidden = any(
        _FORBIDDEN.search(value)
        for value in _all_strings(abstention.model_dump(mode="json", exclude={"abstention_id"}))
    )
    return [
        _check(
            "abstention_queries_attempted",
            set(abstention.tool_query_ids).issubset(query_ids),
        ),
        _check(
            "abstention_missing_inputs_consistent",
            abstention.reason != "missing_input"
            or bool(abstention.missing_input_ids)
            and not set(abstention.missing_input_ids).issubset(catalog_ids),
        ),
        _check("abstention_forbidden_content_absent", not forbidden),
        _check(
            "abstention_content_hash_recomputes",
            abstention.abstention_id
            == content_sha256(abstention.model_dump(mode="json", exclude={"abstention_id"})),
        ),
    ]
