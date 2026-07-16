from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nlp_trader.config import ResearchConfig
from nlp_trader.research_agents.compiler import (
    compile_verified_proposal,
    load_compiled_execution_definition,
)
from nlp_trader.research_agents.contracts import (
    ContractIdentity,
    CounterevidenceSearchRecord,
    ParameterChoice,
    ProposalCheck,
    ProposalVerification,
    ResearchProposal,
    StudyDefinition,
    canonical_json,
)


def _proposal(study: StudyDefinition, attempt_id: str) -> ResearchProposal:
    return ResearchProposal(
        study_id=study.study_id,
        attempt_id=attempt_id,
        bundle_id="1" * 64,
        input_snapshot_hash="2" * 64,
        hypothesis="Causal text may add ranked predictive information.",
        mechanism="Source updates may precede slower market measures.",
        affected_universe_id=study.universe_snapshot_id,
        horizon_sessions=study.horizon_sessions,
        target_family=study.target_family,
        expected_direction="positive",
        direction_is_hypothesis=True,
        supporting_evidence_ids=("3" * 64,),
        counterevidence_searches=(
            CounterevidenceSearchRecord(
                query_id="4" * 64,
                normalized_query_hash="5" * 64,
                filters_hash="6" * 64,
                result_count=0,
                pages_inspected=1,
                reason="no_result",
            ),
        ),
        falsification_conditions=("Matched diagnostics do not support the relation.",),
        invalidation_conditions=("Source availability cannot be established.",),
        required_input_ids=("causal-text",),
        availability_requirements=("Every source satisfies available_at <= asof_ts per row.",),
        experiment_template_id="matched_feature_ablation_v1",
        parameter_choices=(ParameterChoice(parameter_id="text_decay_days", value=5),),
        required_learned_families=study.required_learned_families,
        required_fixed_benchmarks=study.required_fixed_benchmarks,
        negative_controls=study.required_negative_controls,
        sensitivity_checks=study.required_robustness_checks,
        expected_failure_modes=("Sparse evidence may weaken stability.",),
        known_limitations=("The evidence snapshot is bounded.",),
        expected_output_metrics=study.required_metrics,
        acceptance_interpretation="Results remain hypothetical and require human review.",
    )


def test_compiler_maps_only_frozen_typed_template_fields(
    tmp_path: Path,
    generated_config: ResearchConfig,
    research_study_definition: StudyDefinition,
) -> None:
    study = research_study_definition
    proposal = _proposal(study, "7" * 64)
    encoded = (proposal.canonical_json() + "\n").encode()
    proposal_hash = hashlib.sha256(encoded).hexdigest()
    verification = ProposalVerification(
        study_id=study.study_id,
        attempt_id=proposal.attempt_id,
        terminal_artifact_hash=proposal_hash,
        registry_head_hash="8" * 64,
        bundle_id=proposal.bundle_id,
        verifier_contract=study.verifier_contract,
        passed=True,
        checks=(ProposalCheck(check_id="all-hard-checks", passed=True),),
    )
    definition_path, definition = compile_verified_proposal(
        artifact_root=(tmp_path / "agent-artifacts").resolve(),
        study=study,
        proposal=proposal,
        proposal_artifact_hash=proposal_hash,
        verification=verification,
        verification_artifact_hash="9" * 64,
        base_config=generated_config,
        compiler_contract=ContractIdentity(
            contract_id="matched-template-compiler",
            version="v1",
            sha256="a" * 64,
        ),
    )

    assert definition.evaluation_scope == "development_only"
    assert definition.reserved_decision_boundary == study.reserved_holdout_decisions
    assert definition.typed_patches[0].field_path == "features.text_decay_half_life_days"
    assert {path.name for path in definition_path.iterdir()} == {
        "execution_definition.json",
        "base_config.snapshot.json",
        "compiler_provenance.json",
        "manifest.json",
    }
    assert (
        load_compiled_execution_definition(
            definition_path.parent.parent,
            definition.definition_id,
        )
        == definition
    )

    bad_payload = proposal.model_dump(mode="python", exclude={"proposal_id"})
    bad_payload["parameter_choices"] = (ParameterChoice(parameter_id="text_decay_days", value=99),)
    bad = ResearchProposal.model_validate(bad_payload)
    bad_hash = hashlib.sha256((bad.canonical_json() + "\n").encode()).hexdigest()
    bad_verification = verification.model_copy(
        update={"terminal_artifact_hash": bad_hash, "verification_id": ""}
    )
    with pytest.raises(ValueError, match="maximum"):
        compile_verified_proposal(
            artifact_root=(tmp_path / "other-artifacts").resolve(),
            study=study,
            proposal=bad,
            proposal_artifact_hash=bad_hash,
            verification=bad_verification,
            verification_artifact_hash="9" * 64,
            base_config=generated_config,
            compiler_contract=definition.compiler_contract,
        )

    snapshot_path = definition_path / "base_config.snapshot.json"
    manifest_path = definition_path / "manifest.json"
    snapshot = json.loads(snapshot_path.read_bytes())
    snapshot["mode"] = "full"
    tampered_snapshot = (canonical_json(snapshot) + "\n").encode("utf-8")
    snapshot_path.write_bytes(tampered_snapshot)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_compiled_execution_definition(
            definition_path.parent.parent,
            definition.definition_id,
        )

    manifest = json.loads(manifest_path.read_bytes())
    snapshot_entry = next(
        value
        for value in manifest["files"]
        if value["relative_path"] == "base_config.snapshot.json"
    )
    snapshot_entry["bytes"] = len(tampered_snapshot)
    snapshot_entry["sha256"] = hashlib.sha256(tampered_snapshot).hexdigest()
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base config snapshot"):
        load_compiled_execution_definition(
            definition_path.parent.parent,
            definition.definition_id,
        )
