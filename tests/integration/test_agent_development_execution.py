from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nlp_trader.config import ResearchConfig
from nlp_trader.data.parquet import read_partitioned_parquet
from nlp_trader.experiment_execution import run_approved_development
from nlp_trader.research import sha256_file
from nlp_trader.research_agents.approvals import (
    approve_development_execution,
    freeze_candidate,
)
from nlp_trader.research_agents.compiler import (
    ExperimentExecutionDefinition,
    compile_verified_proposal,
)
from nlp_trader.research_agents.contracts import (
    ContractIdentity,
    CounterevidenceSearchRecord,
    ParameterChoice,
    ProposalCheck,
    ProposalVerification,
    ResearchProposal,
    StudyDefinition,
    TimeRange,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_agents.registry import ResearchRegistryLedger
from nlp_trader.timestamps import parse_utc


def _aligned_study(template: StudyDefinition) -> StudyDefinition:
    payload = template.model_dump(mode="python", exclude={"study_id"})
    payload.update(
        {
            "created_at": datetime(2026, 7, 16, tzinfo=UTC),
            "analysis_cutoff": datetime(2026, 7, 14, 23, 59, tzinfo=UTC),
            "development_decisions": TimeRange(
                start=datetime(2026, 6, 29, tzinfo=UTC),
                end=datetime(2026, 7, 14, 23, 59, tzinfo=UTC),
            ),
            "reserved_holdout_decisions": TimeRange(
                start=datetime(2026, 7, 15, tzinfo=UTC),
                end=datetime(2026, 7, 31, tzinfo=UTC),
            ),
            "reserved_holdout_outcomes": TimeRange(
                start=datetime(2026, 7, 16, tzinfo=UTC),
                end=datetime(2026, 8, 3, tzinfo=UTC),
            ),
        }
    )
    return StudyDefinition.model_validate(payload)


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


def test_approved_development_run_emits_no_reserved_result_and_freezes_once(
    tmp_path: Path,
    generated_config: ResearchConfig,
    research_study_definition: StudyDefinition,
) -> None:
    study = _aligned_study(research_study_definition)
    ledger = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve())
    ledger.register_study(
        study,
        expected_head_hash=ledger.head_hash(),
        actor_label="reviewer",
    )
    ledger.reserve_proposal_attempt(
        study.study_id,
        expected_head_hash=ledger.head_hash(),
        actor_label="host",
    )
    attempt = ledger.project().studies[study.study_id].attempts[0]
    proposal = _proposal(study, attempt.attempt_id)
    proposal_hash = hashlib.sha256((proposal.canonical_json() + "\n").encode()).hexdigest()
    proposal_path = (
        ledger.artifact_root / "runs" / "verified-proposal-run" / "reports" / "proposal.json"
    )
    proposal_path.parent.mkdir(parents=True)
    proposal_path.write_text(proposal.canonical_json() + "\n", encoding="utf-8")
    ledger.complete_proposal_attempt(
        study.study_id,
        attempt.attempt_id,
        outcome="proposal",
        agent_run_id="verified-proposal-run",
        detail="synthetic verified proposal fixture",
        terminal_artifact_hash=proposal_hash,
        expected_head_hash=ledger.head_hash(),
        actor_label="host",
    )
    verification = ProposalVerification(
        study_id=study.study_id,
        attempt_id=attempt.attempt_id,
        terminal_artifact_hash=proposal_hash,
        registry_head_hash=ledger.head_hash(),
        bundle_id=proposal.bundle_id,
        verifier_contract=study.verifier_contract,
        passed=True,
        checks=(ProposalCheck(check_id="all-hard-checks", passed=True),),
    )
    verification_hash = "9" * 64
    ledger.record_proposal_verification(
        study.study_id,
        attempt.attempt_id,
        terminal_artifact_hash=proposal_hash,
        verification_hash=verification_hash,
        passed=True,
        expected_head_hash=ledger.head_hash(),
        actor_label="host",
    )
    compiled_root, definition = compile_verified_proposal(
        artifact_root=ledger.artifact_root,
        study=study,
        proposal=proposal,
        proposal_artifact_hash=proposal_hash,
        verification=verification,
        verification_artifact_hash=verification_hash,
        base_config=generated_config,
        compiler_contract=ContractIdentity(
            contract_id="matched-template-compiler",
            version="v1",
            sha256="a" * 64,
        ),
    )
    for patch_value in (6, 999):
        forged_payload = definition.model_dump(mode="python", exclude={"definition_id"})
        forged_payload["typed_patches"] = (
            {
                "field_path": "features.text_decay_half_life_days",
                "value": patch_value,
            },
        )
        forged = ExperimentExecutionDefinition.model_validate(forged_payload)
        forged_root = compiled_root.parent / forged.definition_id
        shutil.copytree(compiled_root, forged_root)
        definition_bytes = (forged.canonical_json() + "\n").encode("utf-8")
        (forged_root / "execution_definition.json").write_bytes(definition_bytes)
        manifest_path = forged_root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["definition_id"] = forged.definition_id
        execution_entry = next(
            value
            for value in manifest["files"]
            if value["relative_path"] == "execution_definition.json"
        )
        execution_entry["bytes"] = len(definition_bytes)
        execution_entry["sha256"] = hashlib.sha256(definition_bytes).hexdigest()
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="patches do not match"):
            approve_development_execution(
                ledger,
                study=study,
                definition=forged,
                verification=verification,
                verification_artifact_hash=verification_hash,
                actor_label="reviewer",
                reviewer_reason="A forged patch must not acquire authority.",
            )

    boundary_payload = definition.model_dump(mode="python", exclude={"definition_id"})
    boundary_payload["development_decisions"] = TimeRange(
        start=study.development_decisions.start,
        end=datetime(2026, 7, 13, 23, 59, tzinfo=UTC),
    )
    forged_boundary = ExperimentExecutionDefinition.model_validate(boundary_payload)
    with pytest.raises(ValueError, match="bind one passed proposal"):
        approve_development_execution(
            ledger,
            study=study,
            definition=forged_boundary,
            verification=verification,
            verification_artifact_hash=verification_hash,
            actor_label="reviewer",
            reviewer_reason="A forged date boundary must not acquire authority.",
        )
    assert ledger.project().studies[study.study_id].state == "development_open"

    approval = approve_development_execution(
        ledger,
        study=study,
        definition=definition,
        verification=verification,
        verification_artifact_hash=verification_hash,
        actor_label="reviewer",
        reviewer_reason="Run the exact verified development-only definition.",
    )
    approval_path = (
        ledger.artifact_root
        / "studies"
        / study.study_id
        / "approvals"
        / f"{approval.approval_id}.json"
    )
    approval_bytes = approval_path.read_bytes()
    tampered_approval = json.loads(approval_bytes)
    tampered_approval["reviewer_reason"] = "tampered immutable approval"
    approval_path.write_text(canonical_json(tampered_approval) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="approval artifact does not match"):
        run_approved_development(
            ledger=ledger,
            study=study,
            definition=definition,
            approval=approval,
            base_config=generated_config,
        )
    approval_path.write_bytes(approval_bytes)

    fake_approval_payload = approval.model_dump(mode="python", exclude={"approval_id"})
    fake_approval_payload["approval_event_hash"] = "f" * 64
    fake_approval = type(approval).model_validate(fake_approval_payload)
    with pytest.raises(ValueError, match="approval event is missing"):
        run_approved_development(
            ledger=ledger,
            study=study,
            definition=definition,
            approval=fake_approval,
            base_config=generated_config,
        )
    assert ledger.project().studies[study.study_id].state == "development_locked"

    development_root, result, _ = run_approved_development(
        ledger=ledger,
        study=study,
        definition=definition,
        approval=approval,
        base_config=generated_config,
    )

    development_labels = read_partitioned_parquet(
        generated_config.paths.processed_dir / result.development_run_id / "gold" / "labels"
    )
    assert development_labels
    assert all(
        parse_utc(str(label[field_name])) < study.reserved_holdout_decisions.start
        for label in development_labels
        for field_name in ("label_end_ts", "label_available_at")
    )

    run_id = result.development_run_id
    pipeline_roots = (
        generated_config.paths.interim_dir / run_id,
        generated_config.paths.processed_dir / run_id,
        generated_config.paths.models_dir / run_id,
        generated_config.paths.reports_dir / run_id,
    )
    for root in pipeline_roots:
        for path in root.rglob("*"):
            assert "final_holdout" not in path.name
            if path.suffix == ".json" and path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if path.name == "config.snapshot.json":
                    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                    assert hashlib.sha256(encoded).hexdigest() == result.patched_config_hash
                    assert payload["models"]["final_holdout_periods"] >= 1
                else:
                    assert "final_holdout" not in json.dumps(payload, sort_keys=True)
    evaluation_contract = json.loads(
        (development_root / "required_evaluation_contract.json").read_text(encoding="utf-8")
    )
    assert evaluation_contract["development_execution_coverage"] == {
        "learned_and_fixed_families": "verified_in_backtest_comparison",
        "negative_controls": "predeclared_not_executed",
        "robustness_checks": "predeclared_not_executed",
    }
    assert content_sha256(evaluation_contract) == result.required_evaluation_contract_hash
    result_hash = sha256_file(development_root / "result_manifest.json")
    frozen_hash = sha256_file(development_root / "frozen_model.manifest.json")
    candidate_config = {
        "template_id": definition.template_id,
        "typed_patches": [value.model_dump(mode="json") for value in definition.typed_patches],
        "selected_family": "combined",
    }
    invalid_candidates = (
        {**candidate_config, "selected_family": "no_trade"},
        {
            **candidate_config,
            "typed_patches": [
                {
                    "field_path": "features.text_decay_half_life_days",
                    "value": 6,
                }
            ],
        },
    )
    for invalid_candidate in invalid_candidates:
        with pytest.raises(ValueError, match="candidate config"):
            freeze_candidate(
                ledger,
                study=study,
                proposal_hash=proposal_hash,
                definition=definition,
                approval=approval,
                development_result_manifest_hash=result_hash,
                frozen_model_manifest_hash=frozen_hash,
                candidate_config=invalid_candidate,
                required_evaluation_contract_hash=result.required_evaluation_contract_hash,
                input_snapshot_hashes=("b" * 64,),
                universe_asset_ids=("asset_aaa", "asset_bbb", "asset_ccc"),
                actor_label="reviewer",
                reviewer_reason="An invalid candidate must not poison the freeze event.",
            )
    with pytest.raises(ValueError, match="evaluation contract hash is not authoritative"):
        freeze_candidate(
            ledger,
            study=study,
            proposal_hash=proposal_hash,
            definition=definition,
            approval=approval,
            development_result_manifest_hash=result_hash,
            frozen_model_manifest_hash=frozen_hash,
            candidate_config=candidate_config,
            required_evaluation_contract_hash="0" * 64,
            input_snapshot_hashes=("b" * 64,),
            universe_asset_ids=("asset_aaa", "asset_bbb", "asset_ccc"),
            actor_label="reviewer",
            reviewer_reason="A forged contract hash must not poison the freeze event.",
        )
    evaluation_contract_path = development_root / "required_evaluation_contract.json"
    evaluation_contract_bytes = evaluation_contract_path.read_bytes()
    tampered_contract = json.loads(evaluation_contract_bytes)
    tampered_contract["development_execution_coverage"]["negative_controls"] = "executed"
    evaluation_contract_path.write_text(
        canonical_json(tampered_contract) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluation contract is not derived"):
        freeze_candidate(
            ledger,
            study=study,
            proposal_hash=proposal_hash,
            definition=definition,
            approval=approval,
            development_result_manifest_hash=result_hash,
            frozen_model_manifest_hash=frozen_hash,
            candidate_config=candidate_config,
            required_evaluation_contract_hash=result.required_evaluation_contract_hash,
            input_snapshot_hashes=("b" * 64,),
            universe_asset_ids=("asset_aaa", "asset_bbb", "asset_ccc"),
            actor_label="reviewer",
            reviewer_reason="A tampered contract artifact must not poison the freeze event.",
        )
    evaluation_contract_path.write_bytes(evaluation_contract_bytes)
    assert ledger.project().studies[study.study_id].state == "development_locked"
    assert all(event.payload.kind != "candidate_frozen" for event in ledger.replay())
    freeze = freeze_candidate(
        ledger,
        study=study,
        proposal_hash=proposal_hash,
        definition=definition,
        approval=approval,
        development_result_manifest_hash=result_hash,
        frozen_model_manifest_hash=frozen_hash,
        candidate_config=candidate_config,
        required_evaluation_contract_hash=result.required_evaluation_contract_hash,
        input_snapshot_hashes=("b" * 64,),
        universe_asset_ids=("asset_aaa", "asset_bbb", "asset_ccc"),
        actor_label="reviewer",
        reviewer_reason="Freeze the exact combined candidate after development review.",
    )

    assert ledger.project().studies[study.study_id].state == "candidate_frozen"
    assert freeze.holdout_identity.candidate_hash == freeze.candidate_config_hash
    with pytest.raises(
        ValueError,
        match="completed development|lineage|requires|authoritative study state",
    ):
        freeze_candidate(
            ledger,
            study=study,
            proposal_hash=proposal_hash,
            definition=definition,
            approval=approval,
            development_result_manifest_hash=result_hash,
            frozen_model_manifest_hash=frozen_hash,
            candidate_config={"selected_family": "text"},
            required_evaluation_contract_hash=result.required_evaluation_contract_hash,
            input_snapshot_hashes=("b" * 64,),
            universe_asset_ids=("asset_aaa",),
            actor_label="reviewer",
            reviewer_reason="Replacement must fail.",
        )
