from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_agent_development_execution import _aligned_study, _proposal

from nlp_trader.config import ResearchConfig
from nlp_trader.data.stores import ParquetFeatureStore
from nlp_trader.experiment_execution import run_approved_development
from nlp_trader.holdout_execution import reveal_frozen_holdout
from nlp_trader.research import input_manifest, sha256_file
from nlp_trader.research_agents.approvals import (
    CandidateFreezeRecord,
    approve_development_execution,
    freeze_candidate,
)
from nlp_trader.research_agents.audit import audit_completed_holdout
from nlp_trader.research_agents.compiler import compile_verified_proposal
from nlp_trader.research_agents.contracts import (
    ContractIdentity,
    ProposalCheck,
    ProposalVerification,
    StudyDefinition,
    canonical_json,
)
from nlp_trader.research_agents.registry import ResearchRegistryLedger


def test_one_time_holdout_uses_frozen_model_without_training_update_and_audits(
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
    _, definition = compile_verified_proposal(
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
    approval = approve_development_execution(
        ledger,
        study=study,
        definition=definition,
        verification=verification,
        verification_artifact_hash=verification_hash,
        actor_label="reviewer",
        reviewer_reason="Approve exact development definition.",
    )
    development_root, development, _ = run_approved_development(
        ledger=ledger,
        study=study,
        definition=definition,
        approval=approval,
        base_config=generated_config,
    )
    input_hashes = tuple(
        sorted(
            str(value["sha256"])
            for value in input_manifest(generated_config)
            if value.get("exists") and isinstance(value.get("sha256"), str)
        )
    )
    freeze = freeze_candidate(
        ledger,
        study=study,
        proposal_hash=proposal_hash,
        definition=definition,
        approval=approval,
        development_result_manifest_hash=sha256_file(development_root / "result_manifest.json"),
        frozen_model_manifest_hash=sha256_file(development_root / "frozen_model.manifest.json"),
        candidate_config={
            "template_id": definition.template_id,
            "typed_patches": [value.model_dump(mode="json") for value in definition.typed_patches],
            "selected_family": "combined",
        },
        required_evaluation_contract_hash=development.required_evaluation_contract_hash,
        input_snapshot_hashes=input_hashes,
        universe_asset_ids=("asset_aaa", "asset_bbb", "asset_ccc"),
        actor_label="reviewer",
        reviewer_reason="Freeze one exact candidate for one-time evaluation.",
    )

    for replacement in (
        _replace_freeze(freeze, execution_definition_hash="d" * 64),
        _replace_freeze(freeze, frozen_model_manifest_hash="e" * 64),
        _replace_freeze(freeze, freeze_event_hash="f" * 64),
    ):
        with pytest.raises(ValueError, match="authoritative registry"):
            reveal_frozen_holdout(
                ledger=ledger,
                freeze=replacement,
                development_root=development_root,
                base_config=generated_config,
                actor_label="reviewer",
            )
    alternate_root = tmp_path / "alternate-development-root"
    alternate_root.mkdir()
    with pytest.raises(ValueError, match="registry-authoritative run directory"):
        reveal_frozen_holdout(
            ledger=ledger,
            freeze=freeze,
            development_root=alternate_root,
            base_config=generated_config,
            actor_label="reviewer",
        )
    pristine = ledger.project().studies[study.study_id]
    assert pristine.state == "candidate_frozen"
    assert not ledger.project().holdout_use.overlapping(freeze.holdout_identity)

    holdout_root, result = reveal_frozen_holdout(
        ledger=ledger,
        freeze=freeze,
        development_root=development_root,
        base_config=generated_config,
        actor_label="reviewer",
    )

    assert result.training_updates == 0
    assert result.frozen_model_hash_before == result.frozen_model_hash_after
    assert generated_config.features.text_decay_half_life_days == 1.0
    assert result.base_config_hash == generated_config.content_hash()
    assert result.patched_config_hash != result.base_config_hash
    assert result.patched_config_hash == development.patched_config_hash
    holdout_features = ParquetFeatureStore(
        generated_config.paths.processed_dir / result.reservation_id / "gold" / "features"
    ).read_features()
    assert holdout_features
    assert {
        float(value)
        for row in holdout_features
        for key, value in row.items()
        if key.startswith("text_decay_half_life_days_")
    } == {5.0}
    assert ledger.project().studies[study.study_id].state == "holdout_revealed"
    assert len(ledger.project().holdout_use.overlapping(freeze.holdout_identity)) == 1
    pipeline_roots = (
        generated_config.paths.interim_dir / development.development_run_id,
        generated_config.paths.processed_dir / development.development_run_id,
        generated_config.paths.models_dir / development.development_run_id,
        generated_config.paths.reports_dir / development.development_run_id,
    )
    holdout_pipeline_roots = (
        generated_config.paths.interim_dir / result.reservation_id,
        generated_config.paths.processed_dir / result.reservation_id,
        generated_config.paths.models_dir / result.reservation_id,
        generated_config.paths.reports_dir / result.reservation_id,
    )
    pipeline_final_path = holdout_pipeline_roots[-1] / "run.final.json"
    pipeline_final = json.loads(pipeline_final_path.read_text(encoding="utf-8"))
    assert sha256_file(pipeline_final_path) == result.pipeline_run_final_manifest_hash
    assert pipeline_final["status"] == "complete"
    assert pipeline_final["completed_stage"] == "holdout_evaluation"
    assert pipeline_final["run_id"] == result.reservation_id
    assert pipeline_final["config_hash"] == result.patched_config_hash
    audit = audit_completed_holdout(
        ledger=ledger,
        freeze=freeze,
        development_root=development_root,
        holdout_root=holdout_root,
        holdout_pipeline_roots=holdout_pipeline_roots,
        development_pipeline_roots=pipeline_roots,
    )
    assert audit.passed
    assert next(
        value for value in audit.findings if value.check_id == "development_pipeline_run_bound"
    ).passed

    for artifact_name in ("predictions.json", "backtests.json", "metrics.json"):
        artifact_path = holdout_root / artifact_name
        original_bytes = artifact_path.read_bytes()
        tampered = json.loads(original_bytes)
        family_keys_before = (
            set(tampered["families"])
            if "families" in tampered
            else set(tampered["prediction"]) | set(tampered["portfolio"])
        )
        tampered["reservation_id"] = "canonical-value-tampering"
        artifact_path.write_text(canonical_json(tampered) + "\n", encoding="utf-8")
        reparsed = json.loads(artifact_path.read_bytes())
        family_keys_after = (
            set(reparsed["families"])
            if "families" in reparsed
            else set(reparsed["prediction"]) | set(reparsed["portfolio"])
        )
        assert family_keys_after == family_keys_before
        tampered_audit = audit_completed_holdout(
            ledger=ledger,
            freeze=freeze,
            development_root=development_root,
            holdout_root=holdout_root,
            holdout_pipeline_roots=holdout_pipeline_roots,
            development_pipeline_roots=pipeline_roots,
        )
        hash_finding = next(
            value
            for value in tampered_audit.findings
            if value.check_id == "holdout_result_artifact_hashes_bound"
        )
        assert hash_finding.severity == "critical"
        assert hash_finding.passed is False
        assert tampered_audit.passed is False
        if artifact_name == "backtests.json":
            assert next(
                value
                for value in tampered_audit.findings
                if value.check_id == "required_baselines_present"
            ).passed
        artifact_path.write_bytes(original_bytes)

    with pytest.raises(ValueError, match="candidate_frozen|consumed"):
        reveal_frozen_holdout(
            ledger=ledger,
            freeze=freeze,
            development_root=development_root,
            base_config=generated_config,
            actor_label="reviewer",
        )
    with pytest.raises(ValueError, match="overlaps"):
        ledger.register_external_holdout(
            freeze.holdout_identity,
            reason="overlapping external use must fail",
            expected_head_hash=ledger.head_hash(),
            actor_label="reviewer",
        )

    for pipeline_artifact in (
        pipeline_final_path,
        holdout_pipeline_roots[-1] / "run.initial.json",
    ):
        original_bytes = pipeline_artifact.read_bytes()
        tampered = json.loads(original_bytes)
        tampered["status"] = "tampered"
        pipeline_artifact.write_text(canonical_json(tampered) + "\n", encoding="utf-8")
        pipeline_violation = audit_completed_holdout(
            ledger=ledger,
            freeze=freeze,
            development_root=development_root,
            holdout_root=holdout_root,
            holdout_pipeline_roots=holdout_pipeline_roots,
            development_pipeline_roots=pipeline_roots,
        )
        pipeline_finding = next(
            value
            for value in pipeline_violation.findings
            if value.check_id == "holdout_pipeline_run_bound"
        )
        assert pipeline_finding.severity == "critical"
        assert pipeline_finding.passed is False
        assert pipeline_violation.passed is False
        pipeline_artifact.write_bytes(original_bytes)

    for development_pipeline_artifact in (
        pipeline_roots[-1] / "run.final.json",
        pipeline_roots[-1] / "run.initial.json",
    ):
        original_bytes = development_pipeline_artifact.read_bytes()
        tampered = json.loads(original_bytes)
        tampered["status"] = "tampered"
        development_pipeline_artifact.write_text(canonical_json(tampered) + "\n", encoding="utf-8")
        development_pipeline_violation = audit_completed_holdout(
            ledger=ledger,
            freeze=freeze,
            development_root=development_root,
            holdout_root=holdout_root,
            holdout_pipeline_roots=holdout_pipeline_roots,
            development_pipeline_roots=pipeline_roots,
        )
        development_pipeline_finding = next(
            value
            for value in development_pipeline_violation.findings
            if value.check_id == "development_pipeline_run_bound"
        )
        assert development_pipeline_finding.severity == "critical"
        assert development_pipeline_finding.passed is False
        assert development_pipeline_violation.passed is False
        development_pipeline_artifact.write_bytes(original_bytes)

    (development_root / "frozen_model.json").write_bytes(b"seeded audit violation")
    violated = audit_completed_holdout(
        ledger=ledger,
        freeze=freeze,
        development_root=development_root,
        holdout_root=holdout_root,
        holdout_pipeline_roots=holdout_pipeline_roots,
        development_pipeline_roots=pipeline_roots,
    )
    finding = next(
        value for value in violated.findings if value.check_id == "frozen_model_unchanged"
    )
    assert finding.passed is False
    assert violated.passed is False

    seeded_holdout_artifact = pipeline_roots[-1] / "final_holdout_seeded.json"
    seeded_holdout_artifact.write_text("{}\n", encoding="utf-8")
    development_violation = audit_completed_holdout(
        ledger=ledger,
        freeze=freeze,
        development_root=development_root,
        holdout_root=holdout_root,
        holdout_pipeline_roots=holdout_pipeline_roots,
        development_pipeline_roots=pipeline_roots,
    )
    assert (
        next(
            value
            for value in development_violation.findings
            if value.check_id == "development_reserved_result_absent"
        ).passed
        is False
    )
    seeded_agent_artifact = ledger.artifact_root / "runs" / "seeded-violation" / "leak.txt"
    seeded_agent_artifact.parent.mkdir(parents=True)
    seeded_agent_artifact.write_text(f"{result.manifest_id}\n", encoding="utf-8")
    analyst_violation = audit_completed_holdout(
        ledger=ledger,
        freeze=freeze,
        development_root=development_root,
        holdout_root=holdout_root,
        holdout_pipeline_roots=holdout_pipeline_roots,
        development_pipeline_roots=pipeline_roots,
    )
    assert (
        next(
            value
            for value in analyst_violation.findings
            if value.check_id == "analyst_holdout_result_unreachable"
        ).passed
        is False
    )


def _replace_freeze(
    freeze: CandidateFreezeRecord,
    **updates: object,
) -> CandidateFreezeRecord:
    payload = freeze.model_dump(mode="python", exclude={"freeze_id"})
    payload.update(updates)
    return CandidateFreezeRecord.model_validate(payload)
