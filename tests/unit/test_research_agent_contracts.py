from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from nlp_trader.config import ResearchConfig
from nlp_trader.research_agents.artifacts import (
    AgentArtifactError,
    ensure_agent_artifact_root,
    load_agent_manifest,
    write_agent_manifest_exclusive,
)
from nlp_trader.research_agents.config import load_research_agent_config
from nlp_trader.research_agents.contracts import (
    AgentArtifactManifest,
    ArtifactEntry,
    CounterevidenceSearchRecord,
    EvidenceRecord,
    HoldoutIdentity,
    RegistryEvent,
    ResearchAbstention,
    ResearchAgentAction,
    ResearchProposal,
    StudyDefinition,
    StudyRegisteredPayload,
)


def _holdout(
    study: StudyDefinition, *, assets: tuple[str, ...] = ("asset-a", "asset-b")
) -> HoldoutIdentity:
    return HoldoutIdentity(
        data_lineage_id=study.data_lineage_id,
        input_snapshot_hashes=("1" * 64, "2" * 64),
        universe_snapshot_id=study.universe_snapshot_id,
        universe_asset_ids=assets,
        calendar_contract=study.calendar_contract,
        market_data_contract=study.market_data_contract,
        label_contract=study.label_contract,
        target_family=study.target_family,
        horizon_sessions=study.horizon_sessions,
        return_adjustment_contract=study.return_adjustment_contract,
        decision_interval=study.reserved_holdout_decisions,
        outcome_interval=study.reserved_holdout_outcomes,
        study_id=study.study_id,
        candidate_hash="3" * 64,
    )


def test_study_and_holdout_ids_are_canonical_and_utc_normalized(
    research_study_definition: StudyDefinition,
) -> None:
    study = research_study_definition
    replayed = StudyDefinition.model_validate_json(study.canonical_json())
    holdout = _holdout(study)

    assert replayed == study
    assert study.study_id == study.computed_study_id()
    assert study.model_dump(mode="json")["created_at"].endswith("Z")
    assert holdout.holdout_id == holdout.computed_holdout_id()
    assert (
        HoldoutIdentity.model_validate_json(json.dumps(holdout.model_dump(mode="json"))) == holdout
    )


def test_study_contract_rejects_unknown_nonfinite_naive_and_leaking_boundaries(
    research_study_definition: StudyDefinition,
) -> None:
    payload = research_study_definition.model_dump(mode="python", exclude={"study_id"})
    payload["unknown"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        StudyDefinition.model_validate(payload)

    payload = research_study_definition.model_dump(mode="python", exclude={"study_id"})
    payload["analysis_cutoff"] = datetime(2026, 1, 2)
    with pytest.raises(ValidationError, match="timezone-aware"):
        StudyDefinition.model_validate(payload)

    payload = research_study_definition.model_dump(mode="python", exclude={"study_id"})
    payload["analysis_cutoff"] = datetime(2026, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="cannot exceed|precede"):
        StudyDefinition.model_validate(payload)

    payload = research_study_definition.model_dump(mode="python", exclude={"study_id"})
    payload["proposal_budget"] = float("nan")
    with pytest.raises(ValidationError):
        StudyDefinition.model_validate(payload)


def test_holdout_contract_requires_sorted_membership_and_content_hash(
    research_study_definition: StudyDefinition,
) -> None:
    with pytest.raises(ValidationError, match="sorted"):
        _holdout(research_study_definition, assets=("asset-b", "asset-a"))
    payload = _holdout(research_study_definition).model_dump(mode="python")
    payload["candidate_hash"] = "4" * 64
    with pytest.raises(ValidationError, match="holdout_id"):
        HoldoutIdentity.model_validate(payload)


def test_registry_event_ids_authenticate_body_sequence_and_chain(
    research_study_definition: StudyDefinition,
) -> None:
    event = RegistryEvent.create(
        sequence=1,
        previous_event_hash="0" * 64,
        event_ts=datetime(2026, 7, 16, 1, 0, tzinfo=UTC),
        study_id=research_study_definition.study_id,
        actor_kind="human",
        actor_label="local-reviewer",
        payload=StudyRegisteredPayload(study_definition=research_study_definition),
    )

    assert event.event_id == event.computed_event_id()
    assert event.event_hash == event.computed_event_hash()
    changed_sequence = event.model_dump(mode="python")
    changed_sequence["sequence"] = 2
    with pytest.raises(ValidationError, match="event_hash"):
        RegistryEvent.model_validate(changed_sequence)


def test_proposal_abstention_evidence_and_action_contracts_are_content_identified(
    research_study_definition: StudyDefinition,
) -> None:
    study = research_study_definition
    evidence = EvidenceRecord(
        source_item_id="item-1",
        span_id="span-1",
        source_text_hash="1" * 64,
        span_hash="2" * 64,
        source_type="licensed-news",
        content_status="active",
        relationship_type="original",
        license_or_terms_ref="synthetic-fixture-terms",
        retention_permitted=True,
        asset_ids=("asset-a",),
        active_period_valid=True,
        published_at=datetime(2025, 6, 1, tzinfo=UTC),
        available_at=datetime(2025, 6, 1, 0, 1, tzinfo=UTC),
        snapshot_cutoff=study.analysis_cutoff,
        quoted_span="Synthetic demand commentary weakened.",
        source_artifact_id="silver-text-v1",
        source_artifact_hash="3" * 64,
    )
    search = CounterevidenceSearchRecord(
        query_id="4" * 64,
        normalized_query_hash="5" * 64,
        filters_hash="6" * 64,
        result_count=0,
        pages_inspected=1,
        reason="no_result",
    )
    proposal = ResearchProposal(
        study_id=study.study_id,
        attempt_id="7" * 64,
        bundle_id="8" * 64,
        input_snapshot_hash="9" * 64,
        hypothesis="Lower development-period source tone is associated with lower ranked returns.",
        mechanism="The source may update expectations before slower deterministic measures.",
        affected_universe_id=study.universe_snapshot_id,
        horizon_sessions=study.horizon_sessions,
        target_family=study.target_family,
        expected_direction="positive",
        direction_is_hypothesis=True,
        supporting_evidence_ids=(evidence.evidence_id,),
        counterevidence_searches=(search,),
        falsification_conditions=(
            "The matched combined family does not improve rank diagnostics.",
        ),
        invalidation_conditions=("Required source availability cannot be established.",),
        required_input_ids=("text-signals", "market-bars"),
        availability_requirements=("Every source must be available by its historical decision.",),
        experiment_template_id="matched_feature_ablation_v1",
        parameter_choices=(),
        required_learned_families=("traditional", "text", "combined"),
        required_fixed_benchmarks=("equal_weight", "momentum_only", "no_trade"),
        negative_controls=("shuffled_text",),
        sensitivity_checks=("endpoint_shift",),
        expected_failure_modes=("Sparse retained evidence may weaken stability.",),
        known_limitations=("The local model remains a retrospective parser.",),
        expected_output_metrics=("spearman_ic", "max_drawdown"),
        acceptance_interpretation="Any result remains hypothetical and requires human review.",
    )
    abstention = ResearchAbstention(
        study_id=study.study_id,
        attempt_id="a" * 64,
        bundle_id="8" * 64,
        input_snapshot_hash="9" * 64,
        reason="insufficient_evidence",
        explanation="The sealed snapshot has no retained counterevidence.",
        missing_input_ids=("counterevidence",),
        tool_query_ids=("4" * 64,),
        resolvable_in_new_study=True,
    )

    assert evidence.evidence_id == evidence.computed_evidence_id()
    assert len(proposal.proposal_id) == 64
    assert len(abstention.abstention_id) == 64
    action = TypeAdapter(ResearchAgentAction).validate_python(
        {"action_type": "abstention", "abstention": abstention}
    )
    assert action.abstention == abstention
    tool_action = TypeAdapter(ResearchAgentAction).validate_python(
        {
            "action_type": "tool_call",
            "tool_call": {
                "tool_name": "read_feature_catalog",
                "request": {"section": "features"},
            },
        }
    )
    assert tool_action.tool_call.tool_name == "read_feature_catalog"
    with pytest.raises(ValidationError):
        TypeAdapter(ResearchAgentAction).validate_python(
            {
                "action_type": "tool_call",
                "tool_call": {
                    "tool_name": "read_feature_catalog",
                    "request": {"section": "features"},
                    "parallel_calls": [],
                },
            }
        )
    forbidden = proposal.model_dump(mode="python", exclude={"proposal_id"})
    forbidden["order_side"] = "buy"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ResearchProposal.model_validate(forbidden)
    unstructured_claim = proposal.model_dump(mode="python", exclude={"proposal_id"})
    unstructured_claim["hypothesis"] = "The feature improves returns by 2%."
    with pytest.raises(ValidationError, match="numeric tokens"):
        ResearchProposal.model_validate(unstructured_claim)


def test_independent_disabled_agent_config_does_not_change_research_config() -> None:
    config = load_research_agent_config(Path("configs/research_agent.disabled.yaml"))

    assert config.enabled is False
    assert config.model_path is None
    assert config.artifact_root.is_absolute()
    assert len(config.content_hash()) == 64
    assert "research_agent" not in ResearchConfig.model_fields


def test_enabled_agent_config_requires_one_local_model_and_expected_hash(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        Path("configs/research_agent.disabled.yaml").read_text(encoding="utf-8")
    )
    payload["enabled"] = True
    payload["artifact_root"] = str(tmp_path / "agent-artifacts")
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="model_path"):
        load_research_agent_config(path)


def test_agent_artifact_root_and_manifest_are_private_exclusive_and_replayable(
    tmp_path: Path,
) -> None:
    root = ensure_agent_artifact_root((tmp_path / "agent-artifacts").resolve())
    manifest = AgentArtifactManifest(
        artifact_root_id="test-root",
        created_at=datetime(2026, 7, 16, 2, 0, tzinfo=UTC),
        git_commit="1" * 64,
        dirty_worktree=False,
        artifacts=(
            ArtifactEntry(
                role="study-definition",
                relative_path="studies/abc/definition.json",
                sha256="2" * 64,
                bytes=123,
                schema_version="research-study-v1",
            ),
        ),
        limitations=("Local advisory locks only.",),
        next_questions=("Proceed to sealed views?",),
    )
    path = root / "manifest.json"

    write_agent_manifest_exclusive(path, manifest)
    assert load_agent_manifest(path) == manifest
    with pytest.raises(AgentArtifactError, match="already exists"):
        write_agent_manifest_exclusive(path, manifest)

    record = json.loads(path.read_text(encoding="utf-8"))
    record["dirty_worktree"] = True
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(AgentArtifactError, match="contract"):
        load_agent_manifest(path)

    path.write_text('{"bytes":NaN}\n', encoding="utf-8")
    with pytest.raises(AgentArtifactError, match="strict JSON"):
        load_agent_manifest(path)


def test_agent_artifacts_reject_parent_paths_and_symlink_roots(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="relative_path"):
        ArtifactEntry(
            role="escape",
            relative_path="../outside.json",
            sha256="1" * 64,
            bytes=1,
            schema_version="v1",
        )
    with pytest.raises(ValidationError, match="relative_path"):
        ArtifactEntry(
            role="root",
            relative_path=".",
            sha256="1" * 64,
            bytes=1,
            schema_version="v1",
        )
    target = (tmp_path / "target").resolve()
    target.mkdir()
    link = (tmp_path / "linked-root").resolve()
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(AgentArtifactError, match="symlink"):
        ensure_agent_artifact_root(link)
