from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nlp_trader import pipeline
from nlp_trader.agent_study import export_exploratory_standard_run
from nlp_trader.config import ResearchConfig
from nlp_trader.research_agents.config import load_research_agent_config
from nlp_trader.research_agents.contracts import StudyDefinition, TimeRange
from nlp_trader.research_agents.views import load_development_view_bundle
from nlp_trader.timestamps import parse_utc


def test_trusted_standard_run_export_omits_holdout_paths_and_private_authority(
    tmp_path: Path,
    generated_config: ResearchConfig,
    research_study_definition: StudyDefinition,
) -> None:
    outputs = pipeline.report(generated_config)
    final_path = Path(outputs["final_manifest"])
    prediction_path = Path(outputs["model_evaluation"])
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    protocol = prediction["evaluation_protocol"]
    holdout_start = parse_utc(protocol["final_holdout_start"])
    holdout_end = parse_utc(protocol["final_holdout_end"])
    final = json.loads(final_path.read_text(encoding="utf-8"))
    source_start = parse_utc(final["period"]["start"])
    cutoff = holdout_start - timedelta(microseconds=1)
    payload = research_study_definition.model_dump(mode="python", exclude={"study_id"})
    payload.update(
        {
            "created_at": datetime.now(UTC),
            "analysis_cutoff": cutoff,
            "development_decisions": TimeRange(
                start=source_start,
                end=cutoff,
            ),
            "reserved_holdout_decisions": TimeRange(
                start=holdout_start,
                end=holdout_end,
            ),
            "reserved_holdout_outcomes": TimeRange(
                start=holdout_start,
                end=holdout_end + timedelta(days=generated_config.features.horizon_days),
            ),
            "horizon_sessions": generated_config.features.horizon_days,
        }
    )
    study = StudyDefinition.model_validate(payload)
    agent_config = load_research_agent_config(
        Path("configs/research_agent.disabled.yaml")
    ).model_copy(update={"artifact_root": (tmp_path / "agent-artifacts").resolve()})

    bundle_root, manifest = export_exploratory_standard_run(
        agent_config,
        study=study,
        source_run_final=final_path,
    )
    loaded = load_development_view_bundle(bundle_root)

    assert manifest.confirmatory_eligible is False
    assert loaded.development_view.source_mode == "exploratory_standard_run"
    assert loaded.evidence
    model_visible = (
        loaded.development_view.canonical_json()
        + loaded.feature_catalog.canonical_json()
        + "".join(record.canonical_json() for record in loaded.evidence)
    )
    assert "final_holdout" not in model_visible
    assert str(final_path.parent) not in model_visible
    assert "paper" not in model_visible
    assert "broker" not in model_visible

    uncovered_payload = study.model_dump(mode="python", exclude={"study_id"})
    uncovered_payload["development_decisions"] = TimeRange(
        start=source_start - timedelta(days=1),
        end=cutoff,
    )
    uncovered = StudyDefinition.model_validate(uncovered_payload)
    with pytest.raises(ValueError, match="does not cover"):
        export_exploratory_standard_run(
            agent_config,
            study=uncovered,
            source_run_final=final_path,
        )

    final["config_hash"] = "0" * 64
    final_path.write_text(json.dumps(final, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the run config hash"):
        export_exploratory_standard_run(
            agent_config,
            study=study,
            source_run_final=final_path,
        )
