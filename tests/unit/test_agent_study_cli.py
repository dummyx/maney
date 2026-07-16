from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from nlp_trader.cli import app, run_cli
from nlp_trader.research_agents.contracts import StudyDefinition
from nlp_trader.research_agents.registry import ResearchRegistryLedger


def _agent_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(
        Path("configs/research_agent.disabled.yaml").read_text(encoding="utf-8")
    )
    payload["artifact_root"] = str((tmp_path / "agent-artifacts").resolve())
    path = (tmp_path / "agent.yaml").resolve()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_trusted_agent_study_cli_registers_and_reserves_without_model_loading(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
) -> None:
    config = _agent_config(tmp_path)
    definition = (tmp_path / "study.json").resolve()
    definition.write_text(research_study_definition.canonical_json() + "\n", encoding="utf-8")

    assert (
        run_cli(
            [
                "agent-study",
                "create",
                "--definition",
                str(definition),
                "--agent-config",
                str(config),
            ]
        )
        == 0
    )
    assert (
        run_cli(
            [
                "agent-study",
                "reserve-attempt",
                "--study-id",
                research_study_definition.study_id,
                "--agent-config",
                str(config),
            ]
        )
        == 0
    )

    projection = ResearchRegistryLedger((tmp_path / "agent-artifacts").resolve()).project()
    state = projection.studies[research_study_definition.study_id]
    assert state.proposal_budget_consumed == 1
    assert (
        tmp_path
        / "agent-artifacts"
        / "studies"
        / research_study_definition.study_id
        / "definition.json"
    ).is_file()


def test_trusted_agent_study_cli_exposes_complete_core_lifecycle() -> None:
    result = CliRunner().invoke(app, ["agent-study", "--help"])

    assert result.exit_code == 0, result.output
    for command in (
        "create",
        "reserve-attempt",
        "export-view",
        "compile",
        "approve-development",
        "run-development",
        "freeze-candidate",
        "reveal-holdout",
        "register-external-holdout",
        "audit",
        "close",
    ):
        assert command in result.output
