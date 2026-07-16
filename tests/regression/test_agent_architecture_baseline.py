from __future__ import annotations

import ast
import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_trader.config import ResearchConfig
from nlp_trader.nlp.llm_decision_rounds import DecisionRound
from nlp_trader.reports import write_report
from nlp_trader.research import create_run_context, finalize_run

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_AGENT_ROOT = REPOSITORY_ROOT / "src" / "nlp_trader" / "research_agents"
FORBIDDEN_AGENT_IMPORTS = (
    "nlp_trader.cli",
    "nlp_trader.pipeline",
    "nlp_trader.paper",
    "nlp_trader.portfolio",
    "nlp_trader.backtest",
    "nlp_trader.broker",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _decision_round_payload() -> dict[str, object]:
    raw = '{"annotations":[{"asset_id":"asset_aaa"}]}'
    return {
        "run_id": "baseline-run",
        "config_hash": "a" * 64,
        "input_snapshot_hash": "b" * 64,
        "item_id": "item-001",
        "source_text_hash": "c" * 64,
        "source_available_at": "2026-07-15T20:00:00Z",
        "decision_time": "2026-07-16T20:00:00Z",
        "horizon_days": 1,
        "model": {
            "provider": "llama_cpp_gguf",
            "model_id": "local/model",
            "revision": "revision-1",
            "sha256": "d" * 64,
        },
        "prompt": {"version": "prompt-v1", "sha256": "e" * 64},
        "schema_contract": {"version": "schema-v1", "sha256": "f" * 64},
        "sampling": {
            "decoding": "greedy",
            "seed": 7,
            "max_input_tokens": 512,
            "max_new_tokens": 64,
            "temperature": None,
            "top_p": None,
        },
        "retrieval": {"evidence_ids": ["S1"]},
        "raw_generation": {
            "request_id": "request-001",
            "generated_text": raw,
            "metadata": {},
        },
        "structured_output": {"item_id": "item-001", "annotations": [{"asset_id": "asset_aaa"}]},
        "verifier": {
            "version": "verifier-v1",
            "passed": True,
            "checks": [{"check_id": "schema", "passed": True}],
        },
        "inference_source": "generated",
        "usage": {},
        "application_mode": "sidecar",
    }


def test_research_agent_package_has_no_forbidden_imports() -> None:
    imported = {
        module for path in RESEARCH_AGENT_ROOT.rglob("*.py") for module in _imported_modules(path)
    }

    violations = sorted(
        module
        for module in imported
        if any(
            module == denied or module.startswith(f"{denied}.")
            for denied in FORBIDDEN_AGENT_IMPORTS
        )
    )
    assert violations == []


def test_importing_agent_foundation_loads_no_model_or_execution_package() -> None:
    code = "\n".join(
        (
            "import sys",
            "import nlp_trader.research_agents",
            f"denied = {FORBIDDEN_AGENT_IMPORTS!r}",
            "loaded = sorted(module for module in sys.modules if any(",
            "    module == item or module.startswith(item + '.') for item in denied",
            "))",
            "if 'llama_cpp' in sys.modules or loaded:",
            "    raise SystemExit(','.join(loaded + ['llama_cpp']))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_standard_config_and_script_binding_remain_agent_independent() -> None:
    assert tuple(ResearchConfig.model_fields) == (
        "path",
        "mode",
        "paths",
        "features",
        "models",
        "backtest",
        "data",
        "runtime",
        "transformer",
        "llm_annotations",
    )
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["nlp-trader"] == "nlp_trader.cli:main"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool_calls", ({"name": "calculate"},)),
        ("portfolio", {"target_weight": 0.5}),
        ("risk", {"max_position_weight": 0.5}),
        ("orders", ({"side": "buy"},)),
    ],
)
def test_existing_decision_round_rejects_agent_and_execution_authority(
    field: str,
    value: object,
) -> None:
    payload = _decision_round_payload()
    payload[field] = value

    with pytest.raises(ValidationError, match="must remain empty"):
        DecisionRound.model_validate(payload)


def test_standard_manifest_and_report_contract_fields_are_locked(
    generated_config: ResearchConfig,
) -> None:
    context = create_run_context(
        generated_config,
        now=datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC),
        run_id="agent-architecture-baseline",
    )
    initial = json.loads((context.paths.reports / "run.initial.json").read_text(encoding="utf-8"))
    snapshot = json.loads(
        (context.paths.reports / "config.snapshot.json").read_text(encoding="utf-8")
    )
    final_path = finalize_run(
        context,
        universe=["BBB", "AAA"],
        period={"start": "2026-01-01", "end": "2026-06-30"},
        metrics={"development": {"sharpe": 0.0}, "final_holdout": {"sharpe": 0.0}},
        known_limitations=["synthetic only"],
        next_questions=["collect licensed data"],
    )
    final = json.loads(final_path.read_text(encoding="utf-8"))
    report_path = write_report(
        generated_config,
        {"metrics": {"sharpe": 0.0}, "periods": []},
        context.paths.reports / "baseline-report.md",
        report_run_id=context.run_id,
        created_at=context.created_at,
    )
    report = report_path.read_text(encoding="utf-8")

    assert set(snapshot) == {
        "backtest",
        "data",
        "features",
        "llm_annotations",
        "mode",
        "models",
        "paths",
        "runtime",
        "transformer",
    }
    assert set(initial) == {
        "code_version",
        "config_hash",
        "created_at",
        "data_manifest",
        "run_id",
        "status",
    }
    assert set(final) == {
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
    assert set(final["metrics"]) == {"development", "final_holdout"}
    assert "hypothetical research output" in report
    assert "## Cost and Fill Model" in report
    assert "## Portfolio Constraints" in report
    assert "## Backtest Metrics" in report
