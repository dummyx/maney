from __future__ import annotations

import json

import pytest

from nlp_trader.config import ResearchConfig
from nlp_trader.pipeline import backtest


def test_generated_strategy_outputs_are_stable_across_immutable_runs(
    generated_config: ResearchConfig,
) -> None:
    first = backtest(generated_config)
    second = backtest(generated_config)

    first_metrics = json.loads(first["backtest_comparison"].read_text(encoding="utf-8"))
    second_metrics = json.loads(second["backtest_comparison"].read_text(encoding="utf-8"))
    first_holdout = json.loads(
        first["final_holdout_backtest_comparison"].read_text(encoding="utf-8")
    )
    second_holdout = json.loads(
        second["final_holdout_backtest_comparison"].read_text(encoding="utf-8")
    )

    assert first["run_id"] != second["run_id"]
    assert first_metrics["provenance"]["run_id"] == first["run_id"]
    assert second_metrics["provenance"]["run_id"] == second["run_id"]
    assert first_metrics["provenance"]["config_hash"] == second_metrics["provenance"]["config_hash"]
    assert (
        first_metrics["provenance"]["input_manifest"]
        == second_metrics["provenance"]["input_manifest"]
    )
    assert first_metrics["artifact_schema_version"] == "backtest-comparison-v2"
    assert first_holdout["artifact_schema_version"] == "backtest-comparison-v2"
    for key in ("assumptions", "evaluation_protocol", "families"):
        assert first_metrics[key] == second_metrics[key]
        assert first_holdout[key] == second_holdout[key]
    assert first_metrics["evaluation_window"]["name"] == "development"
    assert first_holdout["evaluation_window"]["name"] == "final_holdout"
    assert first_metrics["evaluation_protocol"]["name"] == ("chronological_frozen_final_holdout_v2")
    development_families = first_metrics["families"]
    holdout_families = first_holdout["families"]
    assert set(development_families) == {
        "combined",
        "equal_weight",
        "momentum_only",
        "no_trade",
        "text",
        "traditional",
    }
    assert development_families["combined"]["periods"] == 11
    assert development_families["combined"]["trades"] == 22
    assert development_families["combined"]["total_return"] == pytest.approx(
        0.03006226433080661,
        abs=1e-12,
    )
    assert development_families["combined"]["total_cost_return"] == pytest.approx(
        0.006067573204514206,
        abs=1e-12,
    )
    assert holdout_families["combined"]["periods"] == 1
    assert holdout_families["combined"]["trades"] == 4
    assert holdout_families["combined"]["total_return"] == pytest.approx(
        0.0013393177483371765,
        abs=1e-12,
    )
    assert development_families["no_trade"]["total_return"] == 0.0
