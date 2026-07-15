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

    assert first["run_id"] != second["run_id"]
    assert first_metrics == second_metrics
    assert set(first_metrics) == {
        "combined",
        "equal_weight",
        "momentum_only",
        "no_trade",
        "text",
        "traditional",
    }
    assert first_metrics["combined"]["periods"] == 13
    assert first_metrics["combined"]["trades"] == 38
    assert first_metrics["combined"]["total_return"] == pytest.approx(
        0.014373518183854683,
        abs=1e-12,
    )
    assert first_metrics["combined"]["total_cost_return"] == pytest.approx(
        0.007290480271561932,
        abs=1e-12,
    )
    assert first_metrics["no_trade"]["total_return"] == 0.0
