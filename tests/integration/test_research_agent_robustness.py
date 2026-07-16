from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nlp_trader.config import BacktestConfig
from nlp_trader.research_templates.robustness_evaluation import (
    AttemptedCandidate,
    PeriodAnnotation,
    build_robustness_backtests,
    evaluate_downstream_results,
    required_robustness_scenarios,
)
from nlp_trader.timestamps import format_utc


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        commission_bps=1.0,
        half_spread_bps=2.0,
        slippage_bps=3.0,
        borrow_bps_per_year=0.0,
        max_position_weight=0.5,
        max_gross_exposure=1.0,
        max_net_exposure=0.5,
        max_daily_turnover=1.0,
        max_participation_rate=0.01,
        min_price=1.0,
        min_dollar_volume=1_000.0,
        shorting_allowed=False,
        hard_to_borrow_allowed=False,
    )


def test_predeclared_robustness_scenarios_use_the_same_cost_aware_engine() -> None:
    decisions = tuple(
        datetime(2026, 7, 1, 20, tzinfo=UTC) + timedelta(days=index) for index in range(6)
    )
    candidate: list[dict[str, object]] = []
    control: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for decision_index, decision in enumerate(decisions):
        for asset_index, asset in enumerate(("asset-a", "asset-b", "asset-c"), start=1):
            common = {
                "asset_id": asset,
                "symbol": asset.upper(),
                "asof_ts": format_utc(decision),
                "horizon": "1d",
                "close": 10.0,
                "dollar_volume": 100_000_000.0,
                "sector": "Mixed",
                "beta": 0.0,
                "volatility": 0.01,
            }
            candidate.append({**common, "score": float(asset_index + decision_index)})
            control.append({**common, "score": float(4 - asset_index)})
            labels.append(
                {
                    **{key: common[key] for key in ("asset_id", "symbol", "asof_ts", "horizon")},
                    "forward_return": 0.002 * (asset_index - 1),
                    "label_start_ts": format_utc(decision + timedelta(hours=17, minutes=30)),
                    "label_end_ts": format_utc(decision + timedelta(days=1)),
                    "execution_price": 10.0,
                    "exit_price": 10.0,
                    "execution_dollar_volume": 100_000_000.0,
                    "exit_dollar_volume": 100_000_000.0,
                }
            )
    ablations = {
        "licensed-news": [{**row, "score": float(row["score"]) - 0.5} for row in candidate],
        "permitted-social": [{**row, "score": -float(row["score"])} for row in candidate],
    }
    arguments = {
        "candidate_predictions": candidate,
        "control_predictions": control,
        "source_ablation_predictions": ablations,
        "labels": labels,
        "backtest_config": _backtest_config(),
        "candidate_top_k": 1,
        "control_top_k": 1,
        "endpoint_shift_periods": 1,
        "causal_delay_periods": 1,
        "shuffle_seed": 7,
    }
    candidate_results, control_results = build_robustness_backtests(**arguments)  # type: ignore[arg-type]
    repeated = build_robustness_backtests(**arguments)  # type: ignore[arg-type]

    scenarios = required_robustness_scenarios(tuple(ablations))
    assert (candidate_results, control_results) == repeated
    assert tuple(candidate_results) == scenarios
    assert len(candidate_results["baseline"]["periods"]) == 6
    assert len(candidate_results["endpoint_shift_early"]["periods"]) == 5
    assert len(candidate_results["endpoint_shift_late"]["periods"]) == 5
    assert len(candidate_results["causal_delay"]["periods"]) == 5
    assert all("cost_return" in period for period in candidate_results["shuffled_text"]["periods"])

    report = evaluate_downstream_results(
        candidate_results={"candidate-a": candidate_results},
        control_results=control_results,
        attempts=(
            AttemptedCandidate(
                attempt_id="1" * 64,
                outcome="proposal",
                candidate_id="candidate-a",
                selected=True,
                terminal_artifact_hash="a" * 64,
            ),
        ),
        selected_candidate_id="candidate-a",
        fixed_control_id="fixed-template-v1",
        required_scenarios=scenarios,
        period_annotations={
            format_utc(decision): PeriodAnnotation(
                regime="early" if index < 3 else "late",
                behavior="stable" if index % 2 else "volatile",
            )
            for index, decision in enumerate(decisions)
        },
        block_size=2,
        repetitions=200,
        seed=7,
    )
    assert len(report.comparisons) == len(scenarios)
