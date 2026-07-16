from __future__ import annotations

from nlp_trader.research_templates.robustness_evaluation import (
    AttemptedCandidate,
    PeriodAnnotation,
    evaluate_downstream_results,
    required_robustness_scenarios,
)


def _result(values: tuple[float, ...]) -> dict[str, object]:
    return {
        "periods": [
            {"asof_ts": f"2026-07-{index:02d}T20:00:00Z", "net_return": value}
            for index, value in enumerate(values, start=1)
        ]
    }


def test_downstream_report_retains_all_attempts_scenarios_negatives_and_uncertainty() -> None:
    scenarios = required_robustness_scenarios(("licensed-news", "permitted-social"))
    control_values = (0.01, -0.01, 0.00, 0.01, -0.01, 0.00)
    control = {scenario: _result(control_values) for scenario in scenarios}
    candidate_results = {
        "candidate-a": {
            scenario: _result(tuple(value + 0.002 for value in control_values))
            for scenario in scenarios
        },
        "candidate-b": {
            scenario: _result(tuple(value - 0.001 for value in control_values))
            for scenario in scenarios
        },
    }
    attempts = (
        AttemptedCandidate(
            attempt_id="1" * 64,
            outcome="proposal",
            candidate_id="candidate-a",
            selected=True,
            terminal_artifact_hash="a" * 64,
        ),
        AttemptedCandidate(
            attempt_id="2" * 64,
            outcome="proposal",
            candidate_id="candidate-b",
            terminal_artifact_hash="b" * 64,
        ),
        AttemptedCandidate(attempt_id="3" * 64, outcome="abstention"),
    )
    annotations = {
        f"2026-07-{index:02d}T20:00:00Z": PeriodAnnotation(
            regime="high-vol" if index % 2 else "low-vol",
            behavior="high-turnover" if index <= 3 else "low-turnover",
        )
        for index in range(1, 7)
    }

    report = evaluate_downstream_results(
        candidate_results=candidate_results,
        control_results=control,
        attempts=attempts,
        selected_candidate_id="candidate-a",
        fixed_control_id="fixed-template-v1",
        required_scenarios=scenarios,
        period_annotations=annotations,
        block_size=2,
        repetitions=200,
        seed=7,
        limitations=("Tiny synthetic evaluation fixture.",),
    )
    repeated = evaluate_downstream_results(
        candidate_results=candidate_results,
        control_results=control,
        attempts=attempts,
        selected_candidate_id="candidate-a",
        fixed_control_id="fixed-template-v1",
        required_scenarios=scenarios,
        period_annotations=annotations,
        block_size=2,
        repetitions=200,
        seed=7,
        limitations=("Tiny synthetic evaluation fixture.",),
    )

    assert report == repeated
    assert report.negative_result_candidates == ("candidate-b",)
    assert len(report.comparisons) == 2 * len(scenarios)
    assert {attempt.outcome for attempt in report.attempts} == {"proposal", "abstention"}
    assert all(value.interval.block_size == 2 for value in report.comparisons)
    assert all(value.holm_adjusted_p_value >= value.raw_p_value for value in report.comparisons)
    assert {value.dimension for value in report.comparisons[0].subgroup_effects} == {
        "regime",
        "behavior",
    }
