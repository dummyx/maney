from __future__ import annotations

import hashlib
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from nlp_trader.backtest.engine import DeterministicBacktestEngine
from nlp_trader.config import BacktestConfig
from nlp_trader.research_agents.contracts import Sha256, StrictModel, content_sha256
from nlp_trader.research_templates.selector_signal_matrix import (
    DependenceAwareInterval,
    holm_adjust,
    moving_block_bootstrap_mean,
)
from nlp_trader.timestamps import parse_utc

AttemptOutcome = Literal[
    "proposal",
    "abstention",
    "malformed",
    "rejected",
    "duplicate",
    "exhausted",
    "crashed",
]


class AttemptedCandidate(StrictModel):
    attempt_id: Sha256
    outcome: AttemptOutcome
    candidate_id: str | None = Field(default=None, min_length=1, max_length=256)
    selected: bool = False
    terminal_artifact_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        if self.outcome == "proposal" and (
            self.candidate_id is None or self.terminal_artifact_hash is None
        ):
            raise ValueError("proposal attempts must bind a candidate and terminal artifact")
        if self.outcome != "proposal" and (self.candidate_id is not None or self.selected):
            raise ValueError("non-proposal attempts cannot bind or select a candidate")
        return self


class PeriodAnnotation(StrictModel):
    regime: str = Field(min_length=1, max_length=128)
    behavior: str = Field(min_length=1, max_length=128)


class SubgroupEffect(StrictModel):
    dimension: Literal["regime", "behavior"]
    group: str
    observations: int = Field(ge=1)
    mean_net_return_difference: float


class RobustnessComparison(StrictModel):
    candidate_id: str
    control_id: str
    scenario_id: str
    observations: int = Field(ge=1)
    effect_size_mean_net_return: float
    interval: DependenceAwareInterval
    raw_p_value: float = Field(ge=0.0, le=1.0)
    holm_adjusted_p_value: float = Field(ge=0.0, le=1.0)
    negative_result: bool
    subgroup_effects: tuple[SubgroupEffect, ...]
    uncertainty_limitation: str | None = None

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        if self.negative_result != (self.effect_size_mean_net_return <= 0.0):
            raise ValueError("negative-result flag must follow the predeclared nonpositive rule")
        return self


class DownstreamEvaluationReport(StrictModel):
    artifact_schema_version: Literal["downstream-evaluation-report-v1"] = (
        "downstream-evaluation-report-v1"
    )
    report_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    selected_candidate_id: str
    fixed_control_id: str
    required_scenarios: tuple[str, ...]
    attempts: tuple[AttemptedCandidate, ...]
    comparisons: tuple[RobustnessComparison, ...]
    negative_result_candidates: tuple[str, ...]
    multiplicity_method: Literal["holm_familywise"] = "holm_familywise"
    uncertainty_method: Literal["moving_block_bootstrap"] = "moving_block_bootstrap"
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_completeness(self) -> Self:
        if len({attempt.attempt_id for attempt in self.attempts}) != len(self.attempts):
            raise ValueError("every immutable attempt must appear exactly once")
        proposal_candidates = {
            attempt.candidate_id for attempt in self.attempts if attempt.candidate_id is not None
        }
        comparison_candidates = {value.candidate_id for value in self.comparisons}
        if proposal_candidates != comparison_candidates:
            raise ValueError("all and only attempted proposal candidates must be reported")
        selected = [attempt for attempt in self.attempts if attempt.selected]
        if len(selected) != 1 or selected[0].candidate_id != self.selected_candidate_id:
            raise ValueError("exactly one reported candidate must match the frozen selection")
        expected_pairs = {
            (candidate, scenario)
            for candidate in proposal_candidates
            for scenario in self.required_scenarios
        }
        actual_pairs = {(value.candidate_id, value.scenario_id) for value in self.comparisons}
        if actual_pairs != expected_pairs:
            raise ValueError("every attempted candidate requires every predeclared scenario")
        baseline_negative = tuple(
            sorted(
                value.candidate_id
                for value in self.comparisons
                if value.scenario_id == "baseline" and value.negative_result
            )
        )
        if self.negative_result_candidates != baseline_negative:
            raise ValueError("negative baseline results must be retained explicitly")
        expected = content_sha256(self.model_dump(mode="json", exclude={"report_id"}))
        if self.report_id and self.report_id != expected:
            raise ValueError("downstream report ID does not match canonical content")
        if not self.report_id:
            object.__setattr__(self, "report_id", expected)
        return self


def required_robustness_scenarios(source_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("source ablations require a nonempty unique source set")
    if any(not source_id or ":" in source_id for source_id in source_ids):
        raise ValueError("source IDs must be nonempty and cannot contain ':'")
    return (
        "baseline",
        "endpoint_shift_early",
        "endpoint_shift_late",
        "causal_delay",
        "shuffled_text",
        *(f"source_ablation:{source_id}" for source_id in sorted(source_ids)),
    )


def build_robustness_backtests(
    *,
    candidate_predictions: list[dict[str, Any]],
    control_predictions: list[dict[str, Any]],
    source_ablation_predictions: Mapping[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    backtest_config: BacktestConfig,
    candidate_top_k: int | None,
    control_top_k: int | None,
    endpoint_shift_periods: int,
    causal_delay_periods: int,
    shuffle_seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build every predeclared perturbation and run it through one backtest engine.

    Source-ablation predictions must be rebuilt upstream without the named source. This function
    verifies exact row membership; it never approximates an ablation by editing an outcome.
    """

    source_ids = tuple(sorted(source_ablation_predictions))
    scenarios = required_robustness_scenarios(source_ids)
    candidate_by_key = _unique_predictions(candidate_predictions, label="candidate")
    control_by_key = _unique_predictions(control_predictions, label="fixed control")
    if set(candidate_by_key) != set(control_by_key):
        raise ValueError("candidate and fixed-control prediction membership must match exactly")
    decision_times = tuple(sorted({key[1] for key in candidate_by_key}, key=parse_utc))
    if (
        endpoint_shift_periods < 1
        or endpoint_shift_periods >= len(decision_times)
        or causal_delay_periods < 1
        or causal_delay_periods >= len(decision_times)
    ):
        raise ValueError("endpoint shifts and causal delay must leave at least one decision period")
    early_times = set(decision_times[:-endpoint_shift_periods])
    late_times = set(decision_times[endpoint_shift_periods:])
    candidate_scenarios: dict[str, list[dict[str, Any]]] = {
        "baseline": list(candidate_by_key.values()),
        "endpoint_shift_early": [
            row for key, row in candidate_by_key.items() if key[1] in early_times
        ],
        "endpoint_shift_late": [
            row for key, row in candidate_by_key.items() if key[1] in late_times
        ],
        "causal_delay": _delay_prediction_scores(
            candidate_by_key, delay_periods=causal_delay_periods
        ),
        "shuffled_text": _shuffle_prediction_scores(candidate_by_key, seed=shuffle_seed),
    }
    for source_id, rows in sorted(source_ablation_predictions.items()):
        ablated = _unique_predictions(rows, label=f"source ablation {source_id}")
        if set(ablated) != set(candidate_by_key):
            raise ValueError("every source ablation must preserve exact prediction membership")
        candidate_scenarios[f"source_ablation:{source_id}"] = list(ablated.values())
    if tuple(candidate_scenarios) != scenarios:
        raise RuntimeError("robustness scenario construction order is inconsistent")
    control_scenarios = {
        scenario: _rows_for_keys(
            control_by_key,
            {_prediction_key(row) for row in candidate_scenarios[scenario]},
        )
        for scenario in scenarios
    }
    engine = DeterministicBacktestEngine()
    candidate_results = {
        scenario: engine.run(
            rows,
            labels,
            backtest_config,
            top_k=candidate_top_k,
        )
        for scenario, rows in candidate_scenarios.items()
    }
    control_results = {
        scenario: engine.run(
            control_scenarios[scenario],
            labels,
            backtest_config,
            top_k=control_top_k,
        )
        for scenario in scenarios
    }
    return candidate_results, control_results


def evaluate_downstream_results(
    *,
    candidate_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    control_results: Mapping[str, Mapping[str, Any]],
    attempts: tuple[AttemptedCandidate, ...],
    selected_candidate_id: str,
    fixed_control_id: str,
    required_scenarios: tuple[str, ...],
    period_annotations: Mapping[str, PeriodAnnotation],
    block_size: int,
    repetitions: int,
    seed: int,
    limitations: tuple[str, ...] = (),
) -> DownstreamEvaluationReport:
    if tuple(control_results) != required_scenarios:
        raise ValueError("control results must exactly follow the predeclared scenario order")
    attempted_candidates = tuple(
        sorted(attempt.candidate_id for attempt in attempts if attempt.candidate_id is not None)
    )
    if tuple(sorted(candidate_results)) != attempted_candidates:
        raise ValueError("candidate results must include every attempted proposal exactly once")
    pending: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for candidate_id in attempted_candidates:
        scenario_results = candidate_results[candidate_id]
        if tuple(scenario_results) != required_scenarios:
            raise ValueError("candidate scenarios must exactly follow the predeclared order")
        for scenario_id in required_scenarios:
            candidate_periods = _period_returns(scenario_results[scenario_id])
            control_periods = _period_returns(control_results[scenario_id])
            if tuple(candidate_periods) != tuple(control_periods):
                raise ValueError("candidate and control periods must be exactly matched")
            missing_annotations = set(candidate_periods).difference(period_annotations)
            if missing_annotations:
                raise ValueError("every comparison period requires regime and behavior annotations")
            differences = tuple(
                candidate_periods[period] - control_periods[period] for period in candidate_periods
            )
            effective_block = min(block_size, len(differences))
            comparison_seed = _comparison_seed(seed, candidate_id, scenario_id)
            interval = moving_block_bootstrap_mean(
                differences,
                block_size=effective_block,
                repetitions=repetitions,
                seed=comparison_seed,
            )
            raw_p = _centered_block_bootstrap_p_value(
                differences,
                block_size=effective_block,
                repetitions=repetitions,
                seed=comparison_seed,
            )
            raw_p_values.append(raw_p)
            pending.append(
                {
                    "candidate_id": candidate_id,
                    "control_id": fixed_control_id,
                    "scenario_id": scenario_id,
                    "observations": len(differences),
                    "effect_size_mean_net_return": statistics.fmean(differences),
                    "interval": interval,
                    "raw_p_value": raw_p,
                    "negative_result": statistics.fmean(differences) <= 0.0,
                    "subgroup_effects": _subgroup_effects(
                        tuple(candidate_periods), differences, period_annotations
                    ),
                    "uncertainty_limitation": (
                        "One period yields a degenerate interval; no time-series uncertainty "
                        "can be estimated."
                        if len(differences) == 1
                        else None
                    ),
                }
            )
    adjusted = holm_adjust(tuple(raw_p_values))
    comparisons = tuple(
        RobustnessComparison(**value, holm_adjusted_p_value=adjusted[index])
        for index, value in enumerate(pending)
    )
    negative = tuple(
        sorted(
            value.candidate_id
            for value in comparisons
            if value.scenario_id == "baseline" and value.negative_result
        )
    )
    return DownstreamEvaluationReport(
        selected_candidate_id=selected_candidate_id,
        fixed_control_id=fixed_control_id,
        required_scenarios=required_scenarios,
        attempts=attempts,
        comparisons=comparisons,
        negative_result_candidates=negative,
        limitations=(
            "Effect sizes are hypothetical net-return differences, not evidence of profitability.",
            "Historical model pretraining contamination cannot be ruled out retrospectively.",
            "Human-declared lineage and behavior annotations can still be mislabeled.",
            *limitations,
        ),
    )


def _period_returns(result: Mapping[str, Any]) -> dict[str, float]:
    periods = result.get("periods")
    if not isinstance(periods, list) or not periods:
        raise ValueError("scenario result requires nonempty backtest periods")
    output: dict[str, float] = {}
    for period in periods:
        if not isinstance(period, dict):
            raise ValueError("backtest period must be an object")
        asof_ts = period.get("asof_ts")
        net_return = period.get("net_return")
        if not isinstance(asof_ts, str) or not isinstance(net_return, int | float):
            raise ValueError("backtest periods require string asof_ts and finite net_return")
        value = float(net_return)
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("backtest net returns must be finite")
        if asof_ts in output:
            raise ValueError("backtest scenario repeats a decision period")
        output[asof_ts] = value
    return dict(sorted(output.items()))


PredictionKey = tuple[str, str, str]


def _prediction_key(row: Mapping[str, Any]) -> PredictionKey:
    try:
        return str(row["asset_id"]), str(row["asof_ts"]), str(row["horizon"])
    except KeyError as exc:
        raise ValueError("prediction rows require asset_id, asof_ts, and horizon") from exc


def _unique_predictions(
    rows: list[dict[str, Any]], *, label: str
) -> dict[PredictionKey, dict[str, Any]]:
    if not rows:
        raise ValueError(f"{label} predictions must not be empty")
    output: dict[PredictionKey, dict[str, Any]] = {}
    for row in rows:
        key = _prediction_key(row)
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError(f"{label} predictions require numeric scores")
        if key in output:
            raise ValueError(f"{label} predictions repeat a canonical row")
        output[key] = dict(row)
    return dict(
        sorted(output.items(), key=lambda item: (parse_utc(item[0][1]), item[0][2], item[0][0]))
    )


def _rows_for_keys(
    rows: Mapping[PredictionKey, dict[str, Any]], keys: set[PredictionKey]
) -> list[dict[str, Any]]:
    if not keys.issubset(rows):
        raise ValueError("scenario membership is absent from fixed-control predictions")
    return [row for key, row in rows.items() if key in keys]


def _delay_prediction_scores(
    rows: Mapping[PredictionKey, dict[str, Any]], *, delay_periods: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[PredictionKey, dict[str, Any]]]] = defaultdict(list)
    for key, row in rows.items():
        grouped[(key[0], key[2])].append((key, row))
    delayed: list[dict[str, Any]] = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda value: parse_utc(value[0][1]))
        for index in range(delay_periods, len(ordered)):
            source = ordered[index - delay_periods][1]
            destination = dict(ordered[index][1])
            destination["score"] = source["score"]
            delayed.append(destination)
    if not delayed:
        raise ValueError("causal delay produced no matched prediction rows")
    return sorted(delayed, key=lambda row: (parse_utc(str(row["asof_ts"])), str(row["asset_id"])))


def _shuffle_prediction_scores(
    rows: Mapping[PredictionKey, dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, row in rows.items():
        grouped[(key[1], key[2])].append(row)
    shuffled: list[dict[str, Any]] = []
    for (asof_ts, horizon), values in sorted(
        grouped.items(), key=lambda item: (parse_utc(item[0][0]), item[0][1])
    ):
        ordered = sorted(values, key=lambda row: str(row["asset_id"]))
        scores = [row["score"] for row in ordered]
        local_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{asof_ts}|{horizon}".encode()).digest()[:8], "big"
        )
        random.Random(local_seed).shuffle(scores)
        for row, score in zip(ordered, scores, strict=True):
            copied = dict(row)
            copied["score"] = score
            shuffled.append(copied)
    return shuffled


def _comparison_seed(seed: int, candidate_id: str, scenario_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|{candidate_id}|{scenario_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _centered_block_bootstrap_p_value(
    values: tuple[float, ...], *, block_size: int, repetitions: int, seed: int
) -> float:
    center = statistics.fmean(values)
    centered = tuple(value - center for value in values)
    blocks = tuple(
        centered[index : index + block_size] for index in range(0, len(centered) - block_size + 1)
    )
    rng = random.Random(seed)
    extreme = 0
    for _ in range(repetitions):
        sample: list[float] = []
        while len(sample) < len(centered):
            sample.extend(rng.choice(blocks))
        if abs(statistics.fmean(sample[: len(centered)])) >= abs(center) - 1e-15:
            extreme += 1
    return (extreme + 1.0) / (repetitions + 1.0)


def _subgroup_effects(
    periods: tuple[str, ...],
    values: tuple[float, ...],
    annotations: Mapping[str, PeriodAnnotation],
) -> tuple[SubgroupEffect, ...]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for period, value in zip(periods, values, strict=True):
        annotation = annotations[period]
        grouped[("regime", annotation.regime)].append(value)
        grouped[("behavior", annotation.behavior)].append(value)
    return tuple(
        SubgroupEffect(
            dimension=dimension,  # type: ignore[arg-type]
            group=group,
            observations=len(group_values),
            mean_net_return_difference=statistics.fmean(group_values),
        )
        for (dimension, group), group_values in sorted(grouped.items())
    )
