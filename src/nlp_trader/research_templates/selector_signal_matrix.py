from __future__ import annotations

import hashlib
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from nlp_trader.backtest.engine import DeterministicBacktestEngine
from nlp_trader.config import BacktestConfig
from nlp_trader.research_agents.contracts import (
    Sha256,
    StrictModel,
    content_sha256,
)
from nlp_trader.timestamps import format_utc, parse_utc


class SelectorInput(StrictModel):
    asset_id: str = Field(min_length=1, max_length=256)
    asof_ts: datetime
    available_at: datetime
    eligible: bool
    momentum_score: float
    volatility_score: float = Field(ge=0.0)

    @field_validator("asof_ts", "available_at", mode="before")
    @classmethod
    def normalize_time(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("selector timestamps must be timezone-aware")
            return value.astimezone(UTC)
        if isinstance(value, str):
            return parse_utc(value)
        raise ValueError("selector timestamps must be datetimes or ISO timestamps")

    @field_serializer("asof_ts", "available_at")
    def serialize_time(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.available_at > self.asof_ts:
            raise ValueError("selector input is unavailable at its decision")
        return self


class SelectorAssetDecision(StrictModel):
    asset_id: str
    eligible: bool
    score: float | None
    rank: int | None = Field(default=None, ge=1)
    selected: bool
    input_hash: Sha256


class SelectorDecision(StrictModel):
    selector_decision_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    selector_id: str
    selector_version: Literal["selector_signal_matrix_v1"] = "selector_signal_matrix_v1"
    asof_ts: datetime
    seed: int | None = Field(default=None, ge=0)
    lookback_sessions: int | None = Field(default=None, ge=1)
    skip_sessions: int | None = Field(default=None, ge=0)
    selected_k: int | None = Field(default=None, ge=1)
    assets: tuple[SelectorAssetDecision, ...]

    @field_validator("asof_ts", mode="before")
    @classmethod
    def normalize_time(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("selector decision time must be timezone-aware")
            return value.astimezone(UTC)
        if isinstance(value, str):
            return parse_utc(value)
        raise ValueError("selector decision time must be a datetime or ISO timestamp")

    @field_serializer("asof_ts")
    def serialize_time(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        asset_ids = tuple(value.asset_id for value in self.assets)
        if asset_ids != tuple(sorted(asset_ids)) or len(asset_ids) != len(set(asset_ids)):
            raise ValueError("selector decision assets must be unique and sorted")
        expected = content_sha256(self.model_dump(mode="json", exclude={"selector_decision_id"}))
        if self.selector_decision_id and self.selector_decision_id != expected:
            raise ValueError("selector decision ID does not match canonical content")
        if not self.selector_decision_id:
            object.__setattr__(self, "selector_decision_id", expected)
        return self


class SelectorSignalMatrix(StrictModel):
    artifact_schema_version: Literal["selector-signal-matrix-v1"] = "selector-signal-matrix-v1"
    matrix_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    selected_k: int = Field(ge=1)
    random_seeds: tuple[int, ...]
    momentum_lookback_sessions: int = Field(ge=2)
    momentum_skip_sessions: int = Field(ge=1)
    decisions: tuple[SelectorDecision, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.momentum_skip_sessions >= self.momentum_lookback_sessions:
            raise ValueError("momentum skip must be shorter than its lookback")
        if len(self.random_seeds) != len(set(self.random_seeds)):
            raise ValueError("selector random seeds must be unique")
        keys = tuple(
            (value.asof_ts, value.selector_id, -1 if value.seed is None else value.seed)
            for value in self.decisions
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("selector decisions must be unique and canonically sorted")
        expected = content_sha256(self.model_dump(mode="json", exclude={"matrix_id"}))
        if self.matrix_id and self.matrix_id != expected:
            raise ValueError("selector matrix ID does not match canonical content")
        if not self.matrix_id:
            object.__setattr__(self, "matrix_id", expected)
        return self


class DependenceAwareInterval(StrictModel):
    estimate: float
    lower: float
    upper: float
    block_size: int = Field(ge=1)
    repetitions: int = Field(ge=100)
    seed: int = Field(ge=0)


def build_selector_signal_matrix(
    rows: tuple[SelectorInput, ...],
    *,
    selected_k: int,
    random_seeds: tuple[int, ...],
    momentum_lookback_sessions: int,
    momentum_skip_sessions: int,
) -> SelectorSignalMatrix:
    if not rows:
        raise ValueError("selector matrix requires input rows")
    grouped: dict[datetime, list[SelectorInput]] = defaultdict(list)
    for row in rows:
        grouped[row.asof_ts].append(row)
    decisions: list[SelectorDecision] = []
    for asof_ts, current in sorted(grouped.items()):
        assets = sorted(current, key=lambda value: value.asset_id)
        decisions.append(_ranked_decision("full_eligible", asof_ts, assets, None, None))
        for seed in sorted(random_seeds):
            random_scores = {
                value.asset_id: int.from_bytes(
                    hashlib.sha256(
                        f"{seed}|{format_utc(asof_ts)}|{value.asset_id}".encode()
                    ).digest()[:8],
                    "big",
                )
                for value in assets
            }
            decisions.append(
                _ranked_decision(
                    "seeded_random_k",
                    asof_ts,
                    assets,
                    selected_k,
                    random_scores,
                    seed=seed,
                )
            )
        decisions.append(
            _ranked_decision(
                "causal_momentum",
                asof_ts,
                assets,
                selected_k,
                {value.asset_id: value.momentum_score for value in assets},
                lookback=momentum_lookback_sessions,
                skip=momentum_skip_sessions,
            )
        )
        decisions.append(
            _ranked_decision(
                "causal_low_volatility",
                asof_ts,
                assets,
                selected_k,
                {value.asset_id: -value.volatility_score for value in assets},
                lookback=momentum_lookback_sessions,
                skip=0,
            )
        )
    return SelectorSignalMatrix(
        selected_k=selected_k,
        random_seeds=tuple(sorted(random_seeds)),
        momentum_lookback_sessions=momentum_lookback_sessions,
        momentum_skip_sessions=momentum_skip_sessions,
        decisions=tuple(
            sorted(
                decisions,
                key=lambda value: (
                    value.asof_ts,
                    value.selector_id,
                    -1 if value.seed is None else value.seed,
                ),
            )
        ),
    )


def _ranked_decision(
    selector_id: str,
    asof_ts: datetime,
    rows: list[SelectorInput],
    selected_k: int | None,
    scores: Mapping[str, float | int] | None,
    *,
    seed: int | None = None,
    lookback: int | None = None,
    skip: int | None = None,
) -> SelectorDecision:
    eligible = [value for value in rows if value.eligible]
    if scores is None:
        ordered = eligible
    else:
        ordered = sorted(eligible, key=lambda value: (-scores[value.asset_id], value.asset_id))
    ranks = {value.asset_id: index for index, value in enumerate(ordered, start=1)}
    selected_ids = {
        value.asset_id for value in (ordered if selected_k is None else ordered[:selected_k])
    }
    assets = tuple(
        SelectorAssetDecision(
            asset_id=value.asset_id,
            eligible=value.eligible,
            score=(float(scores[value.asset_id]) if scores is not None else None),
            rank=ranks.get(value.asset_id),
            selected=value.asset_id in selected_ids,
            input_hash=content_sha256(value.model_dump(mode="json")),
        )
        for value in rows
    )
    return SelectorDecision(
        selector_id=selector_id,
        asof_ts=asof_ts,
        seed=seed,
        lookback_sessions=lookback,
        skip_sessions=skip,
        selected_k=selected_k,
        assets=assets,
    )


def evaluate_selector_signal_matrix(
    matrix: SelectorSignalMatrix,
    predictions: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    backtest_config: BacktestConfig,
    *,
    top_k: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    engine = DeterministicBacktestEngine()
    selected_by_selector: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for decision in matrix.decisions:
        selector_key = decision.selector_id + (
            "" if decision.seed is None else f"/seed={decision.seed}"
        )
        for asset in decision.assets:
            if asset.selected:
                selected_by_selector[selector_key].add(
                    (asset.asset_id, format_utc(decision.asof_ts))
                )
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for selector_key, membership in sorted(selected_by_selector.items()):
        output[selector_key] = {}
        for family, rows in sorted(predictions.items()):
            selected_rows = [
                row for row in rows if (str(row["asset_id"]), str(row["asof_ts"])) in membership
            ]
            output[selector_key][family] = engine.run(
                selected_rows,
                labels,
                backtest_config,
                top_k=None if family in {"equal_weight", "no_trade"} else top_k,
            )
    return output


def moving_block_bootstrap_mean(
    values: tuple[float, ...],
    *,
    block_size: int,
    repetitions: int,
    seed: int,
) -> DependenceAwareInterval:
    if not values or block_size < 1 or block_size > len(values) or repetitions < 100:
        raise ValueError("invalid moving-block bootstrap inputs")
    blocks = [
        values[index : index + block_size] for index in range(0, len(values) - block_size + 1)
    ]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sample: list[float] = []
        while len(sample) < len(values):
            sample.extend(rng.choice(blocks))
        estimates.append(statistics.fmean(sample[: len(values)]))
    ordered = sorted(estimates)
    lower = ordered[int(0.025 * (repetitions - 1))]
    upper = ordered[int(0.975 * (repetitions - 1))]
    return DependenceAwareInterval(
        estimate=statistics.fmean(values),
        lower=lower,
        upper=upper,
        block_size=block_size,
        repetitions=repetitions,
        seed=seed,
    )


def holm_adjust(p_values: tuple[float, ...]) -> tuple[float, ...]:
    if any(value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("p-values must be between zero and one")
    ordered = sorted(enumerate(p_values), key=lambda value: value[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return tuple(adjusted)
