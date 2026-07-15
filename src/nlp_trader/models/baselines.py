from __future__ import annotations

import hashlib
import json
import statistics
from bisect import bisect_right
from collections import defaultdict
from math import sqrt
from typing import Any

from nlp_trader.features.build import finite_float
from nlp_trader.timestamps import format_utc, parse_utc

FEATURE_SETS: dict[str, list[str]] = {
    "traditional": [
        "return_1d",
        "return_3d",
        "return_5d",
        "return_20d",
        "return_60d",
        "abnormal_volume_3d",
        "realized_volatility_3d",
    ],
    "text": [
        "text_count_1d",
        "sentiment_mean_1d",
        "sentiment_conf_weighted_1d",
        "novelty_share_1d",
        "text_count_3d",
        "sentiment_mean_3d",
        "sentiment_conf_weighted_3d",
        "novelty_share_3d",
    ],
}
FEATURE_SETS["combined"] = FEATURE_SETS["traditional"] + FEATURE_SETS["text"]

BENCHMARK_FAMILIES = ("equal_weight", "momentum_only", "no_trade")

_TEXT_PREFIXES = (
    "text_",
    "sentiment_",
    "attention_",
    "novelty_",
    "disagreement_",
    "credibility_",
    "event_",
)
_TRADITIONAL_PREFIXES = (
    "return_",
    "momentum_",
    "reversal_",
    "gap_",
    "abnormal_volume_",
    "turnover_",
    "illiquidity_",
    "spread_",
    "realized_volatility_",
    "downside_volatility_",
    "high_low_volatility_",
    "volatility_regime_",
    "market_beta_",
    "sector_return_",
    "residual_return_",
    "size_",
    "value_",
    "quality_",
    "earnings_",
    "ex_dividend_",
)


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["asset_id"]), str(row["asof_ts"]), str(row["horizon"])


def _correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def _horizon_steps(value: object) -> int:
    text = str(value).strip().lower()
    digits = "".join(character for character in text if character.isdigit())
    return max(1, int(digits)) if digits else 1


def _label_availability(labels: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    """Infer when each forward label became observable from ordered asset sessions.

    An explicit ``label_available_at`` or ``available_at`` wins. Otherwise a 5d label is
    considered available at the fifth later row for the same asset and horizon. This mirrors
    the label builder's trading-bar indexing without assuming weekdays are exchange sessions.
    """

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        grouped[(str(row["asset_id"]), str(row["horizon"]))].append(row)

    available: dict[tuple[str, str, str], str] = {}
    for rows in grouped.values():
        rows.sort(key=lambda row: parse_utc(str(row["asof_ts"])))
        for index, row in enumerate(rows):
            explicit = (
                row.get("label_available_at") or row.get("label_end_ts") or row.get("available_at")
            )
            if explicit is not None:
                available[_key(row)] = str(explicit)
                continue
            target_index = index + _horizon_steps(row["horizon"])
            if target_index < len(rows):
                available[_key(row)] = str(rows[target_index]["asof_ts"])
    return available


def _discover_columns(features: list[dict[str, Any]]) -> dict[str, list[str]]:
    keys = {str(key) for row in features for key in row}
    text = sorted(key for key in keys if key.startswith(_TEXT_PREFIXES))
    traditional = sorted(key for key in keys if key.startswith(_TRADITIONAL_PREFIXES))
    for column in FEATURE_SETS["text"]:
        if column in keys and column not in text:
            text.append(column)
    for column in FEATURE_SETS["traditional"]:
        if column in keys and column not in traditional:
            traditional.append(column)
    return {
        "traditional": traditional,
        "text": text,
        "combined": traditional + text,
    }


def _fit_family(
    joined: list[tuple[dict[str, Any], dict[str, Any]]], columns: list[str]
) -> dict[str, Any]:
    ys = [finite_float(label["forward_return"]) for _, label in joined]
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    raw_weights: dict[str, float] = {}
    for column in columns:
        xs = [finite_float(row.get(column)) for row, _ in joined]
        means[column] = statistics.fmean(xs) if xs else 0.0
        scale = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        scales[column] = scale if scale > 0 else 1.0
        raw_weights[column] = _correlation(xs, ys)
    normalizer = sum(abs(weight) for weight in raw_weights.values()) or 1.0
    weights = {column: weight / normalizer for column, weight in raw_weights.items()}
    fitted = [
        sum(
            (finite_float(row.get(column)) - means[column]) / scales[column] * weights[column]
            for column in columns
        )
        for row, _ in joined
    ]
    residuals = [actual - estimate for actual, estimate in zip(ys, fitted, strict=True)]
    residual_scale = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0
    return {
        "features": columns,
        "weights": weights,
        "means": means,
        "scales": scales,
        "training_rows": len(joined),
        "residual_scale": residual_scale,
    }


def _fit_families(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    columns_by_family: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    return {family: _fit_family(joined, columns) for family, columns in columns_by_family.items()}


class _RunningFit:
    """Sufficient statistics for linear correlation baselines in bounded memory."""

    def __init__(self, columns: list[str], *, record_keys: bool = False) -> None:
        self.columns = columns
        self.rows = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_x = {column: 0.0 for column in columns}
        self.sum_x2 = {column: 0.0 for column in columns}
        self.sum_xy = {column: 0.0 for column in columns}
        self.keys: list[tuple[str, str, str]] | None = [] if record_keys else None
        self._key_digest = hashlib.sha256()

    def add(self, feature: dict[str, Any], label: dict[str, Any]) -> None:
        y = finite_float(label["forward_return"])
        self.rows += 1
        self.sum_y += y
        self.sum_y2 += y * y
        for column in self.columns:
            x = finite_float(feature.get(column))
            self.sum_x[column] += x
            self.sum_x2[column] += x * x
            self.sum_xy[column] += x * y
        key = _key(feature)
        self._key_digest.update(json.dumps(key, separators=(",", ":")).encode("utf-8"))
        self._key_digest.update(b"\n")
        if self.keys is not None:
            self.keys.append(key)

    def key_digest(self) -> str:
        return self._key_digest.copy().hexdigest()

    def fit(self, columns: list[str]) -> dict[str, Any]:
        if self.rows == 0:
            return _fit_family([], columns)
        count = float(self.rows)
        centered_y = max(0.0, self.sum_y2 - self.sum_y * self.sum_y / count)
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        raw_weights: dict[str, float] = {}
        for column in columns:
            mean_x = self.sum_x[column] / count
            centered_x = max(
                0.0,
                self.sum_x2[column] - self.sum_x[column] * self.sum_x[column] / count,
            )
            covariance = self.sum_xy[column] - self.sum_x[column] * self.sum_y / count
            denominator = sqrt(centered_x * centered_y)
            means[column] = mean_x
            scale = sqrt(centered_x / count) if centered_x > 0 else 0.0
            scales[column] = scale if scale > 0 else 1.0
            raw_weights[column] = covariance / denominator if denominator else 0.0
        normalizer = sum(abs(weight) for weight in raw_weights.values()) or 1.0
        return {
            "features": columns,
            "weights": {column: weight / normalizer for column, weight in raw_weights.items()},
            "means": means,
            "scales": scales,
            "training_rows": self.rows,
            "residual_scale": sqrt(centered_y / count) if centered_y > 0 else 0.0,
            "uncertainty_proxy": "training_target_standard_deviation",
        }

    def fit_families(self, columns_by_family: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
        return {family: self.fit(columns) for family, columns in columns_by_family.items()}


def _training_key_digest(keys: list[tuple[str, str, str]]) -> str:
    encoded = json.dumps(sorted(keys), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_baselines(
    features: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    model_version: str,
    min_train_rows: int = 0,
    embargo_periods: int = 0,
    record_training_keys: bool = False,
) -> dict[str, Any]:
    """Fit deterministic expanding-window snapshots without future-label leakage."""

    if min_train_rows < 0:
        raise ValueError("min_train_rows must be non-negative")
    if embargo_periods < 0:
        raise ValueError("embargo_periods must be non-negative")
    labels_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for label in labels:
        key = _key(label)
        if key in labels_by_key:
            raise ValueError(f"duplicate label row: {key}")
        if label.get("forward_return") is not None:
            labels_by_key[key] = label

    availability = _label_availability(labels)
    columns_by_family = _discover_columns(features)
    all_columns = sorted({column for columns in columns_by_family.values() for column in columns})
    events = sorted(
        (
            parse_utc(availability[_key(row)]),
            _key(row),
            row,
            labels_by_key[_key(row)],
        )
        for row in features
        if _key(row) in labels_by_key and _key(row) in availability
    )
    running = _RunningFit(all_columns, record_keys=record_training_keys)
    event_index = 0
    decision_times = sorted({str(row["asof_ts"]) for row in features}, key=parse_utc)
    snapshots: list[dict[str, Any]] = []
    for decision_index, decision_time in enumerate(decision_times):
        decision_ts = parse_utc(decision_time)
        cutoff_index = max(0, decision_index - embargo_periods)
        training_cutoff = parse_utc(decision_times[cutoff_index])
        effective_cutoff = min(decision_ts, training_cutoff)
        while event_index < len(events) and events[event_index][0] < effective_cutoff:
            _, _, feature, label = events[event_index]
            running.add(feature, label)
            event_index += 1
        trained = running.rows >= min_train_rows
        active_keys = running.keys if trained and running.keys is not None else []
        snapshot: dict[str, Any] = {
            "asof_ts": decision_time,
            "training_cutoff_exclusive": format_utc(effective_cutoff),
            "eligible_training_rows": running.rows,
            "training_key_count": running.rows if trained else 0,
            "training_key_digest": running.key_digest() if trained else _training_key_digest([]),
            "families": running.fit_families(columns_by_family)
            if trained
            else _fit_families([], columns_by_family),
        }
        if record_training_keys:
            snapshot["training_keys"] = [list(key) for key in sorted(active_keys)]
        snapshots.append(snapshot)

    empty_families = _fit_families([], columns_by_family)
    latest_families = snapshots[-1]["families"] if snapshots else empty_families
    return {
        "model_version": model_version,
        "training_protocol": "incremental_expanding_walk_forward_strict_availability_v3",
        "min_train_rows": min_train_rows,
        "embargo_periods": embargo_periods,
        "families": latest_families,
        "walk_forward_snapshots": snapshots,
        "benchmark_families": list(BENCHMARK_FAMILIES),
    }


def _score(row: dict[str, Any], spec: dict[str, Any]) -> float:
    return sum(
        (
            (finite_float(row.get(column)) - float(spec["means"][column]))
            / float(spec["scales"][column])
            * float(spec["weights"][column])
        )
        for column in spec["features"]
    )


def _momentum_score(row: dict[str, Any]) -> float:
    for column in ("return_20d", "return_5d", "return_3d", "return_1d"):
        if column in row:
            return finite_float(row[column])
    return 0.0


def _prediction_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata_columns = (
        "close",
        "dollar_volume",
        "sector",
        "beta",
        "market_beta",
        "volatility",
        "market_beta_60d_missing",
        "realized_volatility_20d_missing",
        "beta_fallback_used",
        "volatility_fallback_used",
        "realized_volatility_3d",
        "high_low_volatility_20d",
        "short_available",
        "hard_to_borrow",
    )
    return {column: row[column] for column in metadata_columns if column in row}


def _empty_families(families: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        family: {
            "features": list(spec["features"]),
            "weights": {column: 0.0 for column in spec["features"]},
            "means": {column: 0.0 for column in spec["features"]},
            "scales": {column: 1.0 for column in spec["features"]},
            "training_rows": 0,
            "residual_scale": 0.0,
        }
        for family, spec in families.items()
    }


def predict_with_model(
    features: list[dict[str, Any]], model: dict[str, Any], *, family: str = "combined"
) -> list[dict[str, Any]]:
    """Score rows with their own temporal snapshot or a deterministic naive benchmark."""

    valid_families = set(model["families"]) | set(BENCHMARK_FAMILIES)
    if family not in valid_families:
        raise ValueError(
            f"unknown model family {family!r}; expected one of {sorted(valid_families)}"
        )

    snapshots = sorted(
        model.get("walk_forward_snapshots", []),
        key=lambda snapshot: parse_utc(str(snapshot["asof_ts"])),
    )
    snapshot_times = [parse_utc(str(snapshot["asof_ts"])) for snapshot in snapshots]
    empty_families = _empty_families(model["families"])
    predictions: list[dict[str, Any]] = []
    for row in features:
        uncertainty: float | None
        training_rows = 0
        if family == "equal_weight":
            score = 1.0
            uncertainty = None
        elif family == "momentum_only":
            score = _momentum_score(row)
            uncertainty = None
        elif family == "no_trade":
            score = 0.0
            uncertainty = None
        else:
            if snapshots:
                snapshot_index = bisect_right(snapshot_times, parse_utc(str(row["asof_ts"]))) - 1
                families = (
                    snapshots[snapshot_index]["families"] if snapshot_index >= 0 else empty_families
                )
            else:
                families = model["families"]
            spec = families[family]
            score = _score(row, spec)
            uncertainty = float(spec.get("residual_scale", 0.0))
            training_rows = int(spec.get("training_rows", 0))
        prediction = {
            "asset_id": row["asset_id"],
            "symbol": row["symbol"],
            "asof_ts": row["asof_ts"],
            "horizon": row["horizon"],
            "model_version": model["model_version"],
            "model_family": family,
            "score": score,
            "expected_return": None,
            "uncertainty": uncertainty,
            "training_rows": training_rows,
        }
        prediction.update(_prediction_metadata(row))
        predictions.append(prediction)
    return sorted(predictions, key=lambda row: (row["asof_ts"], row["symbol"]))


def predict_all_families(
    features: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    families = list(model["families"]) + list(BENCHMARK_FAMILIES)
    return {family: predict_with_model(features, model, family=family) for family in families}
