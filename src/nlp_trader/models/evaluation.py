from __future__ import annotations

import statistics
from collections import defaultdict
from math import exp, isfinite, sqrt
from typing import Any

from nlp_trader.features.build import finite_float
from nlp_trader.models.baselines import predict_all_families, train_baselines

JoinedRow = tuple[dict[str, Any], dict[str, Any], dict[str, Any]]

LIQUIDITY_BUCKETS = {
    "low": "dollar volume < 5,000,000",
    "medium": "5,000,000 <= dollar volume < 50,000,000",
    "high": "dollar volume >= 50,000,000",
}
VOLATILITY_BUCKETS = {
    "low": "volatility regime < 0.8, or daily volatility < 0.01",
    "normal": "volatility regime 0.8-1.2, or daily volatility 0.01-0.03",
    "high": "volatility regime > 1.2, or daily volatility >= 0.03",
}


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["asset_id"]), str(row["asof_ts"]), str(row["horizon"])


def _unique_index(
    rows: list[dict[str, Any]], *, name: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate {name} row: {key}")
        indexed[key] = row
    return indexed


def _join_rows(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    context_rows: list[dict[str, Any]] | None = None,
) -> list[JoinedRow]:
    labels_by_key = _unique_index(labels, name="label")
    context_by_key = _unique_index(context_rows or [], name="context")
    seen_predictions: set[tuple[str, str, str]] = set()
    joined: list[JoinedRow] = []
    for prediction in predictions:
        key = _key(prediction)
        if key in seen_predictions:
            raise ValueError(f"duplicate prediction row: {key}")
        seen_predictions.add(key)
        label = labels_by_key.get(key)
        if label is None:
            continue
        context = dict(prediction)
        context.update(context_by_key.get(key, {}))
        joined.append((prediction, label, context))
    return joined


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in range(cursor, end):
            result[ordered[index][0]] = rank
        cursor = end
    return result


def _empty_metrics() -> dict[str, float | int]:
    return {
        "rows": 0,
        "dates": 0,
        "ic_dates": 0,
        "pearson_ic": 0.0,
        "spearman_ic": 0.0,
        "mean_daily_pearson_ic": 0.0,
        "mean_daily_spearman_ic": 0.0,
        "hit_rate": 0.0,
        "precision_at_k": 0.0,
        "mean_daily_precision_at_k": 0.0,
        "mean_squared_error": 0.0,
    }


def _regression_metrics(joined: list[JoinedRow], *, top_k: int) -> dict[str, float | int]:
    regression_rows = [
        (prediction, finite_float(label["forward_return"]))
        for prediction, label, _ in joined
        if label.get("forward_return") is not None
    ]
    if not regression_rows:
        return _empty_metrics()
    scores = [finite_float(prediction.get("score")) for prediction, _ in regression_rows]
    outcomes = [outcome for _, outcome in regression_rows]
    ordered = sorted(zip(scores, outcomes, strict=True), key=lambda pair: pair[0], reverse=True)[
        :top_k
    ]

    by_date: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for prediction, outcome in regression_rows:
        by_date[str(prediction["asof_ts"])].append((finite_float(prediction.get("score")), outcome))
    daily_pearson: list[float] = []
    daily_spearman: list[float] = []
    daily_precision: list[float] = []
    for rows in by_date.values():
        date_scores = [score for score, _ in rows]
        date_outcomes = [outcome for _, outcome in rows]
        if len(rows) >= 2:
            daily_pearson.append(_pearson(date_scores, date_outcomes))
            daily_spearman.append(_pearson(_ranks(date_scores), _ranks(date_outcomes)))
        selected = sorted(rows, key=lambda pair: pair[0], reverse=True)[:top_k]
        daily_precision.append(
            statistics.fmean(1.0 if outcome > 0 else 0.0 for _, outcome in selected)
        )

    return {
        "rows": len(regression_rows),
        "dates": len(by_date),
        "ic_dates": len(daily_pearson),
        "pearson_ic": _pearson(scores, outcomes),
        "spearman_ic": _pearson(_ranks(scores), _ranks(outcomes)),
        "mean_daily_pearson_ic": statistics.fmean(daily_pearson) if daily_pearson else 0.0,
        "mean_daily_spearman_ic": statistics.fmean(daily_spearman) if daily_spearman else 0.0,
        "hit_rate": statistics.fmean(
            1.0 if score * outcome > 0 else 0.0
            for score, outcome in zip(scores, outcomes, strict=True)
        ),
        "precision_at_k": statistics.fmean(1.0 if outcome > 0 else 0.0 for _, outcome in ordered),
        "mean_daily_precision_at_k": statistics.fmean(daily_precision),
        "mean_squared_error": statistics.fmean(
            (score - outcome) ** 2 for score, outcome in zip(scores, outcomes, strict=True)
        ),
    }


def _probability(prediction: dict[str, Any]) -> tuple[float, str]:
    explicit = prediction.get("probability_up")
    if explicit is not None:
        try:
            probability = float(explicit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"probability_up must be numeric, got {explicit!r}") from exc
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability_up must be in [0, 1], got {explicit!r}")
        return probability, "probability_up"
    score = finite_float(prediction.get("score"))
    if score >= 0:
        return 1.0 / (1.0 + exp(-score)), "logistic_score"
    scaled = exp(score)
    return scaled / (1.0 + scaled), "logistic_score"


def _binary_target(value: object) -> int:
    if value in (0, False):
        return 0
    if value in (1, True):
        return 1
    raise ValueError(f"binary_up target must be 0 or 1, got {value!r}")


def _classification_metrics(
    joined: list[JoinedRow], *, calibration_bin_count: int
) -> dict[str, Any] | None:
    classification_rows = [
        (prediction, _binary_target(label["binary_up"]))
        for prediction, label, _ in joined
        if label.get("binary_up") is not None
    ]
    if not classification_rows:
        return None
    probabilities: list[float] = []
    targets: list[int] = []
    sources: set[str] = set()
    for prediction, target in classification_rows:
        probability, source = _probability(prediction)
        probabilities.append(probability)
        targets.append(target)
        sources.add(source)

    bins: list[list[tuple[float, int]]] = [[] for _ in range(calibration_bin_count)]
    for probability, target in zip(probabilities, targets, strict=True):
        index = min(calibration_bin_count - 1, int(probability * calibration_bin_count))
        bins[index].append((probability, target))
    calibration: list[dict[str, float | int]] = []
    expected_calibration_error = 0.0
    for index, rows in enumerate(bins):
        if not rows:
            continue
        mean_probability = statistics.fmean(probability for probability, _ in rows)
        observed_rate = statistics.fmean(target for _, target in rows)
        expected_calibration_error += (
            len(rows) / len(classification_rows) * abs(mean_probability - observed_rate)
        )
        calibration.append(
            {
                "lower_bound": index / calibration_bin_count,
                "upper_bound": (index + 1) / calibration_bin_count,
                "rows": len(rows),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return {
        "rows": len(classification_rows),
        "brier_score": statistics.fmean(
            (probability - target) ** 2
            for probability, target in zip(probabilities, targets, strict=True)
        ),
        "expected_calibration_error": expected_calibration_error,
        "probability_source": next(iter(sources)) if len(sources) == 1 else "mixed",
        "calibration_bins": calibration,
    }


def _metrics(joined: list[JoinedRow], *, top_k: int, calibration_bin_count: int) -> dict[str, Any]:
    metrics: dict[str, Any] = _regression_metrics(joined, top_k=top_k)
    classification = _classification_metrics(joined, calibration_bin_count=calibration_bin_count)
    if classification is not None:
        metrics["classification"] = classification
    return metrics


def prediction_metrics(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    top_k: int = 10,
    calibration_bin_count: int = 10,
) -> dict[str, Any]:
    """Return date-safe diagnostics without fitting or selecting a model."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if calibration_bin_count < 2:
        raise ValueError("calibration_bin_count must be at least 2")
    return _metrics(
        _join_rows(predictions, labels),
        top_k=top_k,
        calibration_bin_count=calibration_bin_count,
    )


def _liquidity_bucket(context: dict[str, Any]) -> str | None:
    value = context.get("average_dollar_volume_20d", context.get("dollar_volume"))
    if value is None:
        return None
    dollar_volume = finite_float(value)
    if dollar_volume < 5_000_000.0:
        return "low"
    if dollar_volume < 50_000_000.0:
        return "medium"
    return "high"


def _volatility_bucket(context: dict[str, Any]) -> str | None:
    regime = context.get("volatility_regime")
    if isinstance(regime, str) and regime:
        return regime
    if regime is not None:
        ratio = finite_float(regime)
        if ratio < 0.8:
            return "low"
        if ratio <= 1.2:
            return "normal"
        return "high"
    value = context.get(
        "realized_volatility_20d",
        context.get("realized_volatility_3d", context.get("volatility")),
    )
    if value is None:
        return None
    volatility = finite_float(value)
    if volatility < 0.01:
        return "low"
    if volatility < 0.03:
        return "normal"
    return "high"


def _source_types(context: dict[str, Any]) -> set[str]:
    sources: set[str] = set()
    direct = context.get("source_type") or context.get("source")
    if direct:
        sources.add(str(direct))
    for key, value in context.items():
        if not key.startswith("attention_source_") or "_count_" not in key:
            continue
        if finite_float(value) <= 0:
            continue
        source = key.removeprefix("attention_source_").split("_count_", maxsplit=1)[0]
        if source:
            sources.add(source)
    return sources


def _source_availability(context: dict[str, Any], sources: set[str]) -> str | None:
    if sources:
        return "available"
    missing_flags = [
        bool(value) for key, value in context.items() if key.startswith("source_identity_missing_")
    ]
    source_columns_present = any(
        key.startswith(("attention_source_", "source_identity_missing_")) for key in context
    )
    if not source_columns_present:
        return None
    return "missing" if any(missing_flags) or not sources else "available"


def _event_types(context: dict[str, Any]) -> set[str]:
    events: set[str] = set()
    direct = context.get("event_type")
    if direct:
        events.add(str(direct))
    for key, value in context.items():
        if not key.startswith("event_") or "_count_" not in key:
            continue
        if key.startswith(("event_item_count_", "event_type_diversity_count_")):
            continue
        if finite_float(value) <= 0:
            continue
        event = key.removeprefix("event_").split("_count_", maxsplit=1)[0]
        if event:
            events.add(event)
    return events


def _event_availability(context: dict[str, Any], events: set[str]) -> str | None:
    if events:
        return "available"
    item_counts = [
        finite_float(value) for key, value in context.items() if key.startswith("event_item_count_")
    ]
    if not item_counts:
        return None
    return "available" if any(count > 0 for count in item_counts) else "missing"


def segmented_prediction_metrics(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    context_rows: list[dict[str, Any]] | None = None,
    top_k: int = 10,
    calibration_bin_count: int = 10,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Segment diagnostics using only metadata available on each decision row."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if calibration_bin_count < 2:
        raise ValueError("calibration_bin_count must be at least 2")
    joined = _join_rows(predictions, labels, context_rows)
    grouped: dict[str, dict[str, list[JoinedRow]]] = defaultdict(lambda: defaultdict(list))
    for row in joined:
        context = row[2]
        sector = context.get("sector")
        if sector:
            grouped["sector"][str(sector)].append(row)
        liquidity = _liquidity_bucket(context)
        if liquidity is not None:
            grouped["liquidity_bucket"][liquidity].append(row)
        volatility = _volatility_bucket(context)
        if volatility is not None:
            grouped["volatility_regime"][volatility].append(row)
        sources = _source_types(context)
        for source in sources:
            grouped["source_type"][source].append(row)
        source_availability = _source_availability(context, sources)
        if source_availability is not None:
            grouped["source_availability"][source_availability].append(row)
        events = _event_types(context)
        for event in events:
            grouped["event_type"][event].append(row)
        event_availability = _event_availability(context, events)
        if event_availability is not None:
            grouped["event_availability"][event_availability].append(row)
    return {
        dimension: {
            group: _metrics(
                rows,
                top_k=top_k,
                calibration_bin_count=calibration_bin_count,
            )
            for group, rows in sorted(groups.items())
        }
        for dimension, groups in sorted(grouped.items())
    }


def evaluate_predictions(
    predictions_by_family: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    *,
    context_rows: list[dict[str, Any]] | None = None,
    top_k: int = 10,
    calibration_bin_count: int = 10,
) -> dict[str, Any]:
    """Evaluate existing predictions without fitting on any evaluation outcome."""

    return {
        "families": {
            family: prediction_metrics(
                rows,
                labels,
                top_k=top_k,
                calibration_bin_count=calibration_bin_count,
            )
            for family, rows in predictions_by_family.items()
        },
        "segments": {
            family: segmented_prediction_metrics(
                rows,
                labels,
                context_rows=context_rows,
                top_k=top_k,
                calibration_bin_count=calibration_bin_count,
            )
            for family, rows in predictions_by_family.items()
        },
        "segment_definitions": {
            "liquidity_bucket": LIQUIDITY_BUCKETS,
            "volatility_regime": VOLATILITY_BUCKETS,
            "date_safety": "fixed thresholds and contemporaneous row metadata only",
        },
    }


def evaluate_families(
    features: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    model_version: str,
    top_k: int = 10,
    min_train_rows: int = 0,
    embargo_periods: int = 0,
    calibration_bin_count: int = 10,
) -> dict[str, Any]:
    model = train_baselines(
        features,
        labels,
        model_version=model_version,
        min_train_rows=min_train_rows,
        embargo_periods=embargo_periods,
    )
    predictions = predict_all_families(features, model)
    diagnostics = evaluate_predictions(
        predictions,
        labels,
        context_rows=features,
        top_k=top_k,
        calibration_bin_count=calibration_bin_count,
    )
    return {
        "model_version": model_version,
        "training_protocol": model["training_protocol"],
        **diagnostics,
    }
