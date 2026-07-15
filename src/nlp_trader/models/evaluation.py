from __future__ import annotations

import statistics
from collections import defaultdict
from math import isfinite, sqrt
from typing import Any

from nlp_trader.features.build import finite_float
from nlp_trader.models.baselines import (
    chronological_holdout_split,
    complete_label_cross_sections,
    label_availability_by_key,
    predict_all_families,
    train_baselines,
)
from nlp_trader.timestamps import parse_utc

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
    labels_by_key, observed_keys = complete_label_cross_sections(
        predictions,
        labels,
        row_name="predictions",
    )
    context_by_key = _unique_index(context_rows or [], name="context")
    seen_predictions: set[tuple[str, str, str]] = set()
    joined: list[JoinedRow] = []
    for prediction in predictions:
        key = _key(prediction)
        if key in seen_predictions:
            raise ValueError(f"duplicate prediction row: {key}")
        seen_predictions.add(key)
        if key not in observed_keys:
            continue
        label = labels_by_key[key]
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
    }


def _require_all_or_none(rows: list[dict[str, Any]], field: str) -> bool:
    coverage = [row.get(field) is not None for row in rows]
    if any(coverage) and not all(coverage):
        raise ValueError(f"{field} must be supplied for every evaluated row or for none")
    return bool(coverage) and all(coverage)


def _long_precision_at_k(rows: list[tuple[float, float]], top_k: int) -> float:
    """Return permutation-invariant long precision with fractional cutoff ties."""

    if not rows:
        return 0.0
    depth = min(top_k, len(rows))
    ordered = sorted(rows, key=lambda pair: pair[0], reverse=True)
    cutoff = ordered[depth - 1][0]
    above = [outcome for score, outcome in rows if score > cutoff]
    tied = [outcome for score, outcome in rows if score == cutoff]
    tied_slots = depth - len(above)
    positive_equivalent = float(sum(outcome > 0 for outcome in above))
    positive_equivalent += tied_slots * statistics.fmean(outcome > 0 for outcome in tied)
    return positive_equivalent / depth


def _regression_metrics(joined: list[JoinedRow], *, top_k: int) -> dict[str, float | int]:
    regression_rows = [
        (prediction, finite_float(label["forward_return"])) for prediction, label, _ in joined
    ]
    if not regression_rows:
        return _empty_metrics()
    scores = [finite_float(prediction.get("score")) for prediction, _ in regression_rows]
    outcomes = [outcome for _, outcome in regression_rows]
    scored_outcomes = list(zip(scores, outcomes, strict=True))

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
        daily_precision.append(_long_precision_at_k(rows, top_k))

    metrics: dict[str, float | int] = {
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
        "precision_at_k": _long_precision_at_k(scored_outcomes, top_k),
        "mean_daily_precision_at_k": statistics.fmean(daily_precision),
    }
    if _require_all_or_none([prediction for prediction, _ in regression_rows], "expected_return"):
        expected_returns = [
            (_finite_prediction_value(prediction["expected_return"], "expected_return"), outcome)
            for prediction, outcome in regression_rows
        ]
        metrics["mean_squared_error"] = statistics.fmean(
            (expected - outcome) ** 2 for expected, outcome in expected_returns
        )
    return metrics


def _finite_prediction_value(value: object, name: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _probability(prediction: dict[str, Any]) -> float:
    explicit = prediction.get("probability_up")
    probability = _finite_prediction_value(explicit, "probability_up")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability_up must be in [0, 1], got {explicit!r}")
    return probability


def _binary_target(value: object) -> int:
    if value in (0, False):
        return 0
    if value in (1, True):
        return 1
    raise ValueError(f"binary_up target must be 0 or 1, got {value!r}")


def _classification_metrics(
    joined: list[JoinedRow], *, calibration_bin_count: int
) -> dict[str, Any] | None:
    if not joined:
        return None
    has_probabilities = _require_all_or_none(
        [prediction for prediction, _, _ in joined], "probability_up"
    )
    has_targets = _require_all_or_none([label for _, label, _ in joined], "binary_up")
    if not has_probabilities or not has_targets:
        return None
    classification_rows = [
        (_probability(prediction), _binary_target(label["binary_up"]))
        for prediction, label, _ in joined
    ]
    probabilities = [probability for probability, _ in classification_rows]
    targets = [target for _, target in classification_rows]

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
        "probability_source": "probability_up",
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
    final_holdout_periods: int = 0,
    final_holdout_training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate predictions with an optional reserved chronological final window."""

    if final_holdout_periods < 0:
        raise ValueError("final_holdout_periods must be non-negative")
    if not predictions_by_family:
        raise ValueError("prediction evaluation requires at least one model family")

    candidate_keys: frozenset[tuple[str, str, str]] | None = None
    observed_keys: frozenset[tuple[str, str, str]] | None = None
    for family, rows in sorted(predictions_by_family.items()):
        current_keys = frozenset(_key(row) for row in rows)
        if len(current_keys) != len(rows):
            raise ValueError(f"duplicate prediction row in family {family}")
        if candidate_keys is None:
            candidate_keys = current_keys
            _, observed_keys = complete_label_cross_sections(
                rows,
                labels,
                row_name="predictions",
            )
        elif current_keys != candidate_keys:
            raise ValueError("all model families must cover the same prediction cross-sections")

    pre_holdout_times, configured_holdout_times = chronological_holdout_split(
        observed_keys or frozenset(),
        final_holdout_periods,
    )
    complete_times = [*pre_holdout_times, *configured_holdout_times]
    metric_keys = observed_keys or frozenset()
    _require_all_or_none(
        [label for label in labels if _key(label) in metric_keys],
        "binary_up",
    )
    for _family, rows in sorted(predictions_by_family.items()):
        observed_rows = [row for row in rows if _key(row) in metric_keys]
        _require_all_or_none(observed_rows, "expected_return")
        _require_all_or_none(observed_rows, "probability_up")

    def payload(
        selected_predictions: dict[str, list[dict[str, Any]]],
        selected_labels: list[dict[str, Any]],
        selected_context: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        return {
            "families": {
                family: prediction_metrics(
                    rows,
                    selected_labels,
                    top_k=top_k,
                    calibration_bin_count=calibration_bin_count,
                )
                for family, rows in selected_predictions.items()
            },
            "segments": {
                family: segmented_prediction_metrics(
                    rows,
                    selected_labels,
                    context_rows=selected_context,
                    top_k=top_k,
                    calibration_bin_count=calibration_bin_count,
                )
                for family, rows in selected_predictions.items()
            },
        }

    if final_holdout_periods:
        holdout_times = frozenset(configured_holdout_times)
        holdout_start = min(holdout_times, key=parse_utc)
        availability = label_availability_by_key(labels)
        keys_by_time: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for key in observed_keys or frozenset():
            keys_by_time[key[1]].append(key)
        development_values: list[str] = []
        purged_values: list[str] = []
        purge_suffix_started = False
        for decision_time in pre_holdout_times:
            decision_keys = keys_by_time[decision_time]
            missing_availability = [key for key in decision_keys if key not in availability]
            if missing_availability:
                raise ValueError(
                    "cannot establish final-holdout purge boundary for labels: "
                    f"{missing_availability}"
                )
            wholly_available = all(
                parse_utc(availability[key]) < parse_utc(holdout_start) for key in decision_keys
            )
            if wholly_available and not purge_suffix_started:
                development_values.append(decision_time)
            else:
                purge_suffix_started = True
                purged_values.append(decision_time)
        if not development_values:
            raise ValueError(
                "final_holdout_periods must leave at least one development period after purging "
                "overlapping labels"
            )
        development_times = frozenset(development_values)

        if final_holdout_training is None:
            raise ValueError("final-holdout evaluation requires frozen training protocol metadata")
        holdout_training_protocol = dict(final_holdout_training)
        if holdout_training_protocol.get("enabled") is not True:
            raise ValueError("final-holdout evaluation requires an enabled frozen training rule")
        if holdout_training_protocol.get("name") != "frozen_pre_holdout_snapshot_v1":
            raise ValueError("final-holdout evaluation requires the frozen training protocol")
        if holdout_training_protocol.get("final_holdout_start") != holdout_start:
            raise ValueError("training and evaluation final-holdout boundaries do not match")
        if holdout_training_protocol.get("final_holdout_periods") != len(holdout_times):
            raise ValueError("training and evaluation final-holdout period counts do not match")
        cutoff = holdout_training_protocol.get("training_cutoff_exclusive")
        if not isinstance(cutoff, str) or parse_utc(cutoff) > parse_utc(holdout_start):
            raise ValueError(
                "frozen final-holdout training cutoff must be no later than the holdout start"
            )
        frozen_asof = holdout_training_protocol.get("frozen_snapshot_asof_ts")
        if not isinstance(frozen_asof, str) or parse_utc(frozen_asof) > parse_utc(holdout_start):
            raise ValueError(
                "frozen final-holdout snapshot must be no later than the holdout start"
            )
        if (
            holdout_training_protocol.get("training_key_rule")
            != "training key asof_ts < final_holdout_start"
        ):
            raise ValueError("final-holdout training-key rule does not exclude holdout decisions")
        if (
            holdout_training_protocol.get("update_rule")
            != "no training updates at or after final_holdout_start"
        ):
            raise ValueError("final-holdout training update rule is not frozen")
        holdout_training_protocol["verified_untouched"] = True

        def select_rows(rows: list[dict[str, Any]], times: frozenset[str]) -> list[dict[str, Any]]:
            return [row for row in rows if str(row["asof_ts"]) in times]

        development = payload(
            {
                family: select_rows(rows, development_times)
                for family, rows in predictions_by_family.items()
            },
            select_rows(labels, development_times),
            select_rows(context_rows or [], development_times)
            if context_rows is not None
            else None,
        )
        final_holdout = payload(
            {
                family: select_rows(rows, holdout_times)
                for family, rows in predictions_by_family.items()
            },
            select_rows(labels, holdout_times),
            select_rows(context_rows or [], holdout_times) if context_rows is not None else None,
        )
        result: dict[str, Any] = {
            **development,
            "final_holdout": final_holdout,
            "evaluation_protocol": {
                "name": "chronological_frozen_final_holdout_v2",
                "selection_scope": "development_only",
                "development_periods": len(development_times),
                "pre_holdout_periods": len(pre_holdout_times),
                "purged_development_periods": len(purged_values),
                "purged_development_times": purged_values,
                "final_holdout_periods": len(holdout_times),
                "final_holdout_start": holdout_start,
                "final_holdout_end": max(holdout_times, key=parse_utc),
                "purge_rule": "training labels become eligible only when their availability "
                "timestamp is strictly before the effective cutoff",
                "development_purge_rule": "remove the contiguous pre-holdout suffix beginning "
                "at the first cross-section not wholly available before final_holdout_start",
                "holdout_update_rule": "no training updates at or after final_holdout_start",
                "final_holdout_training": holdout_training_protocol,
            },
        }
    else:
        result = {
            **payload(predictions_by_family, labels, context_rows),
            "evaluation_protocol": {
                "name": "walk_forward_without_reserved_final_holdout",
                "selection_scope": "all_complete_periods",
                "development_periods": len(complete_times),
                "final_holdout_periods": 0,
                "purge_rule": "training labels become eligible only when their availability "
                "timestamp is strictly before the effective cutoff",
            },
        }

    result["segment_definitions"] = {
        "liquidity_bucket": LIQUIDITY_BUCKETS,
        "volatility_regime": VOLATILITY_BUCKETS,
        "date_safety": "fixed thresholds and contemporaneous row metadata only",
    }
    result["metric_definitions"] = {
        "precision_at_k": "global long-side diagnostic: highest raw scores, outcome > 0; "
        "cutoff ties receive fractional positive-rate credit",
        "mean_daily_precision_at_k": (
            "mean of per-decision long-side precision using the highest raw scores and "
            "fractional cutoff-tie credit"
        ),
        "portfolio_selection": (
            "reported by backtests; direction eligibility and absolute-score ranking differ "
            "from long-side precision-at-k"
        ),
        "optional_metric_coverage": (
            "expected_return, probability_up, and binary_up are accepted only with all-or-none "
            "coverage in each evaluated cohort"
        ),
    }
    return result


def evaluate_families(
    features: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    model_version: str,
    top_k: int = 10,
    min_train_rows: int = 0,
    embargo_periods: int = 0,
    calibration_bin_count: int = 10,
    final_holdout_periods: int = 0,
) -> dict[str, Any]:
    model = train_baselines(
        features,
        labels,
        model_version=model_version,
        min_train_rows=min_train_rows,
        embargo_periods=embargo_periods,
        final_holdout_periods=final_holdout_periods,
    )
    predictions = predict_all_families(features, model)
    diagnostics = evaluate_predictions(
        predictions,
        labels,
        context_rows=features,
        top_k=top_k,
        calibration_bin_count=calibration_bin_count,
        final_holdout_periods=final_holdout_periods,
        final_holdout_training=model["final_holdout_training"],
    )
    return {
        "model_version": model_version,
        "training_protocol": model["training_protocol"],
        **diagnostics,
    }
