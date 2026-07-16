from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from nlp_trader.config import ResearchConfig
from nlp_trader.holdout_execution import _record_pipeline_failure, _select_complete_holdout_rows
from nlp_trader.research import create_run_context
from nlp_trader.research_agents.contracts import TimeRange


def _features(*decision_times: str) -> list[dict[str, object]]:
    return [
        {
            "asset_id": asset_id,
            "asof_ts": decision_time,
            "horizon": "1d",
        }
        for decision_time in decision_times
        for asset_id in ("asset_aaa", "asset_bbb")
    ]


def _labels(*decision_times: str) -> list[dict[str, object]]:
    return [
        {
            "asset_id": asset_id,
            "asof_ts": decision_time,
            "horizon": "1d",
            "forward_return": 0.01 if asset_id == "asset_aaa" else -0.01,
            "label_end_ts": "2026-07-16T20:00:00Z",
            "label_available_at": "2026-07-16T21:00:00Z",
            "expected_label_end_ts": "2026-07-16T20:00:00Z",
        }
        for decision_time in decision_times
        for asset_id in ("asset_aaa", "asset_bbb")
    ]


def _outcome_interval() -> TimeRange:
    return TimeRange(
        start=datetime(2026, 7, 16, tzinfo=UTC),
        end=datetime(2026, 7, 17, 23, 59, tzinfo=UTC),
    )


def test_holdout_selection_rejects_partial_decision_horizon_cross_section() -> None:
    decision_time = "2026-07-15T20:00:00Z"
    labels = _labels(decision_time)
    labels[1]["forward_return"] = None

    with pytest.raises(ValueError, match="partial forward-label coverage"):
        _select_complete_holdout_rows(
            _features(decision_time),
            labels,
            outcome_interval=_outcome_interval(),
        )


@pytest.mark.parametrize(
    ("field_name", "timestamp"),
    (
        ("label_end_ts", "2026-07-15T23:59:59Z"),
        ("label_end_ts", "2026-07-18T00:00:00Z"),
        ("label_available_at", "2026-07-15T23:59:59Z"),
        ("label_available_at", "2026-07-18T00:00:00Z"),
    ),
)
def test_holdout_selection_requires_label_times_inside_frozen_outcome_interval(
    field_name: str,
    timestamp: str,
) -> None:
    decision_time = "2026-07-15T20:00:00Z"
    labels = _labels(decision_time)
    labels[0][field_name] = timestamp

    with pytest.raises(ValueError, match=field_name):
        _select_complete_holdout_rows(
            _features(decision_time),
            labels,
            outcome_interval=_outcome_interval(),
        )


def test_holdout_selection_preserves_canonical_trailing_group_censoring() -> None:
    first_decision = "2026-07-15T20:00:00Z"
    trailing_decision = "2026-07-16T20:00:00Z"
    features = _features(first_decision, trailing_decision)
    labels = _labels(first_decision, trailing_decision)
    for label in labels[2:]:
        label.update(
            {
                "forward_return": None,
                "label_end_ts": None,
                "label_available_at": None,
                "expected_label_end_ts": "2026-07-17T20:00:00Z",
            }
        )

    selected_features, selected_labels = _select_complete_holdout_rows(
        features,
        labels,
        outcome_interval=_outcome_interval(),
    )

    assert {row["asof_ts"] for row in selected_features} == {first_decision}
    assert {row["asof_ts"] for row in selected_labels} == {first_decision}


def test_holdout_pipeline_failure_emits_one_canonical_failure_manifest(
    generated_config: ResearchConfig,
) -> None:
    context = create_run_context(generated_config, run_id="failed-holdout-pipeline")

    _record_pipeline_failure(context, RuntimeError("synthetic holdout failure"))

    failed_path = context.paths.reports / "run.failed.json"
    original_bytes = failed_path.read_bytes()
    failed = json.loads(original_bytes)
    assert failed["run_id"] == context.run_id
    assert failed["status"] == "failed"
    assert failed["failed_stage"] == "holdout_evaluation"
    assert failed["error_type"] == "RuntimeError"
    assert not (context.paths.reports / "run.final.json").exists()

    _record_pipeline_failure(context, ValueError("must not replace the first failure"))

    assert failed_path.read_bytes() == original_bytes
