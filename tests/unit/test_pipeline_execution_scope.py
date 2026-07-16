from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nlp_trader.pipeline import _development_rows_before_information_cutoff


def test_development_cutoff_purges_whole_cross_section_when_labels_overlap_holdout() -> None:
    features = _features("2026-07-13T20:00:00Z", "2026-07-14T20:00:00Z")
    labels = _labels("2026-07-13T20:00:00Z", end="2026-07-14T20:00:00Z") + _labels(
        "2026-07-14T20:00:00Z",
        end="2026-07-15T20:00:00Z",
    )

    retained_features, retained_labels = _development_rows_before_information_cutoff(
        features,
        labels,
        cutoff=datetime(2026, 7, 15, tzinfo=UTC),
    )

    assert {str(row["asof_ts"]) for row in retained_features} == {"2026-07-13T20:00:00Z"}
    assert {str(row["asof_ts"]) for row in retained_labels} == {"2026-07-13T20:00:00Z"}


def test_development_cutoff_rejects_partial_asset_availability() -> None:
    features = _features("2026-07-14T20:00:00Z")
    labels = _labels("2026-07-14T20:00:00Z", end="2026-07-14T21:00:00Z")
    labels[1]["label_available_at"] = "2026-07-15T20:00:00Z"

    with pytest.raises(ValueError, match="partially select the asset cross-section"):
        _development_rows_before_information_cutoff(
            features,
            labels,
            cutoff=datetime(2026, 7, 15, tzinfo=UTC),
        )


def _features(*decision_times: str) -> list[dict[str, object]]:
    return [
        {"asset_id": asset_id, "asof_ts": asof_ts, "horizon": "1d"}
        for asof_ts in decision_times
        for asset_id in ("asset_aaa", "asset_bbb")
    ]


def _labels(asof_ts: str, *, end: str) -> list[dict[str, object]]:
    return [
        {
            "asset_id": asset_id,
            "asof_ts": asof_ts,
            "horizon": "1d",
            "forward_return": 0.01,
            "label_end_ts": end,
            "label_available_at": end,
        }
        for asset_id in ("asset_aaa", "asset_bbb")
    ]
