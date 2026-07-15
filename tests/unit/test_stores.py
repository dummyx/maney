from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nlp_trader.data.stores import (
    ContentAddressedRawStore,
    LocalFeatureStore,
    LocalModelRegistry,
    ParquetFeatureStore,
    PointInTimeViolation,
    RawIngestionRequest,
)
from nlp_trader.schemas import FeatureRow


def test_raw_store_is_content_addressed_append_only_and_idempotent(tmp_path: Path) -> None:
    store = ContentAddressedRawStore(tmp_path / "raw")
    request = RawIngestionRequest(
        source="news",
        vendor="synthetic-vendor",
        license_or_terms_ref="synthetic-fixture-v1",
        ingested_at=datetime(2026, 7, 6, 13, tzinfo=UTC),
        request_id="batch-001",
        schema_version="text-v1",
        fetch_params={"symbols": ["AAA"], "limit": 10},
    )

    first = store.ingest_bytes(b'{"ok":true}\n', request, suffix="json")
    second = store.ingest_bytes(b'{"ok":true}\n', request, suffix="json")

    assert first.payload_path == second.payload_path
    assert first.metadata_path == second.metadata_path
    assert first.payload_path.read_bytes() == b'{"ok":true}\n'
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source"] == "news"
    assert metadata["vendor"] == "synthetic-vendor"
    assert metadata["request_id"] == "batch-001"
    assert metadata["sha256"] == first.metadata.sha256
    assert len(metadata["fetch_params_hash"]) == 64


def test_feature_store_rejects_duplicates_and_future_inputs(tmp_path: Path) -> None:
    store = LocalFeatureStore(tmp_path / "features")
    asof_ts = datetime(2026, 7, 6, 20, tzinfo=UTC)
    row = FeatureRow(
        asset_id="asset_aaa",
        symbol="AAA",
        asof_ts=asof_ts,
        horizon="1d",
        feature_set_version="features-v1",
        features={"return_1d": 0.01},
        input_available_at=(asof_ts - timedelta(minutes=1),),
    )

    assert store.write_features([row]) == 1
    assert store.read_features(symbols=["AAA"])[0]["return_1d"] == 0.01
    with pytest.raises(ValueError, match="duplicate feature row"):
        store.write_features([row])

    with pytest.raises(PointInTimeViolation, match="after asof_ts"):
        store.validate_point_in_time(
            [
                {
                    "asset_id": "asset_bbb",
                    "symbol": "BBB",
                    "asof_ts": asof_ts,
                    "horizon": "1d",
                    "feature_set_version": "features-v1",
                    "input_available_at": [asof_ts + timedelta(seconds=1)],
                }
            ]
        )


def test_feature_store_validates_flattened_availability_fields(tmp_path: Path) -> None:
    store = LocalFeatureStore(tmp_path / "features")
    with pytest.raises(PointInTimeViolation):
        store.validate_point_in_time(
            [
                {
                    "asset_id": "asset_aaa",
                    "symbol": "AAA",
                    "asof_ts": "2026-07-06T20:00:00Z",
                    "horizon": "1d",
                    "feature_set_version": "features-v1",
                    "latest_text_available_at_1d": "2026-07-06T20:00:01Z",
                }
            ]
        )


def test_parquet_feature_store_partitions_and_filters_lazily(tmp_path: Path) -> None:
    store = ParquetFeatureStore(tmp_path / "gold")
    rows = [
        FeatureRow(
            asset_id=f"asset_{symbol.casefold()}",
            symbol=symbol,
            asof_ts=datetime(2026, 7, 6, 20, tzinfo=UTC),
            horizon="1d",
            feature_set_version="features-v1",
            features={"return_1d": value},
        )
        for symbol, value in (("AAA", 0.01), ("BBB", -0.02))
    ]

    assert store.write_features(rows) == 2
    assert len(list((tmp_path / "gold").glob("feature_set_version=*/year=*/part-*.parquet"))) == 1
    selected = store.read_features(symbols=["BBB"], feature_set_version="features-v1")
    assert len(selected) == 1
    assert selected[0]["return_1d"] == -0.02


def test_local_model_registry_keeps_versions_immutable(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "models")
    path = registry.save_model("model-v1", {"weights": [0.1, -0.2]}, metadata={"seed": 7})

    assert path.name == "model.json"
    assert registry.load_model("model-v1") == {"weights": [0.1, -0.2]}
    assert registry.save_model("model-v1", {"weights": [0.1, -0.2]}) == path
    with pytest.raises(FileExistsError, match="immutable artifact"):
        registry.save_model("model-v1", {"weights": [0.9]})
    metrics_path = registry.record_metrics("model-v1", {"rank_ic": 0.03})
    assert metrics_path.exists()
