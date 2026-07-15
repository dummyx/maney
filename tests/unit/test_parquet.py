from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nlp_trader.data.parquet import (
    read_partitioned_parquet,
    scan_partitioned_parquet,
    write_partitioned_parquet,
)


def test_partitioned_parquet_is_content_addressed_and_lazy(tmp_path: Path) -> None:
    records = [
        {"symbol": "AAA", "year": 2026, "close": 10.0, "unused": "x"},
        {"symbol": "BBB", "year": 2026, "close": 20.0, "unused": "y"},
    ]
    first = write_partitioned_parquet(records, tmp_path, partition_fields=("symbol", "year"))
    second = write_partitioned_parquet(records, tmp_path, partition_fields=("symbol", "year"))

    assert first == second
    assert len(list(tmp_path.rglob("*.parquet"))) == 2
    selected = (
        scan_partitioned_parquet(tmp_path, columns=["symbol", "close"])
        .filter(pl.col("symbol") == "BBB")
        .collect()
    )
    assert selected.to_dicts() == [{"symbol": "BBB", "close": 20.0}]


def test_nested_values_round_trip_as_structured_json(tmp_path: Path) -> None:
    records = [
        {
            "symbol": "AAA",
            "entities": [
                {"symbol": "AAA", "confidence": 0.9},
                {"symbol": "BBB", "confidence": 0.4},
            ],
        }
    ]

    write_partitioned_parquet(records, tmp_path, partition_fields=("symbol",))
    raw_entities = read_partitioned_parquet(tmp_path)[0]["entities"]

    assert isinstance(raw_entities, str)
    assert json.loads(raw_entities) == records[0]["entities"]


def test_partitions_share_schema_when_early_values_are_all_null(tmp_path: Path) -> None:
    records = [
        {"symbol": "AAA", "year": 2025, "optional_metric": None},
        {"symbol": "AAA", "year": 2026, "optional_metric": 1.25},
    ]

    write_partitioned_parquet(
        records,
        tmp_path,
        partition_fields=("symbol", "year"),
        max_rows_per_file=1,
    )
    rows = scan_partitioned_parquet(tmp_path).collect().sort("year").to_dicts()

    assert rows[0]["optional_metric"] is None
    assert rows[1]["optional_metric"] == 1.25


def test_append_calls_promote_prior_all_null_physical_schema(tmp_path: Path) -> None:
    write_partitioned_parquet(
        [{"family": "a", "optional_metric": None}],
        tmp_path,
        partition_fields=("family",),
    )
    write_partitioned_parquet(
        [{"family": "b", "optional_metric": 1.25}],
        tmp_path,
        partition_fields=("family",),
    )

    rows = scan_partitioned_parquet(tmp_path).collect().sort("family").to_dicts()

    assert rows == [
        {"family": "a", "optional_metric": None},
        {"family": "b", "optional_metric": 1.25},
    ]
