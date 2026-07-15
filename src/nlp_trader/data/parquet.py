from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from nlp_trader.timestamps import format_utc

_SAFE_PARTITION = re.compile(r"[^A-Za-z0-9._-]+")


def _nested_value(value: object) -> Any:
    if isinstance(value, datetime):
        return format_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _nested_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_nested_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported Parquet value: {type(value).__name__}")


def _json_value(value: object) -> Any:
    nested = _nested_value(value)
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return json.dumps(nested, sort_keys=True, separators=(",", ":"))
    return nested


def _record(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in value.items()}


def _component(value: object) -> str:
    normalized = _SAFE_PARTITION.sub("_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"empty partition component derived from {value!r}")
    return normalized


def _stable_schema(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Infer one cross-partition schema without retaining a normalized copy of every row."""

    field_order: list[str] = []
    kinds: dict[str, set[str]] = {}
    for source in records:
        for field, value in _record(source).items():
            if field not in kinds:
                field_order.append(field)
                kinds[field] = set()
            if value is None:
                continue
            if isinstance(value, bool):
                kinds[field].add("bool")
            elif isinstance(value, int):
                kinds[field].add("int")
            elif isinstance(value, float):
                kinds[field].add("float")
            elif isinstance(value, str):
                kinds[field].add("str")
            else:  # pragma: no cover - _record rejects unsupported values first
                raise TypeError(f"unsupported normalized Parquet value: {type(value).__name__}")

    if not field_order:
        return None
    schema: dict[str, Any] = {}
    for field in field_order:
        observed = kinds[field]
        if not observed:
            schema[field] = pl.Null
        elif observed == {"bool"}:
            schema[field] = pl.Boolean
        elif observed == {"int"}:
            schema[field] = pl.Int64
        elif observed <= {"int", "float"}:
            schema[field] = pl.Float64
        elif observed == {"str"}:
            schema[field] = pl.String
        else:
            raise TypeError(
                f"incompatible Parquet value types for {field}: {', '.join(sorted(observed))}"
            )
    return schema


def write_partitioned_parquet(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    partition_fields: tuple[str, ...] = (),
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
    max_rows_per_file: int = 10_000,
) -> list[Path]:
    """Write bounded immutable Parquet batches grouped by Hive partitions."""

    if max_rows_per_file < 1:
        raise ValueError("max_rows_per_file must be positive")
    if not isinstance(records, Sequence):
        raise TypeError(
            "records must be a reusable sequence so one stable Parquet schema can be "
            "inferred across every partition"
        )
    schema = _stable_schema(records)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    paths: list[Path] = []

    def flush() -> None:
        for key, group in sorted(groups.items()):
            canonical = json.dumps(group, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
            directory = root
            for field, value in zip(partition_fields, key, strict=True):
                directory /= f"{field}={value}"
            path = directory / f"part-{digest[:20]}.parquet"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(
                    group,
                    schema=schema,
                    strict=False,
                    infer_schema_length=None,
                ).write_parquet(
                    path,
                    compression=compression,
                    statistics=True,
                )
            paths.append(path)
        groups.clear()

    pending = 0
    for source in records:
        missing = [field for field in partition_fields if field not in source]
        if missing:
            raise ValueError(f"record missing partition fields: {', '.join(missing)}")
        key = tuple(_component(source[field]) for field in partition_fields)
        groups[key].append(_record(source))
        pending += 1
        if pending >= max_rows_per_file:
            flush()
            pending = 0
    if groups:
        flush()
    return sorted(set(paths))


def scan_partitioned_parquet(
    root: Path,
    *,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Return a lazy scan so callers can filter/project before collection."""

    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet artifacts under {root}")
    schema: dict[str, Any] = {}
    for path in files:
        for field, dtype in pl.read_parquet_schema(path).items():
            previous = schema.get(field)
            if previous is None or previous == pl.Null:
                schema[field] = dtype
            elif dtype == pl.Null or dtype == previous:
                continue
            elif {previous, dtype} == {pl.Int64, pl.Float64}:
                schema[field] = pl.Float64
            else:
                raise TypeError(
                    f"incompatible stored Parquet schemas for {field}: {previous} and {dtype}"
                )
    lazy = pl.scan_parquet(
        [str(path) for path in files],
        schema=schema,
        missing_columns="insert",
    )
    return lazy.select(list(columns)) if columns is not None else lazy


def read_partitioned_parquet(root: Path) -> list[dict[str, Any]]:
    return [
        {str(key): value for key, value in row.items()}
        for row in scan_partitioned_parquet(root).collect(engine="streaming").to_dicts()
    ]
