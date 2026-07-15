from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import polars as pl

from nlp_trader.calendars import USEquityCalendar
from nlp_trader.config import BacktestConfig
from nlp_trader.data.local import canonical_text_hash
from nlp_trader.schemas import (
    Asset,
    CorporateAction,
    EarningsCalendarEvent,
    EntityMention,
    FeatureRow,
    FundamentalRecord,
    MarketBar,
    TextItem,
)
from nlp_trader.timestamps import parse_optional_date, parse_utc

Record = dict[str, Any]


@runtime_checkable
class MarketDataProvider(Protocol):
    def fetch_assets(self, *, symbols: Sequence[str] | None = None) -> Sequence[Asset]: ...

    def fetch_bars(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        bar_size: str = "1d",
        limit: int | None = None,
    ) -> Sequence[MarketBar]: ...

    def fetch_corporate_actions(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[CorporateAction]: ...


@runtime_checkable
class TextDataProvider(Protocol):
    def fetch_items(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        sources: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> Sequence[TextItem]: ...

    def normalize_item(self, record: Mapping[str, Any]) -> TextItem: ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    def fetch_fundamentals(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[FundamentalRecord]: ...

    def fetch_earnings_calendar(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[EarningsCalendarEvent]: ...


@runtime_checkable
class CalendarProvider(Protocol):
    def sessions(self, start: date, end: date) -> tuple[date, ...]: ...

    def next_session(self, after: date, *, include_current: bool = False) -> date: ...

    def decision_times(self, start: date, end: date) -> tuple[datetime, ...]: ...

    def next_decision_time(self, available_at: datetime) -> datetime: ...


@runtime_checkable
class FeatureStore(Protocol):
    def write_features(self, rows: Iterable[FeatureRow | Mapping[str, Any]]) -> int: ...

    def read_features(
        self,
        *,
        feature_set_version: str | None = None,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]: ...

    def validate_point_in_time(self, rows: Iterable[FeatureRow | Mapping[str, Any]]) -> None: ...


@runtime_checkable
class ModelRegistry(Protocol):
    def save_model(
        self,
        model_version: str,
        model: bytes | Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path: ...

    def load_model(self, model_version: str) -> bytes | dict[str, Any]: ...

    def record_metrics(self, model_version: str, metrics: Mapping[str, Any]) -> Path: ...


@runtime_checkable
class BacktestEngine(Protocol):
    def run(
        self,
        predictions: list[dict[str, Any]],
        labels: list[dict[str, Any]],
        config: BacktestConfig,
    ) -> dict[str, Any]: ...

    def summarize(self, result: Mapping[str, Any]) -> dict[str, Any]: ...


def read_local_records(path: Path) -> list[Record]:
    """Read local records without initiating network access.

    CSV and JSON/JSONL use the standard library. Parquet is supported when a local
    PyArrow installation is available; the baseline package intentionally does not
    download or import it eagerly.
    """

    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [_mapping(json.loads(line), path) for line in handle if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [_mapping(record, path) for record in value]
        return [_mapping(value, path)]
    if suffix == ".parquet":
        try:
            parquet = import_module("pyarrow.parquet")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "reading Parquet requires a local pyarrow installation; no download was attempted"
            ) from error
        table = parquet.read_table(path)
        return [_mapping(record, path) for record in table.to_pylist()]
    raise ValueError(f"unsupported local data format: {path.suffix or '<none>'}")


def read_filtered_local_records(
    path: Path,
    *,
    equals: Mapping[str, Sequence[str] | str] | None = None,
    timestamp_field: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
    select_fields: Sequence[str] | None = None,
    distinct_sorted_field: str | None = None,
) -> list[Record]:
    """Push filters and limits into lazy local scans when the format supports it."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    suffix = path.suffix.casefold()
    lazy: pl.LazyFrame | None = None
    if path.is_dir():
        parquet_files = tuple(path.rglob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"local data directory contains no Parquet files: {path}")
        source = str(path / "**" / "*.parquet")
        lazy = pl.scan_parquet(source)
    elif suffix == ".parquet":
        source = str(path)
        lazy = pl.scan_parquet(source)
    elif suffix == ".csv":
        lazy = pl.scan_csv(path, infer_schema=False)
    elif suffix == ".jsonl":
        lazy = pl.scan_ndjson(path)
    if lazy is None:
        records = read_local_records(path)
        records = _filter_materialized(
            records,
            equals=equals,
            timestamp_field=timestamp_field,
            start=start,
            end=end,
            limit=None if distinct_sorted_field is not None else limit,
        )
        if distinct_sorted_field is not None:
            deduplicated = {str(record[distinct_sorted_field]): record for record in records}
            records = [deduplicated[key] for key in sorted(deduplicated)]
            if limit is not None:
                records = records[:limit]
        if select_fields is None:
            return records
        return [
            {field: record[field] for field in select_fields if field in record}
            for record in records
        ]

    schema = lazy.collect_schema()
    for field, expected in (equals or {}).items():
        if field not in schema:
            raise ValueError(f"missing filter field {field!r} in {path}")
        values = [expected] if isinstance(expected, str) else list(expected)
        lazy = lazy.filter(pl.col(field).cast(pl.String).is_in(values))
    if timestamp_field is not None and (
        start is not None or end is not None or distinct_sorted_field == timestamp_field
    ):
        if timestamp_field not in schema:
            raise ValueError(f"missing timestamp field {timestamp_field!r} in {path}")
        if schema[timestamp_field] == pl.String:
            lazy = lazy.with_columns(
                pl.col(timestamp_field)
                .str.to_datetime(time_zone="UTC", strict=True)
                .alias(timestamp_field)
            )
        column = pl.col(timestamp_field)
        if start is not None:
            lazy = lazy.filter(column >= pl.lit(start))
        if end is not None:
            lazy = lazy.filter(column <= pl.lit(end))
    if select_fields is not None:
        missing = [field for field in select_fields if field not in schema]
        if missing:
            raise ValueError(f"missing selected fields {missing} in {path}")
        lazy = lazy.select(*select_fields)
    if distinct_sorted_field is not None:
        if distinct_sorted_field not in schema:
            raise ValueError(f"missing distinct field {distinct_sorted_field!r} in {path}")
        lazy = lazy.unique(subset=[distinct_sorted_field]).sort(distinct_sorted_field)
    if limit is not None:
        lazy = lazy.head(limit)
    return [
        {str(key): item for key, item in row.items()}
        for row in lazy.collect(engine="streaming").to_dicts()
    ]


def _filter_materialized(
    records: list[Record],
    *,
    equals: Mapping[str, Sequence[str] | str] | None,
    timestamp_field: str | None,
    start: datetime | None,
    end: datetime | None,
    limit: int | None,
) -> list[Record]:
    selected: list[Record] = []
    for record in records:
        if any(
            str(record.get(field))
            not in ({expected} if isinstance(expected, str) else set(expected))
            for field, expected in (equals or {}).items()
        ):
            continue
        if timestamp_field is not None and (start is not None or end is not None):
            timestamp = _datetime(_required(record, timestamp_field), timestamp_field)
            if not _in_range(timestamp, start, end):
                continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _mapping(value: object, path: Path) -> Record:
    if not isinstance(value, Mapping):
        raise ValueError(f"expected object record in {path}")
    return {str(key): item for key, item in value.items()}


def _required(record: Mapping[str, Any], key: str) -> Any:
    value = record.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field: {key}")
    return value


def _optional_str(value: object) -> str | None:
    return None if value is None or value == "" else str(value)


def _stable_identifier_hash(value: object, *, casefold: bool = False) -> str:
    normalized = str(value).strip()
    if casefold:
        normalized = normalized.casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _hash_from_record(
    record: Mapping[str, Any],
    hash_field: str,
    raw_fields: tuple[str, ...],
    *,
    casefold: bool = False,
) -> str | None:
    existing = _optional_str(record.get(hash_field))
    if existing is not None:
        return existing
    for field in raw_fields:
        value = record.get(field)
        if value is not None and value != "":
            return _stable_identifier_hash(value, casefold=casefold)
    return None


def _optional_float(value: object) -> float | None:
    return None if value is None or value == "" else float(str(value))


def _datetime(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"{field_name} must be timezone-aware")
        return value
    return parse_utc(str(value))


def _optional_datetime(value: object, field_name: str) -> datetime | None:
    if value is None or value == "":
        return None
    return _datetime(value, field_name)


def _date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = parse_optional_date(str(value))
    if parsed is None:
        raise ValueError(f"missing required field: {field_name}")
    return parsed


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _in_range(value: datetime, start: datetime | None, end: datetime | None) -> bool:
    return (start is None or value >= start) and (end is None or value <= end)


class LocalMarketDataProvider:
    def __init__(
        self,
        assets_path: Path,
        bars_path: Path,
        corporate_actions_path: Path | None = None,
    ) -> None:
        self.assets_path = assets_path
        self.bars_path = bars_path
        self.corporate_actions_path = corporate_actions_path

    def fetch_assets(self, *, symbols: Sequence[str] | None = None) -> list[Asset]:
        records = read_filtered_local_records(
            self.assets_path,
            equals={"symbol": symbols} if symbols is not None else None,
        )
        return [self._asset(record) for record in records]

    def fetch_bars(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        bar_size: str = "1d",
        limit: int | None = None,
    ) -> list[MarketBar]:
        records = read_filtered_local_records(
            self.bars_path,
            equals={
                **({"symbol": symbols} if symbols is not None else {}),
                "bar_size": bar_size,
            },
            timestamp_field="ts",
            start=start,
            end=end,
            limit=limit,
        )
        return [self._bar(record) for record in records]

    def fetch_decision_times(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        bar_size: str = "1d",
        limit: int | None = None,
    ) -> tuple[datetime, ...]:
        """Read only bar timestamps to choose whole decision cross-sections."""

        records = read_filtered_local_records(
            self.bars_path,
            equals={
                **({"symbol": symbols} if symbols is not None else {}),
                "bar_size": bar_size,
            },
            timestamp_field="ts",
            start=start,
            end=end,
            limit=limit,
            select_fields=("ts",),
            distinct_sorted_field="ts",
        )
        values = sorted({_datetime(record["ts"], "ts") for record in records})
        return tuple(values)

    def fetch_corporate_actions(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[CorporateAction]:
        if self.corporate_actions_path is None:
            return []
        records = read_filtered_local_records(
            self.corporate_actions_path,
            equals={"symbol": symbols} if symbols is not None else None,
            timestamp_field="event_ts",
            start=start,
            end=end,
        )
        return [self._corporate_action(record) for record in records]

    @staticmethod
    def _asset(record: Mapping[str, Any]) -> Asset:
        return Asset(
            asset_id=str(_required(record, "asset_id")),
            symbol=str(_required(record, "symbol")),
            exchange=str(_required(record, "exchange")),
            currency=str(_required(record, "currency")),
            name=str(_required(record, "name")),
            sector=str(_required(record, "sector")),
            active_from=parse_optional_date(_optional_str(record.get("active_from"))),
            active_to=parse_optional_date(_optional_str(record.get("active_to"))),
            cik=_optional_str(record.get("cik")),
            figi=_optional_str(record.get("figi")),
            isin=_optional_str(record.get("isin")),
            industry=_optional_str(record.get("industry")),
        )

    @staticmethod
    def _bar(record: Mapping[str, Any]) -> MarketBar:
        return MarketBar(
            asset_id=str(_required(record, "asset_id")),
            symbol=str(_required(record, "symbol")),
            ts=_datetime(_required(record, "ts"), "ts"),
            bar_size=str(_required(record, "bar_size")),
            open=float(_required(record, "open")),
            high=float(_required(record, "high")),
            low=float(_required(record, "low")),
            close=float(_required(record, "close")),
            volume=int(_required(record, "volume")),
            vwap=_optional_float(record.get("vwap")),
            adjusted_close=_optional_float(record.get("adjusted_close")),
            corporate_action_adjusted=_bool(record.get("corporate_action_adjusted", False)),
            adjustment_vintage_at=_optional_datetime(
                record.get("adjustment_vintage_at"), "adjustment_vintage_at"
            ),
            return_adjustment_factor=float(_required(record, "return_adjustment_factor")),
        )

    @staticmethod
    def _corporate_action(record: Mapping[str, Any]) -> CorporateAction:
        return CorporateAction(
            asset_id=str(_required(record, "asset_id")),
            symbol=str(_required(record, "symbol")),
            event_ts=_datetime(_required(record, "event_ts"), "event_ts"),
            available_at=_datetime(_required(record, "available_at"), "available_at"),
            action_type=str(_required(record, "action_type")),
            value=_optional_float(record.get("value")),
        )


class LocalTextDataProvider:
    def __init__(self, items_path: Path) -> None:
        self.items_path = items_path

    def fetch_items(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        sources: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[TextItem]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        records = read_filtered_local_records(
            self.items_path,
            equals={"source": sources} if sources is not None else None,
            timestamp_field="available_at",
            start=start,
            end=end,
            limit=limit,
        )
        return [self.normalize_item(record) for record in records]

    def normalize_item(self, record: Mapping[str, Any]) -> TextItem:
        title = _optional_str(record.get("title"))
        body = _optional_str(record.get("body"))
        raw_entities = record.get("entities", [])
        if isinstance(raw_entities, str):
            raw_entities = json.loads(raw_entities)
        if not isinstance(raw_entities, list):
            raise ValueError("entities must be a list")
        entities = tuple(self._entity(_mapping(entity, self.items_path)) for entity in raw_entities)
        return TextItem(
            item_id=str(_required(record, "item_id")),
            source=str(_required(record, "source")),
            source_type=str(_required(record, "source_type")),
            language=str(_required(record, "language")),
            title=title,
            body=body,
            published_at=_datetime(_required(record, "published_at"), "published_at"),
            vendor_received_at=_optional_datetime(
                record.get("vendor_received_at"), "vendor_received_at"
            ),
            ingested_at=_datetime(_required(record, "ingested_at"), "ingested_at"),
            available_at=_datetime(_required(record, "available_at"), "available_at"),
            license_or_terms_ref=str(_required(record, "license_or_terms_ref")),
            raw_text_hash=_optional_str(record.get("raw_text_hash")),
            canonical_text_hash=_optional_str(record.get("canonical_text_hash"))
            or canonical_text_hash(title, body),
            author_hash=_hash_from_record(
                record,
                "author_hash",
                ("author_id", "author_handle", "author"),
                casefold=True,
            ),
            url_hash=_hash_from_record(record, "url_hash", ("url",)),
            entities=entities,
            raw_text_path=_optional_str(record.get("raw_text_path")),
            event_ts=_optional_datetime(record.get("event_ts"), "event_ts"),
            processed_at=_optional_datetime(record.get("processed_at"), "processed_at"),
            event_type=_optional_str(record.get("event_type")),
            relationship_type=str(record.get("relationship_type", "original")),
            parent_item_id_hash=_hash_from_record(
                record,
                "parent_item_id_hash",
                ("parent_item_id",),
            ),
            content_status=str(record.get("content_status", "unknown")),
            retention_permitted=_bool(record.get("retention_permitted", True)),
        )

    @staticmethod
    def _entity(record: Mapping[str, Any]) -> EntityMention:
        return EntityMention(
            asset_id=_optional_str(record.get("asset_id")),
            symbol=_optional_str(record.get("symbol")),
            name=str(_required(record, "name")),
            relevance=float(_required(record, "relevance")),
            mention_type=str(_required(record, "mention_type")),
            confidence=float(_required(record, "confidence")),
        )


class LocalFundamentalsProvider:
    def __init__(
        self,
        fundamentals_path: Path | None = None,
        earnings_calendar_path: Path | None = None,
    ) -> None:
        self.fundamentals_path = fundamentals_path
        self.earnings_calendar_path = earnings_calendar_path

    def fetch_fundamentals(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[FundamentalRecord]:
        if self.fundamentals_path is None:
            return []
        records = read_filtered_local_records(
            self.fundamentals_path,
            equals={"symbol": symbols} if symbols is not None else None,
            timestamp_field="available_at",
            start=start,
            end=end,
        )
        return [self._fundamental(record) for record in records]

    def fetch_earnings_calendar(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[EarningsCalendarEvent]:
        if self.earnings_calendar_path is None:
            return []
        records = read_filtered_local_records(
            self.earnings_calendar_path,
            equals={"symbol": symbols} if symbols is not None else None,
            timestamp_field="event_ts",
            start=start,
            end=end,
        )
        return [self._earnings_event(record) for record in records]

    @staticmethod
    def _fundamental(record: Mapping[str, Any]) -> FundamentalRecord:
        raw_values = record.get("values", {})
        if isinstance(raw_values, str):
            raw_values = json.loads(raw_values)
        values = _mapping(raw_values, Path("<fundamental-record>"))
        return FundamentalRecord(
            asset_id=str(_required(record, "asset_id")),
            symbol=str(_required(record, "symbol")),
            period_end=_date(_required(record, "period_end"), "period_end"),
            available_at=_datetime(_required(record, "available_at"), "available_at"),
            values=values,
            filing_id=_optional_str(record.get("filing_id")),
        )

    @staticmethod
    def _earnings_event(record: Mapping[str, Any]) -> EarningsCalendarEvent:
        status = str(record.get("status", "estimated"))
        if status not in {"estimated", "confirmed", "reported"}:
            raise ValueError(f"invalid earnings status: {status}")
        return EarningsCalendarEvent(
            asset_id=str(_required(record, "asset_id")),
            symbol=str(_required(record, "symbol")),
            event_ts=_datetime(_required(record, "event_ts"), "event_ts"),
            available_at=_datetime(_required(record, "available_at"), "available_at"),
            status=cast(Literal["estimated", "confirmed", "reported"], status),
        )


class LocalCalendarProvider(USEquityCalendar):
    """Local deterministic calendar provider with no external data access."""
