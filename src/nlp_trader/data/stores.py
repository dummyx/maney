from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from nlp_trader.data.parquet import scan_partitioned_parquet
from nlp_trader.schemas import FeatureRow, RawArtifact, RawArtifactMetadata
from nlp_trader.timestamps import format_utc, parse_utc

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FEATURE_KEY_FIELDS = (
    "asset_id",
    "asof_ts",
    "horizon",
    "feature_set_version",
)


class PointInTimeViolation(ValueError):
    """Raised when feature provenance is later than its decision timestamp."""


@dataclass(frozen=True, slots=True)
class RawIngestionRequest:
    source: str
    vendor: str
    license_or_terms_ref: str
    ingested_at: datetime
    request_id: str
    schema_version: str
    fetch_params: Mapping[str, Any]


def fetch_params_hash(fetch_params: Mapping[str, Any]) -> str:
    payload = _canonical_json(fetch_params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ContentAddressedRawStore:
    """Append-only immutable payload store with content-addressed metadata sidecars."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def ingest_bytes(
        self,
        payload: bytes,
        request: RawIngestionRequest,
        *,
        suffix: str = ".bin",
    ) -> RawArtifact:
        self._validate_request(request)
        normalized_suffix = self._suffix(suffix)
        digest = hashlib.sha256(payload).hexdigest()
        source = _safe_component(request.source, "source")
        payload_path = (
            self.root
            / f"source={source}"
            / f"date={request.ingested_at.astimezone(UTC).date().isoformat()}"
            / digest[:2]
            / f"{digest}{normalized_suffix}"
        )
        metadata = RawArtifactMetadata(
            source=request.source,
            vendor=request.vendor,
            license_or_terms_ref=request.license_or_terms_ref,
            ingested_at=request.ingested_at.astimezone(UTC),
            request_id=request.request_id,
            sha256=digest,
            schema_version=request.schema_version,
            fetch_params_hash=fetch_params_hash(request.fetch_params),
        )
        metadata_record = {
            "source": metadata.source,
            "vendor": metadata.vendor,
            "license_or_terms_ref": metadata.license_or_terms_ref,
            "ingested_at": format_utc(metadata.ingested_at),
            "request_id": metadata.request_id,
            "sha256": metadata.sha256,
            "schema_version": metadata.schema_version,
            "fetch_params_hash": metadata.fetch_params_hash,
            "byte_count": len(payload),
            "payload_path": str(payload_path.relative_to(self.root)),
        }
        metadata_bytes = (_canonical_json(metadata_record) + "\n").encode("utf-8")
        metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
        metadata_path = (
            self.root / "_metadata" / f"source={source}" / digest / f"{metadata_digest}.json"
        )
        _write_once(payload_path, payload)
        _write_once(metadata_path, metadata_bytes)
        return RawArtifact(
            payload_path=payload_path,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    def ingest_file(self, path: Path, request: RawIngestionRequest) -> RawArtifact:
        """Ingest a local file in bounded memory while preserving exact bytes."""

        self._validate_request(request)
        normalized_suffix = self._suffix(path.suffix or ".bin")
        digest = hashlib.sha256()
        byte_count = 0
        staging_dir = self.root / "_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, staged_name = tempfile.mkstemp(prefix="ingest-", dir=staging_dir)
        staged_path = Path(staged_name)
        try:
            with path.open("rb") as source, os.fdopen(file_descriptor, "wb") as destination:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    byte_count += len(chunk)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException:
            with suppress(OSError):
                os.close(file_descriptor)
            staged_path.unlink(missing_ok=True)
            raise
        sha256 = digest.hexdigest()
        source_name = _safe_component(request.source, "source")
        payload_path = (
            self.root
            / f"source={source_name}"
            / f"date={request.ingested_at.astimezone(UTC).date().isoformat()}"
            / sha256[:2]
            / f"{sha256}{normalized_suffix}"
        )
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            try:
                os.link(staged_path, payload_path)
            except FileExistsError as error:
                if _sha256_path(payload_path) != sha256:
                    raise FileExistsError(
                        f"immutable raw artifact is corrupted: {payload_path}"
                    ) from error
        finally:
            staged_path.unlink(missing_ok=True)

        metadata = RawArtifactMetadata(
            source=request.source,
            vendor=request.vendor,
            license_or_terms_ref=request.license_or_terms_ref,
            ingested_at=request.ingested_at.astimezone(UTC),
            request_id=request.request_id,
            sha256=sha256,
            schema_version=request.schema_version,
            fetch_params_hash=fetch_params_hash(request.fetch_params),
        )
        metadata_record = {
            "source": metadata.source,
            "vendor": metadata.vendor,
            "license_or_terms_ref": metadata.license_or_terms_ref,
            "ingested_at": format_utc(metadata.ingested_at),
            "request_id": metadata.request_id,
            "sha256": metadata.sha256,
            "schema_version": metadata.schema_version,
            "fetch_params_hash": metadata.fetch_params_hash,
            "byte_count": byte_count,
            "payload_path": str(payload_path.relative_to(self.root)),
        }
        metadata_bytes = (_canonical_json(metadata_record) + "\n").encode("utf-8")
        metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
        metadata_path = (
            self.root / "_metadata" / f"source={source_name}" / sha256 / f"{metadata_digest}.json"
        )
        _write_once(metadata_path, metadata_bytes)
        return RawArtifact(payload_path, metadata_path, metadata)

    @staticmethod
    def _suffix(value: str) -> str:
        suffix = value if value.startswith(".") else f".{value}"
        if not suffix[1:].isalnum():
            raise ValueError(f"unsafe payload suffix: {value!r}")
        return suffix.casefold()

    @staticmethod
    def _validate_request(request: RawIngestionRequest) -> None:
        if request.ingested_at.tzinfo is None:
            raise ValueError("ingested_at must be timezone-aware")
        for name in ("source", "vendor", "license_or_terms_ref", "request_id", "schema_version"):
            if not getattr(request, name).strip():
                raise ValueError(f"{name} must not be empty")


class PointInTimeFeatureStore:
    """Small append-only JSONL feature store with leakage and uniqueness checks."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "features.jsonl"

    def write_features(self, rows: Iterable[FeatureRow | Mapping[str, Any]]) -> int:
        records = [_feature_record(row) for row in rows]
        if not records:
            return 0
        existing = self.read_features()
        self.validate_point_in_time([*existing, *records])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(_canonical_json(record) + "\n")
        return len(records)

    def read_features(
        self,
        *,
        feature_set_version: str | None = None,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        allowed = set(symbols) if symbols is not None else None
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object feature record in {self.path}")
                record = {str(key): item for key, item in value.items()}
                asof_ts = _as_datetime(record.get("asof_ts"), "asof_ts")
                if feature_set_version is not None and (
                    str(record.get("feature_set_version")) != feature_set_version
                ):
                    continue
                if allowed is not None and str(record.get("symbol")) not in allowed:
                    continue
                if start is not None and asof_ts < start:
                    continue
                if end is not None and asof_ts > end:
                    continue
                records.append(record)
        return records

    def validate_point_in_time(self, rows: Iterable[FeatureRow | Mapping[str, Any]]) -> None:
        seen: set[tuple[str, str, str, str]] = set()
        for row in rows:
            record = _feature_record(row)
            missing = [field for field in _FEATURE_KEY_FIELDS if record.get(field) in (None, "")]
            if missing:
                raise ValueError(f"feature row missing key fields: {', '.join(missing)}")
            asof_ts = _as_datetime(record["asof_ts"], "asof_ts")
            key = (
                str(record["asset_id"]),
                format_utc(asof_ts),
                str(record["horizon"]),
                str(record["feature_set_version"]),
            )
            if key in seen:
                raise ValueError(f"duplicate feature row: {key}")
            seen.add(key)
            for name, value in record.items():
                if "available_at" not in name or value is None:
                    continue
                for available_at in _availability_values(value, name):
                    if available_at > asof_ts:
                        raise PointInTimeViolation(
                            f"feature input {name}={format_utc(available_at)} is after "
                            f"asof_ts={format_utc(asof_ts)} for {key}"
                        )


LocalFeatureStore = PointInTimeFeatureStore


class ParquetFeatureStore:
    """Append-only partitioned Parquet feature store with lazy filtered reads."""

    def __init__(
        self,
        root: Path,
        *,
        compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
    ) -> None:
        self.root = root
        self.compression = compression
        self._validator = PointInTimeFeatureStore(root / "_validation")

    def _files(self) -> list[Path]:
        return sorted(self.root.glob("feature_set_version=*/year=*/part-*.parquet"))

    @staticmethod
    def _key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(record["asset_id"]),
            str(record["asof_ts"]),
            str(record["horizon"]),
            str(record["feature_set_version"]),
        )

    def write_features(self, rows: Iterable[FeatureRow | Mapping[str, Any]]) -> int:
        records = [_feature_record(row) for row in rows]
        if not records:
            return 0
        self.validate_point_in_time(records)
        existing_keys = {self._key(record) for record in self.read_features()}
        duplicate = next(
            (self._key(record) for record in records if self._key(record) in existing_keys),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"duplicate feature row already stored: {duplicate}")

        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        schema = pl.DataFrame(records, infer_schema_length=None).schema
        for record in records:
            asof = _as_datetime(record["asof_ts"], "asof_ts")
            groups[(str(record["feature_set_version"]), str(asof.year))].append(record)
        written = 0
        for (version, year), group in sorted(groups.items()):
            canonical = "\n".join(_canonical_json(record) for record in group).encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
            path = (
                self.root
                / f"feature_set_version={_safe_component(version, 'feature_set_version')}"
                / f"year={year}"
                / f"part-{digest[:20]}.parquet"
            )
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pl.DataFrame(
                group,
                schema=schema,
                strict=False,
                infer_schema_length=None,
            )
            frame.write_parquet(path, compression=self.compression, statistics=True)
            written += len(group)
        return written

    def read_features(
        self,
        *,
        feature_set_version: str | None = None,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        files = self._files()
        if not files:
            return []
        lazy = scan_partitioned_parquet(self.root)
        if feature_set_version is not None:
            lazy = lazy.filter(pl.col("feature_set_version") == feature_set_version)
        if symbols is not None:
            lazy = lazy.filter(pl.col("symbol").is_in(list(symbols)))
        if start is not None:
            lazy = lazy.filter(pl.col("asof_ts") >= format_utc(start))
        if end is not None:
            lazy = lazy.filter(pl.col("asof_ts") <= format_utc(end))
        return [
            {str(key): _json_value(value) for key, value in record.items()}
            for record in lazy.collect(engine="streaming").to_dicts()
        ]

    def validate_point_in_time(self, rows: Iterable[FeatureRow | Mapping[str, Any]]) -> None:
        self._validator.validate_point_in_time(rows)


class LocalModelRegistry:
    """Immutable versioned local model artifacts and content-addressed metrics."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save_model(
        self,
        model_version: str,
        model: bytes | Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        version = _safe_component(model_version, "model_version")
        directory = self.root / f"model_version={version}"
        if isinstance(model, bytes):
            path = directory / "model.bin"
            payload = model
        else:
            path = directory / "model.json"
            payload = (_canonical_json(model) + "\n").encode("utf-8")
        _write_once(path, payload)
        if metadata is not None:
            _write_once(
                directory / "metadata.json",
                (_canonical_json(metadata) + "\n").encode("utf-8"),
            )
        return path

    def load_model(self, model_version: str) -> bytes | dict[str, Any]:
        version = _safe_component(model_version, "model_version")
        directory = self.root / f"model_version={version}"
        json_path = directory / "model.json"
        binary_path = directory / "model.bin"
        if json_path.exists():
            value = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"model JSON must contain an object: {json_path}")
            return {str(key): item for key, item in value.items()}
        if binary_path.exists():
            return binary_path.read_bytes()
        raise FileNotFoundError(f"unknown model version: {model_version}")

    def record_metrics(self, model_version: str, metrics: Mapping[str, Any]) -> Path:
        version = _safe_component(model_version, "model_version")
        payload = (_canonical_json(metrics) + "\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        path = self.root / f"model_version={version}" / "metrics" / f"{digest}.json"
        _write_once(path, payload)
        return path


def _safe_component(value: str, name: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{name} contains unsafe path characters: {value!r}")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}") from None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_record(row: FeatureRow | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, FeatureRow):
        return {str(key): _json_value(value) for key, value in row.items()}
    reserved = {"asset_id", "symbol", "asof_ts", "horizon", "feature_set_version"}
    collisions = reserved.intersection(row.features)
    if collisions:
        raise ValueError(f"feature names collide with key fields: {sorted(collisions)}")
    return {
        **{str(key): _json_value(value) for key, value in row.features.items()},
        "asset_id": row.asset_id,
        "symbol": row.symbol,
        "asof_ts": format_utc(row.asof_ts),
        "horizon": row.horizon,
        "feature_set_version": row.feature_set_version,
        "input_available_at": [format_utc(value) for value in row.input_available_at],
    }


def _availability_values(value: object, name: str) -> tuple[datetime, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_as_datetime(item, name) for item in value)
    return (_as_datetime(value, name),)


def _as_datetime(value: object, name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")
        return value.astimezone(UTC)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return parse_utc(value)


def _canonical_json(value: object) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


def _json_value(value: object) -> Any:
    if isinstance(value, datetime):
        return format_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")
