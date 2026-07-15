from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from typing import Literal

type FeatureValue = str | int | float | bool | None

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_TYPES = {"news", "social", "filing", "transcript", "blog", "forum", "other"}
_MENTION_TYPES = {"primary", "secondary", "incidental"}
_SENTIMENT_LABELS = {"positive", "negative", "neutral"}
_RELATIONSHIP_TYPES = {"original", "repost", "quote", "reply"}
_CONTENT_STATUSES = {"active", "deleted", "private", "protected", "unknown"}


def _nonempty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _range(value: float, name: str, lower: float = 0.0, upper: float = 1.0) -> None:
    if not isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _optional_hash(value: str | None, name: str) -> None:
    if value is not None and not _HEX_64.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class Asset:
    asset_id: str
    symbol: str
    exchange: str
    currency: str
    name: str
    sector: str
    active_from: date | None
    active_to: date | None
    cik: str | None = None
    figi: str | None = None
    isin: str | None = None
    industry: str | None = None
    short_available: bool = False
    hard_to_borrow: bool | None = None
    trading_unit: int | None = None

    def __post_init__(self) -> None:
        for name in ("asset_id", "symbol", "exchange", "currency", "name", "sector"):
            _nonempty(getattr(self, name), name)
        if type(self.short_available) is not bool:
            raise ValueError("short_available must be a boolean")
        if self.hard_to_borrow is not None and type(self.hard_to_borrow) is not bool:
            raise ValueError("hard_to_borrow must be a boolean when supplied")
        if self.hard_to_borrow is None:
            object.__setattr__(self, "hard_to_borrow", self.short_available)
        if self.trading_unit is not None and (
            type(self.trading_unit) is not int or self.trading_unit < 1
        ):
            raise ValueError("trading_unit must be a positive integer when supplied")
        if (
            self.active_from is not None
            and self.active_to is not None
            and self.active_to < self.active_from
        ):
            raise ValueError("active_to must be on or after active_from")


@dataclass(frozen=True, slots=True)
class MarketBar:
    asset_id: str
    symbol: str
    ts: datetime
    bar_size: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None
    adjusted_close: float | None
    corporate_action_adjusted: bool
    adjustment_vintage_at: datetime | None = None
    return_adjustment_factor: float = 1.0
    exchange: str | None = None
    currency: str | None = None
    trading_unit: int | None = None
    session_date: date | None = None
    available_at: datetime | None = None
    price_basis: Literal["raw_tradable"] | None = None

    def __post_init__(self) -> None:
        for name in ("asset_id", "symbol", "bar_size"):
            _nonempty(getattr(self, name), name)
        _aware(self.ts, "ts")
        for name in ("exchange", "currency"):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name)
        if self.trading_unit is not None and (
            type(self.trading_unit) is not int or self.trading_unit < 1
        ):
            raise ValueError("trading_unit must be a positive integer when supplied")
        if self.available_at is not None:
            _aware(self.available_at, "available_at")
        if self.price_basis is not None and self.price_basis != "raw_tradable":
            raise ValueError("price_basis must be raw_tradable when supplied")
        if self.adjustment_vintage_at is not None:
            _aware(self.adjustment_vintage_at, "adjustment_vintage_at")
        if self.corporate_action_adjusted:
            if self.adjustment_vintage_at is None:
                raise ValueError("corporate-action-adjusted OHLC requires adjustment_vintage_at")
            if self.adjustment_vintage_at > self.ts and self.available_at is None:
                raise ValueError("adjustment_vintage_at must not be after the bar timestamp")
            if self.available_at is not None and self.adjustment_vintage_at > self.available_at:
                raise ValueError("adjustment_vintage_at must not be after available_at")
        _finite(self.return_adjustment_factor, "return_adjustment_factor")
        if self.return_adjustment_factor <= 0:
            raise ValueError("return_adjustment_factor must be positive")
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            _finite(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("OHLC values are inconsistent")
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")
        for name in ("vwap", "adjusted_close"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
                if value <= 0:
                    raise ValueError(f"{name} must be positive when provided")


@dataclass(frozen=True, slots=True)
class EntityMention:
    name: str
    relevance: float
    mention_type: str
    confidence: float
    asset_id: str | None = None
    symbol: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.name, "name")
        if self.mention_type not in _MENTION_TYPES:
            raise ValueError(f"unsupported mention_type: {self.mention_type}")
        _range(self.relevance, "relevance")
        _range(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class TextItem:
    item_id: str
    source: str
    source_type: str
    language: str
    title: str | None
    body: str | None
    published_at: datetime
    ingested_at: datetime
    available_at: datetime
    license_or_terms_ref: str
    vendor_received_at: datetime | None = None
    raw_text_hash: str | None = None
    canonical_text_hash: str | None = None
    author_hash: str | None = None
    url_hash: str | None = None
    entities: tuple[EntityMention, ...] = field(default_factory=tuple)
    raw_text_path: str | None = None
    event_ts: datetime | None = None
    processed_at: datetime | None = None
    event_type: str | None = None
    relationship_type: str = "original"
    parent_item_id_hash: str | None = None
    content_status: str = "unknown"
    retention_permitted: bool = True

    def __post_init__(self) -> None:
        for name in ("item_id", "source", "language", "license_or_terms_ref"):
            _nonempty(getattr(self, name), name)
        if self.source_type not in _SOURCE_TYPES:
            raise ValueError(f"unsupported source_type: {self.source_type}")
        if self.relationship_type not in _RELATIONSHIP_TYPES:
            raise ValueError(f"unsupported relationship_type: {self.relationship_type}")
        if self.content_status not in _CONTENT_STATUSES:
            raise ValueError(f"unsupported content_status: {self.content_status}")
        if not self.retention_permitted:
            raise ValueError("source terms do not permit retention of this text item")
        for name in ("published_at", "ingested_at", "available_at"):
            _aware(getattr(self, name), name)
        for name in ("vendor_received_at", "event_ts", "processed_at"):
            value = getattr(self, name)
            if value is not None:
                _aware(value, name)
        if self.available_at < self.published_at:
            raise ValueError("available_at must not be before published_at")
        if self.vendor_received_at is not None:
            if self.vendor_received_at < self.published_at:
                raise ValueError("vendor_received_at must not be before published_at")
            if self.available_at < self.vendor_received_at:
                raise ValueError("available_at must not be before vendor_received_at")
        if self.processed_at is not None and self.processed_at < self.ingested_at:
            raise ValueError("processed_at must not be before ingested_at")
        for name in ("raw_text_hash", "canonical_text_hash", "author_hash", "url_hash"):
            _optional_hash(getattr(self, name), name)
        _optional_hash(self.parent_item_id_hash, "parent_item_id_hash")


@dataclass(frozen=True, slots=True)
class TextSignal:
    item_id: str
    asset_id: str
    symbol: str
    asof_ts: datetime
    sentiment_score: float
    sentiment_label: str
    sentiment_confidence: float
    relevance: float
    novelty: float
    source_credibility: float
    model_version: str
    source: str | None = None
    source_type: str | None = None
    author_hash: str | None = None
    duplicate_cluster_id: str | None = None
    available_at: datetime | None = None
    event_type: str | None = None
    spam_score: float | None = None
    disagreement: float | None = None
    llm_semantic_signal: int | None = None
    llm_raw_confidence: float | None = None
    llm_uncertainty: float | None = None
    llm_event_type: str | None = None
    llm_event_confidence: float | None = None
    llm_supporting_evidence_count: int | None = None
    llm_counterevidence_count: int | None = None
    llm_abstained: bool | None = None

    def __post_init__(self) -> None:
        for name in ("item_id", "asset_id", "symbol", "model_version"):
            _nonempty(getattr(self, name), name)
        _aware(self.asof_ts, "asof_ts")
        if self.available_at is not None:
            _aware(self.available_at, "available_at")
        _range(self.sentiment_score, "sentiment_score", -1.0, 1.0)
        if self.sentiment_label not in _SENTIMENT_LABELS:
            raise ValueError(f"unsupported sentiment_label: {self.sentiment_label}")
        for name in ("sentiment_confidence", "relevance", "novelty", "source_credibility"):
            _range(getattr(self, name), name)
        for name in ("spam_score", "disagreement"):
            value = getattr(self, name)
            if value is not None:
                _range(value, name)
        if self.llm_semantic_signal is not None and (
            type(self.llm_semantic_signal) is not int or not -2 <= self.llm_semantic_signal <= 2
        ):
            raise ValueError("llm_semantic_signal must be an integer between -2 and 2")
        for name in ("llm_raw_confidence", "llm_uncertainty", "llm_event_confidence"):
            value = getattr(self, name)
            if value is not None:
                _range(value, name)
        if self.llm_event_type is not None:
            _nonempty(self.llm_event_type, "llm_event_type")
        for name in ("llm_supporting_evidence_count", "llm_counterevidence_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer when supplied")
        if self.llm_abstained is not None and type(self.llm_abstained) is not bool:
            raise ValueError("llm_abstained must be a boolean when supplied")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """Point-in-time feature record with explicit input availability provenance."""

    asset_id: str
    symbol: str
    asof_ts: datetime
    horizon: str
    feature_set_version: str
    features: Mapping[str, FeatureValue] = field(default_factory=dict)
    input_available_at: tuple[datetime, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("asset_id", "symbol", "horizon", "feature_set_version"):
            _nonempty(getattr(self, name), name)
        _aware(self.asof_ts, "asof_ts")
        for value in self.input_available_at:
            _aware(value, "input_available_at")
            if value > self.asof_ts:
                raise ValueError("input_available_at must not be after asof_ts")


@dataclass(frozen=True, slots=True)
class LabelRow:
    """Forward-looking target generated separately from feature construction."""

    asset_id: str
    symbol: str
    asof_ts: datetime
    horizon: str
    label_version: str
    forward_return: float | None = None
    forward_abnormal_return: float | None = None
    forward_sector_neutral_return: float | None = None

    def __post_init__(self) -> None:
        for name in ("asset_id", "symbol", "horizon", "label_version"):
            _nonempty(getattr(self, name), name)
        _aware(self.asof_ts, "asof_ts")
        for name in (
            "forward_return",
            "forward_abnormal_return",
            "forward_sector_neutral_return",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)


@dataclass(frozen=True, slots=True)
class PredictionRow:
    asset_id: str
    symbol: str
    asof_ts: datetime
    horizon: str
    model_version: str
    score: float
    expected_return: float | None = None
    probability_up: float | None = None
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        for name in ("asset_id", "symbol", "horizon", "model_version"):
            _nonempty(getattr(self, name), name)
        _aware(self.asof_ts, "asof_ts")
        _finite(self.score, "score")
        for name in ("expected_return", "probability_up", "uncertainty"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.probability_up is not None:
            _range(self.probability_up, "probability_up")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("uncertainty must be non-negative")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """Research or paper-trading target; it is not a routed broker order."""

    strategy_id: str
    asof_ts: datetime
    symbol: str
    target_weight: float
    side: Literal["buy", "sell", "hold"]
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    risk_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("strategy_id", "symbol"):
            _nonempty(getattr(self, name), name)
        _aware(self.asof_ts, "asof_ts")
        _finite(self.target_weight, "target_weight")
        _range(self.target_weight, "target_weight", -1.0, 1.0)
        if self.side == "buy" and self.target_weight < 0:
            raise ValueError("buy side cannot have a negative target_weight")
        if self.side == "sell" and self.target_weight > 0:
            raise ValueError("sell side cannot have a positive target_weight")


@dataclass(frozen=True, slots=True)
class RawArtifactMetadata:
    """Required provenance persisted beside an immutable raw payload."""

    source: str
    vendor: str
    license_or_terms_ref: str
    ingested_at: datetime
    request_id: str
    sha256: str
    schema_version: str
    fetch_params_hash: str

    def __post_init__(self) -> None:
        for name in ("source", "vendor", "license_or_terms_ref", "request_id", "schema_version"):
            _nonempty(getattr(self, name), name)
        _aware(self.ingested_at, "ingested_at")
        _optional_hash(self.sha256, "sha256")
        _optional_hash(self.fetch_params_hash, "fetch_params_hash")


@dataclass(frozen=True, slots=True)
class RawArtifact:
    payload_path: Path
    metadata_path: Path
    metadata: RawArtifactMetadata


@dataclass(frozen=True, slots=True)
class FundamentalRecord:
    asset_id: str
    symbol: str
    period_end: date
    available_at: datetime
    values: Mapping[str, FeatureValue] = field(default_factory=dict)
    filing_id: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.asset_id, "asset_id")
        _nonempty(self.symbol, "symbol")
        _aware(self.available_at, "available_at")


@dataclass(frozen=True, slots=True)
class EarningsCalendarEvent:
    asset_id: str
    symbol: str
    event_ts: datetime
    available_at: datetime
    status: Literal["estimated", "confirmed", "reported"] = "estimated"

    def __post_init__(self) -> None:
        _nonempty(self.asset_id, "asset_id")
        _nonempty(self.symbol, "symbol")
        _aware(self.event_ts, "event_ts")
        _aware(self.available_at, "available_at")


@dataclass(frozen=True, slots=True)
class CorporateAction:
    asset_id: str
    symbol: str
    event_ts: datetime
    available_at: datetime
    action_type: str
    value: float | None = None

    def __post_init__(self) -> None:
        _nonempty(self.asset_id, "asset_id")
        _nonempty(self.symbol, "symbol")
        _nonempty(self.action_type, "action_type")
        _aware(self.event_ts, "event_ts")
        _aware(self.available_at, "available_at")
        if self.value is not None:
            _finite(self.value, "value")
