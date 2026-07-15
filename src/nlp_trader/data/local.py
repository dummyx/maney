from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from nlp_trader.schemas import Asset, EntityMention, MarketBar, TextItem
from nlp_trader.timestamps import format_utc, parse_optional_date, parse_utc


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _datetime_or_none(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    return parse_utc(value)


def read_assets(path: Path) -> list[Asset]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Asset(
                asset_id=row["asset_id"],
                symbol=row["symbol"],
                exchange=row["exchange"],
                currency=row["currency"],
                name=row["name"],
                sector=row["sector"],
                active_from=parse_optional_date(row.get("active_from")),
                active_to=parse_optional_date(row.get("active_to")),
                cik=row.get("cik") or None,
                figi=row.get("figi") or None,
                isin=row.get("isin") or None,
                industry=row.get("industry") or None,
            )
            for row in csv.DictReader(handle)
        ]


def read_market_bars(path: Path) -> list[MarketBar]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            MarketBar(
                asset_id=row["asset_id"],
                symbol=row["symbol"],
                ts=parse_utc(row["ts"]),
                bar_size=row["bar_size"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                vwap=_float_or_none(row.get("vwap")),
                adjusted_close=_float_or_none(row.get("adjusted_close")),
                corporate_action_adjusted=row["corporate_action_adjusted"].lower() == "true",
                adjustment_vintage_at=_datetime_or_none(row.get("adjustment_vintage_at")),
                return_adjustment_factor=float(row["return_adjustment_factor"]),
            )
            for row in csv.DictReader(handle)
        ]


def canonical_text_hash(title: str | None, body: str | None) -> str:
    text = f"{title or ''}\n{body or ''}".strip().casefold()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text_items(path: Path) -> list[TextItem]:
    items: list[TextItem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            title = raw.get("title")
            body = raw.get("body")
            items.append(
                TextItem(
                    item_id=raw["item_id"],
                    source=raw["source"],
                    source_type=raw["source_type"],
                    language=raw["language"],
                    title=title,
                    body=body,
                    published_at=parse_utc(raw["published_at"]),
                    vendor_received_at=(
                        parse_utc(raw["vendor_received_at"])
                        if raw.get("vendor_received_at")
                        else None
                    ),
                    ingested_at=parse_utc(raw["ingested_at"]),
                    available_at=parse_utc(raw["available_at"]),
                    license_or_terms_ref=raw["license_or_terms_ref"],
                    raw_text_hash=raw.get("raw_text_hash"),
                    canonical_text_hash=raw.get("canonical_text_hash")
                    or canonical_text_hash(title, body),
                    author_hash=raw.get("author_hash"),
                    url_hash=raw.get("url_hash"),
                    raw_text_path=raw.get("raw_text_path"),
                    event_ts=_datetime_or_none(raw.get("event_ts")),
                    processed_at=_datetime_or_none(raw.get("processed_at")),
                    event_type=raw.get("event_type"),
                    relationship_type=raw.get("relationship_type", "original"),
                    parent_item_id_hash=raw.get("parent_item_id_hash"),
                    content_status=raw.get("content_status", "unknown"),
                    retention_permitted=bool(raw.get("retention_permitted", True)),
                    entities=tuple(
                        EntityMention(
                            asset_id=entity.get("asset_id"),
                            symbol=entity.get("symbol"),
                            name=entity["name"],
                            relevance=float(entity["relevance"]),
                            mention_type=entity["mention_type"],
                            confidence=float(entity["confidence"]),
                        )
                        for entity in raw.get("entities", [])
                    ),
                )
            )
    return items


def write_json(path: Path, data: dict[str, Any] | list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def asset_to_record(asset: Asset) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "symbol": asset.symbol,
        "exchange": asset.exchange,
        "currency": asset.currency,
        "name": asset.name,
        "sector": asset.sector,
        "active_from": asset.active_from.isoformat() if asset.active_from else None,
        "active_to": asset.active_to.isoformat() if asset.active_to else None,
        "cik": asset.cik,
        "figi": asset.figi,
        "isin": asset.isin,
        "industry": asset.industry,
    }


def market_bar_to_record(bar: MarketBar) -> dict[str, Any]:
    return {
        "asset_id": bar.asset_id,
        "symbol": bar.symbol,
        "ts": format_utc(bar.ts),
        "bar_size": bar.bar_size,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "vwap": bar.vwap,
        "adjusted_close": bar.adjusted_close,
        "corporate_action_adjusted": bar.corporate_action_adjusted,
        "adjustment_vintage_at": format_utc(bar.adjustment_vintage_at)
        if bar.adjustment_vintage_at
        else None,
        "return_adjustment_factor": bar.return_adjustment_factor,
    }


def text_item_to_record(item: TextItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "source": item.source,
        "source_type": item.source_type,
        "language": item.language,
        "title": item.title,
        "body": item.body,
        "published_at": format_utc(item.published_at),
        "vendor_received_at": format_utc(item.vendor_received_at)
        if item.vendor_received_at
        else None,
        "ingested_at": format_utc(item.ingested_at),
        "available_at": format_utc(item.available_at),
        "license_or_terms_ref": item.license_or_terms_ref,
        "raw_text_hash": item.raw_text_hash,
        "canonical_text_hash": item.canonical_text_hash,
        "author_hash": item.author_hash,
        "url_hash": item.url_hash,
        "raw_text_path": item.raw_text_path,
        "event_ts": format_utc(item.event_ts) if item.event_ts else None,
        "processed_at": format_utc(item.processed_at) if item.processed_at else None,
        "event_type": item.event_type,
        "relationship_type": item.relationship_type,
        "parent_item_id_hash": item.parent_item_id_hash,
        "content_status": item.content_status,
        "retention_permitted": item.retention_permitted,
        "entities": [
            {
                "asset_id": entity.asset_id,
                "symbol": entity.symbol,
                "name": entity.name,
                "relevance": entity.relevance,
                "mention_type": entity.mention_type,
                "confidence": entity.confidence,
            }
            for entity in item.entities
        ],
    }
