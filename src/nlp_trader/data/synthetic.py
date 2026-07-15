from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from nlp_trader.calendars import USEquityCalendar
from nlp_trader.data.local import canonical_text_hash
from nlp_trader.timestamps import format_utc


@dataclass(frozen=True, slots=True)
class SyntheticFixturePaths:
    assets: Path
    market_bars: Path
    text_items: Path


def generate_synthetic_fixture(
    output_dir: Path,
    *,
    seed: int = 7,
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    start: date = date(2026, 6, 29),
    session_count: int = 8,
) -> SyntheticFixturePaths:
    """Create deterministic, redistributable local fixtures without external data."""

    if session_count < 3:
        raise ValueError("session_count must be at least 3")
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be non-empty and unique")
    if any(
        not symbol.isascii() or not symbol.isalnum() or symbol != symbol.upper()
        for symbol in symbols
    ):
        raise ValueError("symbols must be uppercase ASCII alphanumeric values")

    paths = SyntheticFixturePaths(
        assets=output_dir / "assets.csv",
        market_bars=output_dir / "market_bars.csv",
        text_items=output_dir / "text_items.jsonl",
    )
    for path in (paths.assets, paths.market_bars, paths.text_items):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite synthetic fixture: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    calendar = USEquityCalendar()
    available_sessions = calendar.sessions(start, start + timedelta(days=session_count * 3))
    sessions = available_sessions[:session_count]
    if len(sessions) != session_count:
        raise ValueError("could not generate the requested number of sessions")

    _write_assets(paths.assets, symbols, start)
    _write_market_bars(paths.market_bars, symbols, sessions, seed, calendar)
    _write_text_items(paths.text_items, symbols, sessions, calendar)
    return paths


def _write_assets(path: Path, symbols: tuple[str, ...], active_from: date) -> None:
    fields = [
        "asset_id",
        "symbol",
        "exchange",
        "currency",
        "name",
        "sector",
        "industry",
        "active_from",
        "active_to",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, symbol in enumerate(symbols):
            writer.writerow(
                {
                    "asset_id": f"asset_{symbol.casefold()}",
                    "symbol": symbol,
                    "exchange": "XNAS" if index % 2 == 0 else "XNYS",
                    "currency": "USD",
                    "name": f"Synthetic {symbol} Corporation",
                    "sector": "Technology" if index % 2 == 0 else "Financials",
                    "industry": "Synthetic Research",
                    "active_from": active_from.isoformat(),
                    "active_to": "",
                }
            )


def _write_market_bars(
    path: Path,
    symbols: tuple[str, ...],
    sessions: tuple[date, ...],
    seed: int,
    calendar: USEquityCalendar,
) -> None:
    fields = [
        "asset_id",
        "symbol",
        "ts",
        "bar_size",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "adjusted_close",
        "corporate_action_adjusted",
        "adjustment_vintage_at",
        "return_adjustment_factor",
    ]
    rng = random.Random(seed)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for symbol_index, symbol in enumerate(symbols):
            price = 40.0 + 30.0 * symbol_index
            for session_index, session_date in enumerate(sessions):
                opening = price
                direction = 1.0 if (session_index + symbol_index) % 3 else -1.0
                move = direction * rng.uniform(0.002, 0.018)
                closing = max(1.0, opening * (1.0 + move))
                high = max(opening, closing) * (1.0 + rng.uniform(0.001, 0.006))
                low = min(opening, closing) * (1.0 - rng.uniform(0.001, 0.006))
                volume = 1_200_000 + 75_000 * session_index + 100_000 * symbol_index
                vwap = (opening + high + low + closing) / 4.0
                writer.writerow(
                    {
                        "asset_id": f"asset_{symbol.casefold()}",
                        "symbol": symbol,
                        "ts": format_utc(calendar.session_close(session_date)),
                        "bar_size": "1d",
                        "open": f"{opening:.4f}",
                        "high": f"{high:.4f}",
                        "low": f"{low:.4f}",
                        "close": f"{closing:.4f}",
                        "volume": str(volume),
                        "vwap": f"{vwap:.4f}",
                        "adjusted_close": f"{closing:.4f}",
                        "corporate_action_adjusted": "true",
                        "adjustment_vintage_at": format_utc(calendar.session_close(session_date)),
                        "return_adjustment_factor": "1.0",
                    }
                )
                price = closing


def _write_text_items(
    path: Path,
    symbols: tuple[str, ...],
    sessions: tuple[date, ...],
    calendar: USEquityCalendar,
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for index, symbol in enumerate(symbols):
            session_date = sessions[min(index + 1, len(sessions) - 1)]
            published_at = calendar.session_open(session_date) + timedelta(hours=1)
            vendor_received_at = published_at + timedelta(minutes=2)
            ingested_at = vendor_received_at + timedelta(minutes=3)
            title = f"Synthetic {symbol} operating update"
            body = (
                f"Synthetic {symbol} reported improved demand and stable margins. "
                "This text is generated test data and contains no investment conclusion."
            )
            text_hash = canonical_text_hash(title, body)
            author_hash = hashlib.sha256(f"synthetic-author-{index}".encode()).hexdigest()
            url_hash = hashlib.sha256(f"synthetic://{symbol}/{session_date}".encode()).hexdigest()
            record = {
                "item_id": f"synthetic-{index + 1:03d}",
                "source": "synthetic_news",
                "source_type": "news",
                "language": "en",
                "title": title,
                "body": body,
                "published_at": format_utc(published_at),
                "vendor_received_at": format_utc(vendor_received_at),
                "ingested_at": format_utc(ingested_at),
                "available_at": format_utc(vendor_received_at),
                "license_or_terms_ref": "synthetic-fixture-v1",
                "relationship_type": "original",
                "content_status": "active",
                "retention_permitted": True,
                "canonical_text_hash": text_hash,
                "author_hash": author_hash,
                "url_hash": url_hash,
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
