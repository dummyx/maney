from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nlp_trader.config import DataConfig
from nlp_trader.data.japan import (
    JAPAN_CASH_EQUITY_CONTRACT,
    is_canonical_japanese_security_code,
)
from nlp_trader.data.parquet import write_partitioned_parquet
from nlp_trader.providers import LocalMarketDataProvider


def _asset_record() -> dict[str, Any]:
    return {
        "asset_id": "xjpx-130a",
        "symbol": "130A",
        "exchange": "XJPX",
        "currency": "JPY",
        "name": "Permitted Local Export Company",
        "sector": "Industrials",
        "active_from": "2026-01-01",
        "active_to": "",
        "trading_unit": 100,
    }


def _bar_record() -> dict[str, Any]:
    return {
        "asset_id": "xjpx-130a",
        "symbol": "130A",
        "exchange": "XJPX",
        "currency": "JPY",
        "trading_unit": 100,
        "session_date": "2026-07-15",
        "ts": "2026-07-15T06:30:00Z",
        "available_at": "2026-07-15T07:30:00Z",
        "bar_size": "1d",
        "open": 1000.0,
        "high": 1020.0,
        "low": 995.0,
        "close": 1010.0,
        "volume": 125_000,
        "vwap": 1008.0,
        "adjusted_close": "",
        "corporate_action_adjusted": True,
        "adjustment_vintage_at": "2026-07-15T07:30:00Z",
        "return_adjustment_factor": 1.0,
        "price_basis": "raw_tradable",
    }


def _write_record(path: Path, record: dict[str, Any], storage_format: str) -> Path:
    if storage_format == "csv":
        destination = path.with_suffix(".csv")
        csv_record = {
            field: str(value).lower() if type(value) is bool else value
            for field, value in record.items()
        }
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_record))
            writer.writeheader()
            writer.writerow(csv_record)
        return destination
    destination = path
    write_partitioned_parquet([record], destination)
    return destination


@pytest.mark.parametrize("storage_format", ["csv", "parquet"])
def test_japanese_contract_accepts_strict_local_csv_and_parquet(
    tmp_path: Path, storage_format: str
) -> None:
    assets_path = _write_record(tmp_path / "assets", _asset_record(), storage_format)
    bars_path = _write_record(tmp_path / "bars", _bar_record(), storage_format)
    provider = LocalMarketDataProvider(
        assets_path,
        bars_path,
        market_contract=JAPAN_CASH_EQUITY_CONTRACT,
    )

    assets = provider.fetch_assets()
    bars = provider.fetch_bars()

    assert assets[0].symbol == "130A"
    assert assets[0].trading_unit == 100
    assert bars[0].exchange == "XJPX"
    assert bars[0].currency == "JPY"
    assert bars[0].trading_unit == 100
    assert bars[0].session_date is not None
    assert bars[0].available_at is not None and bars[0].available_at > bars[0].ts
    assert bars[0].price_basis == "raw_tradable"


def test_japanese_decision_times_wait_for_complete_session_cross_section(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for day, first_time, second_time in (
        (15, "07:15:00", "07:30:00"),
        (16, "07:20:00", "07:45:00"),
    ):
        for symbol, available_time in (("130A", first_time), ("7203", second_time)):
            row = _bar_record()
            row.update(
                asset_id=f"xjpx-{symbol.lower()}",
                symbol=symbol,
                session_date=f"2026-07-{day:02d}",
                ts=f"2026-07-{day:02d}T06:30:00Z",
                available_at=f"2026-07-{day:02d}T{available_time}Z",
                adjustment_vintage_at=f"2026-07-{day:02d}T{available_time}Z",
            )
            rows.append(row)
    bars_path = tmp_path / "bars.csv"
    with bars_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: str(value).lower() if type(value) is bool else value
                    for field, value in row.items()
                }
            )
    provider = LocalMarketDataProvider(
        tmp_path / "unused-assets.csv",
        bars_path,
        market_contract=JAPAN_CASH_EQUITY_CONTRACT,
    )

    assert provider.fetch_decision_times(limit=1) == (datetime(2026, 7, 15, 7, 30, tzinfo=UTC),)
    assert provider.fetch_decision_times(start=datetime(2026, 7, 15, 7, 31, tzinfo=UTC)) == (
        datetime(2026, 7, 16, 7, 45, tzinfo=UTC),
    )


@pytest.mark.parametrize("symbol", ["1300", "130A", "1A00", "9A7A"])
def test_canonical_japanese_security_codes_include_new_alphanumeric_codes(symbol: str) -> None:
    assert is_canonical_japanese_security_code(symbol)


@pytest.mark.parametrize(
    "symbol",
    ["130a", "130B", "A300", "13A0", "0130", "13000", "130"],
)
def test_canonical_japanese_security_codes_fail_closed(symbol: str) -> None:
    assert not is_canonical_japanese_security_code(symbol)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(exchange="XTKS"), "exchange must be XJPX"),
        (lambda row: row.update(currency="USD"), "currency must be JPY"),
        (lambda row: row.update(trading_unit=0), "trading_unit must be positive"),
        (lambda row: row.update(symbol="130B"), "canonical Japanese"),
        (lambda row: row.pop("trading_unit"), "missing required fields: trading_unit"),
        (lambda row: row.update(unit=100), "unsupported fields: unit"),
    ],
)
def test_japanese_asset_contract_rejects_noncanonical_rows(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    asset = _asset_record()
    mutate(asset)
    provider = LocalMarketDataProvider(
        _write_record(tmp_path / "assets", asset, "csv"),
        tmp_path / "unused-bars.csv",
        market_contract=JAPAN_CASH_EQUITY_CONTRACT,
    )

    with pytest.raises(ValueError, match=message):
        provider.fetch_assets()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(exchange="XTKS"), "exchange must be XJPX"),
        (lambda row: row.update(currency="USD"), "currency must be JPY"),
        (lambda row: row.update(trading_unit=0), "trading_unit must be positive"),
        (lambda row: row.update(volume=1.5), "volume must be an integer"),
        (lambda row: row.update(symbol="130B"), "canonical Japanese"),
        (lambda row: row.update(bar_size="1h"), "bar_size must be 1d"),
        (lambda row: row.update(price_basis="adjusted"), "price_basis must be raw_tradable"),
        (
            lambda row: row.update(corporate_action_adjusted=False),
            "corporate_action_adjusted must be explicitly true",
        ),
        (lambda row: row.pop("available_at"), "missing required fields: available_at"),
        (
            lambda row: row.update(
                available_at="2026-07-15T06:29:59Z",
                adjustment_vintage_at="2026-07-15T06:29:59Z",
            ),
            "available_at must not be before ts",
        ),
        (
            lambda row: row.update(
                available_at="2026-07-15T07:30:00Z",
                adjustment_vintage_at="2026-07-15T07:30:01Z",
            ),
            "adjustment_vintage_at must not be after available_at",
        ),
        (
            lambda row: row.update(adjustment_vintage_at="2026-07-15T06:29:59Z"),
            "adjustment_vintage_at must not be before ts",
        ),
        (lambda row: row.update(session_date="2026-07-16"), "session_date must match"),
        (lambda row: row.update(typo_field="x"), "unsupported fields: typo_field"),
    ],
)
def test_japanese_bar_contract_rejects_ambiguous_or_noncausal_rows(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    asset = _asset_record()
    bar = _bar_record()
    mutate(bar)
    provider = LocalMarketDataProvider(
        _write_record(tmp_path / "assets", asset, "csv"),
        _write_record(tmp_path / "bars", bar, "csv"),
        market_contract=JAPAN_CASH_EQUITY_CONTRACT,
    )

    with pytest.raises(ValueError, match=message):
        provider.fetch_bars()


def test_japanese_bar_contract_binds_each_bar_to_asset_master(tmp_path: Path) -> None:
    bar = _bar_record()
    bar["trading_unit"] = 1
    provider = LocalMarketDataProvider(
        _write_record(tmp_path / "assets", _asset_record(), "parquet"),
        _write_record(tmp_path / "bars", bar, "parquet"),
        market_contract=JAPAN_CASH_EQUITY_CONTRACT,
    )

    with pytest.raises(ValueError, match="trading_unit does not match asset master"):
        provider.fetch_bars()


def test_data_config_binds_xjpx_calendar_to_japanese_contract() -> None:
    japanese = DataConfig(calendar="XJPX", market_contract=JAPAN_CASH_EQUITY_CONTRACT)
    assert japanese.calendar == "XJPX"

    with pytest.raises(ValidationError, match="calendar must be XJPX"):
        DataConfig(calendar="XNYS", market_contract=JAPAN_CASH_EQUITY_CONTRACT)
    with pytest.raises(ValidationError, match="market_contract must be japan_cash_equity_v1"):
        DataConfig(calendar="XJPX", market_contract="generic")
    with pytest.raises(ValidationError, match="calendar"):
        DataConfig(calendar="XTKS")  # type: ignore[arg-type]
