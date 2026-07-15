from __future__ import annotations

import re
from pathlib import Path

import pytest

from nlp_trader.calendars import USEquityCalendar
from nlp_trader.data.synthetic import generate_synthetic_fixture
from nlp_trader.providers import LocalMarketDataProvider, LocalTextDataProvider


def test_synthetic_fixture_is_deterministic_and_provider_compatible(tmp_path: Path) -> None:
    first = generate_synthetic_fixture(tmp_path / "first", seed=11)
    second = generate_synthetic_fixture(tmp_path / "second", seed=11)

    assert first.assets.read_bytes() == second.assets.read_bytes()
    assert first.market_bars.read_bytes() == second.market_bars.read_bytes()
    assert first.text_items.read_bytes() == second.text_items.read_bytes()

    market = LocalMarketDataProvider(first.assets, first.market_bars)
    text = LocalTextDataProvider(first.text_items)
    assets = market.fetch_assets()
    bars = market.fetch_bars(symbols=["AAA"])
    items = text.fetch_items()

    assert [asset.symbol for asset in assets] == ["AAA", "BBB"]
    assert len(bars) == 8
    assert bars[0].ts == USEquityCalendar().session_close(bars[0].ts.date())
    assert all(item.available_at <= item.ingested_at for item in items)
    assert all(item.author_hash is not None for item in items)
    assert all(re.fullmatch(r"[0-9a-f]{64}", item.author_hash or "") for item in items)


def test_synthetic_fixture_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "sample"
    generate_synthetic_fixture(output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_synthetic_fixture(output)
