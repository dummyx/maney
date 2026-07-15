from __future__ import annotations

import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from nlp_trader.backtest.engine import DeterministicBacktestEngine
from nlp_trader.config import BacktestConfig
from nlp_trader.data.local import asset_to_record, read_assets
from nlp_trader.data.parquet import write_partitioned_parquet
from nlp_trader.data.stores import LocalFeatureStore, LocalModelRegistry
from nlp_trader.providers import (
    BacktestEngine,
    CalendarProvider,
    FeatureStore,
    FundamentalsProvider,
    LocalCalendarProvider,
    LocalFundamentalsProvider,
    LocalMarketDataProvider,
    LocalTextDataProvider,
    MarketDataProvider,
    ModelRegistry,
    TextDataProvider,
)
from nlp_trader.schemas import (
    Asset,
    EntityMention,
    LabelRow,
    MarketBar,
    OrderIntent,
    PredictionRow,
    TextItem,
)


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        commission_bps=0.0,
        half_spread_bps=0.0,
        slippage_bps=0.0,
        borrow_bps_per_year=0.0,
        max_position_weight=0.5,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_daily_turnover=1.0,
        max_participation_rate=0.05,
        min_price=1.0,
        min_dollar_volume=0.0,
        shorting_allowed=False,
        hard_to_borrow_allowed=False,
    )


def test_local_implementations_satisfy_provider_contracts(tmp_path: Path) -> None:
    market = LocalMarketDataProvider(tmp_path / "assets.csv", tmp_path / "bars.csv")
    text = LocalTextDataProvider(tmp_path / "text.jsonl")
    fundamentals = LocalFundamentalsProvider()
    calendar = LocalCalendarProvider()
    features = LocalFeatureStore(tmp_path / "features")
    models = LocalModelRegistry(tmp_path / "models")
    backtests = DeterministicBacktestEngine()

    assert isinstance(market, MarketDataProvider)
    assert isinstance(text, TextDataProvider)
    assert isinstance(fundamentals, FundamentalsProvider)
    assert isinstance(calendar, CalendarProvider)
    assert isinstance(features, FeatureStore)
    assert isinstance(models, ModelRegistry)
    assert isinstance(backtests, BacktestEngine)
    typed_backtests: BacktestEngine = backtests
    result = typed_backtests.run(
        [],
        [],
        _backtest_config(),
        top_k=1,
        rebalance_offset=0,
    )
    assert result["assumptions"]["top_k"] == 1
    assert result["assumptions"]["rebalance_offset"] == 0


def test_empty_text_jsonl_is_a_valid_market_only_input(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert LocalTextDataProvider(path).fetch_items() == []


def test_required_research_schemas_preserve_versions_and_identifiers() -> None:
    asof_ts = datetime(2026, 7, 6, 20, tzinfo=UTC)
    asset = Asset(
        asset_id="asset_aaa",
        symbol="AAA",
        exchange="XNAS",
        currency="USD",
        name="Alpha Analytics",
        sector="Technology",
        active_from=date(2026, 1, 1),
        active_to=None,
        cik="0000000001",
        figi="BBG000TEST01",
        industry="Software",
    )
    label = LabelRow(
        asset_id=asset.asset_id,
        symbol=asset.symbol,
        asof_ts=asof_ts,
        horizon="1d",
        label_version="labels-v1",
        forward_return=0.01,
    )
    prediction = PredictionRow(
        asset_id=asset.asset_id,
        symbol=asset.symbol,
        asof_ts=asof_ts,
        horizon="1d",
        model_version="model-v1",
        score=0.4,
        uncertainty=0.2,
    )
    intent = OrderIntent(
        strategy_id="research-only",
        asof_ts=asof_ts,
        symbol=asset.symbol,
        target_weight=0.05,
        side="buy",
        reason_codes=("positive-score",),
        risk_flags=("paper-only",),
    )

    assert asset.cik == "0000000001"
    assert asset.short_available is False
    assert asset.hard_to_borrow is False
    assert label.label_version == "labels-v1"
    assert prediction.model_version == "model-v1"
    assert intent.risk_flags == ("paper-only",)


def test_asset_borrow_fields_reject_non_booleans_and_fail_closed_when_htb_is_missing() -> None:
    def asset_with_borrow_fields(
        *, short_available: Any = False, hard_to_borrow: Any = None
    ) -> Asset:
        return Asset(
            asset_id="asset_aaa",
            symbol="AAA",
            exchange="XNAS",
            currency="USD",
            name="Alpha Analytics",
            sector="Technology",
            active_from=date(2026, 1, 1),
            active_to=None,
            short_available=short_available,
            hard_to_borrow=hard_to_borrow,
        )

    with pytest.raises(ValueError, match="short_available must be a boolean"):
        asset_with_borrow_fields(short_available="false")
    with pytest.raises(ValueError, match="hard_to_borrow must be a boolean"):
        asset_with_borrow_fields(hard_to_borrow="false")

    shortable = asset_with_borrow_fields(short_available=True)
    assert shortable.hard_to_borrow is True


@pytest.mark.parametrize("storage_format", ["csv", "json", "parquet"])
@pytest.mark.parametrize("omit_hard_to_borrow", [False, True])
def test_asset_borrow_fields_round_trip_through_local_providers(
    tmp_path: Path,
    storage_format: str,
    omit_hard_to_borrow: bool,
) -> None:
    asset = Asset(
        asset_id="asset_aaa",
        symbol="AAA",
        exchange="XNAS",
        currency="USD",
        name="Alpha Analytics",
        sector="Technology",
        active_from=date(2026, 1, 1),
        active_to=None,
        short_available=True,
        hard_to_borrow=True,
    )
    record = asset_to_record(asset)
    if omit_hard_to_borrow:
        record.pop("hard_to_borrow")
    if storage_format == "csv":
        assets_path = tmp_path / "assets.csv"
        with assets_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
        assert read_assets(assets_path) == [asset]
    elif storage_format == "json":
        assets_path = tmp_path / "assets.json"
        assets_path.write_text(json.dumps([record]), encoding="utf-8")
    else:
        assets_path = tmp_path / "assets"
        write_partitioned_parquet([record], assets_path)

    provider = LocalMarketDataProvider(assets_path, tmp_path / "unused-bars.csv")
    assert provider.fetch_assets() == [asset]


def test_boundary_schemas_reject_invalid_market_and_text_records() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketBar(
            asset_id="asset_aaa",
            symbol="AAA",
            ts=datetime(2026, 7, 6, 20),
            bar_size="1d",
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.0,
            volume=100,
            vwap=10.0,
            adjusted_close=10.0,
            corporate_action_adjusted=True,
        )
    with pytest.raises(ValueError, match="OHLC"):
        MarketBar(
            asset_id="asset_aaa",
            symbol="AAA",
            ts=datetime(2026, 7, 6, 20, tzinfo=UTC),
            bar_size="1d",
            open=10.0,
            high=9.0,
            low=8.0,
            close=10.0,
            volume=100,
            vwap=None,
            adjusted_close=None,
            corporate_action_adjusted=True,
            adjustment_vintage_at=datetime(2026, 7, 6, 20, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="open must be finite"):
        MarketBar(
            asset_id="asset_aaa",
            symbol="AAA",
            ts=datetime(2026, 7, 6, 20, tzinfo=UTC),
            bar_size="1d",
            open=float("nan"),
            high=11.0,
            low=9.0,
            close=10.0,
            volume=100,
            vwap=None,
            adjusted_close=None,
            corporate_action_adjusted=True,
            adjustment_vintage_at=datetime(2026, 7, 6, 20, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="adjustment_vintage_at must not be after"):
        MarketBar(
            asset_id="asset_aaa",
            symbol="AAA",
            ts=datetime(2026, 7, 6, 20, tzinfo=UTC),
            bar_size="1d",
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=100,
            vwap=None,
            adjusted_close=None,
            corporate_action_adjusted=True,
            adjustment_vintage_at=datetime(2026, 7, 7, 20, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="relevance"):
        EntityMention(
            name="Alpha",
            relevance=1.1,
            mention_type="primary",
            confidence=0.9,
        )

    timestamp = datetime(2026, 7, 6, 20, tzinfo=UTC)
    with pytest.raises(ValueError, match="author_hash"):
        TextItem(
            item_id="item",
            source="fixture",
            source_type="social",
            language="en",
            title=None,
            body="text",
            published_at=timestamp,
            ingested_at=timestamp,
            available_at=timestamp,
            license_or_terms_ref="synthetic",
            author_hash="raw-handle",
        )
    with pytest.raises(ValueError, match="available_at must not be before vendor_received_at"):
        TextItem(
            item_id="item",
            source="fixture",
            source_type="news",
            language="en",
            title=None,
            body="text",
            published_at=timestamp,
            vendor_received_at=timestamp.replace(minute=5),
            ingested_at=timestamp.replace(minute=10),
            available_at=timestamp,
            license_or_terms_ref="synthetic",
        )


def test_social_provider_hashes_identifiers_and_enforces_retention(tmp_path: Path) -> None:
    provider = LocalTextDataProvider(tmp_path / "social.jsonl")
    timestamp = "2026-07-06T20:00:00Z"
    item = provider.normalize_item(
        {
            "item_id": "social-1",
            "source": "user-export",
            "source_type": "social",
            "language": "en",
            "body": "permitted synthetic post",
            "published_at": timestamp,
            "ingested_at": timestamp,
            "available_at": timestamp,
            "license_or_terms_ref": "synthetic",
            "author_handle": "ExampleUser",
            "relationship_type": "reply",
            "parent_item_id": "parent-1",
            "content_status": "active",
        }
    )

    assert item.author_hash is not None and len(item.author_hash) == 64
    assert item.author_hash != "ExampleUser"
    assert item.parent_item_id_hash is not None
    assert item.relationship_type == "reply"

    with pytest.raises(ValueError, match="do not permit retention"):
        provider.normalize_item(
            {
                "item_id": "social-2",
                "source": "user-export",
                "source_type": "social",
                "language": "en",
                "body": "protected content",
                "published_at": timestamp,
                "ingested_at": timestamp,
                "available_at": timestamp,
                "license_or_terms_ref": "synthetic",
                "content_status": "protected",
                "retention_permitted": False,
            }
        )
