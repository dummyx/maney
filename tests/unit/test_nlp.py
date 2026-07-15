from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

from nlp_trader.nlp.simple import link_entities, sentiment_score
from nlp_trader.schemas import Asset, TextItem


def _asset() -> Asset:
    return Asset(
        asset_id="asset_aaa",
        symbol="AAA",
        exchange="XNAS",
        currency="USD",
        name="Alpha Analytics",
        sector="Technology",
        active_from=None,
        active_to=None,
    )


def _item(title: str, body: str) -> TextItem:
    now = datetime(2026, 7, 1, 12, tzinfo=UTC)
    return TextItem(
        item_id="item",
        source="sample",
        source_type="news",
        language="en",
        title=title,
        body=body,
        published_at=now,
        vendor_received_at=now,
        ingested_at=now,
        available_at=now,
        license_or_terms_ref="synthetic-fixture",
    )


def test_link_entities_requires_name_title_symbol_or_cashtag() -> None:
    assert not link_entities(_item("Market update", "AAA appears only in body text."), [_asset()])
    assert link_entities(_item("Market update", "$AAA backlog improved."), [_asset()])
    assert link_entities(_item("Alpha Analytics update", "Demand improved."), [_asset()])


def test_link_entities_uses_the_assets_exchange_local_active_date() -> None:
    available_at = datetime(2026, 7, 15, 15, 30, tzinfo=UTC)
    asset = replace(
        _asset(),
        asset_id="asset_7203",
        symbol="7203",
        exchange="XJPX",
        currency="JPY",
        name="Toyota Motor",
        active_from=date(2026, 7, 16),
    )
    item = replace(
        _item("Toyota Motor update", "Demand improved."),
        published_at=available_at,
        vendor_received_at=available_at,
        ingested_at=available_at,
        available_at=available_at,
    )

    assert link_entities(item, [asset])


def test_sentiment_handles_simple_negation() -> None:
    score, label, confidence = sentiment_score("Not weak", "Margins improved.")

    assert score > 0
    assert label == "positive"
    assert confidence > 0
