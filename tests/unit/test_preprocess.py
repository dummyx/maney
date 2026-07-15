from __future__ import annotations

from datetime import UTC, datetime

from nlp_trader.nlp.preprocess import cluster_near_duplicates, preprocess_text
from nlp_trader.nlp.simple import analyze_sentiment, build_text_signals, link_entities
from nlp_trader.schemas import Asset, EntityMention, TextItem


def _asset(asset_id: str, symbol: str, name: str) -> Asset:
    return Asset(
        asset_id=asset_id,
        symbol=symbol,
        exchange="XNAS",
        currency="USD",
        name=name,
        sector="Technology",
        active_from=None,
        active_to=None,
    )


def _item(item_id: str, title: str, body: str, *, minute: int = 0) -> TextItem:
    timestamp = datetime(2026, 7, 1, 12, minute, tzinfo=UTC)
    return TextItem(
        item_id=item_id,
        source="fixture",
        source_type="news",
        language="en",
        title=title,
        body=body,
        published_at=timestamp,
        ingested_at=timestamp,
        available_at=timestamp,
        license_or_terms_ref="synthetic-fixture",
    )


def test_preprocess_normalizes_and_extracts_market_tokens() -> None:
    result = preprocess_text(
        "Ａｌｐｈａ beats estimates 🚀 https://example.test/story @Trader #Growth $AAA\nRead more"
    )

    assert result.cashtags == ("AAA",)
    assert "AAA" in result.ticker_candidates
    assert result.urls == ("https://example.test/story",)
    assert result.mentions == ("trader",)
    assert result.hashtags == ("growth",)
    assert result.emojis == ("🚀",)
    assert "read more" not in result.normalized_text
    assert "<url>" in result.normalized_text
    assert "<mention>" in result.normalized_text
    assert "<emoji>" in result.normalized_text


def test_exact_and_near_duplicates_share_a_deterministic_cluster() -> None:
    clusters = cluster_near_duplicates(
        [
            ("first", "Alpha raises guidance after a strong quarter."),
            ("near", "Alpha raises guidance after the strong quarter."),
            ("exact", "Alpha raises guidance after a strong quarter."),
            ("other", "Beta warns that credit losses may rise."),
        ],
        threshold=0.8,
    )

    assert clusters["first"] == clusters["near"] == clusters["exact"]
    assert clusters["first"] != clusters["other"]


def test_entity_linker_rejects_ambiguous_symbol_without_name() -> None:
    assets = [
        _asset("first", "ABC", "Alpha Brands"),
        _asset("second", "ABC", "Atlas Bancorp"),
    ]

    assert not link_entities(_item("ambiguous", "$ABC update", "Results released."), assets)
    mentions = link_entities(
        _item("named", "Alpha Brands $ABC update", "Results released."), assets
    )

    assert [mention.asset_id for mention in mentions] == ["first"]


def test_sentiment_tracks_negation_uncertainty_and_duplicate_novelty() -> None:
    clear = analyze_sentiment("Alpha did not miss estimates", "Management raised guidance.")
    uncertain = analyze_sentiment(
        "Alpha did not miss estimates", "Management may possibly raise uncertain guidance."
    )

    assert clear.score > 0.0
    assert uncertain.uncertainty > clear.uncertainty
    assert uncertain.confidence < clear.confidence

    asset = _asset("alpha", "AAA", "Alpha Analytics")
    signals = build_text_signals(
        [
            _item("original", "Alpha Analytics raises guidance", "Demand is strong."),
            _item(
                "duplicate",
                "Alpha Analytics raises guidance",
                "Demand is very strong.",
                minute=1,
            ),
        ],
        [asset],
    )

    assert [signal.novelty for signal in signals] == [1.0, 0.0]


def test_future_bridge_document_cannot_rewrite_prior_novelty() -> None:
    asset = _asset("alpha", "AAA", "Alpha Analytics")
    mention = EntityMention(
        asset_id=asset.asset_id,
        symbol=asset.symbol,
        name=asset.name,
        relevance=1.0,
        mention_type="primary",
        confidence=1.0,
    )

    def linked(item_id: str, body: str, minute: int) -> TextItem:
        item = _item(item_id, "", body, minute=minute)
        return TextItem(
            item_id=item.item_id,
            source=item.source,
            source_type=item.source_type,
            language=item.language,
            title=item.title,
            body=item.body,
            published_at=item.published_at,
            ingested_at=item.ingested_at,
            available_at=item.available_at,
            license_or_terms_ref=item.license_or_terms_ref,
            entities=(mention,),
        )

    first = linked("a", "a b c d e f g h i j", 0)
    second = linked("b", "a b c d e f g h k l", 1)
    bridge = linked("c", "a b c d e f g h i k", 2)

    early = build_text_signals([first, second], [asset])
    extended = build_text_signals([first, second, bridge], [asset])

    assert [(row.item_id, row.novelty, row.duplicate_cluster_id) for row in early] == [
        (row.item_id, row.novelty, row.duplicate_cluster_id) for row in extended[:2]
    ]
    assert [row.novelty for row in early] == [1.0, 1.0]
    assert extended[2].novelty == 0.0
