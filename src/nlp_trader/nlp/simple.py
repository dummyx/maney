from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from zoneinfo import ZoneInfo

from nlp_trader.nlp.preprocess import (
    AMBIGUOUS_UPPERCASE_TOKENS,
    cluster_near_duplicates_incremental,
    preprocess_text,
)
from nlp_trader.schemas import Asset, EntityMention, TextItem, TextSignal

POSITIVE_TERMS: dict[str, float] = {
    "accelerate": 1.0,
    "accelerated": 1.0,
    "beat": 1.0,
    "beats": 1.0,
    "better": 0.75,
    "bullish": 1.0,
    "expand": 0.75,
    "expanded": 1.0,
    "growth": 0.75,
    "improved": 1.0,
    "improvement": 1.0,
    "outperform": 1.0,
    "profitable": 1.0,
    "profit": 0.75,
    "profits": 0.75,
    "raise": 0.75,
    "raised": 1.0,
    "rebound": 0.75,
    "stable": 0.5,
    "strong": 1.0,
    "upgrade": 1.0,
    "upgraded": 1.0,
}
NEGATIVE_TERMS: dict[str, float] = {
    "bankruptcy": -1.5,
    "bearish": -1.0,
    "cut": -0.75,
    "cuts": -0.75,
    "decline": -0.75,
    "declined": -1.0,
    "default": -1.5,
    "downgrade": -1.0,
    "downgraded": -1.0,
    "fraud": -1.5,
    "impairment": -1.0,
    "investigation": -0.75,
    "loss": -1.0,
    "losses": -1.0,
    "miss": -1.0,
    "missed": -1.0,
    "risk": -0.5,
    "risks": -0.5,
    "weak": -1.0,
    "weaker": -1.0,
    "warned": -1.0,
    "warning": -1.0,
}
NEGATIONS = {"hardly", "neither", "never", "no", "not", "without"}
UNCERTAINTY_TERMS = {
    "approximately",
    "could",
    "estimate",
    "estimated",
    "may",
    "might",
    "possible",
    "possibly",
    "potential",
    "uncertain",
    "uncertainty",
    "unclear",
}
NEGATION_BREAKERS = {"although", "but", "however", "nevertheless", "though", "yet"}
SOURCE_CREDIBILITY = {
    "filing": 1.0,
    "news": 0.9,
    "transcript": 0.85,
    "blog": 0.6,
    "forum": 0.45,
    "social": 0.35,
    "other": 0.5,
}
PROMOTIONAL_PHRASES = ("buy now", "can't lose", "guaranteed return", "going to the moon", "100x")
LEGAL_SUFFIXES = (" corp", " corporation", " inc", " incorporated", " limited", " llc", " plc")
_EXCHANGE_TIMEZONES = {
    "XNYS": ZoneInfo("America/New_York"),
    "XNAS": ZoneInfo("America/New_York"),
    "XJPX": ZoneInfo("Asia/Tokyo"),
}


@dataclass(frozen=True, slots=True)
class SentimentResult:
    score: float
    label: str
    confidence: float
    uncertainty: float
    positive_hits: int
    negative_hits: int


def normalize_text(value: str) -> str:
    """Compatibility wrapper for the deterministic preprocessing normalizer."""

    return preprocess_text(value).normalized_text


def _asset_is_active(asset: Asset, item: TextItem) -> bool:
    timezone = _EXCHANGE_TIMEZONES.get(asset.exchange.upper(), UTC)
    item_date = item.available_at.astimezone(timezone).date()
    return not (
        (asset.active_from is not None and item_date < asset.active_from)
        or (asset.active_to is not None and item_date > asset.active_to)
    )


def _name_aliases(asset: Asset) -> tuple[str, ...]:
    name = normalize_text(asset.name).strip()
    aliases = [name]
    for suffix in LEGAL_SUFFIXES:
        if name.endswith(suffix):
            alias = name[: -len(suffix)].strip()
            if len(alias) >= 5:
                aliases.append(alias)
            break
    return tuple(dict.fromkeys(aliases))


def _contains_name(text: str, aliases: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text) for alias in aliases)


def link_entities(item: TextItem, assets: Iterable[Asset]) -> tuple[EntityMention, ...]:
    """Link explicit company references while rejecting ambiguous bare ticker tokens."""

    asset_list = list(assets)
    symbol_counts = Counter(asset.symbol.upper() for asset in asset_list)
    full_text = f"{item.title or ''} {item.body or ''}"
    normalized = normalize_text(full_text)
    normalized_title = normalize_text(item.title or "")
    processed = preprocess_text(full_text)
    title_processed = preprocess_text(item.title or "")
    mentions: list[EntityMention] = []
    for asset in asset_list:
        if not _asset_is_active(asset, item):
            continue
        symbol = asset.symbol.upper()
        aliases = _name_aliases(asset)
        title_name_hit = _contains_name(normalized_title, aliases)
        name_hit = title_name_hit or _contains_name(normalized, aliases)
        cashtag_hit = symbol in processed.cashtags
        bare_title_hit = symbol in title_processed.ticker_candidates
        symbol_is_ambiguous = symbol_counts[symbol] > 1

        # A duplicated symbol needs a company-name reference. Common prose tokens such
        # as IT or ON are never accepted as bare title tickers.
        if symbol_is_ambiguous and not name_hit:
            continue
        if (
            symbol in AMBIGUOUS_UPPERCASE_TOKENS
            and bare_title_hit
            and not (cashtag_hit or name_hit)
        ):
            continue
        if not (name_hit or cashtag_hit or bare_title_hit):
            continue

        if title_name_hit:
            relevance, confidence = 1.0, 0.99
        elif cashtag_hit and name_hit:
            relevance, confidence = 1.0, 0.98
        elif cashtag_hit:
            relevance, confidence = 0.9, 0.88
        elif name_hit:
            relevance, confidence = 0.85, 0.96
        else:
            relevance, confidence = 0.6, 0.75
        mentions.append(
            EntityMention(
                asset_id=asset.asset_id,
                symbol=asset.symbol,
                name=asset.name,
                relevance=relevance,
                mention_type="primary" if relevance >= 0.9 else "secondary",
                confidence=confidence,
            )
        )
    return tuple(mentions)


def _is_negated(tokens: tuple[str, ...], index: int) -> bool:
    scope = tokens[max(0, index - 3) : index]
    active_scope: list[str] = []
    for token in reversed(scope):
        if token in NEGATION_BREAKERS or token in POSITIVE_TERMS or token in NEGATIVE_TERMS:
            break
        active_scope.append(token)
    return sum(token in NEGATIONS for token in active_scope) % 2 == 1


def analyze_sentiment(title: str | None, body: str | None) -> SentimentResult:
    """Score finance-oriented dictionary sentiment with local negation and uncertainty."""

    tokens = tuple(
        token for token in preprocess_text(f"{title or ''} {body or ''}").tokens if token.isalpha()
    )
    signed_hits: list[float] = []
    positive_hits = 0
    negative_hits = 0
    for index, token in enumerate(tokens):
        value = POSITIVE_TERMS.get(token, NEGATIVE_TERMS.get(token, 0.0))
        if value == 0.0:
            continue
        if _is_negated(tokens, index):
            value *= -1.0
        signed_hits.append(value)
        if value > 0:
            positive_hits += 1
        else:
            negative_hits += 1

    uncertainty_hits = sum(token in UNCERTAINTY_TERMS for token in tokens)
    uncertainty = min(1.0, uncertainty_hits / max(1, len(signed_hits) + uncertainty_hits))
    if not signed_hits:
        return SentimentResult(0.0, "neutral", 0.0, uncertainty, 0, 0)

    raw_score = sum(signed_hits) / sum(abs(value) for value in signed_hits)
    score = max(-1.0, min(1.0, raw_score * (1.0 - 0.35 * uncertainty)))
    label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    agreement = abs(sum(signed_hits)) / sum(abs(value) for value in signed_hits)
    evidence = min(1.0, len(signed_hits) / 3.0)
    confidence = min(1.0, (0.2 + 0.55 * agreement + 0.25 * evidence) * (1.0 - 0.5 * uncertainty))
    return SentimentResult(
        score=score,
        label=label,
        confidence=confidence,
        uncertainty=uncertainty,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
    )


def sentiment_score(title: str | None, body: str | None) -> tuple[float, str, float]:
    """Return the historical tuple API backed by the richer sentiment result."""

    result = analyze_sentiment(title, body)
    return result.score, result.label, result.confidence


def infer_event_type(title: str | None, body: str | None) -> str | None:
    """Infer a conservative event category from explicit finance vocabulary."""

    tokens = set(preprocess_text(f"{title or ''} {body or ''}").tokens)
    rules = (
        ("bankruptcy", {"bankruptcy", "insolvency", "chapter"}),
        ("merger_acquisition", {"acquisition", "acquire", "merger", "takeover"}),
        ("guidance", {"forecast", "guidance", "outlook"}),
        ("earnings", {"earnings", "eps", "revenue", "quarter"}),
        ("dividend", {"dividend", "distribution"}),
        ("litigation", {"lawsuit", "litigation", "settlement"}),
        ("regulatory", {"regulator", "regulatory", "sec"}),
        ("capital_raise", {"offering", "placement", "raise"}),
    )
    for event_type, terms in rules:
        if tokens & terms:
            return event_type
    return None


def estimate_spam_score(title: str | None, body: str | None) -> float:
    """Estimate obvious promotional content without profiling its author."""

    value = f"{title or ''} {body or ''}"
    processed = preprocess_text(value)
    score = 0.0
    folded = processed.normalized_text
    score += 0.3 * sum(phrase in folded for phrase in PROMOTIONAL_PHRASES)
    score += 0.1 * max(0, len(processed.urls) - 1)
    score += 0.1 * max(0, len(processed.cashtags) - 3)
    return min(1.0, score)


def build_text_signals(items: Iterable[TextItem], assets: Iterable[Asset]) -> list[TextSignal]:
    """Build point-in-time text signals with duplicate-aware novelty."""

    item_list = list(items)
    asset_list = list(assets)
    assets_by_id = {asset.asset_id: asset for asset in asset_list}
    ordered_items = sorted(item_list, key=lambda text: (text.available_at, text.item_id))
    duplicate_clusters = cluster_near_duplicates_incremental(
        (item.item_id, f"{item.title or ''}\n{item.body or ''}") for item in ordered_items
    )
    first_seen: set[tuple[str, str]] = set()
    signals: list[TextSignal] = []
    for item in ordered_items:
        entities = item.entities or link_entities(item, asset_list)
        sentiment = analyze_sentiment(item.title, item.body)
        event_type = item.event_type or infer_event_type(item.title, item.body)
        spam_score = estimate_spam_score(item.title, item.body)
        for entity in entities:
            if not entity.asset_id or entity.asset_id not in assets_by_id:
                continue
            novelty_key = (entity.asset_id, duplicate_clusters[item.item_id])
            novelty = 0.0 if novelty_key in first_seen else 1.0
            first_seen.add(novelty_key)
            signals.append(
                TextSignal(
                    item_id=item.item_id,
                    asset_id=entity.asset_id,
                    symbol=entity.symbol or assets_by_id[entity.asset_id].symbol,
                    asof_ts=item.available_at,
                    sentiment_score=sentiment.score,
                    sentiment_label=sentiment.label,
                    sentiment_confidence=sentiment.confidence,
                    relevance=entity.relevance,
                    novelty=novelty,
                    source_credibility=SOURCE_CREDIBILITY.get(item.source_type, 0.5),
                    model_version="finance-dictionary-v2",
                    source=item.source,
                    source_type=item.source_type,
                    author_hash=item.author_hash,
                    duplicate_cluster_id=duplicate_clusters[item.item_id],
                    available_at=item.available_at,
                    event_type=event_type,
                    spam_score=spam_score,
                )
            )
    return signals
