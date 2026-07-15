from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9_]{1,30})")
HASHTAG_RE = re.compile(r"(?<![\w#])#([^\W_]{1,64})", re.UNICODE)
CASHTAG_RE = re.compile(r"(?<![\w$])\$([A-Za-z][A-Za-z0-9.]{0,9})(?![\w])")
UPPER_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5}(?:\.[A-Z])?)(?![A-Za-z0-9])")
TOKEN_RE = re.compile(r"\$[a-z][a-z0-9.]{0,9}|<[a-z]+>|[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

# Uppercase prose tokens are candidates, never confirmed entities. These common words
# are excluded before candidates reach the asset-master linker.
AMBIGUOUS_UPPERCASE_TOKENS = {
    "A",
    "ALL",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BY",
    "CAN",
    "CEO",
    "CFO",
    "COO",
    "EPS",
    "ETF",
    "FOR",
    "FROM",
    "GDP",
    "I",
    "IN",
    "IPO",
    "IS",
    "IT",
    "NO",
    "NOT",
    "OF",
    "ON",
    "OR",
    "SEC",
    "THE",
    "TO",
    "US",
    "USD",
    "WAS",
    "WE",
    "WITH",
}

_BOILERPLATE_PREFIXES = (
    "advertisement",
    "all rights reserved",
    "click here",
    "cookie policy",
    "copyright ",
    "for more information visit",
    "privacy policy",
    "read more",
    "sign up for our newsletter",
    "subscribe to our newsletter",
    "terms of use",
)


@dataclass(frozen=True, slots=True)
class PreprocessedText:
    """Deterministic, model-free representation of a text payload."""

    normalized_text: str
    tokens: tuple[str, ...]
    ticker_candidates: tuple[str, ...]
    cashtags: tuple[str, ...]
    urls: tuple[str, ...]
    hashtags: tuple[str, ...]
    mentions: tuple[str, ...]
    emojis: tuple[str, ...]


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _is_emoji(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0x2300 <= codepoint <= 0x23FF
    )


def _strip_boilerplate(value: str) -> str:
    kept: list[str] = []
    for line in value.splitlines():
        compact = " ".join(line.split()).strip()
        folded = compact.casefold().strip(" .:;-—")
        if compact and len(compact) <= 160 and folded.startswith(_BOILERPLATE_PREFIXES):
            continue
        if compact:
            kept.append(compact)
    return "\n".join(kept)


def preprocess_text(value: str) -> PreprocessedText:
    """Normalize text while retaining separately extracted market-relevant tokens.

    URLs, social handles, and emoji are replaced with stable placeholders. Hashtag
    content and cashtags remain in the normalized text so deterministic NLP can use
    their words without retaining raw handles or URLs.
    """

    unicode_text = unicodedata.normalize("NFKC", value)
    stripped = _strip_boilerplate(unicode_text)
    urls = _unique(match.group(0).rstrip(".,;:!?") for match in URL_RE.finditer(stripped))
    mentions = _unique(match.group(1).casefold() for match in MENTION_RE.finditer(stripped))
    hashtags = _unique(match.group(1).casefold() for match in HASHTAG_RE.finditer(stripped))
    cashtags = _unique(match.group(1).upper() for match in CASHTAG_RE.finditer(stripped))
    emojis = _unique(character for character in stripped if _is_emoji(character))

    uppercase = (
        match.group(1)
        for match in UPPER_TICKER_RE.finditer(stripped)
        if match.group(1) not in AMBIGUOUS_UPPERCASE_TOKENS
    )
    ticker_candidates = _unique((*cashtags, *uppercase))

    normalized = URL_RE.sub(" <url> ", stripped)
    normalized = MENTION_RE.sub(" <mention> ", normalized)
    normalized = HASHTAG_RE.sub(lambda match: f" {match.group(1)} ", normalized)
    normalized = "".join(
        " <emoji> " if _is_emoji(character) else character for character in normalized
    )
    normalized = normalized.replace("\ufe0f", "").replace("\u200d", " ")
    normalized = " ".join(normalized.casefold().split())
    tokens = tuple(TOKEN_RE.findall(normalized))
    return PreprocessedText(
        normalized_text=normalized,
        tokens=tokens,
        ticker_candidates=ticker_candidates,
        cashtags=cashtags,
        urls=urls,
        hashtags=hashtags,
        mentions=mentions,
        emojis=emojis,
    )


def canonicalize_text(value: str) -> str:
    """Return the canonical text used for hashing and duplicate comparison."""

    return preprocess_text(value).normalized_text


def exact_text_hash(value: str) -> str:
    """Return a content hash after deterministic preprocessing."""

    return hashlib.sha256(canonicalize_text(value).encode("utf-8")).hexdigest()


def _comparison_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in preprocess_text(value).tokens if not token.startswith("<"))


def near_duplicate_similarity(left: str, right: str) -> float:
    """Score near-duplicate text using token overlap and ordered token similarity."""

    left_tokens = _comparison_tokens(left)
    right_tokens = _comparison_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    union = left_set | right_set
    jaccard = len(left_set & right_set) / len(union) if union else 1.0
    ordered = SequenceMatcher(a=left_tokens, b=right_tokens, autojunk=False).ratio()
    return max(jaccard, ordered)


def cluster_near_duplicates(
    texts: Iterable[tuple[str, str]], *, threshold: float = 0.82
) -> dict[str, str]:
    """Assign exact and near-duplicate texts to deterministic content clusters."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be between 0 and 1")
    records = sorted(texts, key=lambda record: record[0])
    identifiers = [identifier for identifier, _ in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate text identifier")
    canonical = [canonicalize_text(value) for _, value in records]
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left_index, left_text in enumerate(canonical):
        for right_index in range(left_index + 1, len(canonical)):
            right_text = canonical[right_index]
            if (
                left_text == right_text
                or near_duplicate_similarity(left_text, right_text) >= threshold
            ):
                union(left_index, right_index)

    members: dict[int, list[str]] = {}
    for index, text in enumerate(canonical):
        members.setdefault(find(index), []).append(text)
    cluster_ids = {
        root: "text-" + hashlib.sha256(min(values).encode("utf-8")).hexdigest()[:16]
        for root, values in members.items()
    }
    return {identifier: cluster_ids[find(index)] for index, identifier in enumerate(identifiers)}


def cluster_near_duplicates_incremental(
    texts: Iterable[tuple[str, str]],
    *,
    threshold: float = 0.82,
    max_candidates: int = 256,
) -> dict[str, str]:
    """Assign stable clusters in input order without using future documents.

    Existing clusters are never merged when a later bridge document resembles more
    than one of them. This makes each prior assignment reproducible using only the
    corpus that was available at that point in time.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be between 0 and 1")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    seen_ids: set[str] = set()
    prior: list[tuple[str, str]] = []
    exact_clusters: dict[str, str] = {}
    postings: dict[tuple[str, ...], list[int]] = {}
    assignments: dict[str, str] = {}
    for identifier, value in texts:
        if identifier in seen_ids:
            raise ValueError("duplicate text identifier")
        seen_ids.add(identifier)
        canonical = canonicalize_text(value)
        tokens = _comparison_tokens(canonical)
        shingles = (
            {tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)}
            if len(tokens) >= 3
            else {tokens}
        )
        candidate_indexes: set[int] = set()
        for shingle in sorted(shingles, key=lambda value: (len(postings.get(value, [])), value)):
            candidate_indexes.update(postings.get(shingle, []))
            if len(candidate_indexes) >= max_candidates:
                break
        ordered_candidates = sorted(candidate_indexes)[-max_candidates:]
        candidates: list[tuple[float, str]] = []
        for index in ordered_candidates:
            prior_text, cluster_id = prior[index]
            similarity = (
                1.0 if canonical == prior_text else near_duplicate_similarity(canonical, prior_text)
            )
            if similarity >= threshold:
                candidates.append((similarity, cluster_id))
        if canonical in exact_clusters:
            cluster_id = exact_clusters[canonical]
        elif candidates:
            best_similarity = max(similarity for similarity, _ in candidates)
            cluster_id = min(
                cluster for similarity, cluster in candidates if similarity == best_similarity
            )
        else:
            cluster_id = "text-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        assignments[identifier] = cluster_id
        exact_clusters.setdefault(canonical, cluster_id)
        index = len(prior)
        prior.append((canonical, cluster_id))
        for shingle in shingles:
            postings.setdefault(shingle, []).append(index)
    return assignments
