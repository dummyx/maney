# Input Data Guide

This guide describes the local files accepted by the current providers. It is intended for the
person preparing a full-mode dataset.

> Data that parses successfully is not automatically point-in-time correct or legally usable. You
> remain responsible for source rights, historical availability, revision history, and universe
> construction.

## Supported containers

Each configured input may be:

| Format | Notes |
|---|---|
| CSV | Header row required. Nested values such as `entities` or `values` must be JSON strings. |
| JSON | One object or an array of objects. Materialized before filtering. |
| JSONL | One object per nonblank line. |
| Parquet | One file with matching field names and compatible scalar/list values. |
| Parquet directory | A nonempty directory scanned recursively for `*.parquet`. |

Parquet, CSV, and JSONL inputs support filter pushdown where the format permits it. The later
feature/model stages still materialize their filtered working sets, so size full-mode intervals to
available memory.

## Universal rules

- All timestamps must include a timezone. Use UTC `Z` timestamps when possible.
- `asset_id` is the canonical identity. `symbol` must agree with the asset master.
- Symbols used in runtime filters are uppercase.
- Numeric values must be finite.
- Required strings must be nonempty.
- Market bars and asset-ID prelinked text entities must fall within the asset’s exchange-local active
  interval. Optional fundamentals/events currently receive asset-ID and symbol checks but not an
  active-interval check; validate that condition in the source preparation process.
- Do not place input files inside raw/interim/processed/model/report artifact roots.
- Do not edit an input while a run is capturing it; the run will detect a hash mismatch and fail.

## Required input 1: asset master

Minimal CSV:

```csv
asset_id,symbol,exchange,currency,name,sector,active_from,active_to,short_available,hard_to_borrow
asset_aapl,AAPL,XNAS,USD,Apple Inc.,Technology,1980-12-12,,false,false
```

| Field | Required | Type | Meaning |
|---|---:|---|---|
| `asset_id` | yes | string | Stable canonical identity. Do not recycle it across companies. |
| `symbol` | yes | string | Human-readable tradable symbol. |
| `exchange` | yes | string | Exchange identifier associated with the asset. |
| `currency` | yes | string | Trading/reporting currency identifier. |
| `name` | yes | string | Canonical company or asset name used by baseline entity linking. |
| `sector` | yes | string | Sector used by features, constraints, and diagnostics. |
| `active_from` | no | ISO date | First active exchange-local date. |
| `active_to` | no | ISO date | Final active exchange-local date. Must not precede `active_from`. |
| `cik`, `figi`, `isin` | no | string | External identifiers when available. |
| `industry` | no | string | More detailed classification. |
| `trading_unit` | no | positive integer | Tradable lot size. Required by `japan_cash_equity_v1` and checked against every Japanese bar. |
| `short_available` | no | boolean | Point-in-time-valid permission to model a short. Missing values default to `false`. |
| `hard_to_borrow` | no | boolean | Marks an available short as hard to borrow. When omitted, defaults to `true` for a shortable asset and `false` otherwise. |

Keep delisted assets and historical symbol/identity changes when the research period requires them.
A current-only asset list creates survivorship bias.

For every exchange session between the first and last in-scope market bar, each asset whose active
interval includes that session must have a bar. The feature and label builders fail on a missing
active asset rather than silently shrinking the cross-section. Borrow fields are static within one
asset-master input; split runs or provide properly vintaged inputs when availability changes over
time. Present-day locate status must not be projected backward.

## Required input 2: daily market bars

Minimal CSV:

```csv
asset_id,symbol,ts,bar_size,open,high,low,close,volume,vwap,adjusted_close,corporate_action_adjusted,adjustment_vintage_at,return_adjustment_factor
asset_aapl,AAPL,2026-07-13T20:00:00Z,1d,210.00,214.00,209.50,213.25,50000000,212.10,,true,2026-07-13T20:00:00Z,1.0
```

| Field | Required | Type | Meaning |
|---|---:|---|---|
| `asset_id`, `symbol` | yes | string | Must match one asset-master row. |
| `ts` | yes | aware timestamp | Official session-close timestamp for the market event. It may precede the decision when delivery is delayed. |
| `bar_size` | yes | string | Current pipeline requests `1d`. |
| `open`, `high`, `low`, `close` | yes | positive float | Raw tradable prices. OHLC consistency is validated. |
| `volume` | yes | non-negative integer | Raw volume. |
| `vwap` | no | positive float | Optional reference value. |
| `adjusted_close` | no | positive float | Stored in silver only; it does not satisfy the causal adjustment contract. |
| `corporate_action_adjusted` | yes for feature/label runs | boolean | Certifies that the causal adjustment metadata is present and complete. It does not mean raw OHLC was rewritten. |
| `adjustment_vintage_at` | yes when the flag is true | aware timestamp | When that causal factor vintage became usable. Without a separate bar `available_at`, it must be no later than `ts`; otherwise it must be no later than `available_at`. |
| `return_adjustment_factor` | yes | positive float | Causal per-bar factor used to compare returns across corporate actions. |
| `exchange`, `currency` | no generally | string | Required as `XJPX` and `JPY` by `japan_cash_equity_v1`. |
| `trading_unit` | no generally | positive integer | Required by the Japanese contract and must match the asset master. |
| `session_date` | no generally | ISO date | Required by the Japanese contract and must match `ts` in Asia/Tokyo. |
| `available_at` | no generally | aware timestamp | Required by the Japanese contract; earliest defensible availability of this exact bar payload. |
| `price_basis` | no generally | `raw_tradable` | Required by the Japanese contract; certifies the OHLC fields are unadjusted tradable values. |

The two price concepts are intentionally separate:

- simulated fills and liquidity use raw `open`, `close`, and raw price × volume;
- return features and labels compare `raw_price * return_adjustment_factor`.

The provider must construct the factor from information available at the recorded vintage. A
retrospectively adjusted series without revision provenance is not acceptable.

Bars must be unique and internally contiguous on the configured exchange calendar. A missing
session inside an asset series or a missing active asset in a session cross-section fails
feature/label construction; the pipeline does not invent a bar or silently remove an asset.

### Strict Japanese cash-equity input

Select `data.calendar: XJPX` together with
`data.market_contract: japan_cash_equity_v1`. This contract requires exact XJPX/JPY metadata, a
canonical four-character Japanese security code, an explicit trading unit, session date, official
close `ts`, later-or-equal `available_at`, raw-tradable price basis, and causal adjustment vintage.
It rejects unknown columns rather than silently discarding source ambiguity.

For the exact field lists, J-Quants V2 mapping, after-close delivery semantics, and an empty-text
market-only starting point, see [Japan cash-equity baseline](japan_baseline.md).

## Required input 3: text items

Minimal JSONL record:

```json
{"item_id":"news-001","source":"licensed_news","source_type":"news","language":"en","title":"Apple updates guidance","body":"Management raised its revenue outlook.","published_at":"2026-07-13T14:00:00Z","vendor_received_at":"2026-07-13T14:00:03Z","ingested_at":"2026-07-13T14:01:00Z","available_at":"2026-07-13T14:00:03Z","license_or_terms_ref":"vendor-contract-2026"}
```

### Required fields

| Field | Type | Meaning |
|---|---|---|
| `item_id` | string | Stable item identity. |
| `source` | string | Source/vendor channel used for partitions and diagnostics. |
| `source_type` | enum | `news`, `social`, `filing`, `transcript`, `blog`, `forum`, or `other`. |
| `language` | string | Language identifier. |
| `published_at` | aware timestamp | Source-declared publication time. |
| `ingested_at` | aware timestamp | When the local system recorded it. |
| `available_at` | aware timestamp | Earliest defensible strategy availability. Must not precede publication or vendor receipt. |
| `license_or_terms_ref` | string | Rights/terms reference for this item. |

At least one of `title`, `body`, `raw_text_hash`, or an appropriately resolvable `raw_text_path`
should carry the content identity needed by your workflow. The baseline feature path uses supplied
title/body text.

Timestamp ordering is validated: `vendor_received_at` may not precede `published_at`, `available_at`
may not precede either publication or vendor receipt, and `processed_at` may not precede
`ingested_at`.

### Common optional fields

| Field | Type | Notes |
|---|---|---|
| `title`, `body` | string | Retain only when permitted. |
| `vendor_received_at` | aware timestamp | Stronger source-availability evidence. |
| `processed_at`, `event_ts` | aware timestamp | Derived processing or underlying event time. |
| `raw_text_hash`, `canonical_text_hash` | lowercase SHA-256 | Content identity. Canonical hash is computed from title/body when absent. |
| `author_hash`, `url_hash` | lowercase SHA-256 | Preferred privacy-preserving identifiers. |
| `author_id`, `author_handle`, `author`, `url` | string | Accepted convenience fields; the local provider hashes them before silver output. Raw bronze still retains them. |
| `relationship_type` | enum | `original`, `repost`, `quote`, or `reply`; default `original`. |
| `parent_item_id_hash` | lowercase SHA-256 | Parent relationship identity. Raw `parent_item_id` is accepted and hashed. |
| `content_status` | enum | `active`, `deleted`, `private`, `protected`, or `unknown`; default `unknown`. |
| `retention_permitted` | boolean | Defaults true. An explicit false is rejected. This is a declared fact, not legal proof. |
| `event_type` | string | Optional event classification. |
| `entities` | list | Optional prelinked `EntityMention` records. |

Entity example:

```json
{
  "asset_id": "asset_aapl",
  "symbol": "AAPL",
  "name": "Apple",
  "relevance": 0.95,
  "mention_type": "primary",
  "confidence": 0.98
}
```

`relevance` and `confidence` must be between 0 and 1. `mention_type` is `primary`, `secondary`, or
`incidental`. A prelink needs `asset_id` to contribute a text signal. Asset-ID prelinks are checked
for a matching optional symbol, membership in the filtered asset master, and asset activity. A
nonempty `entities` list suppresses automatic linking, so a symbol-only entry is ignored rather than
resolved. If `entities` is absent or empty, the deterministic linker uses the filtered asset master,
company aliases, symbols, and cashtags.

### Optional generative annotation input

The optional local generative annotator receives one normalized `TextItem` at a time, the
deterministically linked and historically active asset candidates, and host-numbered source-text
spans. It does not receive labels, prices, returns, later documents, or external retrieval results.
The model must select evidence span IDs from that supplied text; it cannot introduce another asset
or cite an invented span. Title/body content must therefore be present and permitted for local model
processing when this path is enabled.

The annotation inherits the source item’s `available_at` for research feature timing. The actual
annotation-stage completion time is audit metadata, not evidence that the chosen model existed or
ran at that historical instant. This is a retrospective parsing assumption; see
[Research protocol](research_protocol.md) before interpreting a historical run.

The model itself is also a local input. `paths.llm_model` must point directly to one immutable GGUF
file. The runtime records its exact SHA-256 as `model_file_sha256` and does not download or resolve a
model-hub selector. The bundled default expects `Qwen3.6-27B-UD-Q4_K_XL.gguf` from selector
`unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL`, revision
`5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf`, with SHA-256
`4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095`. The file is about
17.9 GB; model weights plus inference working memory make 32 GB or more of unified memory a practical
starting point. See [Workflows](workflows.md) for the explicit download and verification steps.

### Social-data warning

Hashing raw identifiers in silver does not erase them from bronze. The bronze store preserves the
entire configured file byte-for-byte. Ingest only content whose local retention is permitted, and
keep the raw root private.

## Optional input: fundamentals

JSONL example:

```json
{"asset_id":"asset_aapl","symbol":"AAPL","period_end":"2026-03-31","available_at":"2026-05-01T12:00:00Z","filing_id":"10q-2026-q1","values":{"book_to_market":0.18,"return_on_equity":0.31,"market_cap":3200000000000}}
```

| Field | Required | Type |
|---|---:|---|
| `asset_id`, `symbol` | yes | string |
| `period_end` | yes | ISO date |
| `available_at` | yes | aware timestamp |
| `values` | no | mapping, or JSON object string in a flat container; defaults to empty |
| `filing_id` | no | string |

The current feature builder recognizes these aliases and retains availability provenance:

- value: `book_to_market`, `value_proxy`, or `value`;
- quality: `return_on_equity`, `quality_proxy`, or `gross_profitability`; and
- size: `market_cap`.

An empty `values` mapping parses but produces no useful fundamental value. Provider acceptance does
not prove that a revised fundamental series is point-in-time.

## Optional input: earnings calendar

```json
{"asset_id":"asset_aapl","symbol":"AAPL","event_ts":"2026-07-30T20:00:00Z","available_at":"2026-06-15T12:00:00Z","status":"confirmed"}
```

Required fields are `asset_id`, `symbol`, `event_ts`, and `available_at`. `status` is `estimated`,
`confirmed`, or `reported` and defaults to `estimated`.

A future `event_ts` can create a proximity feature only when the event record was already available
at the decision. The configured event lookahead bounds how far the provider scans.

## Optional input: corporate-action events

```json
{"asset_id":"asset_aapl","symbol":"AAPL","event_ts":"2026-08-10T13:30:00Z","available_at":"2026-07-01T12:00:00Z","action_type":"ex_dividend","value":0.25}
```

Required fields are `asset_id`, `symbol`, `event_ts`, `available_at`, and `action_type`; `value` is an
optional finite float. `dividend`, `ex_dividend`, and `cash_dividend` currently create the
ex-dividend proximity feature. Other action types are preserved in silver but ignored by the current
feature builder. Event records never calculate or replace the market bar’s causal
`return_adjustment_factor`.

## Pre-run checklist

- [ ] Required files exist and parse locally.
- [ ] Every timestamp is timezone-aware.
- [ ] `available_at` reflects historical usability, not the current download time.
- [ ] Asset IDs and symbols agree across files.
- [ ] The asset master covers historical listings and delistings needed by the study.
- [ ] Market bars contain raw tradable OHLC and causal adjustment provenance.
- [ ] Social identifiers and retention status follow source terms.
- [ ] License/terms references are meaningful and current.
- [ ] Input paths are outside artifact roots.
- [ ] If generative annotation is enabled, its direct local GGUF path, SHA-256, exact revision, and
  model license/terms reference are recorded, and the file is outside artifact roots.
- [ ] The requested interval includes enough source history and future label bars.

Run:

```bash
uv run nlp-trader validate-config --config configs/local.yaml
```

Then read [Data contracts](data_contracts.md) for the exact derived-data and point-in-time rules.

Return to the [documentation home](README.md).
