# Data Contracts

All in-memory timestamps are timezone-aware `datetime` values and all serialized timestamps are UTC
ISO-8601 strings. Parsers reject naive timestamps. Dates needed for exchange or partition semantics
are stored separately from timestamps.

Use this page for the invariants the pipeline enforces after parsing. If you are preparing source
files, start with [Input data](input_data.md), which lists accepted containers, required fields, and
examples in supplier-friendly form.

## Contract summary

| Boundary | Required guarantee |
|---|---|
| Input to bronze | Source bytes and provenance are captured without mutation. |
| Bronze to silver | Records are typed, UTC-normalized, canonically linked, and partitioned. |
| Silver to gold features | Every contributing value is available no later than the decision. |
| Silver to gold labels | The outcome starts strictly after the decision on official sessions. |
| Gold to model | Training includes only labels observable before the embargoed cutoff. |
| Prediction to replay | The whole decision cross-section is evaluated with costs and constraints. |

## Timestamp semantics

- `event_ts`: when the underlying event occurred, if known.
- `published_at`: the source-declared publication time.
- `vendor_received_at`: when the vendor made the item available.
- `ingested_at`: when this system recorded the input.
- `processed_at`: when a derived representation was produced, when recorded.
- `available_at`: the earliest defensible time the strategy could have used the information.
- `asof_ts`: the strategy decision timestamp to which a signal or feature belongs.
- `label_start_ts`: the exact next XNYS session open used as the entry timestamp.
- `label_end_ts`: the exact configured-horizon XNYS session close used as the exit timestamp.

Modeling uses `available_at`, not `published_at`. The pipeline supports close decisions only: the
decision is made after the full daily OHLC bar is complete. An item available by an official
XNYS close is assigned to that close, while a later item moves to the next session close. The
versioned exchange calendar handles holidays, exceptional closures, early closes, and daylight-
saving transitions. Calendar requests outside explicit bounds fail. A next-session-open fill is a
future execution assumption, not evidence that the closing feature row was available at the open.

Runtime `start_date` and `end_date` are inclusive filters on `asof_ts`. A date-only start resolves to
the beginning of that UTC date and a date-only end to its end; the actual rows are still official XNYS
close timestamps. The fetch interval expands backward by `market_warmup_sessions` exchange sessions
and `text_warmup_days` calendar days, and forward by `horizon_days` exchange sessions. Context is
available to builders and silver storage, but only decision rows inside the requested interval enter
gold outputs.

Known earnings-calendar and corporate-action events have a separate bounded future context: their
fetch ends `event_lookahead_days` calendar days after the last selected decision (30 by default).
An event with a later `event_ts` contributes to a decision only when its own
`available_at <= asof_ts`; the lookahead bound does not relax point-in-time availability.

`runtime.limit` counts complete decision timestamps after date and symbol filtering. It retains every
asset row at each selected timestamp and does not limit contextual market/text input records.

Every text-derived feature is built from signals satisfying `available_at <= asof_ts`. Feature rows
also record latest input availability by window, and both feature construction and the Parquet
feature store reject future provenance or duplicate canonical keys.

## Core records

- `Asset`: stable asset ID, symbol, exchange, currency, name, sector/industry, active interval, and
  optional CIK/FIGI/ISIN identifiers.
- `MarketBar`: asset, UTC official-close timestamp, raw tradable daily OHLCV, optional VWAP/adjusted
  close, and explicit causal return-adjustment metadata. The daily feature/label path requires
  `corporate_action_adjusted=true`, timezone-aware `adjustment_vintage_at <= ts`, and a positive
  `return_adjustment_factor` for every row. Return features and label outcomes compare
  `raw_price * return_adjustment_factor`; raw open/close remain the execution and trade-log prices,
  and raw price times volume remains the liquidity basis. The flag certifies that the adjustment
  metadata is complete; it does not mean the raw OHLC fields are rewritten. `adjusted_close` does
  not substitute for the required per-bar factor.
- `TextItem`: source/type/language, permitted text or raw hash/path, publication/vendor/ingestion/
  availability timestamps, SHA-256 author/URL identifiers, entities, event fields, relationship type,
  hashed parent item, content status, retention permission, and terms reference.
- `EntityMention`: optional linked asset/symbol, name, relevance, mention type, and confidence.
- `TextSignal`: asset-linked sentiment, confidence, relevance, novelty, credibility, source,
  deduplication, spam/disagreement/event fields, availability, and model version.
- `FeatureRow`, `LabelRow`, and `PredictionRow`: explicit as-of time, horizon, and feature/label/model
  version.
- `OrderIntent`: target-weight research intent with reason and risk fields. `PaperOrderIntent` carries
  reason codes; the simulator or CLI snapshot may attach risk flags separately. These are simulation
  records, not broker orders.
- `FundamentalRecord`, `EarningsCalendarEvent`, and `CorporateAction`: optional point-in-time records
  consumed end to end when their local paths are configured. They contribute only after
  `available_at <= asof_ts`. Corporate-action records provide known-event features; they do not
  adjust market prices on behalf of the provider.

The canonical feature key is:

```text
(asset_id, asof_ts, horizon, feature_set_version)
```

Labels and predictions carry `label_version` and `model_version`, respectively. Within a run, label
rows are unique on `(asset_id, asof_ts, horizon)`, and prediction rows are unique on that key within
each `model_family`; family partitions distinguish predictions produced by the same model version.
Duplicate market timestamps, feature keys, labels, and per-family predictions fail rather than being
silently deduplicated. Text duplication is different: items are ordered by
`(available_at, item_id)` and assigned incrementally to stable exact/near-duplicate clusters using
only prior documents and a bounded candidate set. A later bridge item never merges two historical
clusters.

Asset references are checked against the asset master. Market bars and supplied text entities with
an `asset_id` must match the canonical symbol and fall within the asset's exchange-local
`active_from`/`active_to` interval. A text entity without `asset_id` cannot emit a signal; a nonempty
supplied entity list also prevents automatic relinking. Optional point-in-time records are checked
for asset ID and symbol consistency; their feature use is still gated by the decision bar and their
own period/event and availability fields.

## Bronze: immutable source bytes

Local source files are hashed and copied byte-for-byte into the configured raw root in bounded-size
chunks, with SHA-256 filenames and exclusive create semantics. Identical payload/metadata ingestion
is idempotent; differing content is never written over an existing path. Every metadata sidecar
contains:

- `source`
- `vendor`
- `license_or_terms_ref`
- UTC `ingested_at`
- `request_id`
- payload `sha256` and byte count
- `schema_version`
- canonical `fetch_params_hash`
- relative payload path

Per-run bronze-reference manifests point to these payloads and sidecars. Raw files must only be
created through the ingestion store and must not be edited in place. Capture checks file or
partition-directory hashes against the initial run manifest before providers decode data. Providers
read the captured bytes, preventing a source edit during a run from silently changing the dataset.

## Silver: normalized records

Silver data is typed, normalized, and partitioned for pruning:

```text
silver/market/bar_size=1d/symbol=AAPL/year=2026/part-*.parquet
silver/text/source=news/date=2026-07-09/part-*.parquet
silver/text_signals/symbol=AAPL/year=2026/part-*.parquet
silver/fundamentals/symbol=AAPL/year=2026/part-*.parquet
silver/earnings_calendar/symbol=AAPL/year=2026/part-*.parquet
silver/corporate_actions/symbol=AAPL/year=2026/part-*.parquet
```

Text partitions may retain supplied titles and bodies. Callers must remove content they are not
licensed to retain or transform. Social author and URL identities should be stable hashes, not raw
handles or account URLs. The local pipeline hashes supplied raw author identifiers, URLs, and parent
item IDs before silver materialization; supplied hash fields must be lowercase SHA-256 values.
`relationship_type` is one of `original`, `repost`, `quote`, or `reply`; `content_status` is one of
`active`, `deleted`, `private`, `protected`, or `unknown`. A record explicitly marked
`retention_permitted=false` is rejected.

## Gold: features, labels, and predictions

Gold Parquet is partitioned by version/family and year. Feature and label generation are separate.
For a close decision, a label records the raw tradable `open` of the exact next XNYS session and the
raw tradable `close` of the exact `horizon_days` session as its execution prices. Its forward return
compares those prices after multiplying each by that bar's causal `return_adjustment_factor`, and
records `price_basis=causal_return_adjustment_factor`. The bar series must contain contiguous
official sessions; an internal missing or duplicate session is rejected. A trailing horizon that is
not yet complete is emitted with null outcomes and
`missing_required_session`, not shortened or filled. Supported generated targets include forward,
abnormal, sector-neutral, binary, rank, volatility, and volume outcomes.

Feature manifests are represented by the immutable run config, input hashes, version fields, and
final artifact manifest. Every feature/label/prediction table is local, content-named Parquet;
`storage_format` is fixed to `parquet`, while compression is configurable and may be disabled.
Evaluation, replay, model, and manifest records may be JSON. Filterable Parquet, CSV, and JSONL
source reads use lazy scans and streaming collection.
Those filtered results and downstream feature, label, and model rows are currently materialized in
Python, so full-mode windows must be sized to available local memory; the pipeline is not end-to-end
out-of-core.

## Local input requirements

CSV, JSON, JSONL, and Parquet source decoding is local-only. `configs/local.yaml` is a template for
full mode and expects the user to provide appropriately licensed files at its paths. Assets, market
bars, and text items are required. Fundamentals, earnings calendar, and corporate actions are
optional. Each configured source must be an existing file or a nonempty Parquet directory. Input
paths may not overlap any artifact root, and raw/interim/processed/model/report roots must be
mutually non-overlapping. The system does not download, revise, adjust, or infer missing vendor data.

See [Outputs](outputs.md) for the materialized tree and [Research protocol](research_protocol.md) for
the human review checklist.

Return to the [documentation home](README.md).
