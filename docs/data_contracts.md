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
| Source to optional LLM signal | The source is available by the assigned decision, every cited span belongs to that source, the horizon matches configuration, and deterministic verification passes. |
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
- `label_start_ts`: the first configured-exchange session open strictly after `asof_ts`, used as the
  entry timestamp.
- `label_end_ts`: the exact configured-horizon exchange-session close used as the exit timestamp.
- `label_available_at`: when the complete outcome cross-section became observable. Generated labels
  set this to the latest required future bar availability, not merely `label_end_ts`. For external or
  legacy labels where it is absent, a generic `available_at` takes precedence over `label_end_ts`.

Modeling uses `available_at`, not `published_at`. The pipeline supports completed daily-bar decisions
only. Under the generic contract, a bar without separate availability metadata is treated as
available at its official close. Under `japan_cash_equity_v1`, the bar keeps that close as `ts` but
must supply its later-or-equal `available_at`; a complete session cross-section cannot be decided
before its latest required availability. The versioned XNYS/XJPX calendar handles holidays,
exceptional closures, early closes, and venue close changes. Calendar requests outside explicit
bounds fail. Entry is the first official open strictly after the decision, so data delayed past an
open rolls to a later session rather than being backdated.

Runtime `start_date` and `end_date` are inclusive filters on `asof_ts`. A date-only start resolves to
the beginning of that UTC date and a date-only end to its end; actual rows use the safe decision
timestamp, which may be later than the source bar close. The fetch interval expands backward by
`market_warmup_sessions` exchange sessions
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

For optional generative signals, the host assigns the first safe market decision at or after the
source `available_at` and supplies `features.horizon_days` as the target horizon. The verifier rejects
`source_available_at > decision_time` or a returned horizon that differs from the request. Annotation
stage completion time is audit metadata and never becomes historical feature availability.

## Core records

- `Asset`: stable asset ID, symbol, exchange, currency, name, sector/industry, active interval,
  optional CIK/FIGI/ISIN identifiers, and conservative short-availability/hard-to-borrow flags.
- `MarketBar`: asset, UTC official-close timestamp, raw tradable daily OHLCV, optional VWAP/adjusted
  close, and explicit causal return-adjustment metadata. The daily feature/label path requires
  `corporate_action_adjusted=true`, a timezone-aware causal adjustment vintage, and a positive
  `return_adjustment_factor` for every row. Without separate bar availability, the vintage cannot
  follow `ts`; when `available_at` is present, it cannot follow that timestamp. Return features and
  label outcomes compare
  `raw_price * return_adjustment_factor`; raw open/close remain the execution and trade-log prices,
  and raw price times volume remains the liquidity basis. The flag certifies that the adjustment
  metadata is complete; it does not mean the raw OHLC fields are rewritten. `adjusted_close` does
  not substitute for the required per-bar factor.

The Japanese cash-equity specialization additionally requires XJPX/JPY identity, canonical security
codes, explicit trading units and session dates, `price_basis=raw_tradable`, and
`ts <= adjustment_vintage_at <= available_at`. Its field-level and J-Quants V2 normalization
contract is documented in [Japan cash-equity baseline](japan_baseline.md).
- `TextItem`: source/type/language, permitted text or raw hash/path, publication/vendor/ingestion/
  availability timestamps, SHA-256 author/URL identifiers, entities, event fields, relationship type,
  hashed parent item, content status, retention permission, and terms reference.
- `EntityMention`: optional linked asset/symbol, name, relevance, mention type, and confidence.
- `TextSignal`: asset-linked conventional sentiment, confidence, relevance, novelty, credibility,
  source, deduplication, spam/disagreement/event fields, availability, and model version, plus
  optional separate LLM semantic, raw-confidence, uncertainty, event-confidence, evidence-count, and
  abstention fields.
- `EntityAnnotation`: one strict per-item/per-asset LLM result containing stance, integer semantic
  signal in `[-2, 2]`, explicitly uncalibrated raw confidence, uncertainty, configured horizon,
  primary event/confidence, source-local supporting and counterevidence span IDs, mechanism,
  invalidation conditions, and abstention state.
- `DecisionRound`: a content-identified audit record for one LLM request. It freezes source/model/
  prompt/schema/sampling identity, current-source evidence IDs, exact raw and structured output,
  verifier checks, generated/cache/deduplicated origin, and available token/latency/configured-cost
  usage. Its tool, calibration, portfolio, risk, and order fields are empty, and it has no realized
  outcome field.
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

Baseline-facing text counts, sentiment, event, source, velocity, and abnormal-attention features use
only the novelty-filtered independent evidence window. Raw repetition remains observable under
explicit `raw_*` diagnostics plus novelty/duplicate counts; `raw_*` fields are excluded from baseline
feature discovery.

Asset references are checked against the asset master. Market bars and supplied text entities with
an `asset_id` must match the canonical symbol and fall within the asset's exchange-local
`active_from`/`active_to` interval. A text entity without `asset_id` cannot emit a signal; a nonempty
supplied entity list also prevents automatic relinking. Optional point-in-time records are checked
for asset ID and symbol consistency; their feature use is still gated by the decision bar and their
own period/event and availability fields.

Between the first and last supplied market session, every asset active on a session must have a bar.
This whole-universe check prevents a missing bar from silently changing market/sector context or the
investable cross-section. Training and evaluation apply the same principle to outcomes: a complete
decision-and-horizon cross-section is included, a partial one fails, and a wholly censored group is
omitted only when every expected label end is beyond the final decision boundary.

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
silver/llm_annotations/symbol=AAPL/year=2026/part-*.parquet
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

Optional generative annotations are sidecar records keyed by `(item_id, asset_id)`. Raw model output
must be one strict JSON object whose only top-level field is `annotations`. Each valid per-asset v2
record contains:

- a closed-set stance and a sign-consistent integer `semantic_signal` from -2 through 2;
- `raw_confidence` and uncertainty in `[0, 1]`, with raw confidence explicitly uncalibrated;
- the exact configured positive `horizon_days`;
- at most one closed-set primary event plus confidence;
- unique, disjoint supporting and counterevidence IDs drawn from the current source's numbered
  `S<number>` spans;
- a nonempty mechanism and at least one unique invalidation condition for a non-abstained result; and
- strict empty/default values plus a reason for abstention.

The host supplies item, asset, source availability, decision time, horizon, source type/quality, and
provenance identity. Source quality is a noisy feature, not proof. Unknown/duplicate/missing assets,
unrecognized labels/events, invalid signal direction, non-finite or out-of-range values, and unknown,
duplicate, or overlapping evidence IDs fail validation. Every newly generated raw attempt is written
before strict parsing so a malformed response remains diagnosable; malformed output still fails the
stage before a validated cache record, Silver annotation, or DecisionRound is written.

The deterministic verifier checks exact item and candidate coverage, source availability no later
than decision time, requested/returned horizon equality, evidence-reference membership, and whether
each numeric token in a mechanism or invalidation condition occurs in the cited spans. Passing these
checks does not prove that the cited prose supports the interpretation or that a causal mechanism is
true.

Annotation availability never overrides source availability: every annotation retains the source
`available_at`, and feature construction still enforces `available_at <= asof_ts`. Generated text is
not accepted as factual input. In `sidecar` mode, annotations cannot change gold features. In
`augment` mode, valid annotations add distinct `llm_*` values; they never overwrite conventional
sentiment/event fields. Deterministic linking, relevance, novelty/deduplication, credibility, spam,
source, and availability fields remain intact. Transformer sentiment may therefore coexist with LLM
augmentation.

## Gold: features, labels, and predictions

Gold Parquet is partitioned by version/family and year. Feature and label generation are separate.
For a completed-bar decision, a label records the raw tradable `open` of the first
configured-exchange session strictly after `asof_ts` and the raw tradable `close` of the exact
`horizon_days` session as its execution prices. Its forward return
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

When `feature_mode: augment` is enabled, every configured text window also carries separate LLM
coverage, non-abstention/abstention, semantic, raw-confidence, uncertainty, event-confidence,
supporting/counterevidence, agreement, and explicit missingness aggregates. Raw confidence remains a
model feature, not a probability, semantic-signal magnitude, position size, or portfolio weight.

Enabled LLM runs write canonical JSONL DecisionRounds under
`<models root>/<run_id>/llm_decisions/rounds.jsonl`, then immediately replay and verify the file.
Replay rejects malformed/noncanonical JSON, schema or timestamp violations, duplicate round IDs, and
content hashes that do not match `round_id`. It reuses the stored stochastic output; it does not call
the model again or establish the semantic truth of that output. This ledger is distinct from both the
hash-chained paper-event ledger and the deterministic portfolio backtest replay.

## Local input requirements

CSV, JSON, JSONL, and Parquet source decoding is local-only. `configs/local.yaml` is a template for
full mode and expects the user to provide appropriately licensed files at its paths. Assets, market
bars, and text items are required. Fundamentals, earnings calendar, and corporate actions are
optional. Each configured source must be an existing file or a nonempty Parquet directory. Input
paths may not overlap any artifact root, and raw/interim/processed/model/report roots must be
mutually non-overlapping. The system does not download, revise, adjust, or infer missing vendor data.
When generative annotation is enabled, `paths.llm_model` must be a nonempty local model directory and
must not overlap an artifact root.

See [Outputs](outputs.md) for the materialized tree and [Research protocol](research_protocol.md) for
the human review checklist.

Return to the [documentation home](README.md).
