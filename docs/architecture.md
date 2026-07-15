# Architecture

NLP Trader is a dependency-aware local pipeline. A pipeline execution command creates one unique
`run_id`, snapshots the typed config and input hashes, executes the requested stage and its
prerequisites, then writes an exclusive final or failure manifest. Existing runs are never resumed
or overwritten.

This page explains component boundaries and data flow. For the practical command sequence, start
with [Workflows](workflows.md). For file formats and exact artifact locations, see
[Input data](input_data.md) and [Outputs](outputs.md).

## At a glance

- The baseline and bundled configs run from local files; imports and constructors do not fetch data.
- Bronze source bytes are shared and append-only. Silver, gold, models, and reports belong to one
  immutable run.
- Features are completed-daily-bar decision snapshots. Labels begin at the first configured-exchange
  open strictly after the safe decision timestamp and are built separately.
- Each modeling family uses an expanding walk-forward baseline, then the same constrained and
  cost-aware replay.
- Optional local generative annotation is a validated sidecar stage. It is disabled by default and
  changes features only when application is separately enabled.
- The kabuS broker adapter is a standalone operator-invoked boundary, not a pipeline stage or a
  consumer of research predictions or paper intents.
- The current filtered working set is materialized in Python, so full mode is not end-to-end
  out-of-core.

## Pipeline

```text
licensed local files or synthetic fixtures
        |
        v
append-only bronze payloads + provenance sidecars
        |
        v
silver assets, market bars + causal return-adjustment metadata, normalized text,
optional fundamentals/calendar/actions, optional generative annotation sidecars, and text signals
        |
        v
gold availability-safe daily decisions + later-open-to-horizon-close labels
        |
        v
incremental expanding walk-forward models and predictions
        |
        v
constrained next-open round trips / pending paper intents
        |
        v
research note + immutable run manifest
```

The stages are `ingest_market`, `ingest_text`, `annotate_text`, `build_features`, `build_labels`,
`train`, `predict`, `backtest`, `paper`, and `report`. `annotate_text` depends on both ingestion
stages, and `build_features` depends on it. Disabled annotation is a no-op. `smoke` targets `report`;
`paper` is deliberately separate. Calling an individual CLI stage still runs all of that stage's
dependencies in the same new run.

## Storage layout

Each configured artifact root is extended with the unique timestamp-and-content run ID. The bundled
YAML files include the mode in those roots, but the runtime does not insert it automatically:

```text
<configured raw root>/
  source=<role>/date=<UTC-date>/<sha-prefix>/<sha256>.<ext>
  _metadata/source=<role>/<payload-sha>/<metadata-sha>.json
<configured interim root>/<run_id>/
  bronze_refs/*.json
  silver/assets/*.parquet
  silver/market/bar_size=<size>/symbol=<symbol>/year=<year>/part-*.parquet
  silver/text/source=<source>/date=<date>/part-*.parquet
  silver/fundamentals/symbol=<symbol>/year=<year>/part-*.parquet
  silver/earnings_calendar/symbol=<symbol>/year=<year>/part-*.parquet
  silver/corporate_actions/symbol=<symbol>/year=<year>/part-*.parquet
<configured processed root>/<run_id>/
  silver/llm_annotations/symbol=<symbol>/year=<year>/part-*.parquet
  silver/text_signals/symbol=<symbol>/year=<year>/part-*.parquet
  gold/features/feature_set_version=<version>/year=<year>/part-*.parquet
  gold/labels/label_version=<version>/year=<year>/part-*.parquet
  gold/predictions/model_family=<family>/year=<year>/part-*.parquet
  evaluation/*.json
  backtests/<family>/backtest.json
  backtests/<family>/final_holdout.json
  paper/snapshot.json
  paper/events.jsonl
<configured model root>/<run_id>/
  llm_annotations/prompt.txt
  llm_annotations/schema.json
  llm_annotations/provenance.json
  llm_annotations/responses/*.json
  model_version=<version>/
<configured model root>/_cache/llm_annotations/*.json
<configured report root>/<run_id>/
  config.snapshot.json
  run.initial.json
  run.final.json | run.failed.json
  research_note.md
```

Bronze payloads and sidecars use create-once writes. File ingestion hashes and copies in one-megabyte
chunks so preserving a large source does not require loading it all into memory. Parquet part names
are content-derived. Compression is configurable and defaults to Zstandard in the bundled configs.
Derived analytical storage is Parquet only; JSON files in the tree are manifests, metrics, or replay
records rather than alternative table storage. The shared row writer used for silver tables and gold
labels/predictions flushes immutable Parquet batches at `data.write_batch_rows` rows—10,000 in the
bundled configs—and infers its stable schema without retaining a second normalized copy of the full
table. Feature materialization writes its grouped partitions separately.

Local Parquet files or partition directories, CSV, and JSONL providers push equality, time, and row-
limit filters into Polars lazy scans and collect with the streaming engine. Plain JSON falls back to
a materialized local read. Configured inputs remain user-managed; ingestion preserves an immutable
byte-for-byte copy in bronze. Validation prevents input paths and the five write roots from
overlapping, and prevents write roots from containing one another. Capture verifies each file against
the run-start input manifest; providers then read the immutable bronze payload (or a symlinked
captured Parquet tree), not the mutable configured path.

The provider interfaces and current feature, label, and walk-forward stages still return materialized
Python row collections. Full mode should therefore use date, symbol, and decision limits sized to
local memory; it is not yet an end-to-end out-of-core pipeline. Lazy scans, bounded bronze capture,
and batched writes reduce I/O pressure but do not remove this working-set limit.

Every dependency stage logs its start, completion, elapsed time, and current asset/bar/text/feature/
label counts so a long local run exposes forward progress without mutating its config or artifacts.

## Module boundaries

- `config.py` owns strict configuration, local path resolution, the reserved environment-secret
  schema, and config hashes. Current local providers require no credentials.
- `providers.py` defines `MarketDataProvider`, `TextDataProvider`, `FundamentalsProvider`, and
  `CalendarProvider`, plus local file implementations. The pipeline wires required asset/market/text
  files and optional fundamentals, earnings-calendar, and corporate-action files end to end.
- `calendars.py` maps XNYS and XJPX sessions using explicit calendar bounds. Decisions occur only
  after a completed official close and required market-data availability; label execution begins at
  the first configured-exchange open strictly after the decision.
- `data/` owns local decoding, immutable raw storage, Parquet materialization, feature-store
  validation, model artifacts, deterministic synthetic fixtures, and the strict local Japanese
  cash-equity record contract. It contains no vendor downloader or bundled real data.
- `nlp/` owns deterministic preprocessing, entity linking, sentiment, incremental point-in-time
  deduplication with a bounded candidate search, optional cached local transformer inference, and
  strict local generative entity/event annotation.
- `features/` owns market/text aggregation and label generation. Labels never run inside feature
  construction. The daily builders reject OHLC without the required positive causal return factor,
  point-in-time adjustment provenance, or complete/unique internal exchange sessions. Return features
  and labels use `raw_price * return_adjustment_factor`; execution prices remain raw.
- `models/` owns incremental expanding development snapshots, the frozen pre-boundary final-holdout
  fit, the fixed traditional/text/combined families, naive benchmarks, calibration tables, and
  segmented prediction diagnostics.
- `portfolio/` owns eligibility, construction, exposure calculation, and constraint enforcement.
- `backtest/` owns independent non-overlapping next-open-to-horizon-close rounds, two-sided cost
  accounting, forced liquidation, period diagnostics, position/trade logs, and performance metrics.
- The pipeline's `paper` stage owns pending next-open intents plus append-only hash-chained evidence
  and does not call a fill simulator. `paper/` contains the reusable ledger and a separate in-memory
  simulator utility; neither path invokes the standalone broker adapter.
- `broker/` owns the strict limit-order contract, fixed loopback client, preflight controls,
  reconciliation, and fixed current-user safety state. Its audit ledger, kill switch, and stable
  operation lock are shared across broker configs and environments. It is reached only through the
  `broker` CLI group and has no dependency edge from the research pipeline.
- `research.py` and `reports.py` own run provenance, artifact hashes, final manifests, and readable
  reports.

Provider and store protocols keep vendor, storage, calendar, and model implementations replaceable
without coupling them to feature or backtest logic. No constructor or import performs external data
access.

## Execution modes

`sample` mode is a minutes-scale synthetic smoke path. `full` mode uses the same pipeline with
user-provided licensed files, larger local datasets, runtime date/symbol/decision filters, and more
conservative research defaults. Optional point-in-time data paths remain unset in the bundled
configs, and `paths.llm_model` defaults to null; users supply them only when appropriate. Full mode
does not provision data, credentials, a model, or a vendor connection.

Runtime dates bound emitted decision rows. For a requested start, ingestion expands backward by the
configured market-session warm-up and text calendar-day warm-up. For a requested end, market
ingestion expands forward by the label horizon in exchange sessions. Features and labels are built
with that context, then filtered back to the inclusive runtime interval. This keeps rolling features
and terminal labels intact while allowing silver context and text signals to sit outside the gold
decision period.

Known future earnings and corporate-action events are fetched only through
`event_lookahead_days` calendar days after the last selected decision (30 by default). Their event
timestamps may be later than a decision, but the feature path still requires
`available_at <= asof_ts`; the bounded fetch supplies proximity context without an unbounded future
event scan.

The runtime row limit selects the earliest complete decision timestamps, not raw source rows. The
pipeline then fetches the required context around those decisions and keeps all selected assets at
each timestamp, so development limits do not truncate warm-up history or split cross-sections.

Optional transformer dependencies are isolated behind the `nlp` extra. Device selection is
centralized: MPS is used when available, otherwise CPU. CUDA is not assumed. The smoke CLI's
transformer flag is folded into the typed config before run creation, so config hashing and snapshots
record the enabled state.

The generative annotator uses the same optional dependency/device boundary. Its host constructs one
source-grounded request per text item from deterministic, historically active asset candidates and
numbered evidence spans. No price, label, later-document, web, or retrieval context enters the
request. Strict validation rejects malformed responses and evidence/asset mismatches. Oversized
requests abstain rather than silently truncating source text. In sidecar mode nothing downstream is
changed; apply mode replaces only validated, non-abstained per-entity sentiment/event fields while
retaining deterministic provenance, relevance, novelty/deduplication, credibility, and spam fields.

Model loading is local-files-only with remote custom code disabled. The local model directory is a
hashed run input. Cache keys include exact request/candidate/model-directory and generation-contract
identity, and every consumed response is copied under the run model root so the final artifact
manifest can hash it. The source timestamp controls feature availability; annotation-stage completion
time is audit metadata. Because pretraining may contain later facts, this is recorded as a retrospective
parser rather than proof of historically deployable inference.

There is no scraping, data-vendor adapter, or autonomous strategy-to-order path. The standalone
kabuS command group can transmit operator-prepared cash-equity orders, but the pipeline, backtests,
reports, and paper stage never invoke it or convert their outputs into its order schema. It must run
on the same Windows PC as kabuStation; validation uses fixed responses, while production can place
real orders. See [Broker integration](broker.md).

Optional transformer loading can contact a model hub only when a user deliberately sets
`local_files_only: false`; bundled configs keep it true. The raw next-session opening price is a
backtest fill assumption, not an open-time strategy decision or auction model. Return calculations
use the bar's causal return-adjustment factor, while fill and trade-log prices remain raw. Open or
intraday decisions are outside the current daily-bar pipeline and require an intraday-safe provider,
feature path, and execution model.

## Related documentation

- [Data contracts](data_contracts.md) — invariants at every storage boundary
- [Features and models](features_and_models.md) — feature and training behavior
- [Backtesting](backtesting.md) — replay, costs, and constraints
- [Broker integration](broker.md) — standalone live-order boundary and operational safeguards
- [Development](development.md) — repository map and extension checklists

Return to the [documentation home](README.md).
