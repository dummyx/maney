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
- Features are close-decision snapshots. Labels begin at the next XNYS open and are built
  separately.
- Each modeling family uses an expanding walk-forward baseline, then the same constrained and
  cost-aware replay.
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
optional fundamentals/calendar/actions, and text signals
        |
        v
gold close-decision features + next-open-to-horizon-close labels
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

The stages are `ingest_market`, `ingest_text`, `build_features`, `build_labels`, `train`, `predict`,
`backtest`, `paper`, and `report`. `smoke` targets `report`; `paper` is deliberately separate. Calling
an individual CLI stage still runs all of that stage's dependencies in the same new run.

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
  silver/text_signals/symbol=<symbol>/year=<year>/part-*.parquet
  gold/features/feature_set_version=<version>/year=<year>/part-*.parquet
  gold/labels/label_version=<version>/year=<year>/part-*.parquet
  gold/predictions/model_family=<family>/year=<year>/part-*.parquet
  evaluation/*.json
  backtests/<family>/backtest.json
  paper/snapshot.json
<configured model root>/<run_id>/model_version=<version>/
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
- `calendars.py` maps information availability to official XNYS closes using explicit calendar
  bounds. Decisions occur after a completed official close; label execution begins at the next
  session open.
- `data/` owns local decoding, immutable raw storage, Parquet materialization, feature-store
  validation, model artifacts, and deterministic synthetic fixtures.
- `nlp/` owns deterministic preprocessing, entity linking, sentiment, incremental point-in-time
  deduplication with a bounded candidate search, and optional cached local transformer inference.
- `features/` owns market/text aggregation and label generation. Labels never run inside feature
  construction. The daily builders reject OHLC without the required positive causal return factor,
  point-in-time adjustment provenance, or complete/unique internal XNYS sessions. Return features
  and labels use `raw_price * return_adjustment_factor`; execution prices remain raw.
- `models/` owns incremental expanding walk-forward baseline snapshots, the fixed traditional/text/
  combined families, naive benchmarks, calibration tables, and segmented prediction diagnostics.
- `portfolio/` owns eligibility, construction, exposure calculation, and constraint enforcement.
- `backtest/` owns independent non-overlapping next-open-to-horizon-close rounds, two-sided cost
  accounting, forced liquidation, period diagnostics, position/trade logs, and performance metrics.
- The pipeline's `paper` stage owns pending next-open intents and does not call a fill simulator.
  `paper/` contains a separate in-memory simulator utility; neither path exposes a broker adapter.
- `research.py` and `reports.py` own run provenance, artifact hashes, final manifests, and readable
  reports.

Provider and store protocols keep vendor, storage, calendar, and model implementations replaceable
without coupling them to feature or backtest logic. No constructor or import performs external data
access.

## Execution modes

`sample` mode is a minutes-scale synthetic smoke path. `full` mode uses the same pipeline with
user-provided licensed files, larger local datasets, runtime date/symbol/decision filters, and more
conservative research defaults. The optional local point-in-time paths are not present in the bundled
configs; users add them when such datasets are available. Full mode does not provision data,
credentials, or a vendor connection.

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

There is no scraping, data-vendor adapter, broker API, live order route, or autonomous execution
path. Optional transformer loading can contact a model hub only when a user deliberately sets
`local_files_only: false`; bundled configs keep it true. The raw next-session opening price is a
backtest fill assumption, not an open-time strategy decision or auction model. Return calculations
use the bar's causal return-adjustment factor, while fill and trade-log prices remain raw. Open or
intraday decisions are outside the current daily-bar pipeline and require an intraday-safe provider,
feature path, and execution model.

## Related documentation

- [Data contracts](data_contracts.md) — invariants at every storage boundary
- [Features and models](features_and_models.md) — feature and training behavior
- [Backtesting](backtesting.md) — replay, costs, and constraints
- [Development](development.md) — repository map and extension checklists

Return to the [documentation home](README.md).
