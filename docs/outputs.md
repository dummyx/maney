# Outputs and Artifacts

Each pipeline execution command creates a unique run. This page shows where its files live and which
artifact answers which question.

## Find the run

The CLI prints the `run_id` and every output path produced by the requested stage. A run ID combines
UTC creation time with a config/content-derived suffix.

Artifact paths are:

```text
<configured interim root>/<run_id>/...
<configured processed root>/<run_id>/...
<configured models root>/<run_id>/...
<configured reports root>/<run_id>/...
```

The bundled configs already include `sample`, `backtest`, or `local` in their configured roots, so
their paths look like `reports/sample/<run_id>/`. This naming comes from the YAML paths, not from an
extra automatic mode directory.

The shared content-addressed raw root is different: identical source bytes can be referenced by many
runs without being copied into every run directory.

## Start with these files

| Question | Artifact |
|---|---|
| Did the run complete? | `reports.../<run_id>/run.final.json` |
| What config actually ran? | `reports.../<run_id>/config.snapshot.json` |
| What should a human read? | `reports.../<run_id>/research_note.md` |
| Which captured source inputs and per-run outputs were used? | Final manifest and bronze-reference JSON |
| What did the optional generative parser return? | `processed.../<run_id>/silver/llm_annotations/`, its processing/verification summaries, and per-run prompt/schema/provenance/generation/response records |
| Can I audit and replay-check each LLM request? | `models.../<run_id>/llm_decisions/rounds.jsonl` |
| How did the LLM feature families compare? | `processed.../<run_id>/evaluation/llm_ablation_comparison.json` in augment-mode backtests |
| How did model families compare? | `processed.../<run_id>/evaluation/backtest_comparison.json` |
| How good were predictions? | `processed.../<run_id>/evaluation/prediction_metrics.json` |
| Which trades, costs, and constraints occurred? | `processed.../<run_id>/backtests/<family>/backtest.json` |
| Which paper intents were emitted, and is their evidence chain valid? | `processed.../<run_id>/paper/snapshot.json` and `paper/events.jsonl` |
| Where are standalone broker actions audited? | Current-user kabuS state `audit.jsonl`; this is outside research run artifacts and paper evidence |
| How many rows trained each snapshot? | `model.json` snapshot counts, role, and training-key digest; its final-holdout protocol records the frozen boundary/cutoff, while exact keys are not stored by default |
| Why did a run fail? | `reports.../<run_id>/run.failed.json` plus stage logs |

## Full layout

```text
<configured raw root>/
  source=<role>/date=<date>/<sha-prefix>/<sha256>.<ext>
  _metadata/source=<role>/<payload-sha>/<metadata-sha>.json

<interim root>/<run_id>/
  bronze_refs/market.json
  bronze_refs/text.json
  captured_inputs/<role>/**              # symlinks for partitioned-Parquet source directories
  silver/assets/*.parquet
  silver/market/bar_size=1d/symbol=<symbol>/year=<year>/part-*.parquet
  silver/text/source=<source>/date=<date>/part-*.parquet
  silver/fundamentals/symbol=<symbol>/year=<year>/part-*.parquet
  silver/earnings_calendar/symbol=<symbol>/year=<year>/part-*.parquet
  silver/corporate_actions/symbol=<symbol>/year=<year>/part-*.parquet

<processed root>/<run_id>/
  silver/llm_annotations/symbol=<symbol>/year=<year>/part-*.parquet
  silver/text_signals/symbol=<symbol>/year=<year>/part-*.parquet
  gold/features/feature_set_version=<version>/year=<year>/part-*.parquet
  gold/labels/label_version=<version>/year=<year>/part-*.parquet
  gold/predictions/model_family=<family>/year=<year>/part-*.parquet
  evaluation/prediction_metrics.json
  evaluation/llm_annotation_summary.json
  evaluation/llm_verification_summary.json
  evaluation/llm_ablation_comparison.json       # augment-mode backtest only
  evaluation/backtest_comparison.json
  evaluation/final_holdout_backtest_comparison.json
  backtests/<family>/backtest.json
  backtests/<family>/final_holdout.json
  paper/snapshot.json
  paper/events.jsonl

<models root>/<run_id>/
  llm_annotations/prompt.txt
  llm_annotations/schema.json
  llm_annotations/provenance.json
  llm_annotations/generation_attempts/<cache-key>.json
  llm_annotations/responses/<cache-key>.json
  llm_decisions/rounds.jsonl
  model_version=<version>/model.json
  model_version=<version>/metadata.json
  model_version=<version>/metrics/*.json

<models root>/_cache/transformer/*.json       # optional shared inference cache
<models root>/_cache/llm_annotations/*.json  # optional content-addressed generation cache

<reports root>/<run_id>/
  config.snapshot.json
  run.initial.json
  run.final.json | run.failed.json
  research_note.md
```

Not every stage produces every path. For example, `build-features` stops before labels/models, and
the `paper` and `report` branches write different final products.

The broker adapter does not write into this per-run layout. Its append-only `audit.jsonl` lives in
fixed current-user kabuS state: `%LOCALAPPDATA%\nlp-trader\kabus` on Windows,
`~/Library/Application Support/nlp-trader/kabus` on macOS, or the platform's XDG state location on
Linux. The ledger is shared by all broker configs and environments, distinct from
`paper/events.jsonl`, and not covered by a research run manifest. Keep it private and follow the
retention and verification guidance in [Broker integration](broker.md).

Silver asset rows preserve asset-master `short_available` and `hard_to_borrow` values. The gold
feature rows and their derived prediction rows carry those values into backtest and paper portfolio
eligibility. Missing short availability is written as `false`; a shortable record with missing
hard-to-borrow status is written as `true`. These flags are not a historical borrow-inventory series.

Gold label rows separate `label_end_ts`, the official outcome close, from `label_available_at`, the
latest required exit-session bar availability across the complete cross-section. Walk-forward
training uses the latter with an exclusive cutoff.

## Manifests

### `run.initial.json`

Written when a validated pipeline run starts. It records:

- run identity and creation time;
- config hash;
- input paths, SHA-256 values, and byte counts;
- Git commit/dirty status; and
- initial `running` status.

The config filename itself is not part of the content hash; the resolved config values, including
absolute paths, are.

### `config.snapshot.json`

The complete resolved runtime configuration, including CLI overrides. Use this file—not the source
YAML—to reproduce what actually ran.

### `run.final.json`

Written exclusively on success. It contains completion status/stage, code version, config hash,
input manifest, artifact hashes, universe, period, versions, cost model, constraints, metrics,
limitations, and next questions.

The artifact manifest covers per-run interim, processed, model, and report artifacts other than the
final manifest itself. Raw payload provenance is represented through the input and bronze-reference
records. An optional transformer model’s weights and the shared transformer cache sit outside the
per-run roots and are not hashed by the current input or artifact manifest. The config records the
transformer name/version and inference settings, but exact model-byte provenance must be managed
separately for transformer research.

In contrast, an enabled generative annotator’s local model directory is hashed as a run input. Its
logical ID, exact revision, license/terms reference, prompt and schema versions, decoding settings,
verifier version, optional configured token rates, and `retrospective_parser` status appear in
annotation provenance. The shared cache is outside the run artifact roots, but every consumed raw
response is copied to that run’s `models.../llm_annotations/responses/` directory. Per-run generation
attempts and DecisionRounds also sit under the model run root, so all of them enter the final artifact
manifest on success.

### Generative semantic/evidence artifacts

`silver/llm_annotations/` contains verified sidecar rows keyed by `(item_id, asset_id)`. A v2 row
records source availability and assigned decision time, stance, integer semantic signal, explicitly
uncalibrated raw confidence, uncertainty, configured horizon, primary event/confidence, source-local
supporting and counterevidence span IDs, mechanism, invalidation conditions, abstention information,
source type/quality, verifier identity/result, and model/prompt/schema provenance.

`evaluation/llm_annotation_summary.json` is a processing and inference-usage summary. It reports
request/annotation/non-abstention/abstention/evidence/event counts; generated, cache-hit,
deduplicated, and raw-attempt counts; generated token counts and latency; configured estimated cost
when calculable; and the selected `feature_mode`. It is not an extraction-quality, performance, or
profitability metric. Token totals and configured cost are null if any newly generated response lacks
the required token counts. Run-level latency sums observed batch wall time once per batch; each
generated DecisionRound retains that request's full observed batch latency rather than a divided
allocation.

`evaluation/llm_verification_summary.json` reports pass/fail counts for item identity, candidate
coverage, temporal validity, horizon alignment, evidence-reference validity, and numeric-token
grounding. The verifier establishes those deterministic contracts only. It does not prove that cited
prose semantically supports a claim or that a mechanism is true.

`prompt.txt` and `schema.json` freeze the exact instruction/output contracts. `provenance.json`
records model/config/cache/verifier identities, optional configured token rates, and the retrospective
parsing assumption. Every newly generated response is first written to
`generation_attempts/<cache-key>.json`, before parsing or verification, so malformed/truncated output
survives a failed run for diagnosis. A successful generated or cache-backed response copied under
`responses/<cache-key>.json` includes the exact raw generation, parsed annotation payload,
verification result, and provenance used by the run. These files can repeat licensed or private
source text and remain gitignored local artifacts. Cache replay strictly reparses the stored raw
generation and requires it to reproduce the stored annotation payload before the response can affect
features or a DecisionRound.

Configured token rates are paired estimates, not vendor invoices. Without rates, the run estimate is
null; a generated DecisionRound also keeps cost null when its required token counts are unavailable.
Cache and in-run deduplicated DecisionRounds record zero new generation usage when rates are
configured, rather than charging the original generation again.

### LLM ablation comparison

An augment-mode backtest writes `evaluation/llm_ablation_comparison.json` for both development and
final holdout. It records fixed family semantics and three comparisons:

- `llm` versus conventional `text`;
- `traditional_llm` versus numeric `traditional`; and
- `all` versus numeric-plus-conventional `combined`.

Each comparison contains complete baseline/enhanced metric mappings and numeric
`enhanced_minus_baseline` deltas. The artifact also repeats generated token, latency, and configured
estimated-cost totals. These are arithmetic comparisons only—not statistical significance, causal
attribution, profitability evidence, or an automatic promotion decision.

### `run.failed.json`

Written for an exception after run creation. It records the requested target stage, exception type,
and error message and preserves partial artifacts. A dependency can fail before that target; use the
last `stage_start`/`stage_complete` log messages to locate it. Invalid config or missing required
input paths fail before run creation, so those cases do not produce a failure manifest.

## Research note

`research_note.md` is the human summary. Read it in this order:

1. provenance and data manifest;
2. fill/cost assumptions;
3. portfolio constraints;
4. raw and cost-adjusted metrics;
5. per-period diagnostics;
6. model-family evaluation;
7. known limitations and next questions.

The note intentionally labels results hypothetical. It is a report of assumptions and calculations,
not an investment recommendation.

## LLM DecisionRound ledger

`models.../<run_id>/llm_decisions/rounds.jsonl` contains one immutable canonical-JSON object per LLM
source request. Each record has a SHA-256 `round_id` over its canonical content and includes:

- run/item/source identity, source availability, assigned decision time, and configured horizon;
- exact model bytes identity, prompt/schema versions and hashes, and greedy sampling settings;
- `source_scope: current_source_only` plus every numbered span ID exposed to the model;
- exact raw generation and strict structured output;
- every deterministic verifier check and result;
- `generated`, `cache`, or `deduplicated` inference origin; and
- available input/output tokens, latency, and configured estimated USD cost.

The current schema deliberately requires empty tool-call, calibration, portfolio, risk, and order
fields, and it has no realized-outcome field. A DecisionRound therefore audits the semantic parser;
it is not a full portfolio decision trace and cannot be read as an order record.

The pipeline writes the ledger exclusively and immediately replays it. Replay rejects blank or
partial lines, invalid UTF-8/JSON, duplicate keys, noncanonical encodings, schema/timestamp failures,
duplicate round IDs, a missing ledger path, and content whose hash does not match `round_id`. Round
validation also binds every present structured output, including a failed-verifier trace, to strict
raw JSON. A truncated generation cannot carry structured output or a passing verifier result, and
cache/deduplicated rounds cannot report new inference usage. Replay validates the exact stored
stochastic output instead of calling the model again. Passing replay does not establish semantic
truth. This file is also distinct from the backtest replay below and the separate hash-chained paper
event ledger.

## Backtest JSON

Each family’s `backtest.json` contains the development window. `final_holdout.json` has the same
schema for the reserved final periods, and the two comparison JSON files keep their family metrics
separate. Each comparison file is an envelope with `evaluation_window`, the complete
`evaluation_protocol`, and a `families` metrics mapping. It also carries an artifact schema version,
run/config/code identity, input role hashes, feature/label/model versions, and complete configured
cost, constraint, horizon, and selection assumptions. The final manifest remains canonical for the
full data and generated-artifact manifests.

Each replay JSON contains:

| Key | Contents |
|---|---|
| `metrics` | Aggregated return, risk, exposure, cost, liquidity, and activity values. |
| `periods` | One record per replay period with timing, returns, both turnover legs, exposures, costs, capacity proxy, rejects, and flags. |
| `trades` | Entry and forced-exit records with raw fill price, decision-time liquidity, participation, reason codes, and cost breakdown. |
| `positions` | Target and post-return weights plus per-asset contribution. |
| `final_positions` | Empty for the current forced-liquidation round trips. |
| `assumptions` | Execution clock, label coverage, `top_k` selection depth, liquidity and turnover bases, buffer, and unmodeled effects. |
| `evaluation_window` | Development end-exclusive or final-holdout start-inclusive boundary. |

## Paper snapshot and event ledger

`paper/snapshot.json` is an intent-only current view. It records the decision and intended execution
timestamps, unchanged initial equity, empty trades and positions, constrained intents with rejection
or risk reason codes, and a `ledger` reference containing the event path, sequence, and event hash.
Its `research_protocol` records the run/config identity, horizon, `top_k`, same-day buffer, and exact
effective entry constraints used to size the intents; the hash-chained event repeats that protocol.
The pipeline uses `models.top_k` and the same portfolio constructor as model-family backtests, so the
paper intents are capped and constrained by the same selection semantics. Equal-weight and no-trade
remain backtest reference paths rather than paper intent families.

`paper/events.jsonl` is append-only evidence. The pipeline writes a `paper_intent_batch` event; a
direct `PaperSimulator` can optionally append `paper_rebalance` and `paper_mark_to_market` events.
Every record requires `simulation_only=true` and a timezone-aware `asof_ts`, which is normalized to
UTC, and includes:

| Key | Meaning |
|---|---|
| `sequence` | Contiguous one-based event position. |
| `previous_event_hash` | SHA-256 link to the preceding event, or the fixed genesis hash for event one. |
| `event_hash` | SHA-256 of the canonical event including its sequence and previous-hash link, but excluding this field itself. |

Use `PaperEventLedger(path).replay()` to read the file. Replay rejects malformed or partial lines,
duplicate JSON keys, noncanonical record encodings, noncanonical or regressing timestamps,
missing/false simulation markers, sequence gaps, broken hash links, and payload tampering. Append
first replays the existing chain, so a detected modification prevents extending it. The hash chain
is tamper-evident rather than an authenticated signature: a party able to rewrite the whole file can
recompute its suffix, and no trusted external timestamp is added.

`PaperSimulator` requires a missing or empty ledger when it is constructed. It does not recover
equity, positions, metadata, or configuration from prior records, so a new simulator refuses to
extend a nonempty ledger rather than silently starting a new state sequence in the same chain.

The ledger supports one serialized writer per file. It does not lock the file or provide an atomic
cross-process sequence reservation; concurrent writers can derive the same next sequence and prior
hash. Route all appends for one ledger path through one process or an external lock. `snapshot.json`
is a derived convenience artifact and should not be used to validate the chain independently.

## Metric guide

| Metric | Interpretation |
|---|---|
| `total_return` | Compounded normalized equity return after modeled costs. |
| `cost_adjusted_return` | Alias of `total_return` in the current implementation. |
| `gross_total_return` | Compounded period returns before transaction/borrow costs. |
| `annualized_return` | Mechanical annualization using the configured horizon frequency; unreliable on short samples. |
| `annualized_volatility` | Population standard deviation of period net returns × square root of periods per year. |
| `sharpe` | Mean net period return divided by net-return volatility × square root of periods per year; zero when volatility is zero. |
| `sortino` | Mean net period return divided by downside deviation × square root of periods per year; zero when downside deviation is zero. |
| `max_drawdown` | Worst peak-to-trough normalized equity loss, reported as a non-positive value. |
| `tail_loss_5pct` | Mean of the worst `max(1, floor(5% × periods))` net period returns. |
| `average_turnover` | Mean round-trip turnover on the recorded period basis. |
| `total_*_return` | Sum of commission, spread, slippage, impact, or borrow return deductions. |
| `average_*_exposure` | Mean target gross, net, or beta exposure across replay periods. |
| `max_participation_rate` | Largest modeled entry/exit participation in any period. |
| `minimum_capacity_proxy_equity` | Smallest participation-based screening equity across entry/exit legs; not deployable capacity. |
| `average_holding_period_days` | Mean elapsed timestamp duration between recorded entry and exit; daily open-to-close values are fractional days. |

Borrow cost uses at least one cost day even when open-to-close elapsed time is less than one calendar
day. This is why `average_holding_period_days` and the borrow-cost day basis need not be identical.

## Prediction evaluation

`prediction_metrics.json` contains aggregate and mean-daily Pearson/Spearman IC, hit rate,
precision-at-k, and available breakdowns by sector, liquidity, volatility, source, and event metadata.
Top-level `families` and `segments` cover development periods; `final_holdout` contains the same
structure for the configured final fully observed periods, and `evaluation_protocol` records the
boundary, the contiguous whole-cross-section development suffix purged from the first overlap, and
the predeclared frozen-training rule. The nested final-holdout training record includes its exclusive
cutoff, training row count, membership digest, and `verified_untouched` marker.

Mean squared error appears only when predictions provide explicit `expected_return`. Binary Brier and
calibration diagnostics appear only with explicit `probability_up`. The built-in rank baseline emits
neither, and its raw score is never transformed into a claimed probability. Optional expected-return,
probability, and binary-target fields require all-or-none coverage across all observed development and
holdout rows for a family.
`metric_definitions` records that precision-at-k is a raw-score, long-side diagnostic rather than the
absolute-score constrained portfolio selection used by backtests, and records the fractional cutoff-
tie policy that makes the diagnostic independent of input row order.

## Read Parquet locally

Polars example:

```python
from pathlib import Path

import polars as pl

root = Path("data/processed/sample/<run_id>/gold/features")
features = pl.scan_parquet(str(root / "**" / "*.parquet"))
print(
    features.select("symbol", "asof_ts", "return_5d", "text_count_5d")
    .sort("asof_ts", "symbol")
    .collect()
)
```

DuckDB example:

```python
import duckdb

rows = duckdb.sql(
    """
    select symbol, asof_ts, score
    from read_parquet(
      'data/processed/sample/<run_id>/gold/predictions/**/*.parquet',
      hive_partitioning = true
    )
    where model_family = 'combined'
    order by asof_ts, symbol
    """
)
print(rows)
```

Run snippets from the project environment with `uv run python`.

## Retention and cleanup

The bundled per-run interim, processed, model, and report roots are gitignored. Custom roots inside a
different repository location remain the user’s responsibility. Preserve useful positive and
negative experiments long enough to avoid accidental result selection. Never edit raw payloads or
their metadata in place; ingest a new version instead.

Related documentation:

- [Architecture](architecture.md)
- [Research protocol](research_protocol.md)
- [Backtesting](backtesting.md)
- [Broker integration](broker.md)
- [Troubleshooting](troubleshooting.md)

Return to the [documentation home](README.md).
