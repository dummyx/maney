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
| How did model families compare? | `processed.../<run_id>/evaluation/backtest_comparison.json` |
| How good were predictions? | `processed.../<run_id>/evaluation/prediction_metrics.json` |
| Which trades, costs, and constraints occurred? | `processed.../<run_id>/backtests/<family>/backtest.json` |
| How many rows trained each snapshot? | `model.json` snapshot counts and training-key digest; exact keys are not stored by default |
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
  silver/text_signals/symbol=<symbol>/year=<year>/part-*.parquet
  gold/features/feature_set_version=<version>/year=<year>/part-*.parquet
  gold/labels/label_version=<version>/year=<year>/part-*.parquet
  gold/predictions/model_family=<family>/year=<year>/part-*.parquet
  evaluation/prediction_metrics.json
  evaluation/backtest_comparison.json
  backtests/<family>/backtest.json
  paper/snapshot.json

<models root>/<run_id>/
  model_version=<version>/model.json
  model_version=<version>/metadata.json
  model_version=<version>/metrics/*.json

<models root>/_cache/transformer/*.json       # optional shared inference cache

<reports root>/<run_id>/
  config.snapshot.json
  run.initial.json
  run.final.json | run.failed.json
  research_note.md
```

Not every stage produces every path. For example, `build-features` stops before labels/models, and
the `paper` and `report` branches write different final products.

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
model name/version and inference settings, but exact model-byte provenance must be managed
separately for transformer research.

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

## Backtest JSON

Each family’s `backtest.json` contains:

| Key | Contents |
|---|---|
| `metrics` | Aggregated return, risk, exposure, cost, liquidity, and activity values. |
| `periods` | One record per replay period with timing, returns, both turnover legs, exposures, costs, capacity proxy, rejects, and flags. |
| `trades` | Entry and forced-exit records with raw fill price, decision-time liquidity, participation, reason codes, and cost breakdown. |
| `positions` | Target and post-return weights plus per-asset contribution. |
| `final_positions` | Empty for the current forced-liquidation round trips. |
| `assumptions` | Execution clock, label coverage, liquidity basis, turnover basis, buffer, and unmodeled effects. |

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
precision-at-k, mean squared error, optional binary calibration diagnostics, and available breakdowns
by sector, liquidity, volatility, source, and event metadata.

When an explicit `probability_up` is absent, calibration uses a logistic transform of the raw score.
Treat it as a score diagnostic, not a calibrated probability claim.

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
- [Troubleshooting](troubleshooting.md)

Return to the [documentation home](README.md).
