# NLP Trader

NLP Trader is a local-first research pipeline for studying whether permitted text data—news,
filings, transcripts, social exports, and commentary—adds useful information beyond traditional
market data.

> This repository produces hypothetical, assumption-dependent research. It is not financial advice,
> does not establish that a strategy is profitable, and contains no live-order route.

## The project in one minute

The pipeline:

1. captures local source files byte-for-byte;
2. normalizes market and text records into typed Parquet tables;
3. builds features using only information available by each decision time;
4. creates forward labels separately from features;
5. trains expanding walk-forward baseline models;
6. compares traditional-only, text-only, combined, and naive strategies;
7. runs a constrained, two-sided-cost backtest; and
8. writes a readable research note plus an auditable run manifest.

The implemented strategy clock is deliberately narrow:

```text
official close decision  ->  next official open entry  ->  horizon close exit
       features known              assumed fill                assumed fill
```

Raw open and close prices remain the simulated fill prices. Returns use only the supplied causal
per-bar adjustment factors. See [Backtesting](docs/backtesting.md) for the exact contract.

## Five-minute tour

Requirements:

- macOS or another Python 3.12 environment;
- [`uv`](https://docs.astral.sh/uv/); and
- no market-data credentials for the sample run.

Install and run the deterministic synthetic example:

```bash
uv sync --locked
uv run nlp-trader validate-config --config configs/sample.yaml
uv run nlp-trader smoke --config configs/sample.yaml
```

A successful run prints a unique `run_id`, artifact paths, a research report, and a final manifest.
The generated files live under gitignored `data/`, `models/`, and `reports/` run directories.

Start with:

- `reports/sample/<run_id>/research_note.md` for the human-readable result;
- `reports/sample/<run_id>/run.final.json` for provenance and reproducibility; and
- `data/processed/sample/<run_id>/backtests/` for detailed replay records.

The fixtures are tiny and synthetic. Their metrics test implementation behavior only; they are not
evidence of expected returns.

## What works today

| Area | Current implementation |
|---|---|
| Inputs | Local CSV, JSON, JSONL, Parquet files, or partitioned Parquet directories |
| Storage | Content-addressed bronze; typed, partitioned Parquet silver and gold |
| Timing | Timezone-aware UTC, XNYS close decisions, exchange-aware label windows |
| Text | Deterministic preprocessing, entity linking, deduplication, sentiment, attention, novelty, disagreement, credibility, and event features |
| Market | Returns, momentum, liquidity, volatility, market/sector, and optional point-in-time event/fundamental features |
| Models | Incremental expanding walk-forward traditional, text, and combined baselines |
| Benchmarks | Equal-weight, momentum-only, and no-trade |
| Backtest | Raw-price next-open entry, horizon-close liquidation, costs, constraints, logs, and reports |
| Paper | Pending simulation-only intents; no fills, account state, or broker connection |
| Optional NLP | Transformer inference with a bundled local-files-only default, cache, MPS detection, and CPU fallback |
| Quality | Ruff, strict mypy, unit/property/integration/regression tests, and offline/no-sync CI checks on Ubuntu and Apple Silicon |

## Choose a mode

| Config | Purpose | Data |
|---|---|---|
| `configs/sample.yaml` | Fast end-to-end smoke test | Checked-in synthetic fixtures |
| `configs/backtest.yaml` | Synthetic run with stricter research assumptions | Checked-in synthetic fixtures |
| `configs/local.yaml` | Template for larger local research | User-supplied licensed files |

`configs/local.yaml` is expected to fail validation until you provide its asset, market-bar, and text
files. The repository does not download or bundle vendor data.

## Common commands

Every pipeline command creates a new immutable run and executes its prerequisites in that run.
Commands do not resume or mutate an older run.

```bash
uv run nlp-trader validate-config --config configs/sample.yaml
uv run nlp-trader ingest-market --config configs/sample.yaml
uv run nlp-trader ingest-text --config configs/sample.yaml
uv run nlp-trader build-features --config configs/sample.yaml
uv run nlp-trader build-labels --config configs/sample.yaml
uv run nlp-trader train --config configs/sample.yaml
uv run nlp-trader predict --config configs/sample.yaml
uv run nlp-trader backtest --config configs/sample.yaml
uv run nlp-trader paper --config configs/sample.yaml
uv run nlp-trader report --config configs/sample.yaml
uv run nlp-trader smoke --config configs/sample.yaml
```

Limit a local development run without losing required warm-up or splitting an asset cross-section:

```bash
uv run nlp-trader backtest \
  --config configs/local.yaml \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --symbol AAPL \
  --limit 100
```

`--limit` counts complete decision timestamps, not raw rows. The pipeline still loads configured
market/text warm-up and label/event lookahead around the requested interval.

## Documentation

The [documentation home](docs/README.md) offers reading paths for first-time users, researchers, and
contributors.

| If you want to… | Read |
|---|---|
| Run the sample and understand success | [Getting started](docs/getting_started.md) |
| Configure local research | [Configuration reference](docs/configuration.md) |
| Prepare input files | [Input data guide](docs/input_data.md) |
| Choose and run a CLI stage | [Workflows and commands](docs/workflows.md) |
| Find reports, manifests, and Parquet tables | [Outputs and artifacts](docs/outputs.md) |
| Understand point-in-time rules | [Data contracts](docs/data_contracts.md) |
| Understand features and walk-forward scores | [Features and models](docs/features_and_models.md) |
| Interpret a backtest | [Backtesting](docs/backtesting.md) |
| Design a defensible experiment | [Research protocol](docs/research_protocol.md) |
| Diagnose a failure | [Troubleshooting](docs/troubleshooting.md) |
| Change the code safely | [Development guide](docs/development.md) |
| Review licensing and safety boundaries | [Compliance](docs/compliance.md) |

## Important limitations

- Full mode is not end-to-end out-of-core. Source scans are filtered lazily, but downstream working
  sets are materialized; use date, symbol, and decision limits sized to local memory.
- The baseline does not create a final untouched holdout or purged cross-validation study for you.
- Fill, spread, impact, borrow, and capacity calculations are configurable research proxies, not a
  venue or broker simulator.
- Provider-specific revision history, survivorship-aware universes, and licensed historical source
  quality remain the researcher’s responsibility.
- With the bundled `local_files_only: true` setting, the optional transformer path requires a model
  already present on the machine. Tests never download one.
- There is no scraping, external vendor adapter, account connection, or live execution path.

## Development checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
uv run nlp-trader smoke --config configs/sample.yaml
```

The [quality workflow](.github/workflows/quality.yml) runs on every push and pull request. Ubuntu
runs formatting, lint, strict type checks, the full test suite, and the deterministic sample; an
Apple Silicon runner repeats the baseline tests and sample. Each job installs the lockfile once,
then uses uv's offline, no-sync mode for the checks. The optional PyTorch/MPS path is not installed
or claimed by this workflow.

See [Development](docs/development.md) for repository structure, test layers, and extension
checklists.
