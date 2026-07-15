# Workflows and Commands

This page explains what each CLI command does, which prerequisites it runs, and which files it
produces.

## Two command categories

These commands do not create a research run:

| Command | Purpose |
|---|---|
| `validate-config` | Validate the typed YAML and required local paths. |
| `generate-synthetic` | Create deterministic asset, market, and text fixture files. |

Every other command is a pipeline execution command. It validates first, creates a new immutable
`run_id`, executes its dependency graph, and writes either `run.final.json` or `run.failed.json`.
There is no resume-in-place behavior.

## Stage dependency graph

```text
ingest-market ─────┬──────────────> build-labels ──┐
                  │                                │
                  └──> build-features <── ingest-text
                              │                    │
                              └────────> train <───┘
                                           │
                                        predict
                                       /       \
                                  backtest     paper
                                     │
                                   report
```

`smoke` is an alias for running through `report`. The `paper` branch is separate from `report` and
the backtest.

## Command reference

| Command | Prerequisites run in the same new run | Primary result |
|---|---|---|
| `ingest-market` | none | Bronze references and silver assets, bars, and configured optional records |
| `ingest-text` | loads/captures the filtered asset master internally | Bronze references and normalized silver text |
| `build-features` | market + text ingestion | Silver text signals and gold feature table |
| `build-labels` | market ingestion | Gold label table |
| `train` | features + labels | Versioned model artifact |
| `predict` | training | Predictions and model-evaluation JSON |
| `backtest` | prediction | Per-family replay JSON and comparison metrics |
| `paper` | prediction | Latest pending simulation-only intent snapshot |
| `report` | backtest | Markdown research note |
| `smoke` | same path as `report` | Smallest complete end-to-end run |

Every successful pipeline command also writes a final manifest.

## Global and shared options

Show all commands:

```bash
uv run nlp-trader --help
```

Enable debug logging by putting the global option before the command:

```bash
uv run nlp-trader --verbose backtest --config configs/sample.yaml
```

Pipeline commands and `validate-config` accept:

```text
--config FILE
--start-date ISO_DATE_OR_AWARE_TIMESTAMP
--end-date ISO_DATE_OR_AWARE_TIMESTAMP
--symbol SYMBOL          # repeatable; --symbols is an alias
--limit N
```

The overrides become part of the immutable typed config and therefore affect the config hash.

## Runtime filter semantics

### Dates

Dates bound emitted close-decision rows. They do not simply slice every source at the same points.
The pipeline adds:

- market-session warm-up before the requested start;
- text calendar-day warm-up before the requested start;
- exact-session label bars after the requested end; and
- bounded known-event context after the final selected decision.

It builds with that context, then filters gold outputs back to the requested decision interval.

### Symbols

Symbols restrict the asset master and market universe. Raw and silver text records are not filtered
merely because their prose lacks a selected symbol; they are linked against the filtered asset
master during feature preparation. A supplied asset-ID prelink outside the filtered universe fails
validation rather than being silently removed. A symbol-only prelink does not resolve automatically;
see [Input data](input_data.md).

### Limit

`--limit N` selects the earliest `N` complete market decision timestamps after date and symbol
filters. It preserves the whole asset cross-section and required context.

The standalone `ingest-text` stage does not apply this market-decision limit to source text rows.
Use a downstream stage such as `build-features` when you want limit semantics tied to selected market
decisions.

## Recommended workflows

### Fast implementation check

```bash
uv run nlp-trader smoke --config configs/sample.yaml
```

Use this after setup and after substantive code changes. It is not a research experiment.

### Synthetic assumption comparison

```bash
uv run nlp-trader backtest --config configs/backtest.yaml
```

This still uses synthetic data, but applies a nonzero embargo and stricter cost/constraint settings
than the sample config. It is useful for regression and report inspection, not performance claims.

### Bounded local-data run

```bash
uv run nlp-trader backtest \
  --config configs/local.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --symbol AAPL \
  --symbol MSFT \
  --limit 100
```

Start small, inspect the manifest and feature counts, then widen the period. Full mode currently
materializes downstream working sets in memory.

### Generate paper intents

```bash
uv run nlp-trader paper --config configs/sample.yaml
```

The CLI paper stage:

- looks at the latest combined-model decision;
- marks up to `models.top_k` positive-score assets as selected;
- emits one `BUY` or `FLAT` intent for every asset in the latest cross-section;
- applies portfolio constraints to selected target weights; and
- writes empty `positions` and `trades` with unchanged initial `equity`.

It does not fill anything, charge costs, maintain account state, or connect to a broker. The separate
`PaperSimulator` class is an in-memory programmatic utility for simulated rebalances and
mark-to-market events from caller-supplied returns; it also has no broker adapter or price-level fill
model.

### Optional local transformer sentiment

```bash
uv sync --extra nlp
uv run nlp-trader smoke \
  --config configs/sample.yaml \
  --enable-transformer-sentiment
```

Set `transformer.model_name` to a model already present locally first. Keep
`transformer.local_files_only: true` for an offline, reproducible run. Outputs are cached by normalized
text/model identity.

## Model selection details

`models.top_k` controls precision-at-k evaluation and latest paper-intent selection. It does not cap
the number of backtest candidates. The current portfolio constructor requests every positive score
for long-only runs and every negative score as well when shorting is enabled, then applies eligibility,
exposure, turnover, and participation constraints.

## Validation and failures

An invalid config fails before a run directory is created. A failure after run creation writes
`run.failed.json` and preserves partial artifacts for diagnosis. The failure manifest records the
exception type and message but not a traceback.

See [Troubleshooting](troubleshooting.md) for common errors and [Outputs](outputs.md) for the artifact
layout.

Return to the [documentation home](README.md).
