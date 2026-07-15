# Getting Started

This guide takes you from a fresh checkout to a complete synthetic research run. The sample execution
uses no vendor data, credentials, data/model network calls, PyTorch, or GPU. A fresh
`uv sync --locked` may contact package indexes to install the exact locked dependencies.

## 1. Check the prerequisites

You need:

- Python 3.12;
- [`uv`](https://docs.astral.sh/uv/); and
- a writable checkout of the repository.

From the repository root, install the baseline and development environment:

```bash
uv sync --locked
```

The baseline intentionally excludes deep-learning packages.

It also excludes the optional GGUF language-model runtime. A normal `uv run pytest` or sample smoke
uses injected/fake LLM paths and does not prove that a real model can load, use Metal, or generate a
valid response on this Mac. The opt-in model is a separate 17.9 GB download and is most practical on
a Mac with at least 32 GB of unified memory. Complete the baseline sample first, then follow
[Optional local generative semantic/evidence annotation](workflows.md#optional-local-generative-semanticevidence-annotation)
for the explicit install, download, configuration, and gated real-model acceptance test.

## 2. Validate the sample configuration

```bash
uv run nlp-trader validate-config --config configs/sample.yaml
```

Expected result:

```text
ok: True
```

Validation checks both the typed YAML and the required local paths. It also rejects overlapping
input/output roots, invalid date ranges, unsupported modes, and inconsistent horizon settings.

## 3. Run the complete sample

```bash
uv run nlp-trader smoke --config configs/sample.yaml
```

`smoke` runs these stages in dependency order:

```text
ingest market + ingest text
            |
            v
       annotate text (disabled no-op in sample)
            |
            v
       build features + build labels
                    |
                    v
    train -> predict -> backtest -> report
```

Stage logs show elapsed time and current row counts. The final lines include a `run_id`, the research
report path, and the final manifest path.

## 4. Inspect the result

Replace `<run_id>` with the printed value.

Read these in order:

1. `reports/sample/<run_id>/research_note.md` — human-readable assumptions, metrics, limitations,
   and per-period diagnostics.
2. `reports/sample/<run_id>/run.final.json` — machine-readable provenance, hashes, config hash,
   metrics, and completion status. The full resolved config is in `config.snapshot.json`.
3. `data/processed/sample/<run_id>/evaluation/backtest_comparison.json` — side-by-side development
   model and benchmark results with their evaluation boundary; the separately named final-holdout
   comparison has the same envelope.
4. `data/processed/sample/<run_id>/backtests/<family>/backtest.json` — trades, positions, costs,
   exposures, and period records for one family.

> Do not interpret the sample’s annualized metrics. The dataset is synthetic and only a few sessions
> long, so annualization is mechanically unstable.

## 5. Run a narrower stage

Every pipeline execution command creates a new run and automatically executes its prerequisites.
Utility commands such as `validate-config` and `generate-synthetic` do not. For example:

```bash
uv run nlp-trader build-features --config configs/sample.yaml
```

This ingests the required sources and stops after feature materialization. It does not reuse the
artifacts from your earlier smoke run.

See [Workflows](workflows.md) for the complete dependency table.

## 6. Generate another deterministic fixture

The generator may use an existing output directory, but refuses to overwrite any existing
`assets.csv`, `market_bars.csv`, or `text_items.jsonl` file.

```bash
uv run nlp-trader generate-synthetic \
  --output-dir /tmp/nlp-trader-example-17 \
  --seed 17 \
  --session-count 20 \
  --symbol AAA \
  --symbol BBB
```

It creates:

```text
/tmp/nlp-trader-example-17/assets.csv
/tmp/nlp-trader-example-17/market_bars.csv
/tmp/nlp-trader-example-17/text_items.jsonl
```

The generator does not create a config. Copy `configs/sample.yaml`, point the three input paths at
the new files, and use distinct writable artifact roots. Paths in YAML are resolved relative to the
config file.

## 7. Move to local data

Before using `configs/local.yaml`:

1. read [Input data](input_data.md);
2. place only licensed, permitted local files under paths you control;
3. update the three required paths and any optional point-in-time paths;
4. update the license/terms references;
5. choose a date range and symbols that fit local memory; and
6. validate before running a research stage.

```bash
uv run nlp-trader validate-config --config configs/local.yaml
```

The template is expected to report missing files until you provide them.

For permitted Japanese cash-equity exports, start from `configs/japan_baseline.yaml` instead of
loosening the generic schema. It selects the XJPX calendar and strict local contract, and is also
expected to fail until its private input files exist. Follow [Japan cash-equity baseline](japan_baseline.md)
for J-Quants V2 normalization, availability timestamps, and the market-only empty-text setup.

The research and paper commands above never connect to a broker. The optional kabuS API integration
is a separate, explicitly gated workflow that must run on the same Windows PC as kabuStation; it is
not usable from this Mac for live routing. Start with the fixed-response validation environment and
follow [kabuS API broker operations](broker.md) before enabling any production command.

## Next steps

- [Configuration reference](configuration.md)
- [Japan cash-equity baseline](japan_baseline.md)
- [kabuS API broker operations](broker.md)
- [Input data guide](input_data.md)
- [Outputs and artifacts](outputs.md)
- [Troubleshooting](troubleshooting.md)

Return to the [documentation home](README.md).
