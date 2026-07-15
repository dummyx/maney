# NLP Trader

[日本語で読む](README.ja.md)

NLP Trader runs repeatable trading-strategy experiments from local files. It helps answer one
question: does text you are allowed to use add useful information beyond price and volume data?

The project is designed for research on a laptop. It records the inputs, settings, code version,
and results for every run so another engineer can inspect or repeat the work.

> All results are hypothetical. They are not financial advice and do not show that a strategy will
> make money. Research and simulated paper-trading (`paper`) commands never send orders. The
> separate
> [kabuS commands](docs/broker.md) can place real stock orders when explicitly configured and
> confirmed.

## Quick start

You need Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run nlp-trader validate-config --config configs/sample.yaml
uv run nlp-trader smoke --config configs/sample.yaml
```

The sample uses small, made-up files included in the repository. It needs no market-data account,
credentials, GPU, or downloaded language model.

When the run finishes, it prints a `run_id` and the paths it created. Start with:

- `reports/sample/<run_id>/research_note.md` — the readable result and its limitations;
- `reports/sample/<run_id>/run.final.json` — the settings, input hashes, code version, and output
  list; and
- `data/processed/sample/<run_id>/backtests/` — detailed simulated positions, trades, and costs.

The sample numbers only confirm that the program works. They say nothing about expected returns.

## What the program does

For each run, NLP Trader:

1. reads market and text files without changing the originals;
2. makes sure each historical decision uses only information available at that time;
3. builds model inputs and later outcomes as separate data;
4. repeatedly trains on past periods and scores the next period;
5. simulates entries, exits, trading costs, and limits on how much it can hold or trade; and
6. saves a report and a machine-readable record of the run.

The current daily simulation makes a decision only after the required data is available. A simulated
trade starts no earlier than the next market open and ends after the configured number of trading
days. The exact price and cost rules are documented in [Backtesting](docs/backtesting.md).

The program compares market-data methods, text methods, combined methods, and simple reference
methods such as equal weight, recent price trend, and no trade.

## Choose a configuration

| File | Use it for | Data source |
|---|---|---|
| `configs/sample.yaml` | Quick end-to-end check | Included made-up files |
| `configs/backtest.yaml` | More demanding check with the included data | Included made-up files |
| `configs/local.yaml` | Research with your own files | Local files you are allowed to use |
| `configs/japan_baseline.yaml` | Japanese stock research | Your permitted Japanese market exports |

The two local templates fail validation until you provide their input files. The project does not
download or include vendor data. See [Input data](docs/input_data.md) for accepted files and
[Japan cash-equity baseline](docs/japan_baseline.md) for the Japanese format.

## Use your own data

1. Copy `configs/local.yaml` or `configs/japan_baseline.yaml`.
2. Point it to your asset, market, and text files.
3. Replace the example license references with records that match your data rights.
4. Choose a date range and symbols that fit in local memory.
5. Validate the configuration before running a backtest.

```bash
uv run nlp-trader validate-config --config configs/local.yaml
uv run nlp-trader backtest --config configs/local.yaml --limit 100
```

To narrow a development run:

```bash
uv run nlp-trader backtest \
  --config configs/local.yaml \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --symbol AAPL \
  --limit 100
```

Each research command creates a new run directory. It does not overwrite or continue an older run.
See [Workflows](docs/workflows.md) for all commands and the work each one performs first.

## Optional language-model text analysis

Large language model (LLM) support is off by default. The normal sample and baseline do not install
an LLM library, download a model, or require PyTorch. The optional runtime is
`llama-cpp-python==0.3.34`, installed through the separate `llm` extra. It reads one local GGUF
model file—the format llama.cpp uses—directly from the path you configure. It never downloads a
model while the pipeline is running.

The bundled settings identify
`unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` at revision
`5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf` and expect the local file
`Qwen3.6-27B-UD-Q4_K_XL.gguf`. That file is about 17.9 GB. In practice, use a Mac with at least
32 GB of unified memory and process one text item at a time; the model file is not the program's
only memory use. You must download the file explicitly, review its license/terms, and set
`paths.llm_model`. See [Workflows](docs/workflows.md#optional-local-generative-semanticevidence-annotation)
for the exact install, download, checksum, configuration, and real-model acceptance commands.

When enabled, the model reads one text item at a time and returns a structured interpretation for
each company that the program matched in that item. The result includes positive or negative
direction, uncertainty, a time window, cited passages, possible events, and reasons the
interpretation could be wrong.

The program checks the output format, timing, whether every expected company has a result, and
cited passage IDs. It also checks that any number in the model's explanation appears in a cited
passage. These checks catch malformed or inconsistent output; they cannot prove that the source or
the model's interpretation is true.

LLM values are stored separately from the normal text score. A research run can compare market data,
normal text analysis, LLM analysis, and their combinations without replacing the original inputs.
The model's confidence number is kept as a separate input. It is not treated as a tested probability
or a position size.

The LLM never creates orders. It receives no prices, future outcomes, current simulated holdings,
web results, or other documents. A modern model may already know facts learned after the historical
date, so this test cannot show what the same model would have produced at that past date.

The model runs directly on your Mac through llama.cpp. Although its repository name includes `MTP`,
the current Python path does not use that optional speed-up. Regular automated tests substitute
fixed model replies to check the surrounding code; only the separate real-model test proves that
the full model file can load and produce a valid reply on your machine.

For setup and output details, read [Configuration](docs/configuration.md),
[Features and models](docs/features_and_models.md), and [Outputs](docs/outputs.md).

## Important limits

- A good historical result does not imply a good future result.
- Trading fees, the difference between buy and sell prices, trade-driven price movement,
  short-selling costs, and practical trade size are estimates. This is not a full exchange or broker
  simulator.
- Repeatedly changing a strategy after looking at its final evaluation period makes that evaluation
  unreliable.
- The files you provide must correctly show which stocks existed, which version of each report was
  available, and when the program could have seen each value.
- Large runs eventually load intermediate tables into RAM. Start with date, stock symbol, and
  `--limit` filters.
- Optional language-model tools use local model files only. The default sample needs neither
  PyTorch nor `llama-cpp-python`.
- The project does not scrape websites or provide a market-data downloader.

## Documentation

You do not need to read every document. Start with the page that matches your task:

| Task | Guide |
|---|---|
| Install and run the sample | [Getting started](docs/getting_started.md) |
| Prepare local files | [Input data](docs/input_data.md) |
| Change settings | [Configuration](docs/configuration.md) |
| Run a command | [Workflows](docs/workflows.md) |
| Find and understand results | [Outputs](docs/outputs.md) |
| Understand historical-data rules | [Data contracts](docs/data_contracts.md) |
| Understand the simulation | [Backtesting](docs/backtesting.md) |
| Design a sound experiment | [Research protocol](docs/research_protocol.md) |
| Change the code | [Development](docs/development.md) |
| Review data rights and safety | [Compliance](docs/compliance.md) |

The [documentation home](docs/README.md) contains the full index.

## Development checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
uv run nlp-trader smoke --config configs/sample.yaml
```

Automated checks run the same commands for every code change. See
[Development](docs/development.md) for the repository layout and contribution guidance.

## Separate broker commands

The research pipeline does not call a broker. The kabuS commands are separate, owner-operated, and
must be invoked explicitly on the Windows PC running kabuStation.

```bash
uv run nlp-trader broker --help
```

The validation setup returns fixed test responses. A production setup can place real orders. Read
the [broker guide](docs/broker.md) before configuring or running it.
