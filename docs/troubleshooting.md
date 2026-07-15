# Troubleshooting

Start with configuration validation:

```bash
uv run nlp-trader validate-config --config <your-config.yaml>
```

That command is for research configs. For a standalone broker config, use
`uv run nlp-trader broker validate-config --config <broker.yaml>` and see
[Broker integration](broker.md).

If validation succeeds but a stage fails, use the printed `run_id`, stage logs, and
`reports.../<run_id>/run.failed.json`. Its `failed_stage` is the requested target, not necessarily the
dependency where the exception arose; the final `stage_start` log identifies that dependency.

## Setup and command problems

### `uv: command not found`

Install `uv`, then reopen the shell or ensure its install directory is on `PATH`. Confirm with:

```bash
uv --version
```

### Wrong Python version

The project requires Python 3.12 and rejects 3.11/3.13 through `pyproject.toml`. Let `uv` provision
the declared interpreter or install Python 3.12 locally.

### `No such command`

Run from the repository root and inspect:

```bash
uv run nlp-trader --help
```

Global `--verbose` goes before the command:

```bash
uv run nlp-trader --verbose smoke --config configs/sample.yaml
```

## Configuration problems

### `missing assets`, `missing market_bars`, or `missing text_items`

`configs/local.yaml` is a template and intentionally points to files that are not bundled. Supply
licensed local files and update the paths, or use `configs/sample.yaml` for an implementation test.

Paths are resolved relative to the YAML file.

### `write roots must not overlap`

Give raw, interim, processed, model, and report roots distinct locations. One may not be a parent of
another.

Bad:

```yaml
processed_dir: ../artifacts
reports_dir: ../artifacts/reports
```

Good:

```yaml
processed_dir: ../artifacts/processed
reports_dir: ../artifacts/reports
```

### Input path overlaps an artifact root

Move the source outside all configured output roots. Otherwise a run could ingest its own derived
files.

### Unknown configuration field

Configuration is strict. Check the field name against [Configuration](configuration.md); do not
silently ignore or invent keys.

### Start/end date validation error

Use an ISO date (`2026-07-01`) or a timezone-aware ISO timestamp
(`2026-07-01T20:00:00Z`). The end must not precede the start. Config validation compares UTC dates,
so you must also check exact ordering when both timestamps fall on the same UTC date; a reversed
same-day interval can pass initial validation but yield no usable decisions.

### Horizon/rebalance mismatch

Set `features.horizon_days: N` and `backtest.rebalance_frequency: Nd` to the same positive integer.

### Text warm-up validation error

`text_warmup_days` must be at least twice the largest `windows_days` value. For a 20-day text window,
the minimum warm-up is 40 calendar days.

## Input and timestamp problems

### `must be timezone-aware`

Add an explicit UTC offset or `Z`. Do not use a bare timestamp such as `2026-07-01 20:00:00`.

### Text availability ordering error

Check that:

```text
published_at <= vendor_received_at <= available_at
ingested_at <= processed_at
```

The vendor timestamp is optional, but when present it may not precede publication.

### Hash field rejected

`author_hash`, `url_hash`, content hashes, and parent hashes must be lowercase 64-character SHA-256
hex strings. Alternatively provide a supported raw convenience identifier and let the local provider
hash it for silver output; remember that bronze still stores the raw source bytes.

### `retention_permitted=false`

The local text provider refuses the item. Do not work around this check; remove content whose source
terms do not permit local retention.

### Unknown or mismatched asset

Ensure every `asset_id` exists in the filtered asset master and its `symbol` agrees. Market bars and
asset-ID prelinked text entities must also fall inside the asset’s active interval.

### Prelinked entity fails after `--symbol`

The symbol filter narrows the asset master, while source text rows are still normalized. Remove
invalid asset-ID prelinks, supply a consistent selected universe, or use an empty/absent `entities`
list so the baseline linker can work against the filtered assets. A nonempty symbol-only prelink is
ignored and also suppresses automatic linking.

## Market-bar and label problems

### OHLC values are inconsistent

For each bar:

- all prices must be positive and finite;
- `low <= open/close <= high`;
- `low <= high`; and
- volume must be non-negative.

### Corporate-action contract error

Daily feature/label runs require:

```text
corporate_action_adjusted = true
adjustment_vintage_at <= available_at (when bar availability is explicit)
return_adjustment_factor > 0
```

Raw OHLC remains unadjusted for fills. Do not substitute a modern adjusted-close column for the
causal factor/vintage contract. For a generic bar without explicit `available_at`, the adjustment
vintage must not follow `ts`. The Japanese contract instead requires
`ts <= adjustment_vintage_at <= available_at` because delivery normally follows the close.

### Bar is not at the official close

`ts` must be the configured calendar’s actual session close, including early closes, DST shifts,
and venue close-time changes. Do not hardcode one UTC close for every date. `available_at` is a
separate delivery timestamp and must not replace `ts`.

### Missing or duplicate internal session

Supply one unique bar per expected exchange session for each asset over its covered range. The
builder also requires every asset active between the first and last supplied session to appear in
that session's cross-section; it does not invent a bar or silently shrink the universe.

### Trailing label is missing

The source needs the exact future session bars through the configured horizon. Runtime end handling
loads this context automatically only when those bars exist in the input.

### Partial cross-sectional labels

Training, evaluation, and backtesting refuse a decision where only some candidate assets later have
outcomes. Fix the input coverage or explicitly end the requested decision interval earlier. The
pipeline will not choose surviving assets based on future label availability.

## Model and result surprises

### Early model scores are zero

This is expected until at least `models.min_train_rows` labels have become observable strictly before
the walk-forward cutoff. Embargoes delay fitting further.

### Backtest trades more than `top_k`

Model-scored paths select at most `top_k` positions per decision, but each selected round trip can
produce both an entry and a forced-exit trade row. Equal-weight is intentionally uncapped. Count
distinct entry assets per period when checking selection depth.

### Final holdout leaves no development period

Reduce `models.final_holdout_periods` or widen the decision range. The value counts fully observed
whole cross-sections and must leave at least one development period. Do not solve this by partially
dropping assets or by moving missing future outcomes into the development window.

### No trades despite nonzero scores

Inspect `rejected`, `risk_flags`, and trade/period records. Common causes are minimum price/dollar
volume, participation, turnover, net/beta/sector limits, or shorting disabled.

### Very large annualized sample metrics

The synthetic sample spans only a few periods. Read raw total return, period count, turnover, costs,
and diagnostics. Never interpret its annualized values as expected performance.

### Capacity value looks surprisingly large

It is a participation-based screening proxy using decision-time daily dollar volume. It is not
calibrated deployable capacity and does not model an order book or intraday liquidity.

## Transformer problems

### Optional dependency error

Install the extra:

```bash
uv sync --extra nlp
```

### `transformer.model_name is required`

Set the path/identifier of a model already available locally before enabling transformer sentiment.

### Model tries to download or cannot be found

Keep `local_files_only: true` and place the model in the local Hugging Face cache or use a local model
path. Tests deliberately do not download models.

### MPS is unavailable

The centralized device helper falls back to CPU. This is supported behavior, not an error.

## GGUF generative LLM problems

### `Generative LLM annotations require llama-cpp-python`

The generative backend uses the separate `llm` extra, not the transformer `nlp` extra. On Apple
Silicon, install the pinned `llama-cpp-python==0.3.34` build with Metal enabled:

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv sync --extra llm --locked
```

Installing the extra does not download model weights. Follow the explicit model download and
checksum steps in [Workflows](workflows.md#optional-local-generative-semanticevidence-annotation).

### `llm_model must be an existing local GGUF file`

Set `paths.llm_model` to the file itself, for example:

```yaml
paths:
  llm_model: /absolute/path/to/Qwen3.6-27B-UD-Q4_K_XL.gguf
```

A directory, model-hub selector, missing path, or non-`.gguf` file is rejected. The runtime never
downloads or resolves the configured model ID.

### `default Qwen GGUF SHA-256 mismatch`

The default selector `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` and revision
`5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf` require
`Qwen3.6-27B-UD-Q4_K_XL.gguf` with this checksum:

```text
4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095
```

Check it with `shasum -a 256 /absolute/path/Qwen3.6-27B-UD-Q4_K_XL.gguf`. Do not relabel a different
file or revision to bypass the check. Download the pinned file again from the licensed source if the
local copy is incomplete or different.

### `local GGUF model must contain an embedded chat template`

The backend uses the chat template stored in the GGUF metadata and records its SHA-256. It does not
guess or fetch a template. Use the exact pinned file, or choose another GGUF whose metadata includes
`tokenizer.chat_template` and record that model's real ID, revision, checksum, and terms.

### The model runs out of memory or Metal is not used

The default file is about 17.9 GB, and inference also needs context, key/value cache, and working
buffers. At least 32 GB of unified memory is a practical starting point, not a guarantee. Keep
`batch_size: 1`, close memory-heavy applications, and reduce `context_tokens` only while preserving
`max_input_tokens + max_new_tokens <= context_tokens`.

`gpu_layers: -1` requests all possible llama.cpp Metal layers. If the installed native binding
reports no GPU-offload support, the host uses CPU with zero GPU layers. This is llama.cpp Metal, not
PyTorch MPS. `use_mmap: true` avoids one eager file copy but does not make the 17.9 GB model fit into
insufficient memory.

### Metal context creation fails

First close memory-heavy applications and check that the configured context fits in unified memory.
A Metal-enabled llama.cpp build may initialize Metal even when `gpu_layers: 0`, so retrying the same
build with zero offloaded layers is not a dependable CPU fallback. To run without Metal, rebuild the
optional dependency as CPU-only:

```bash
CMAKE_ARGS="-DGGML_METAL=OFF" \
  uv sync --extra llm --locked --reinstall-package llama-cpp-python --no-cache
```

`--no-cache` matters when changing build modes because it prevents reuse of the existing
Metal-enabled wheel. The runtime then detects that GPU offload is unavailable and uses CPU. Expect
the 27B model to be much slower than Metal inference.

The model repository name includes `MTP`, but this backend uses ordinary inference. There is no MTP
speculative-decoding acceleration to enable in the current implementation.

### The real-model test is skipped

Normal tests intentionally do not load the large licensed model. After installing the `llm` extra
and verifying the file, opt in explicitly:

```bash
NLP_TRADER_RUN_REAL_LLM=1 \
NLP_TRADER_LLM_MODEL_PATH=/absolute/path/Qwen3.6-27B-UD-Q4_K_XL.gguf \
uv run pytest tests/acceptance/test_llama_cpp_qwen.py -v
```

This acceptance test checks real local model loading and inference. The normal fake/injected tests
cannot establish that, and neither test establishes extraction quality or trading performance.

## Broker problems

### Cannot connect to kabuStation or responses look artificial

Run the broker command and kabuStation on the same Windows PC, keep kabuStation logged in, and do not
proxy or expose its loopback API. The validation environment intentionally returns fixed test values
and cannot place real orders; it is not a realistic account or fill simulator. Production uses a
different loopback endpoint and can place real orders. Do not switch environments merely to bypass a
failed validation preflight; follow [Broker integration](broker.md).

### Cannot find the broker audit ledger

Broker commands do not write into a research run or `paper/events.jsonl`. Run `broker
validate-config` to display the fixed current-user `audit.jsonl`, `KILL_SWITCH`, and `operation.lock`
paths. On Windows they are under `%LOCALAPPDATA%\nlp-trader\kabus`; changing config files does not
change or reset them. Consult [Broker integration](broker.md) before reconciling or resolving an
ambiguous mutation, and never edit or truncate the ledger by hand.

### `operation.lock` still exists after the command

This is normal. The file is stable and must never be deleted. The operating system lock on its open
descriptor provides exclusivity and is released on normal close or process exit. If a command
reports lock contention, find the other running broker process; deleting the file can create a
second lock domain and permit unsafe concurrent operations.

## Performance and artifact problems

### Full run uses too much memory

The pipeline is not end-to-end out-of-core. Narrow `--start-date`, `--end-date`, repeated `--symbol`,
and `--limit`; confirm the bounded run first, then widen deliberately.

### Synthetic generator refuses the output

At least one of `assets.csv`, `market_bars.csv`, or `text_items.jsonl` already exists in the output
directory. Choose another directory. The generator does not overwrite those files.

### Dirty or missing Git commit in a manifest

The run records the repository’s actual Git state. Make an intentional initial commit or commit your
research code before a reproducibility-sensitive run. Do not falsify the manifest.

### No `run.failed.json`

Config/path validation occurs before run creation. A failure at that point has no run directory.
Only a stage failure after context creation writes `run.failed.json`.

### Failure message contains sensitive text

Failure manifests store the exception type and message verbatim. Providers and errors must not embed
secrets, raw restricted payloads, or personal data in exception messages.

## Still stuck?

Collect:

- the exact command;
- config validation output;
- failing stage and error text;
- `run.failed.json` if one exists;
- relevant schema/header names, without restricted data or credentials; and
- the Python/`uv` versions.

Then compare against [Input data](input_data.md), [Configuration](configuration.md),
[Data contracts](data_contracts.md), and, for broker commands, [Broker integration](broker.md).

Return to the [documentation home](README.md).
