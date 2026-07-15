# Configuration Reference

NLP Trader uses strict, immutable YAML configuration. Unknown keys are rejected, numeric values must
be finite, and CLI overrides are validated before a run ID or config hash is created.

The bundled research examples are:

- `configs/sample.yaml` — fast synthetic smoke configuration;
- `configs/backtest.yaml` — synthetic data with stricter backtest assumptions; and
- `configs/local.yaml` — a generic template for user-provided licensed data; and
- `configs/japan_baseline.yaml` — a strict XJPX/Japanese cash-equity template for permitted local
  exports. It does not include or download market data.

The broker adapter uses its own strict schema and CLI category rather than a section in a research
config. `configs/kabus.validation.yaml` is the standalone validation example; validate it with
`uv run nlp-trader broker validate-config --config configs/kabus.validation.yaml`. Broker audit,
kill-switch, and operation-lock paths are fixed current-user safety state shared by all broker
configs and both environments; they are not YAML fields. See [Broker integration](broker.md) for the
broker schema, password handling, fixed state paths, and production gates.

## Path rules

Relative paths are resolved from the directory containing the YAML file, not from the shell’s
current directory.

The five artifact roots—raw, interim, processed, models, and reports—must be distinct and must not
contain one another. Input files may not sit inside an artifact root. These rules prevent a run from
capturing its own outputs as source data.

## Research top-level fields

| Field | Values | Meaning |
|---|---|---|
| `mode` | `sample` or `full` | Labels the run and selects the intended operating profile. It does not fetch data. |
| `paths` | mapping | Required inputs, optional point-in-time inputs, and artifact roots. |
| `data` | mapping | Storage, calendar, schema, and licensing metadata. |
| `features` | mapping | Windows, warm-up, horizon, decay, and version identifiers. |
| `models` | mapping | Canonical baseline or LLM-ablation families and walk-forward training controls. |
| `backtest` | mapping | Cost model, constraints, capital, and rebalance settings. |
| `runtime` | mapping | Optional date, symbol, and complete-decision limits. |
| `transformer` | mapping | Optional local transformer inference settings. |
| `llm_annotations` | mapping | Optional local generative semantic/evidence signal, verifier, and inference-accounting settings. |

## `paths`

| Field | Required | Purpose |
|---|---:|---|
| `assets` | yes | Asset-master CSV, JSON, JSONL, Parquet file, or Parquet directory. |
| `market_bars` | yes | Daily market-bar records. |
| `text_items` | yes | Permitted natural-language records. |
| `fundamentals` | no | Point-in-time fundamental records. |
| `earnings_calendar` | no | Point-in-time known earnings events. |
| `corporate_actions` | no | Point-in-time known corporate-action events used as features. |
| `raw_dir` | no | Shared append-only, content-addressed bronze root. Defaults to `../data/raw` if omitted. |
| `interim_dir` | yes | Base for per-run normalized silver artifacts. |
| `processed_dir` | yes | Base for per-run signals, gold tables, evaluations, and replay records. |
| `models_dir` | yes | Base for per-run model and metric artifacts. |
| `reports_dir` | yes | Base for config snapshots, manifests, and research notes. |
| `llm_model` | no | Direct path to one local GGUF model file. Required only when `llm_annotations.enabled` is true. |

Optional input example:

```yaml
paths:
  assets: ../data/local/assets.parquet
  market_bars: ../data/local/market_bars.parquet
  text_items: ../data/local/text_items.parquet
  fundamentals: ../data/local/fundamentals.parquet
  earnings_calendar: ../data/local/earnings_calendar.parquet
  corporate_actions: ../data/local/corporate_actions.parquet
  raw_dir: ../data/raw
  interim_dir: ../data/interim/local
  processed_dir: ../data/processed/local
  models_dir: ../models/local
  reports_dir: ../reports/local
  llm_model: /absolute/path/to/Qwen3.6-27B-UD-Q4_K_XL.gguf
```

## `data`

| Field | Default or allowed values | Notes |
|---|---|---|
| `storage_format` | `parquet` | Derived analytical tables are Parquet. |
| `compression` | `zstd`, `snappy`, `uncompressed` | `zstd` is the bundled default. |
| `write_batch_rows` | `10000`, integer ≥ 1 | Maximum pending rows per shared Parquet-writer flush. |
| `calendar` | `XNYS` or `XJPX` | Exchange calendar used for daily sessions and label windows. |
| `market_contract` | `generic` or `japan_cash_equity_v1` | Exact local market-input contract. `XJPX` and `japan_cash_equity_v1` must be selected together. |
| `schema_version` | `2`, nonempty string | Input/bronze provenance version. |
| `market_license_or_terms_ref` | nonempty string | Human-resolvable rights/terms reference for market inputs. |
| `text_license_or_terms_ref` | nonempty string | Human-resolvable rights/terms reference for text inputs. |

A terms reference records provenance; it does not prove that the source is licensed. The Japanese
template contains conspicuous placeholder references that must be replaced with human-resolvable
records for the exact local exports. See [Japan cash-equity baseline](japan_baseline.md).

## `features`

| Field | Rule | Meaning |
|---|---|---|
| `windows_days` | unique positive integers | Text aggregation windows. Bundled configs use `1, 3, 5, 20`. |
| `market_warmup_sessions` | at least 60 | Prior exchange sessions loaded before a requested start. |
| `text_warmup_days` | at least twice the largest text window | Prior calendar days loaded for text history and rolling baselines. |
| `event_lookahead_days` | positive integer | Bounded future event scan after the last decision; availability rules still apply. |
| `horizon_days` | positive integer | Number of configured-exchange sessions from entry open to exit close. |
| `feature_set_version` | nonempty string | Part of the canonical feature key and storage partition. |
| `label_version` | nonempty string | Version recorded on generated labels. |
| `model_version` | nonempty string | Registry and prediction version. |
| `text_decay_half_life_days` | positive float | Age decay applied to text contributions. |
| `decision_time` | `close` | The only implemented daily-bar clock. The strict Japanese contract may set `asof_ts` after close when the bar becomes available. |

`horizon_days` must match `backtest.rebalance_frequency`. A one-session horizon therefore uses
`rebalance_frequency: 1d`.

## `models`

| Field | Rule | Meaning |
|---|---|---|
| `families` | one of the two exact ordered sets below | The canonical conventional baseline or LLM-ablation families. |
| `min_train_rows` | integer ≥ 2 | Minimum eligible historical rows before a snapshot is fitted. |
| `embargo_periods` | integer ≥ 0 | Recent decision periods excluded from the training cutoff. |
| `final_holdout_periods` | integer ≥ `features.horizon_days` | Number of final fully observed decision periods reserved from development diagnostics and reported separately. |
| `top_k` | integer ≥ 1 | Ranking depth used by evaluation and model-scored backtest/paper selection. Equal-weight and no-trade references are uncapped. |

The accepted family sets are:

```yaml
# Deterministic/conventional baseline
families: [traditional, text, combined]

# Required when llm_annotations.feature_mode is augment
families: [traditional, text, combined, llm, traditional_llm, all]
```

The first three keep fixed meanings: `traditional` is deterministic numeric data, `text` is
conventional text only, and `combined` is their union. The additional families isolate LLM-only,
numeric-plus-LLM, and all-three feature paths without changing the conventional columns. Equal-weight,
momentum-only, and no-trade benchmarks are added automatically; do not list them in `families`.

The holdout boundary is chronological and counts only fully observed whole cross-sections. It is an
evaluation boundary, not protection against a researcher repeatedly inspecting and tuning to the same
terminal period. Development purges the contiguous whole-cross-section suffix beginning with the
first decision whose labels are not all available before the holdout start. The trainer freezes the
embargo-adjusted fit at the holdout start: every holdout prediction uses the same training-key
membership, and no holdout outcome updates it.

## `backtest`

### Cost settings

| Field | Meaning |
|---|---|
| `commission_bps` | Commission or fee per traded notional. |
| `half_spread_bps` | Half-spread crossing cost. |
| `slippage_bps` | Base slippage. |
| `volatility_slippage_multiplier` | Adds slippage as decision-time volatility rises. |
| `participation_slippage_bps` | Adds slippage as participation rises. |
| `market_impact_multiplier` | Volatility/participation market-impact proxy. |
| `borrow_bps_per_year` | Annualized short borrow proxy. |

All cost inputs must be non-negative. See [Backtesting](backtesting.md) for the formulas.

### Portfolio and liquidity settings

| Field | Rule | Meaning |
|---|---|---|
| `max_position_weight` | `0 < value <= 1` | Maximum absolute weight per asset. |
| `max_gross_exposure` | positive | Sum of absolute target weights. |
| `max_net_exposure` | non-negative | Absolute long-minus-short exposure. |
| `max_sector_weight` | `0 < value <= 1` | Maximum gross weight in one sector. |
| `max_beta_exposure` | non-negative | Maximum absolute portfolio beta. |
| `missing_beta_fallback` | non-negative | Conservative beta used when history is insufficient. |
| `missing_volatility_floor` | positive | Minimum volatility used when the 20-session estimate is missing. |
| `max_daily_turnover` | positive | Daily entry/exit turnover budget. |
| `same_day_exit_notional_buffer` | non-negative | Entry reserve for a one-session exit whose notional may grow. |
| `max_participation_rate` | `0 < value <= 1` | Maximum trade notional divided by decision-time dollar volume. |
| `min_price` | positive | Candidate eligibility floor. |
| `min_dollar_volume` | non-negative | Candidate liquidity floor. |
| `shorting_allowed` | boolean | Enables negative requested weights only when other short checks pass. |
| `hard_to_borrow_allowed` | boolean | Allows assets explicitly marked hard to borrow. |

`max_position_weight` and `max_net_exposure` may not exceed `max_gross_exposure`.

### Run settings

| Field | Default | Meaning |
|---|---:|---|
| `initial_capital` | `1000000` | Converts weights into notional for participation and capacity proxies. |
| `rebalance_frequency` | `1d` | Positive integer days; must match the configured feature horizon. |
| `benchmark` | `equal_weight` | Recorded benchmark identifier. It does not select which families run; every configured learned family plus equal-weight, momentum-only, and no-trade is replayed. |

## `runtime`

| Field | Meaning |
|---|---|
| `start_date` | Inclusive lower bound on emitted decision timestamps. |
| `end_date` | Inclusive upper bound on emitted decision timestamps. |
| `symbols` | Unique uppercase symbols. Empty means the configured universe. |
| `limit` | Number of earliest complete decision timestamps to retain after filters. |

For prediction/backtest/report stages, the filtered range must contain more fully observed periods
than `models.final_holdout_periods`; otherwise no development window remains.

Dates may be ISO dates or timezone-aware ISO timestamps. Validation rejects an end whose UTC
calendar date precedes the start date. For two timestamps on the same UTC date, also ensure the exact
end instant is not earlier; that finer ordering is not currently rejected by config validation.
Plain dates are usually clearer for this availability-aware daily-decision pipeline.

Runtime bounds do not cut off required context. The pipeline loads market/text warm-up before the
start, market label context after the end, and bounded known-event context after the final selected
decision. Gold outputs are filtered back to the requested decision interval.

CLI options override these fields before config hashing:

```bash
uv run nlp-trader backtest \
  --config configs/local.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbol AAPL \
  --symbol MSFT \
  --limit 100
```

## `transformer`

| Field | Default | Meaning |
|---|---:|---|
| `enabled` | `false` | Replaces baseline sentiment fields with optional transformer results. |
| `model_name` | `null` | Local model identifier/path; required when enabled. |
| `model_version` | `local-transformer-v1` | Included in cache identity and signals. |
| `batch_size` | `32` | Inference batch size. |
| `max_sequence_length` | `256` | Tokenizer truncation length. |
| `local_files_only` | `true` | Prevents model/tokenizer download in the bundled configs. Keep this true for reproducible local runs. |

Install the optional packages and ensure the model already exists locally:

```bash
uv sync --extra nlp
uv run nlp-trader smoke \
  --config configs/sample.yaml \
  --enable-transformer-sentiment
```

The CLI flag is folded into the typed config before the run snapshot is created.

The run records `model_name`, `model_version`, and inference settings, but it does not currently hash
the external model weights. Keep your own immutable model revision/checksum record for substantive
transformer experiments. Setting `local_files_only: false` may contact a model hub and should be an
explicit, licensed, reproducible choice.

## `llm_annotations`

This separate optional component runs one local GGUF model through `llama-cpp-python==0.3.34` to
produce validated per-entity semantic/evidence signals. It is disabled in every bundled config and
`paths.llm_model` defaults to null. The deterministic sample and baseline therefore require neither
the `llm` extra nor a model file.

| Field | Default or allowed value | Meaning |
|---|---|---|
| `enabled` | `false` | Runs the annotation stage when true. |
| `feature_mode` | `sidecar` or `augment` | `sidecar` records verified output without adding LLM feature values. `augment` adds separate `llm_*` columns and requires `enabled: true` plus the six canonical LLM-ablation model families. |
| `backend` | `llama_cpp_gguf` | The only supported generative backend. It loads the configured GGUF directly in the Python process. |
| `model_id` | `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` | Logical model selector recorded separately from the local file path. |
| `model_revision` | `5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf` | Exact pinned repository revision. |
| `model_license_or_terms_ref` | pinned Hugging Face revision URL | Human-resolvable license/terms reference. Recording it does not grant model or source-data rights. |
| `prompt_version` | `semantic-evidence-v2` | Version included in prompt provenance and cache identity. |
| `schema_version` | `semantic-signal-v2` | Version of the validated structured-output contract. |
| `verifier_version` | `semantic-evidence-verifier-v1` | Version of the deterministic request/response verifier and cache identity. |
| `batch_size` | `1`, integer ≥ 1 | Number of host requests grouped for accounting. The backend still submits each request through ordinary in-process generation; it does not imply simultaneous token generation. |
| `max_input_tokens` | `2048`, integer ≥ 1 | Maximum prompt size. An oversized request abstains rather than silently truncating source text. |
| `max_new_tokens` | `384`, integer ≥ 1 | Maximum reply size for one request. |
| `context_tokens` | `8192`, integer ≥ 1 | Total model window available to the prompt and reply. It must be at least `max_input_tokens + max_new_tokens`. |
| `prompt_batch_tokens` | `512`, integer ≥ 1 | Internal llama.cpp prompt-evaluation chunk size, not the number of documents. It cannot exceed `context_tokens`. |
| `gpu_layers` | `-1`, integer ≥ -1 | `-1` requests all possible layers on llama.cpp Metal, `0` forces CPU, and a positive value requests that many layers. If the binding reports no GPU-offload support, the runtime uses `0`. |
| `flash_attention` | `true` | Requests llama.cpp flash attention when the installed build/model supports it. |
| `use_mmap` | `true` | Memory-maps the GGUF instead of eagerly reading the whole file. It does not remove the need for runtime working memory. |
| `decoding` | `greedy` | Deterministic decoding policy. |
| `seed` | `7`, integer ≥ 1 | Recorded generation seed. |
| `input_cost_per_million_tokens_usd` | `null` or non-negative float | Optional configured input-token rate used only to estimate cost for newly generated responses with recorded token counts. |
| `output_cost_per_million_tokens_usd` | `null` or non-negative float | Optional configured output-token rate. It must be set together with the input rate. |

The default setup is pinned to the direct local file
`Qwen3.6-27B-UD-Q4_K_XL.gguf`. For the default model ID and revision, startup requires this exact
SHA-256:

```text
4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095
```

A mismatch fails before inference. The file is about 17.9 GB; 32 GB or more of unified memory is a
practical starting point because context, key/value cache, and other working buffers need additional
memory. Keep `batch_size: 1` first and reduce the context if necessary. `use_mmap` can reduce eager
copying but does not guarantee that a lower-memory Mac can run the model.

Example enabled sidecar configuration:

```yaml
paths:
  llm_model: /absolute/path/to/Qwen3.6-27B-UD-Q4_K_XL.gguf
llm_annotations:
  enabled: true
  feature_mode: sidecar
  backend: llama_cpp_gguf
  model_id: unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL
  model_revision: 5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf
  model_license_or_terms_ref: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/tree/5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf
  batch_size: 1
  max_input_tokens: 2048
  max_new_tokens: 384
  context_tokens: 8192
  prompt_batch_tokens: 512
  gpu_layers: -1
  flash_attention: true
  use_mmap: true
```

The runtime never resolves the logical selector or downloads weights. Download the GGUF yourself,
review its license/terms, and provide the direct file path. The run input manifest and annotation
provenance record its exact bytes as `model_file_sha256`. Provenance also records the
`llama-cpp-python` version, requested/effective GPU layers, the GGUF-embedded chat-template hash,
context/runtime settings, and `backend: llama_cpp_gguf`. A missing embedded chat template is an
error; the runtime does not silently substitute an unversioned template.

Metal here means llama.cpp layer offload, not PyTorch MPS. The native binding decides whether GPU
offload is available. The runtime uses CPU when the installed binding reports no GPU support or
when `gpu_layers: 0` is set. A Metal-enabled build can still initialize Metal with zero offloaded
layers, so a machine that cannot create a Metal context needs a CPU-only build. The repository/model
name contains `MTP`, but the in-process API performs ordinary inference and records
`mtp_speculative_decoding: false`; this integration makes no MTP acceleration claim.

Normal tests inject a generator and never load or download a real model. Passing them proves the
host contracts, cache, verifier, and fake-backend integration, not that this GGUF loads or generates
on a particular Mac. Use the gated acceptance test in [Workflows](workflows.md) for that check.
Configured costs are estimates, not invoices; if rates or token counts are unavailable for a newly
generated response, its per-round estimated cost remains null rather than being invented. If any
newly generated response lacks token counts, run-level token totals and configured cost are null
rather than partial. Without configured rates, the run-level cost estimate is also null.

The two enabled modes are intentionally distinct:

- `enabled: true`, `feature_mode: sidecar` writes verified sidecars and decision-round audit records
  for review without adding LLM values to gold features;
- `enabled: true`, `feature_mode: augment` adds separate LLM semantic, confidence, uncertainty,
  event, evidence, coverage, abstention, and missingness columns. It does not replace conventional
  sentiment or event values.

Transformer sentiment may coexist with either mode. When enabled, it supplies the conventional text
sentiment fields while LLM augmentation remains isolated under `llm_*` columns.

The v2 structured output contains stance, an integer `semantic_signal` from -2 through 2,
uncalibrated `raw_confidence`, uncertainty, the configured horizon, primary event and confidence,
supporting and counterevidence span IDs, mechanism, invalidation conditions, and explicit abstention.
Raw confidence is a feature only: it is not a probability, semantic-signal magnitude, position size,
or portfolio weight.

The deterministic verifier requires exact item/candidate coverage, source availability no later than
the decision, the configured horizon, known source-local evidence span references, and numeric tokens
in mechanisms/invalidation conditions to occur in cited spans. These checks do not prove that a prose
claim, event interpretation, or mechanism is semantically true. The current prompt uses only the
current source item; no RAG, external retrieval, tools, or model router is implemented.

Do not switch `feature_mode` within one experiment or treat a sidecar-only run as an applied-LLM
performance comparison. Predeclare sidecar review and augment ablations with distinct feature/model
versions; see [Research protocol](research_protocol.md).

## Validate before running

```bash
uv run nlp-trader validate-config --config configs/local.yaml
```

For the strict Japanese template:

```bash
uv run nlp-trader validate-config --config configs/japan_baseline.yaml
```

That command is expected to report missing inputs until permitted local files have been prepared.

Validation is intentionally strict. Fix the first reported contract issue rather than weakening a
schema or timestamp rule to accommodate ambiguous data.

This command validates research configs. Broker configs use `nlp-trader broker validate-config` and
must remain separate; the validation environment is a fixed-response endpoint, whereas production
can submit real orders from the same Windows PC as kabuStation. See [Broker integration](broker.md).

Related documentation:

- [Input data](input_data.md)
- [Workflows](workflows.md)
- [Backtesting](backtesting.md)
- [Broker integration](broker.md)
- [Troubleshooting](troubleshooting.md)

Return to the [documentation home](README.md).
