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
| `models` | mapping | Fixed baseline families and walk-forward training controls. |
| `backtest` | mapping | Cost model, constraints, capital, and rebalance settings. |
| `runtime` | mapping | Optional date, symbol, and complete-decision limits. |
| `transformer` | mapping | Optional local transformer inference settings. |
| `llm_annotations` | mapping | Optional local generative entity-stance/event annotation settings. |

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
| `llm_model` | no | Local generative model directory. Required only when `llm_annotations.enabled` is true. |

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
  llm_model: /absolute/path/to/immutable/local-model
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
| `families` | exactly `[traditional, text, combined]` | The canonical baselines, in this order. |
| `min_train_rows` | integer ≥ 2 | Minimum eligible historical rows before a snapshot is fitted. |
| `embargo_periods` | integer ≥ 0 | Recent decision periods excluded from the training cutoff. |
| `final_holdout_periods` | integer ≥ `features.horizon_days` | Number of final fully observed decision periods reserved from development diagnostics and reported separately. |
| `top_k` | integer ≥ 1 | Ranking depth used by evaluation and model-scored backtest/paper selection. Equal-weight and no-trade references are uncapped. |

Equal-weight, momentum-only, and no-trade benchmarks are added automatically; do not list them in
`families`.

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
| `benchmark` | `equal_weight` | Recorded benchmark identifier. It does not select which families run; all six current families are replayed. |

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

This separate optional component runs a local causal language model to produce validated per-entity
stance/event annotations. It is disabled in every bundled config, so the deterministic sample and
baseline do not require a model or PyTorch.

| Field | Default or allowed value | Meaning |
|---|---|---|
| `enabled` | `false` | Runs the annotation stage when true. |
| `apply_to_features` | `false` | When true, applies valid non-abstained entity annotations to text-signal sentiment/event fields. Requires `enabled: true`. |
| `backend` | `transformers_causal_lm` | Local Hugging Face causal-LM backend. |
| `model_id` | `null` | Stable human/model-registry identifier; required when enabled. It is recorded separately from `paths.llm_model`. |
| `model_revision` | `null` | Exact immutable model revision/version; required when enabled. |
| `model_license_or_terms_ref` | `null` | Human-resolvable license/terms reference for local model use; required when enabled. |
| `prompt_version` | `entity-event-v1` | Version included in prompt provenance and cache identity. |
| `schema_version` | `entity-event-v1` | Version of the validated structured-output contract. |
| `batch_size` | `1`, integer ≥ 1 | Configurable generation batch size. |
| `max_input_tokens` | `2048`, integer ≥ 1 | Context budget; an oversized request abstains rather than silently truncating source text. |
| `max_new_tokens` | `384`, integer ≥ 1 | Maximum generated tokens per request. |
| `decoding` | `greedy` | Deterministic decoding policy. |
| `seed` | `7`, integer | Recorded generation seed. |
| `local_files_only` | `true` | Forbids model/tokenizer downloads; keep true for this local-only component. |
| `trust_remote_code` | `false` | Prevents execution of model-repository custom code; keep false. |

`paths.llm_model` must name the local model directory when enabled. The run input manifest hashes
its exact files, while the config and annotation provenance retain the logical ID/revision/license,
prompt/schema versions, decoding settings, and context limits. Inference uses central MPS-or-CPU
device selection. Tests inject a generator and never load or download a real model.

The two enabled modes are intentionally distinct:

- `enabled: true`, `apply_to_features: false` writes a sidecar for review without changing the
  deterministic feature path;
- `enabled: true`, `apply_to_features: true` replaces only valid, non-abstained per-entity
  sentiment/event fields. Explicit abstentions retain the deterministic values and are counted.

`transformer.enabled: true` may coexist with sidecar-only annotation, but it cannot be combined with
`llm_annotations.apply_to_features: true`: both would compete to replace sentiment fields, so config
validation rejects that combination.

Do not switch `apply_to_features` within one experiment or treat a sidecar-only run as an LLM trading
comparison. Use matched disabled/applied runs with distinct feature/model versions; see
[Research protocol](research_protocol.md).

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
