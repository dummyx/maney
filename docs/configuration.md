# Configuration Reference

NLP Trader uses strict, immutable YAML configuration. Unknown keys are rejected, numeric values must
be finite, and CLI overrides are validated before a run ID or config hash is created.

The bundled examples are:

- `configs/sample.yaml` — fast synthetic smoke configuration;
- `configs/backtest.yaml` — synthetic data with stricter backtest assumptions; and
- `configs/local.yaml` — a template for user-provided licensed data.

## Path rules

Relative paths are resolved from the directory containing the YAML file, not from the shell’s
current directory.

The five artifact roots—raw, interim, processed, models, and reports—must be distinct and must not
contain one another. Input files may not sit inside an artifact root. These rules prevent a run from
capturing its own outputs as source data.

## Top-level fields

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
```

## `data`

| Field | Default or allowed values | Notes |
|---|---|---|
| `storage_format` | `parquet` | Derived analytical tables are Parquet. |
| `compression` | `zstd`, `snappy`, `uncompressed` | `zstd` is the bundled default. |
| `write_batch_rows` | `10000`, integer ≥ 1 | Maximum pending rows per shared Parquet-writer flush. |
| `calendar` | `XNYS` | Exchange calendar used for current daily decisions and label windows. |
| `schema_version` | nonempty string | Input/bronze provenance version. |
| `market_license_or_terms_ref` | nonempty string | Human-resolvable rights/terms reference for market inputs. |
| `text_license_or_terms_ref` | nonempty string | Human-resolvable rights/terms reference for text inputs. |

A terms reference records provenance; it does not prove that the source is licensed.

## `features`

| Field | Rule | Meaning |
|---|---|---|
| `windows_days` | unique positive integers | Text aggregation windows. Bundled configs use `1, 3, 5, 20`. |
| `market_warmup_sessions` | at least 60 | Prior exchange sessions loaded before a requested start. |
| `text_warmup_days` | at least twice the largest text window | Prior calendar days loaded for text history and rolling baselines. |
| `event_lookahead_days` | positive integer | Bounded future event scan after the last decision; availability rules still apply. |
| `horizon_days` | positive integer | Number of XNYS sessions from entry open to exit close. |
| `feature_set_version` | nonempty string | Part of the canonical feature key and storage partition. |
| `label_version` | nonempty string | Version recorded on generated labels. |
| `model_version` | nonempty string | Registry and prediction version. |
| `text_decay_half_life_days` | positive float | Age decay applied to text contributions. |
| `decision_time` | `close` | The only implemented decision clock. |

`horizon_days` must match `backtest.rebalance_frequency`. A one-session horizon therefore uses
`rebalance_frequency: 1d`.

## `models`

| Field | Rule | Meaning |
|---|---|---|
| `families` | exactly `[traditional, text, combined]` | The canonical baselines, in this order. |
| `min_train_rows` | integer ≥ 2 | Minimum eligible historical rows before a snapshot is fitted. |
| `embargo_periods` | integer ≥ 0 | Recent decision periods excluded from the training cutoff. |
| `top_k` | integer ≥ 1 | Ranking depth used by evaluation and latest paper-intent selection. |

Equal-weight, momentum-only, and no-trade benchmarks are added automatically; do not list them in
`families`.

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

Dates may be ISO dates or timezone-aware ISO timestamps. Validation rejects an end whose UTC
calendar date precedes the start date. For two timestamps on the same UTC date, also ensure the exact
end instant is not earlier; that finer ordering is not currently rejected by config validation.
Plain dates are usually clearer for this daily close-decision pipeline.

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

## Validate before running

```bash
uv run nlp-trader validate-config --config configs/local.yaml
```

Validation is intentionally strict. Fix the first reported contract issue rather than weakening a
schema or timestamp rule to accommodate ambiguous data.

Related documentation:

- [Input data](input_data.md)
- [Workflows](workflows.md)
- [Backtesting](backtesting.md)
- [Troubleshooting](troubleshooting.md)

Return to the [documentation home](README.md).
