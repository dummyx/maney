# Research Protocol

The system preserves experiments; it does not decide whether a strategy is economically valid. A
successful smoke test proves that the implementation runs, not that a signal has value.

The shortest responsible workflow is: define the question, freeze inputs and assumptions, validate
the config, run the walk-forward comparison, inspect implementation diagnostics, and only then form a
research conclusion. [Getting started](getting_started.md) explains how to execute the pipeline;
this page explains how to judge a run.

## Before a run

1. Confirm that every input is licensed for the intended research and local retention.
2. Use a survivorship-aware asset universe where the research question requires one; document any
   universe limitation.
3. Verify that source timestamps support a defensible `available_at`. Do not use a source when this
   cannot be established.
4. Choose the horizon, feature/label/model versions, date range, symbol filters, execution costs, and
   constraints before inspecting final results. The implemented decision is after the completed
   official-close bar; entry is the raw next official open and exit is the raw exact horizon close.
   Return outcomes compare those tradable prices on the causal per-bar adjustment basis.
5. Reserve an untouched terminal test period outside this pipeline when making a final research
   claim. The implemented expanding walk-forward evaluation does not create a separate terminal
   holdout automatically.
6. Validate the config and retain the exact local inputs; do not edit a source file during a run.

Runtime dates define the decisions being studied. The implementation automatically loads the
configured market/text warm-up before the start and exact-session label lookahead after the end, then
filters feature and label outputs back to the inclusive decision interval. Confirm that the source
contains that context; the pipeline does not fabricate missing history or future bars.

Known earnings and corporate-action event context is bounded separately to
`event_lookahead_days` calendar days after the last selected decision (30 by default), and remains
gated by `available_at <= asof_ts`.

## Immutable run record

Every pipeline execution command creates a new run directory. `validate-config` and
`generate-synthetic` do not create runs. At run start, the system records the UTC creation time,
config snapshot/hash, input paths, SHA-256 values, byte counts, and Git commit/dirty state. At
completion it writes an exclusive final manifest containing:

- `run_id`, creation/completion time, status, and completed stage
- code version and dirty-worktree status
- config hash and a separately written full `config.snapshot.json`
- input data manifest and hashes for every materialized artifact other than the final manifest itself
- universe, period, rebalance frequency, and feature/label/model versions
- cost model, portfolio constraints, metrics, known limitations, and next questions

CLI overrides are applied to the typed config before run creation. This includes runtime filters and
the smoke command's transformer enable flag, so they affect the config hash and snapshot.

Failures write a separate failure manifest with the exception type and message, but no traceback, and
preserve partial artifacts. Because provider exception text is retained, adapters must never place
credentials or restricted payloads in error messages. Existing run IDs are never overwritten. Do
not delete losing runs merely because their results are inconvenient. Source capture verifies the
configured file or Parquet-directory manifest against these run-start hashes, then provider decoding
uses the immutable captured bytes. A mutation between manifesting and capture fails rather than
silently changing the experiment.

## Model evaluation

The baseline trainer creates incremental expanding walk-forward snapshots. At each decision time it
adds only labels whose `label_end_ts` is strictly earlier than the effective training cutoff. The
configured embargo removes recent decision periods, and no model is fitted until `min_train_rows` is
met. Coefficient fitting uses running sufficient statistics instead of rebuilding a historical
feature matrix for every snapshot. Default training also updates an ordered membership digest
incrementally, so it does not retain every eligible key; explicit key lists are available only through
the opt-in diagnostic `record_training_keys` path.

The typed config requires these three canonical model families, in this order:

- traditional-only
- text-only
- combined traditional plus text

It also evaluates fixed naive benchmarks:

- equal-weight score
- momentum-only score
- no-trade score

Prediction diagnostics include aggregate and mean-daily Pearson/Spearman IC, hit rate, precision at
k, and mean squared error. Binary labels also produce Brier score, expected calibration error, and
calibration bins. An explicit `probability_up` is used when present; the current baseline predictions
otherwise apply a logistic transform to their score, so those calibration values are diagnostics of
an uncalibrated score proxy rather than a calibrated probability model.

Each family is also evaluated, when contemporaneous metadata is present, by sector, fixed liquidity
bucket, fixed volatility regime, source availability/type, and event availability/type. Thresholds
are fixed in code and use only decision-row context. Every family is then passed through the same
portfolio, two-sided cost, forced-liquidation, and backtest machinery. Compare the combined model
with traditional-only and naive results; a positive combined backtest alone is not evidence that text
added value.

The current pipeline does not automatically create purged cross-validation folds or a final untouched
holdout. It emits calibration tables and segmented metrics, not calibration plots, statistical
significance tests, or a causal attribution of performance to a segment. Those analyses must be
added or performed separately before a substantive conclusion.

## Acceptance checklist

Before accepting a result, verify:

- No feature input has `available_at > asof_ts`; after-hours text uses the next configured decision.
- Every daily OHLC row retains raw tradable prices, is marked
  `corporate_action_adjusted=true`, has `adjustment_vintage_at <= ts`, and supplies a positive causal
  `return_adjustment_factor`; optional corporate-action events and `adjusted_close` do not substitute
  for that point-in-time contract.
- Label windows record the raw exact next-session open and horizon-session close as execution
  prices, calculate returns from each `raw_price * return_adjustment_factor`, and enter training only
  after the label is observable.
- The universe is not unintentionally survivorship-only and does not drop delisted assets as missing.
- Fundamentals and corporate actions are genuinely point-in-time; provider contracts alone do not
  make a dataset point-in-time correct.
- Syndicated/duplicate text is not counted as independent evidence; the incremental cluster assignment
  uses no future document.
- Historical social metadata existed at the decision time and is licensed for retention.
- Model selection did not use the reserved final test period.
- Commission, spread, slippage, impact, borrow, turnover, participation, liquidity, and fill
  assumptions are appropriate for the data and strategy.
- Shorts have reliable availability, hard-to-borrow, and borrow-cost inputs. The bundled pipeline
  defaults them unavailable and uses long-only configs.
- Results are stable enough across time, sectors, liquidity, volatility, source, and event type to
  justify further research.
- Reports show drawdown, exposures, round-trip turnover, both execution legs, costs, tail loss,
  per-period diagnostics, and the capacity proxy, not only cumulative return or Sharpe.

## Interpreting sample output

Sample fixtures are deterministic, synthetic, tiny, and intentionally easy to inspect. Their
annualized metrics can be numerically extreme because the period is short. They are regression and
leakage tests only and must never be presented as expected performance.

Use [Outputs](outputs.md) to locate the evidence for each checklist item and
[Backtesting](backtesting.md) to interpret fills, costs, constraints, and replay metrics.

Return to the [documentation home](README.md).
