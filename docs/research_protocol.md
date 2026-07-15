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
5. Set `models.final_holdout_periods` before inspecting results. The pipeline reserves those final
   fully observed decision periods from top-level development diagnostics and reports them separately.
   Do not repeatedly tune against that output; after inspection, a new untouched period is required
   for another confirmatory claim.
6. Validate the config and retain the exact local inputs; do not edit a source file during a run.

Runtime dates define the decisions being studied. The implementation automatically loads the
configured market/text warm-up before the start and exact-session label lookahead after the end, then
filters feature and label outputs back to the inclusive decision interval. Confirm that the source
contains that context; the pipeline does not fabricate missing history or future bars.

Known earnings and corporate-action event context is bounded separately to
`event_lookahead_days` calendar days after the last selected decision (30 by default), and remains
gated by `available_at <= asof_ts`.

### Predeclare a generative-annotation experiment

Treat optional generative annotation as a separate, frozen retrospective-parser experiment. First
run `feature_mode: sidecar` and evaluate a fixed labeled annotation set with stance/event macro-F1,
supporting- and counterevidence precision, horizon accuracy, abstention and invalid-output rates, and
raw-confidence calibration diagnostics. Freeze the local model bytes, prompt version, output schema,
verifier version, decoding settings, token-cost assumptions, and labeled set before market
evaluation.

For the bundled default, freeze the direct file `Qwen3.6-27B-UD-Q4_K_XL.gguf`, logical selector
`unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL`, revision
`5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf`, and SHA-256
`4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095`. Retain the
`model_file_sha256`, `backend: llama_cpp_gguf`, `llama-cpp-python` version, embedded chat-template
hash, context sizes, and requested/effective GPU layers. Changing any of those creates a different
model/runtime condition and requires a new experiment identity.

Run the environment-gated real-model acceptance test on the target Mac before collecting sidecars.
The normal test suite uses injected generators and does not prove that the 17.9 GB model loads or
generates there. Conversely, a successful acceptance test proves only local loading/inference, not
annotation accuracy. The default model name includes `MTP`, but this backend performs ordinary
in-process inference; do not attribute a result or speed claim to MTP speculative decoding.

Sidecar review must include the processing and deterministic-verification summaries, raw generation
attempts, exact response artifacts, and replay-verified DecisionRound ledger. The verifier must pass
its identity/coverage, timing, horizon, evidence-reference, and cited-numeric-token checks. Passing
does not establish semantic truth; manually audit whether cited spans actually support the stance,
event, mechanism, and invalidation conditions.

For an applied experiment, create a new versioned config with `feature_mode: augment` and the exact
six learned families:

```text
traditional, text, combined, llm, traditional_llm, all
```

Their meanings are fixed: numeric only, conventional text only, numeric plus conventional text, LLM
only, numeric plus LLM, and all three feature groups. Conventional text is never overwritten. This
single run produces matched development/final-holdout arithmetic comparisons for `llm` versus
`text`, `traditional_llm` versus `traditional`, and `all` versus `combined`. Also retain a matched
LLM-disabled baseline run with the same data, universe, dates, costs, constraints, horizon, holdout,
and seeds when validating that enabling the subsystem did not affect conventional paths.

Use distinct feature/model versions. A sidecar run is useful for annotation review but is not an
applied-LLM performance comparison. Preserve positive and negative results; do not tune the prompt or
model on Sharpe or repeatedly inspect the final holdout. The ablation artifact reports arithmetic
deltas, not significance, causality, profitability, or automatic promotion.

`semantic_signal` carries the model's discrete source interpretation. `raw_confidence` is explicitly
uncalibrated and remains a separate feature; never treat it as a probability, signal magnitude,
position size, or portfolio weight.

Point-in-time prompts are necessary but not sufficient for historical validity. A modern pretrained
model can encode facts learned after a source document’s `available_at`, even without retrieval or
future text in the prompt. Record the run as a `retrospective_parser`; where historical deployment
claims matter, use the exact model/version actually available then and consider blinding issuer names
and dates. Otherwise describe the result only as retrospective extraction, not a contemporaneously
deployable signal.

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

An enabled generative run additionally retains newly generated attempts before parsing, exact
successful/cache response records, `model_file_sha256`, backend/llama.cpp/runtime and embedded-template
provenance, prompt/schema/verifier provenance, verified Silver rows, processing/verification
summaries, and a canonical DecisionRound ledger. The ledger is written once and replay-verified from
the stored output; it does not regenerate the model response. Its current scope ends at semantic
parsing and deliberately contains no tools, calibration, portfolio, risk, orders, or realized
outcome.

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
adds only labels whose actual availability is strictly earlier than the effective training cutoff.
Availability resolves in conservative precedence order: explicit `label_available_at`, generic
`available_at`, then `label_end_ts` only as a fallback for caller-supplied legacy labels. Generated
market labels record the complete exit cross-section's actual availability. The configured embargo
removes recent decision periods, and no model is fitted until `min_train_rows` is met. Coefficient
fitting uses running sufficient statistics instead of rebuilding a historical
feature matrix for every snapshot. Default training also updates an ordered membership digest
incrementally, so it does not retain every eligible key; explicit key lists are available only through
the opt-in diagnostic `record_training_keys` path. At the first configured final-holdout decision, the
trainer freezes the embargo-adjusted state built only from labels available before its exclusive
cutoff. Every decision at or after that boundary reuses the same membership digest and coefficients.

Candidate outcomes are admitted only as whole decision-and-horizon cross-sections. A missing label,
partial outcome, or non-terminal wholly censored group fails rather than selecting assets using future
outcome availability. Only a common trailing group whose expected label ends all exceed the final
decision boundary may be omitted.

The typed config requires one of two canonical learned-family sets. Disabled and sidecar runs use
these three, in order:

- traditional-only
- text-only
- combined traditional plus text

Augment runs append:

- LLM semantic/evidence only
- traditional plus LLM
- traditional plus conventional text plus LLM (`all`)

It also evaluates fixed naive benchmarks:

- equal-weight score
- momentum-only score
- no-trade score

Prediction diagnostics include aggregate and mean-daily Pearson/Spearman IC, hit rate, and precision
at k. Mean squared error is emitted only for predictions with explicit `expected_return`; Brier score,
expected calibration error, and calibration bins require explicit `probability_up`. The current
baseline produces neither field because its score is directional ranking information, not a calibrated
return or probability. Optional prediction/target fields must have all-or-none coverage across the
entire observed family before development/holdout splitting. Precision-at-k is a long-side raw-score diagnostic; constrained portfolio selection
is evaluated separately by the backtest and can rank eligible long/short candidates by absolute score.
Precision cutoff ties use their tied positive rate as fractional credit, making results invariant to
input row order.

Each family is also evaluated, when contemporaneous metadata is present, by sector, fixed liquidity
bucket, fixed volatility regime, source availability/type, and event availability/type. Thresholds
are fixed in code and use only decision-row context. Every family is then passed through the same
portfolio, two-sided cost, forced-liquidation, and backtest machinery. Compare the combined model
with traditional-only and naive results; a positive combined backtest alone is not evidence that text
added value.

For augment runs, also inspect `llm_ablation_comparison.json` and its inference-usage block. A positive
portfolio or prediction delta must not be described as LLM promotion: the current artifact performs
no significance test, multiple-testing correction, causal attribution, or inference-cost subtraction
from portfolio returns.

The last configured fully observed periods are reported in a separate `final_holdout` section; the
top-level family and segment metrics cover development periods only. Label availability strictly
before each effective training cutoff supplies the walk-forward purge rule. Development removes the
contiguous whole-cross-section suffix beginning with the first decision whose labels are not all
available before the holdout boundary; this preserves a fixed multi-session replay phase even when
vendor availability is delayed out of order. Within the holdout, one frozen pre-boundary model is
reused; earlier holdout outcomes never update later holdout predictions.
The saved model and evaluation protocols record the boundary, exclusive training cutoff, key count,
and membership digest. The pipeline does not create combinatorial purged folds, calibration plots,
statistical significance tests, or causal attribution of performance to a segment. The software
cannot prevent repeated human inspection from invalidating the holdout.

## Acceptance checklist

Before accepting a result, verify:

- No feature input has `available_at > asof_ts`; after-hours text uses the first configured market
  decision at or after its availability.
- Every daily OHLC row retains raw tradable prices, is marked
  `corporate_action_adjusted=true`, proves its adjustment vintage was usable by the decision, and
  supplies a positive causal `return_adjustment_factor`; optional corporate-action events and
  `adjusted_close` do not substitute for that point-in-time contract. If a bar's explicit
  `available_at` follows `ts`, its vintage may follow `ts` but not `available_at`.
- Label windows record the raw exact next-session open and horizon-session close as execution
  prices, calculate returns from each `raw_price * return_adjustment_factor`, and enter training only
  after the label is observable.
- The universe is not unintentionally survivorship-only and does not drop delisted assets as missing.
- Fundamentals and corporate actions are genuinely point-in-time; provider contracts alone do not
  make a dataset point-in-time correct.
- Syndicated/duplicate text is not counted as independent evidence; the incremental cluster assignment
  uses no future document.
- Historical social metadata existed at the decision time and is licensed for retention.
- A generative-annotation run is source-grounded, has valid evidence spans, declares its retrospective
  status, and has no price, label, later-document, RAG, tool, router, return-forecast, portfolio, or
  order context in its prompts.
- Every enabled generative request has exact item/candidate coverage, `source_available_at <=
  decision_time`, the configured horizon, valid source-local supporting/counterevidence references,
  and no uncited numeric token in its mechanism or invalidation conditions. These checks are not a
  substitute for human semantic-grounding review.
- Raw LLM confidence is labeled uncalibrated and is used only as a feature; it is not a probability,
  signal magnitude, position size, or portfolio weight.
- The DecisionRound file replays successfully and is interpreted within its actual scope: stored
  generation and verifier audit only, with no tool, calibration, portfolio, risk, order, or outcome
  trace.
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
