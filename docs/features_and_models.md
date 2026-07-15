# Features and Models

This page explains what the baseline learns from, how missing data is represented, and how the
walk-forward score is produced. The implementation is intentionally transparent and does not require
a large language model.

## Feature-row identity

One gold feature row represents:

```text
(asset_id, asof_ts, horizon, feature_set_version)
```

`symbol` is retained for readability. Builders reject duplicate keys and future input provenance.
Labels are created by a separate builder.

## Price basis

Traditional return features use:

```text
causal_price = raw_price * return_adjustment_factor
return = causal_price_now / causal_price_before - 1
```

Raw prices remain available for fills and liquidity. Every daily bar used by the feature/label path
must carry causal adjustment provenance.

## Traditional features

The builder produces these families:

| Family | Representative fields | Notes |
|---|---|---|
| Returns | `return_1d`, `return_3d`, `return_5d`, `return_20d`, `return_60d` | Causal-price close-to-close returns with matching missingness flags. |
| Momentum/reversal | `short_term_reversal_1d`, `momentum_5d`, `momentum_20d`, `gap_return_1d`, `intraday_return` | Gap compares current raw open/factor with prior raw close/factor. |
| Volume/liquidity | `dollar_volume`, `average_dollar_volume_20d`, abnormal volume, Amihud-style illiquidity, high-low spread estimate | Turnover is currently always missing because no shares-outstanding input is wired. |
| Volatility | 3/20/60-session realized volatility, downside volatility, high-low estimator, regime ratio | Short history is flagged. |
| Market/sector | market return, 60-session beta, market residual, sector return/residual | Requires contemporaneous cross-sectional context. |
| Proxies | log dollar-volume size, optional value/quality/market-cap values | Fundamental values are point-in-time gated. |
| Known events | earnings and ex-dividend proximity/blackout fields | Requires a record already available at the decision. |

The feature table is wider than the baseline model’s selected columns. The saved model artifact lists
the exact columns used by each family.

## Text signal pipeline

### 1. Normalize

The deterministic path normalizes Unicode and whitespace, tokenizes finance-relevant text, and
computes a canonical content identity. Constructors and tests perform no implicit network access.

### 2. Link entities

Supplied entities are validated. Otherwise, the baseline linker checks the filtered asset master,
symbols/cashtags, canonical company names, simple name aliases, title/body context, asset activity,
and confidence thresholds. An uppercase token alone is not assumed to be an asset.

### 3. Deduplicate causally

Items are processed in `(available_at, item_id)` order. Exact and near-duplicate clusters compare a
new item only with prior candidates. A later bridge document never merges two historical clusters.

Only signals with `novelty > 0.5` contribute to baseline-facing sentiment, event, source, independent
count, velocity, or abnormal-attention features. Copy volume is retained separately as `raw_*`
diagnostics and duplicate/novelty counts; the baseline does not automatically discover `raw_*`
columns.

### 4. Score sentiment and context

The finance dictionary baseline handles positive/negative terms, nearby negation, uncertainty, event
terms, promotional phrases, source credibility, relevance, and spam penalties. It emits a bounded
score, label, confidence, event type, and novelty value.

This is a transparent baseline, not a claim of semantic completeness.

## Text features

For every configured window such as `1d`, `3d`, `5d`, or `20d`, features are built only from signals
whose actionable time is no later than `asof_ts`.

| Family | Representative fields |
|---|---|
| Availability | independent and raw item counts, latest available time/age, time since first seen, explicit missingness |
| Sentiment | mean, relevance/confidence/credibility/decay-weighted means, max positive/negative, dispersion, disagreement, ratio, acceleration |
| Attention | total/unique count, velocity, prior-window abnormal attention, source-specific counts |
| Novelty | novelty share, novel count, duplicate count |
| Diversity | source count, unique-author count, author/source disagreement |
| Credibility | mean source credibility, spam mean, credible-attention sum |
| Events | event count/share/diversity/sentiment, event-source interaction, event-type counts |
| Optional LLM | separate coverage/abstention, semantic, raw-confidence, uncertainty, event-confidence, supporting/counterevidence, agreement, and missingness aggregates |

Age decay uses the configured half-life. A signal’s contribution halves once its age reaches
`text_decay_half_life_days`.

The optional `llm_*` columns never replace these conventional features. In each configured window,
the LLM aggregation records annotation and non-abstention counts, annotation coverage, abstention
rate, semantic mean, a raw-confidence-weighted semantic diagnostic, mean raw confidence and
uncertainty, mean event confidence, supporting/counterevidence counts, evidence agreement, and
explicit missingness. Only point-in-time-admissible, novelty-filtered independent signals enter these
aggregates. Raw confidence is explicitly uncalibrated: it remains a feature and must not be read as a
probability, semantic-signal magnitude, position size, or portfolio weight.

## Missingness behavior

When a window has no text, the builder writes:

- zero for count/aggregate fields where a numeric identity is defined;
- `None` where a value such as latest timestamp does not exist; and
- explicit fields such as `text_missing_5d` and source/author missingness flags.

Optional LLM fields use the same discipline. Counts and defined aggregate identities are zero, while
`llm_missing_*` and field-specific missingness flags distinguish absent annotations/evidence from a
genuine neutral semantic value. An abstention is counted but does not contribute a semantic value.

Traditional short-history values follow the same principle. Risk metadata is handled conservatively:

- missing beta becomes `backtest.missing_beta_fallback`;
- missing 20-session volatility becomes the maximum of the configured floor, available short-window
  realized volatility, and the high-low estimator; and
- both substitutions add risk flags to predictions/trades/periods.

## Labels

The label builder uses the first configured-exchange open strictly after `asof_ts` as entry and the
exact horizon session close as exit. Generated outcomes include forward return,
abnormal/sector-neutral variants, binary direction, rank, forward volatility, and forward volume. A
label records when it becomes observable.

An incomplete internal session fails. A trailing decision without its complete future window is
marked missing rather than shortened.

## Baseline model

The current model is not a scikit-learn estimator. At each expanding snapshot it maintains sufficient
statistics for each selected feature:

1. calculate the feature mean and population scale on eligible historical rows;
2. calculate its correlation with forward return;
3. normalize feature correlations so their absolute weights sum to one; and
4. score a row as the weighted sum of standardized feature values.

This gives an inspectable directional baseline. It is not designed to capture complex nonlinear
relationships.

## Walk-forward eligibility

At a decision time, training can include only labels whose outcome was observable strictly before
the effective cutoff. The configured embargo moves that cutoff further back. No family is fitted
until `min_train_rows` is reached.

Training and evaluation require whole decision-and-horizon cross-sections. Missing labels, partial
outcomes, or non-terminal wholly missing groups fail. A wholly censored trailing group is omitted only
when every expected label end is beyond the final decision boundary.

Development decisions retain the causal expanding-window behavior above. At the configured final
holdout boundary, the embargo-adjusted fit is frozen. Every holdout and trailing paper prediction uses
that same training-key membership and coefficients, so no outcome from the holdout can enter a later
holdout snapshot.

Before that threshold, model-family weights and scores are zero. Early no-trade periods are therefore
expected in small runs.

The trainer stores:

- snapshot decision and cutoff timestamps;
- whether a snapshot is development walk-forward or frozen final holdout;
- eligible/training row counts;
- a deterministic digest of training keys;
- per-family feature names, means, scales, weights, and uncertainty proxy.

It does not retain full key lists unless the internal diagnostic option is explicitly enabled.

## Family definitions

| Family | Score |
|---|---|
| `traditional` | Selected market/fundamental/event columns |
| `text` | Conventional text/sentiment/attention/novelty/event columns; never `llm_*` columns |
| `combined` | Union of traditional and conventional text columns |
| `llm` | Optional `llm_*` semantic/evidence aggregate columns only |
| `traditional_llm` | Union of traditional and LLM columns |
| `all` | Union of traditional, conventional text, and LLM columns |
| `equal_weight` | Constant positive score |
| `momentum_only` | `return_20d`, with 5d, 3d, then 1d fallback only if an earlier column is absent |
| `no_trade` | Zero score |

The three LLM families exist only in `llm_annotations.feature_mode: augment`; typed configuration
then requires all six learned families in the order shown. Sidecar and disabled runs retain only the
first three. This keeps the conventional comparison fixed instead of changing what `text` means when
an LLM is enabled.

`models.top_k` sets the depth of the long-side precision diagnostics and independently caps
model-scored backtests and combined-model paper intents. The portfolio path first removes
direction-ineligible rows, then ranks by absolute score; precision-at-k instead ranks raw score
descending, gives cutoff ties fractional credit at their tied positive rate, and should not be read as
a reconstruction of the traded portfolio. Equal-weight and no-trade references remain uncapped.

## Evaluation

Prediction evaluation includes:

- aggregate and mean-daily Pearson and Spearman IC;
- hit rate and precision-at-k;
- mean squared error only when predictions supply an explicit `expected_return`;
- Brier score, calibration error, and bins only when predictions supply an explicit
  `probability_up`; and
- available breakdowns by sector, liquidity, volatility, source, and event metadata.

Optional expected-return, probability, and binary-target fields must cover every observed row across
development and holdout or none of them; partial coverage fails instead of silently changing the
metric population.
The baseline emits a directional rank score, not an expected return or calibrated probability, so it
does not receive MSE or calibration metrics. The final `models.final_holdout_periods` fully observed
decision periods are reported under `final_holdout`; top-level family and segment diagnostics cover
development periods only. Development removes the contiguous suffix starting at the first
boundary-overlapping label cross-section, preserving multi-session replay phase even under
out-of-order vendor delays. Inside the holdout, every prediction uses the same frozen pre-boundary
fit; the evaluation protocol verifies and reports that untouched-training rule.

An augment-mode backtest also writes `llm_ablation_comparison.json`. It compares `llm` with `text`,
`traditional_llm` with `traditional`, and `all` with `combined` for both development and final
holdout. Reported differences are arithmetic metric deltas only. They are not significance tests,
causal attribution, profitability evidence, or an automatic promotion decision. The artifact also
records generated token counts, generation latency, and configured estimated inference cost when
available. Incomplete generated token accounting makes the token totals and configured cost null
rather than zero or a partial estimate.

## Optional transformer sentiment

When explicitly enabled, the local transformer replaces sentiment score/label/confidence in text
signals. It:

- batches inference;
- detects MPS centrally and falls back to CPU;
- truncates to the configured sequence length;
- validates finite bounded outputs; and
- caches by normalized text, model identity/version, and sequence length.

Tests use an injected predictor and a small golden fixture; they never download a model.

## Optional generative semantic/evidence annotations

The local generative path addresses one narrow baseline limitation: the conventional scorer assigns
one document-level sentiment/event result to every linked entity. A document such as “Company A
gains share while Company B cuts guidance” can instead receive separate, source-grounded entity
signals without changing the conventional result.

This path is disabled by default and is not part of the deterministic baseline. When enabled, the
`llama_cpp_gguf` backend uses `llama-cpp-python==0.3.34` to load one direct local GGUF file; it does
not download or resolve model weights. The bundled logical selector is
`unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` at revision
`5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf`, with local filename
`Qwen3.6-27B-UD-Q4_K_XL.gguf` and expected SHA-256
`4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095`. The file is about
17.9 GB, so 32 GB or more of unified memory is a practical starting point after inference working
memory is included.

The runtime records the exact bytes as `model_file_sha256`, requires and hashes the chat template
embedded in the GGUF, and records the llama.cpp version, context sizes, and requested/effective GPU
layers. A Metal-enabled binding may offload layers; otherwise inference falls back to CPU. This is
llama.cpp Metal rather than PyTorch MPS. Despite `MTP` in the model name, the current in-process
chat-completion API performs ordinary inference and does not use MTP speculative acceleration.

For each text item, the host supplies only deterministically linked, historically active asset
candidates, numbered spans from that item, noisy source type/quality metadata, the first safe market
decision at or after source availability, and the configured prediction horizon. There is no RAG,
other-document retrieval, tool call, model router, price, label, return, portfolio, or order context.

The strict v2 result contains, per asset:

- `positive`, `negative`, `neutral`, or `abstain` stance;
- a sign-consistent integer `semantic_signal` in `[-2, 2]`;
- `raw_confidence` and uncertainty in `[0, 1]`, with raw confidence explicitly uncalibrated;
- the exact configured `horizon_days`;
- at most one of `bankruptcy`, `merger_acquisition`, `guidance`, `earnings`, `dividend`,
  `litigation`, `regulatory`, or `capital_raise`, or no event, plus event confidence;
- disjoint supporting and counterevidence IDs drawn from the supplied source-local spans;
- a concise mechanism and one or more invalidation conditions; and
- strict empty/default fields plus an explicit reason for abstention.

An oversized request produces `input_too_long` abstentions instead of silently truncating the source.
Every newly generated attempt is stored before parsing, including malformed or truncated output.
Strict JSON/schema failure still fails the stage; the system does not silently repair it.

The deterministic verifier checks:

1. response item identity and exact unique candidate coverage;
2. `source_available_at <= decision_time`;
3. returned horizon equality with the configured request horizon;
4. membership of every supporting/counterevidence ID in the current source spans; and
5. that each numeric token in a mechanism or invalidation condition occurs in the cited spans.

These checks establish structural and limited lexical grounding only. They do not prove that a cited
span semantically supports the claim, that an event interpretation is correct, or that the proposed
mechanism is true.

The component is disabled by default. `feature_mode: sidecar` writes verified Silver records, raw
response/provenance artifacts, a verification summary, and replay-checked DecisionRounds without
adding LLM values to gold features. `feature_mode: augment` adds separate `llm_*` fields and the six
family ablation. Conventional sentiment/event values are never overwritten, so optional transformer
sentiment can coexist with augmentation. An abstention remains visible through coverage, abstention,
and missingness fields.

Before augmentation, evaluate a frozen human-labeled set with `evaluate_annotation_set(...)`. The
local evaluator reports stance and primary-event macro-F1, supporting- and counterevidence precision,
horizon accuracy, abstention and invalid-response rates, plus raw-confidence Brier and calibration
diagnostics. These are extraction-quality metrics only; raw confidence is not promoted to a
calibrated probability and the evaluator consumes no price, return, portfolio, or backtest outcome.

Normal automated tests use an injected generator and do not establish real GGUF loading or
inference. The environment-gated acceptance command in [Workflows](workflows.md) performs that local
check; neither it nor the fake tests establish research usefulness.

See [Research protocol](research_protocol.md) for the sidecar-first experiment and pretrained-memory
limitation, and [Outputs](outputs.md) for the DecisionRound boundary.

## Current modeling limits

- No nonlinear tree model is implemented yet.
- No automatic purged-fold study or statistical significance test is produced.
- The chronological holdout boundary cannot stop a researcher from repeatedly inspecting and tuning
  to the same terminal sample.
- Segmented metrics describe association, not causal attribution.
- Full-mode training still consumes materialized filtered feature/label rows.
- Generative annotation is retrospective parsing; source-time gating cannot prove that a modern
  pretrained model lacked later historical knowledge.
- The LLM verifier does not establish semantic truth, and no RAG, external tools, model router, or
  calibrated LLM meta-model is implemented.

Use [Research protocol](research_protocol.md) to design evaluation and [Outputs](outputs.md) to find
the exact model columns and metrics for a run.

Return to the [documentation home](README.md).
