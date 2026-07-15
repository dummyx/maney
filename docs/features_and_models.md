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

Age decay uses the configured half-life. A signal’s contribution halves once its age reaches
`text_decay_half_life_days`.

## Missingness behavior

When a window has no text, the builder writes:

- zero for count/aggregate fields where a numeric identity is defined;
- `None` where a value such as latest timestamp does not exist; and
- explicit fields such as `text_missing_5d` and source/author missingness flags.

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
| `text` | Selected text/sentiment/attention/novelty/event columns |
| `combined` | Union of traditional and text columns |
| `equal_weight` | Constant positive score |
| `momentum_only` | `return_20d`, with 5d, 3d, then 1d fallback only if an earlier column is absent |
| `no_trade` | Zero score |

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

## Optional transformer sentiment

When explicitly enabled, the local transformer replaces sentiment score/label/confidence in text
signals. It:

- batches inference;
- detects MPS centrally and falls back to CPU;
- truncates to the configured sequence length;
- validates finite bounded outputs; and
- caches by normalized text, model identity/version, and sequence length.

Tests use an injected predictor and a small golden fixture; they never download a model.

## Optional generative entity/event annotations

The local generative path addresses one narrow baseline limitation: the deterministic scorer assigns
one document-level sentiment/event result to every linked entity. A document such as “Company A
gains share while Company B cuts guidance” can instead receive separate entity-level stances and
events.

For each text item, the host supplies only deterministically linked, historically active asset
candidates and numbered spans from the item itself. The validated structured result contains, per
asset:

- `positive`, `negative`, `neutral`, or `abstain` stance plus confidence and uncertainty;
- at most one of `bankruptcy`, `merger_acquisition`, `guidance`, `earnings`, `dividend`,
  `litigation`, `regulatory`, or `capital_raise`, or no event, plus confidence; and
- one or more supplied evidence span IDs, or an explicit abstention reason.

The host derives a sentiment score as direction times stance confidence. It rejects unknown or
duplicate assets, bad evidence references, values outside their allowed ranges, and malformed output.
Inputs that do not fit the configured context budget produce an explicit `input_too_long`
abstention for every candidate instead of being silently truncated.

The component is disabled by default. With annotation enabled but feature application disabled, it
produces an auditable sidecar without changing any text signal or gold feature. Explicit application
replaces only the matching `(item_id, asset_id)` sentiment/event fields for a valid, non-abstained
annotation. An abstention keeps the deterministic sentiment/event result and is counted in the
annotation summary. Entity linking, deduplication/novelty, relevance, source credibility, spam,
source identity, and source availability stay deterministic in both modes.

Before applying a model, evaluate a frozen human-labeled set with
`evaluate_annotation_set(...)`. The local evaluator reports stance and primary-event macro-F1,
evidence precision, abstention and invalid-response rates, plus stance-confidence Brier and
calibration diagnostics. These are extraction-quality metrics only; they consume no price, return,
portfolio, or backtest outcome.

Applied generative annotation and transformer sentiment are mutually exclusive. Transformer
sentiment can still run while annotations are sidecar-only; in applied generative mode, abstained or
missing annotations fall back specifically to the deterministic dictionary/event baseline.

This path performs local structured extraction. It does not retrieve documents, use RAG, forecast
returns, select a portfolio, or generate orders. See [Research protocol](research_protocol.md) for
the matched-run evaluation and pretrained-memory limitation.

## Current modeling limits

- No nonlinear tree model is implemented yet.
- No automatic purged-fold study or statistical significance test is produced.
- The chronological holdout boundary cannot stop a researcher from repeatedly inspecting and tuning
  to the same terminal sample.
- Segmented metrics describe association, not causal attribution.
- Full-mode training still consumes materialized filtered feature/label rows.
- Generative annotation is retrospective parsing; source-time gating cannot prove that a modern
  pretrained model lacked later historical knowledge.

Use [Research protocol](research_protocol.md) to design evaluation and [Outputs](outputs.md) to find
the exact model columns and metrics for a run.

Return to the [documentation home](README.md).
