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
| Availability | item count, latest available time/age, time since first seen, explicit missingness |
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

The label builder uses the exact next XNYS open as entry and the exact horizon XNYS close as exit.
Generated outcomes include forward return, abnormal/sector-neutral variants, binary direction, rank,
forward volatility, and forward volume. A label records when it becomes observable.

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

Before that threshold, model-family weights and scores are zero. Early no-trade periods are therefore
expected in small runs.

The trainer stores:

- snapshot decision and cutoff timestamps;
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

`models.top_k` affects precision-at-k and paper-intent selection. It does not cap the backtest; the
portfolio layer receives every signed nonzero candidate permitted by the long/short settings.

## Evaluation

Prediction evaluation includes:

- aggregate and mean-daily Pearson and Spearman IC;
- hit rate, mean squared error, and precision-at-k;
- Brier score, calibration error, and bins for binary direction diagnostics; and
- available breakdowns by sector, liquidity, volatility, source, and event metadata.

If `probability_up` is absent, the calibration path applies a logistic transform to the raw score.
That result is an uncalibrated score diagnostic.

## Optional transformer sentiment

When explicitly enabled, the local transformer replaces sentiment score/label/confidence in text
signals. It:

- batches inference;
- detects MPS centrally and falls back to CPU;
- truncates to the configured sequence length;
- validates finite bounded outputs; and
- caches by normalized text, model identity/version, and sequence length.

Tests use an injected predictor and a small golden fixture; they never download a model.

## Current modeling limits

- No nonlinear tree model is implemented yet.
- No final untouched holdout is created automatically.
- No automatic purged-fold study or statistical significance test is produced.
- Segmented metrics describe association, not causal attribution.
- Full-mode training still consumes materialized filtered feature/label rows.

Use [Research protocol](research_protocol.md) to design evaluation and [Outputs](outputs.md) to find
the exact model columns and metrics for a run.

Return to the [documentation home](README.md).
