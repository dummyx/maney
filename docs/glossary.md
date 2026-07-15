# Concepts and Glossary

This page gives the mental model behind the system before the detailed contracts and formulas.

## The central question

For every historical decision, ask:

> What information could the strategy genuinely have used at that moment?

NLP Trader encodes the answer with `available_at` on source information and `asof_ts` on the feature
row. A source value can contribute only when:

```text
available_at <= asof_ts
```

This is point-in-time correctness. It prevents a backtest from quietly using later publication,
vendor, revision, corporate-action, or outcome knowledge.

## One decision timeline

For the current daily pipeline:

```text
before/at XNYS close       after the close        next XNYS open       horizon XNYS close
---------------------|------------------------|--------------------|----------------------
usable inputs arrive     feature/asof decision      assumed entry          assumed exit
                                                    label starts           label ends
```

An after-hours text item is not moved backward into the completed close. It becomes actionable at a
later configured decision.

## Storage layers

| Layer | Plain-language meaning | Examples |
|---|---|---|
| Bronze | Exact evidence received from a source | Original CSV/JSONL/Parquet bytes and provenance sidecar |
| Silver | Clean, typed, normalized records | Assets, bars, text, signals, fundamentals, events |
| Gold | Research-ready decision tables | Features, labels, predictions |

Bronze is shared, content-addressed, and append-only. Silver and gold belong to a unique run.

## The six comparison families

| Family | Purpose |
|---|---|
| Traditional | Baseline using market-derived features |
| Text | Baseline using text-derived features |
| Combined | Traditional and text features together |
| Equal weight | Naive positive score for every candidate |
| Momentum only | Naive score from the 20-session return, with shorter column fallbacks |
| No trade | Zero score sanity check |

The research question is not “did combined make money?” It is “did combined improve robustly over
traditional and naive alternatives after the same costs and constraints?”

## Key terms

| Term | Meaning in this project |
|---|---|
| `asset_id` | Stable canonical asset identity. Symbols are descriptive and can change. |
| `published_at` | Time declared by the source. Not necessarily when the strategy received it. |
| `vendor_received_at` | Time the vendor says it received or exposed the item. |
| `ingested_at` | Time the local system recorded the input. |
| `available_at` | Earliest defensible strategy-use time. This gates feature inclusion. |
| `asof_ts` | Strategy decision timestamp represented by a feature/signal row. |
| `event_ts` | Time the real-world event occurred or is scheduled to occur. |
| `label_start_ts` | Exact next-session open used for the outcome window. |
| `label_end_ts` | Exact configured-horizon close used for the outcome window. |
| Horizon | Number of exchange sessions from entry open to exit close. |
| Warm-up | Pre-start source context loaded so rolling features are complete. |
| Lookahead context | Post-end bars/events loaded only to complete labels or known-event proximity. It is not permission to use future availability. |
| Feature row | One asset/decision/horizon/version record containing model inputs and provenance. |
| Label row | Separately generated future outcome for the same asset/decision/horizon. |
| Missingness flag | Explicit boolean showing that history or data required for a value was unavailable. |
| Leakage | Any use of information unavailable at the historical decision. |
| Embargo | Recent decision periods excluded from an expanding training cutoff. |
| Walk-forward | Refit/recompute model state through time using only already observable labels. |
| Turnover | Absolute weight traded; current periods record both entry and forced exit. |
| Participation | Trade notional divided by decision-time dollar-volume proxy. |
| Capacity proxy | Screening equity implied by the participation cap; not deployable capital. |
| Paper intent | Simulation-only target-weight record, not an executable order. |
| `run_id` | Unique identity for one immutable pipeline execution and its artifacts. |

## Corporate-action vocabulary

Raw OHLC is the price that could have been observed and used as an assumed fill. A
`return_adjustment_factor` is separate metadata used to compare returns across corporate actions.
Its `adjustment_vintage_at` must establish that the factor was causal at that bar.

The system does not treat a modern hindsight-adjusted close as a substitute for that contract.

## Missing data versus zero

Zero can be a legitimate observation. Missing means the system lacked sufficient history or source
information. Feature tables therefore retain explicit `_missing` fields rather than relying only on
sentinel numeric zeros.

For portfolio risk, an unavailable beta or long-window volatility is replaced by configured
conservative values and flagged. Missing risk is not treated as zero risk.

## A successful run means

A successful run means:

- all enforced schemas and timing checks passed;
- artifacts were written and hashed;
- the configured calculations completed.

It does not mean:

- the data license is valid;
- the universe is survivorship-free;
- the strategy is profitable or statistically significant;
- the fill or capacity assumptions are realistic; or
- the result is ready for paper or live capital.

Continue with [Data contracts](data_contracts.md), [Features and models](features_and_models.md), or
[Backtesting](backtesting.md).

Return to the [documentation home](README.md).
