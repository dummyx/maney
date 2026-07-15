# AGENTS.md

## Purpose

This repository is a local-first research system for trading strategies that combine market data
with permitted natural-language data. Optimize for reproducible, point-in-time-correct, auditable
research on an Apple Silicon MacBook.

This file defines durable agent guardrails. [`README.md`](README.md) and [`docs/`](docs/README.md)
explain the current system to humans. When behavior changes, update its tests and canonical human
documentation in the same change.

## Operating boundaries

1. Treat every output as hypothetical research, not financial advice or evidence of profitability.
2. Default to backtesting, simulation, and intent-only paper workflows.
3. Do not add live order routing or broker connectivity unless the user requests it in a dedicated
   task. Live execution also requires a separate design covering broker sandboxing, kill switches,
   limits, credential isolation, idempotency, reconciliation, audit logs, approval gates, and
   explicit confirmation.
4. Do not fabricate data, API responses, credentials, citations, benchmark results, metrics, or
   conclusions.
5. Use official APIs, licensed vendors, redistributable data, or user-provided exports. Do not scrape
   in violation of platform terms.
6. Never commit or expose secrets, account identifiers, paid/vendor payloads, raw private social
   data, or personally identifying social-media data.
7. Prefer small, reviewable changes. Do not re-architect the project without an explicit request.
8. When a user correction establishes a durable convention, update this file or the closest nested
   `AGENTS.md`.
9. For kabuS API workflows, treat code developed by the repository owner with Codex assistance as
   the owner's own program. Scope broker connectivity to the owner's sole use and operation; do not
   give any other person credentials or the ability to use or operate it.

## Data integrity and point-in-time correctness

### Raw data

- `data/raw/` is immutable and append-only.
- Ingestion must capture exact source bytes before parsing, preferably under a content-addressed
  name.
- Ingestion may add payloads and metadata; it may never overwrite or modify an existing raw object.
- Record source, vendor, licensing reference, ingestion timestamp, request or batch ID, SHA-256,
  schema version, and fetch-parameter hash where applicable.
- Keep raw, interim, processed, report, model, cache, and local-secret artifacts out of git.

### Timestamps

Use timezone-aware UTC internally. Preserve source timezone and exchange-local dates separately when
they matter.

A model feature row must have `asof_ts`. Every source value contributing to that row must have a
provable availability time satisfying:

```text
available_at <= asof_ts
```

Use `available_at`, not merely `published_at`, unless equivalence is proven. Respect exchange
calendars and asset active periods. Map after-hours information to the next tradable decision when
appropriate. Add tests that reject future information entering past features.

### Storage layers and identity

- Bronze: exact immutable source payloads and metadata.
- Silver: normalized, typed, partitioned Parquet.
- Gold: point-in-time feature, label, and prediction tables plus manifests.

Use `asset_id` as canonical identity and retain `symbol` as a human-readable field. A feature row is
uniquely keyed by:

```text
(asset_id, asof_ts, horizon, feature_set_version)
```

Generate labels separately from features. Labels must begin strictly after the decision and follow
the exchange calendar. Never choose a cross-section based on which assets later receive outcomes;
fail on partial label coverage or apply an explicit, documented terminal-censoring rule.

For corporate actions:

- use raw tradable OHLC prices for simulated fills and liquidity;
- use only causal return-adjustment factors whose vintage was available at the relevant bar; and
- never use a hindsight-adjusted series in a historical decision.

Do not use revised fundamentals, future corporate-action knowledge, unavailable historical social
metadata, survivorship-only universes, or indefinitely forward-filled text without explicit
point-in-time handling.

## Research validity

- Implement and retain a strong traditional-market baseline before claiming incremental text value.
- Compare traditional-only, text-only, combined, equal-weight, momentum-only, and no-trade paths.
- Use walk-forward evaluation. Purge or embargo overlapping labels when needed, and keep a final test
  period untouched until final evaluation.
- Keep feature windows, label horizons, prediction horizons, and rebalance settings aligned.
- Preserve missingness indicators and explicit text-signal decay.
- Deduplicate syndicated/copied text before treating items as independent evidence.
- Do not optimize or report only Sharpe. Include prediction quality, return, risk, drawdown, turnover,
  exposure, cost, liquidity, capacity, and stability diagnostics appropriate to the run.
- Do not delete losing experiments merely because their results are negative.

Every experiment must emit a reproducible run ID, code version, config snapshot/hash, data manifest,
period and universe, feature/label/model versions, cost and constraint assumptions, metrics,
limitations, and next questions.

## Backtesting and paper workflows

Backtests must be deterministic, point-in-time, and cost-aware. Model, configure, and report at
least:

- fees or commissions;
- half-spread;
- volatility- and participation-aware slippage;
- a market-impact proxy;
- borrow cost and availability when shorting; and
- applicable position, gross/net, sector, beta, turnover, price, liquidity, and participation
  constraints.

Separate signal generation, portfolio construction, fill simulation, and reporting. Retain position
and trade logs. Reports must show costs, drawdowns, turnover, exposures, and per-period diagnostics,
not only cumulative return.

Paper output must remain simulated or intent-only and must not connect to a broker.

## Natural-language guardrails

The useful baseline must remain deterministic and must not require a large language model.

Optional transformer or LLM components must:

- support batching, CPU fallback, and MPS where available;
- avoid downloads during tests;
- cache outputs by canonical text hash and model/config version;
- version prompts, models, decoding settings, and output schemas;
- validate structured output and retain uncertainty;
- preserve deterministic source timestamps; and
- never generate orders, invent missing facts, or use future historical context.

For social data:

- hash or omit author identifiers unless raw values are necessary, permitted, and explicitly
  requested;
- do not build unrelated user-profiling features;
- preserve repost/quote/reply and content-status fields when supplied;
- do not retain deleted/private/protected content unless terms allow it; and
- treat engagement, followers, credibility, and spam/bot signals as noisy research features, not
  truth.

## Security and data rights

- Read secrets only from environment variables or an ignored local `.env`.
- Never log secrets or place them in configs, manifests, notebooks, reports, or model artifacts.
- Do not make network calls during imports or constructors.
- Gate external access behind explicit commands with rate limiting, retry/backoff, and licensed
  caching rules.
- Remember that hashing identifiers in silver does not remove raw identifiers from immutable bronze.

## Engineering baseline

Use Python 3.12, `uv`, `pyproject.toml`, and `uv.lock`. Production code belongs under `src/`;
notebooks may call production code, but production code must not depend on notebooks.

Preferred local stack:

- Polars, PyArrow, Parquet, and DuckDB for analytics;
- Pydantic for configuration and boundary schemas;
- NumPy, SciPy, and scikit-learn for baseline modeling;
- Typer for the CLI; and
- pytest, Ruff, and mypy for quality gates.

Keep deep-learning dependencies optional. The baseline must run without PyTorch. Use the existing
provider/store protocols and implement local CSV/JSONL/Parquet adapters plus fixture contracts before
adding an external provider.

Runtime behavior belongs in typed configuration. Do not hardcode local paths, symbols, credentials,
API hosts, model names, or strategy parameters in business logic.

Use structured logging in library code. Type public functions, keep I/O at boundaries, avoid global
mutable state, and raise explicit actionable errors.

Avoid heavyweight services such as Spark, Kafka, Airflow, Kubernetes, CUDA-only packages, broker
integrations, or remote vector databases unless a demonstrated requirement justifies them.

## Apple Silicon and local performance

- Do not assume CUDA, x86, or Rosetta.
- Centralize optional device selection in `src/nlp_trader/utils/device.py`: MPS when available,
  otherwise CPU.
- Keep model/inference batch sizes configurable.
- Use conservative macOS-safe worker defaults and guard process pools with
  `if __name__ == "__main__":`.
- Use `pl.scan_parquet(...)` for large Parquet inputs so filters and projections can be pushed down.
- Filter and select before joins; use partition pruning and streaming/batched writes where practical.
- Cache expensive NLP outputs and reuse artifacts only when manifests match.
- Expose bounded development runs through date, symbol, and decision limits.
- Keep deterministic `sample` mode fast and `full` mode usable with filtered user-provided data.
- Do not claim the current full path is out-of-core while downstream rows are materialized.

## Tests and quality gates

Tests must not require network access, vendor credentials, paid APIs, CUDA, or MPS. Use tiny
synthetic/redistributable fixtures, fixed seeds, and tolerant floating-point assertions.

Maintain coverage for:

- schemas, UTC normalization, and exchange-calendar behavior;
- entity linking, deduplication, and sentiment aggregation;
- `available_at <= asof_ts` leakage sentinels;
- label off-by-one, terminal coverage, and partial cross-sections;
- corporate-action causality;
- cost calculations and portfolio constraints;
- local fixture ingestion and the complete feature/train/predict/backtest/report path;
- stable sample regression output; and
- optional injected/cache-backed transformer/device paths without requiring MPS.

Before finalizing a substantive change, run:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
uv run nlp-trader smoke --config configs/sample.yaml
```

If a full gate cannot run, run targeted checks and state exactly what was skipped and why.

## Documentation ownership

Use [`README.md`](README.md) as the entry point and [`docs/README.md`](docs/README.md) as the
documentation map.

Write `README.md` for an average engineer. Keep it practical and use plain language. Avoid
unexplained trading, research, storage, or machine-learning jargon; define any term that must remain
and link detailed contracts to their canonical document.

| Change | Canonical documentation |
|---|---|
| Setup or top-level capability | `README.md`, `docs/getting_started.md` |
| Config field or validation | `docs/configuration.md`, bundled configs |
| Input/derived schema or timestamp | `docs/input_data.md`, `docs/data_contracts.md` |
| CLI stage or dependency | `docs/workflows.md` |
| Artifact or metric | `docs/outputs.md` |
| Component/data flow | `docs/architecture.md` |
| Feature/model behavior | `docs/features_and_models.md`, `docs/research_protocol.md` |
| Fill, cost, constraint, or metric semantics | `docs/backtesting.md` |
| Licensing, privacy, security, or trading boundary | `docs/compliance.md` |

Do not duplicate detailed schema, feature, command, or metric catalogs in this file. Link to their
canonical human document.

## Definition of done

A change is complete only when:

- it satisfies the requested scope without an unnecessary rewrite;
- point-in-time, raw-data, security, and research-validity invariants remain intact;
- behavior changes have focused tests and updated documentation;
- relevant quality gates pass, or skipped checks are named with reasons;
- the sample path remains runnable without CUDA or data/model network access; and
- no secret, paid/raw data, or generated research artifact was added to git.

The final response should state what changed, how to run it, which checks ran, and any assumptions,
limitations, or incomplete work without overstating research performance.
