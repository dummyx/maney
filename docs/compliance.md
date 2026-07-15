# Compliance and Safety Boundary

This repository is a research tool. It is not financial advice, an investment adviser, or an
authorization to trade. Backtests are hypothetical and depend on data, model, liquidity, execution,
and cost assumptions. Pending paper intents do not claim that execution occurred.

## Data rights

Use only official APIs, licensed vendors, redistributable data, or user-provided exports that permit
the intended collection, retention, transformation, and research use. Do not bypass access controls
or scrape sites and social platforms contrary to their terms.

Each ingested source requires a `license_or_terms_ref`, vendor/source identity, schema version,
request ID, ingestion time, payload hash, and fetch-parameter hash. These fields record provenance;
they do not grant rights. The user remains responsible for verifying the underlying license and any
redistribution, deletion, or retention requirement.

Full mode requires user-provided licensed local files. The repository supplies no paid text, real
social corpus, account data, vendor credential, or external API adapter. Synthetic fixtures are
clearly marked and may be regenerated without network access.

## Social and natural-language data

- Hash or omit author handles and profile URLs unless raw identifiers are necessary, permitted, and
  protected outside git.
- Do not create unrelated user-profiling features.
- Preserve `original`/`repost`/`quote`/`reply` relationships and
  `active`/`deleted`/`private`/`protected`/`unknown` content status when a licensed source supplies
  them. Parent item identifiers must be hashed.
- Honor source-specific deletion and retention terms; append-only technical storage is not permission
  to retain content indefinitely.
- Treat engagement, follower counts, source credibility, bot/spam flags, and historical author
  quality as noisy research features, not truth.
- Store text bodies in processed data only when retention and transformation are permitted. The local
  provider hashes supplied raw author identifiers, URLs, and parent item IDs before silver output,
  validates supplied hashes as lowercase SHA-256, and rejects records explicitly marked
  `retention_permitted=false`. These checks do not prove that `true` is legally correct.
- The bronze store preserves the original source file byte-for-byte. If that file contains raw
  identifiers or restricted text, hashing in silver does not remove it from bronze; keep the raw root
  private and ingest only data whose retention is permitted.

## Secrets and local artifacts

The code reserves `NLP_TRADER_*` environment-backed secret settings for future external adapters, but
the current local providers do not instantiate them or require credentials. Future adapters must read
secrets only from environment variables or an ignored local `.env`, never from research configs.
Secrets must not appear in snapshots, manifests, reports, logs, or exception messages. Never commit
credentials, account IDs, raw vendor data, paid content, raw social data, generated models, caches,
or reports.

The bundled raw, interim, processed, model, and report roots are gitignored. If you configure roots
elsewhere, keep them out of version control too. Bronze ingestion is append-only and
content-addressed. Do not edit raw payloads or sidecars; ingest a new version and preserve both
records.

## Research, paper, and live boundaries

- Research stages create features, labels, models, predictions, hypothetical backtests, and reports.
- `paper` converts the latest combined-model close decision into constrained, simulation-only pending
  next-session-open intents. The snapshot has `status: pending_unfilled_intents`, unchanged initial
  capital/equity, empty positions, and no trades. It records no fill, cost, cash balance, position
  mutation, or automatic horizon-close liquidation, and has no broker credentials, network client,
  external account state, or order-routing adapter.
- `OrderIntent` and `PaperOrderIntent` are target-weight research records, not executable orders.
- No code path in this repository places or transmits a live order.

Live execution must not be added as an incidental extension of paper simulation. A dedicated future
task would require broker sandboxing, explicit approval gates, kill switches, position and loss
limits, credential isolation, idempotent order handling, reconciliation, audit logs, operational
monitoring, and separate user confirmation. Until that design and approval exist, all output remains
research or paper simulation.

## Reporting language

Reports must state that results are hypothetical and assumption-dependent. Do not use language that
guarantees returns or disguises synthetic/backtested results as realized performance. Show costs,
slippage, borrow and liquidity assumptions, turnover, drawdown, exposure, limitations, and sample
size. Preserve negative experiments and disclose missing point-in-time, survivorship, corporate-
action return-factor provenance, bounded event context, and execution data.

For operational details, see [Input data](input_data.md), [Research protocol](research_protocol.md),
and [Backtesting](backtesting.md).

Return to the [documentation home](README.md).
