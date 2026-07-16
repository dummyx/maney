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

Full mode requires user-provided licensed local files. The research pipeline supplies no paid text,
real social corpus, account data, vendor credential, or external market-data adapter. The separate
kabuS broker adapter accesses only operator-authorized account data when explicitly invoked.
Synthetic fixtures are clearly marked and may be regenerated without network access.

The strict Japanese baseline likewise accepts only a repository-owner-supplied permitted local
export. A J-Quants subscription or API entitlement does not automatically grant redistribution
rights. Verify the current terms for collection, local retention, transformation, derived outputs,
and any required deletion before use; keep vendor payloads and bronze copies private and out of git.
The config's `license_or_terms_ref` is an audit pointer, not permission. The repository contains no
J-Quants credential, client, or dataset. See the [official J-Quants site](https://jpx-jquants.com/en)
and the [Japan baseline preparation guide](japan_baseline.md).

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
- Optional generative prompts, raw responses, and caches can repeat licensed/private source text.
  Generation-attempt records, verified Silver rows, and DecisionRound ledgers can do the same. Keep
  them local and gitignored, apply the source’s retention/transformation terms, and record the local
  model’s license or terms reference. Model access does not expand the rights granted by the source
  license.

The bundled generative settings identify
`unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` at revision
`5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf` and record this terms reference:

```text
https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/tree/5c641ee6f93ccf8b1f01824455bfdbbdd7d658bf
```

That reference, the expected SHA-256
`4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095`, and the
project’s download instructions are provenance, not permission or legal advice. Before downloading
the 17.9 GB `Qwen3.6-27B-UD-Q4_K_XL.gguf`, review the quantized artifact’s terms, applicable upstream
model license, access conditions, and any output/use restrictions for the intended jurisdiction and
research. Keep required notices and an immutable copy of the terms used for the experiment.

Model acquisition is an explicit operator action. The pipeline accepts a direct local GGUF path and
never downloads weights at runtime or during normal tests. `model_file_sha256` proves which bytes the
run used; it does not prove that possession, processing of source text, redistribution, or output use
is permitted.

The generative component is a constrained semantic/evidence parser. It receives only the current
source item, host-linked candidates, source-local numbered spans, noisy source type/quality, and the
configured horizon. The host records a safe decision time, but that date is omitted from the model
prompt. The model receives no external RAG/web context, other documents, tools, router, prices,
labels, portfolio state, or account state. It may cite only supplied source spans and must abstain
rather than invent missing facts. Its integer semantic signal and explicitly uncalibrated raw
confidence are research features, not return forecasts, probabilities, position sizes, portfolio
weights, recommendations, or orders.

The deterministic verifier checks identity/candidate coverage, source timing, horizon, evidence
references, and whether numeric tokens in claims occur in cited spans. It does not prove that the
source is true, that cited prose semantically supports the claim, or that the mechanism is correct.
Source-quality metadata is noisy research context, not an endorsement. Historical source timestamps
also do not eliminate knowledge embedded during model pretraining; reports must identify this as a
retrospective-parser limitation unless the exact model is historically valid.

DecisionRounds freeze exact model/prompt/schema/sampling identity, current-source evidence IDs, raw
and structured output, verifier results, generated/cache/deduplicated origin, and available usage
metadata. Their current schema intentionally contains no tools, calibration, portfolio, risk, orders,
or realized outcome. Replay validates the stored record; it does not rerun the model, authenticate the
ledger externally, or establish semantic truth.

## Secrets and local artifacts

Local research providers require no credentials. The kabuS adapter reads its API password from
`NLP_TRADER_KABUS_API_PASSWORD` or an ignored local `.env`, never from research or broker YAML. The
returned token remains memory-only. Authenticated operations are restricted to the same Windows PC
as kabuStation. Prefer a secure interactive PowerShell prompt over a literal assignment or command-
line secret; an `.env` is plaintext and requires a current-user ACL. Secrets must not appear in
snapshots, manifests, audit records, reports, logs, exception messages, or shell history. Never
commit credentials, account IDs, operator-modified broker configs or intents, raw vendor data, paid
content, raw social data, generated models, caches, or reports. The bundled validation config
contains no secret. See [Broker integration](broker.md).

The bundled raw, interim, processed, model, and report roots are gitignored. If you configure roots
elsewhere, keep them out of version control too. Bronze ingestion is append-only and
content-addressed. Do not edit raw payloads or sidecars; ingest a new version and preserve both
records.

## Research-agent threat model

The optional research-agent sidecar implements a local proposal workflow plus separately authorized
development and one-time holdout evaluation. It remains disabled by default and exposes no paper or
broker action. Its controls address these threats:

- Prompt injection: source text is untrusted data and cannot name tools, change host instructions,
  request hidden data, or define executable actions. The gateway dispatches only strict typed action
  variants from a literal mapping.
- Source rights: the exporter requires availability, retention, license/terms, content-status, and
  hashed-identity assertions. Those assertions are audit metadata and are not legal permission.
- Final-holdout leakage: a strict field-level exporter constructs a new bundle; it does not forward
  generic artifacts. Future, final-holdout, paper, broker, order, account, position, target-weight,
  secret, and path fields are excluded and tested with misleading and nested fixtures.
- Paper and broker isolation: model-capable modules have no import or callable path to pipeline,
  backtest, portfolio, paper, broker, account, position, intent, or order code. Proposals cannot
  authorize execution.
- Environment and filesystem leakage: the action catalog contains no environment, arbitrary path,
  shell, Python, SQL, network, clock, or secret-store tool. A versioned environment scrub is
  defense-in-depth; it is not an operating-system sandbox.
- Local resource exhaustion: model steps, tool calls, evidence pages, bytes, context/output tokens,
  wall time, and retained artifacts are bounded before generation. Exhaustion fails the attempt and
  never yields a partially accepted proposal.
- Corrupt storage and concurrent writers: authoritative JSONL replays fail closed on malformed,
  noncanonical, duplicate-key, incomplete, non-finite, reordered, or broken-chain data. One
  nonblocking global advisory lock serializes registry transitions. Automatic truncation is
  forbidden.
- Human contamination: software cannot prevent a person from pasting holdout facts into a question,
  inspecting results outside the registry, or misdeclaring lineage. The workflow records this as a
  limitation and requires conservative external holdout-use registration for confirmatory claims.

The sidecar is a model-capability boundary, not an OS sandbox. Stronger process isolation is
a separate hardening task and must not be implied by verifier or import-firewall tests.

## Research, paper, and live boundaries

- Research stages create features, labels, models, predictions, hypothetical backtests, and reports.
- `paper` converts the latest combined-model daily decision into constrained, simulation-only pending
  next-session-open intents. The snapshot has `status: pending_unfilled_intents`, unchanged initial
  capital/equity, empty positions, and no trades. It records no fill, cost, cash balance, position
  mutation, or automatic horizon-close liquidation, and has no broker credentials, network client,
  external account state, or order-routing adapter.
- `OrderIntent` and `PaperOrderIntent` are target-weight research records, not executable orders.
- No research, backtest, report, or paper code path places or transmits a live order or converts its
  output into one.
- Only the standalone `nlp-trader broker ...` group can contact kabuStation. It accepts a separately
  prepared strict cash-order document and requires explicit operator confirmations; production can
  transmit real orders.

The kabuS integration is scoped to a private, single-user installation where the account holder owns
and controls the resulting program and is its sole operator. Codex-assisted development does not, by
itself, change that engineering assumption; no other person may use or operate the program or its
API access, and credentials and account data remain private. This does not assert that source-code
distribution is prohibited. The operator remains responsible for confirming that their own use
complies with the provider's current service rules; this project does not make a legal
determination. The adapter must run on the same
Windows PC as kabuStation. Its validation endpoint returns fixed test values and cannot place a real
order, while production can move real money. Local limits, confirmations, the kill switch,
reconciliation, and audit evidence reduce operational risk but do not guarantee correctness or
prevent loss. Read [Broker integration](broker.md) before enabling it.

## Reporting language

Reports must state that results are hypothetical and assumption-dependent. Do not use language that
guarantees returns or disguises synthetic/backtested results as realized performance. Show costs,
slippage, borrow and liquidity assumptions, turnover, drawdown, exposure, limitations, and sample
size. Preserve negative experiments and disclose missing point-in-time, survivorship, corporate-
action return-factor provenance, bounded event context, and execution data.

For operational details, see [Input data](input_data.md), [Research protocol](research_protocol.md),
[Backtesting](backtesting.md), and [Broker integration](broker.md).

Return to the [documentation home](README.md).
