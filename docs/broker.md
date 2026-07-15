# kabuS Cash-Equity Broker Adapter

The kabuS adapter is a standalone live-trading boundary for a private, single-user installation.
It is not part of the research pipeline, backtests, or paper workflow, and it never converts their
outputs into orders. Only an operator-prepared cash-order JSON document passed to an explicit
`nlp-trader broker` command can reach the adapter.

This component can place and cancel real orders. Its checks reduce operational risk; they do not
make an order correct, guarantee execution, prevent loss, or replace review in kabuStation. Review
the current [kabuStation API service rules][service-rules], account rules, [official portal][portal],
and [API reference][api-reference] before use.

## Scope and non-goals

The adapter supports:

- Japanese cash-equity buys and sales of existing cash holdings;
- limit orders with an explicit expiry date;
- validation and production kabuStation API environments;
- sanitized status, bounded reconciliation, and cancellation;
- local quantity, notional, loss, freshness, and open-order limits;
- one current-user kill switch and operation lock shared by every config and environment; and
- a current-user append-only, hash-chained audit ledger.

It does not support market orders, margin trading, short selling, futures, options, funds, foreign
securities, amendments, automatic strategy execution, unattended scheduling, remote operation, or
broker-managed algorithms. It is not a complete fill, lot, realized-P/L, or portfolio-accounting
system. The ordinary `smoke`, `backtest`, `report`, and `paper` commands do not load broker secrets,
contact kabuStation, or invoke this adapter.

## Private single-user and deployment assumptions

This integration assumes all of the following:

1. The account holder owns and controls the resulting program and is its sole operator for their
   own private use. Using Codex as a development assistant does not, by itself, change that
   engineering assumption: no other person may use or operate the program or its API access.
   Credentials and account data remain private. This is an engineering assumption, not a legal
   opinion; it does not assert that source-code distribution is prohibited, and the operator remains
   responsible for confirming compliance with the current [service rules][service-rules].
2. Every authenticated command runs on the same Windows PC as kabuStation. The adapter rejects
   authenticated operation on other operating systems. kabuStation must be running, logged in, and
   configured for API access.
3. The API remains local. Do not proxy, port-forward, tunnel, container-publish, or remotely expose
   it. Validation is fixed to `http://127.0.0.1:18081/kabusapi`; production is fixed to
   `http://127.0.0.1:18080/kabusapi`. Neither host nor port is configurable.
4. The Windows PC uses Japan time and the date conventions in the official
   [time-zone guide][time-zone]. Intent timestamps are timezone-aware and normalized to UTC.
5. Only one kabuStation instance and one operating-system account operate this adapter for the
   brokerage account.

The production endpoint can expose real account information and place or cancel real orders. The
provider's [validation environment][validation-stub] returns fixed values and cannot place a real
order or move money. It is useful for protocol and failure-path exercises, but it does not simulate
realistic prices, holdings, buying power, fills, transitions, or losses. Its fixed quote timestamps
may be stale under the adapter's freshness rule, so a validation submission may correctly fail
closed; do not weaken freshness checks to force it through.

## Local security setup on Windows

Keep modified broker configs and intent files outside the repository. One reasonable location is a
private operator subdirectory beside the adapter's automatically selected state directory:

```powershell
$StateRoot = Join-Path $env:LOCALAPPDATA "nlp-trader\kabus"
$OperatorRoot = Join-Path $StateRoot "operator"
$IntentRoot = Join-Path $OperatorRoot "intents"
New-Item -ItemType Directory -Force -Path $IntentRoot | Out-Null
$Account = [Security.Principal.WindowsIdentity]::GetCurrent().Name
icacls $StateRoot /inheritance:r /grant:r "${Account}:(OI)(CI)F"
```

Review the resulting ACL with `icacls $StateRoot`. Windows ACL inheritance, backup tools,
administrators, malware, and another process running as the same user can still expose or alter
these files. The adapter attempts private file modes where the platform supports them, but it does
not prove that the Windows ACL is safe.

Copy the bundled validation config before changing it:

```powershell
Copy-Item configs\kabus.validation.yaml "$OperatorRoot\kabus.validation.yaml"
```

Do not put the API password in YAML, JSON, command-line arguments, or a PowerShell string literal.
Prompt for it and keep the authenticated command inside the `try` block so the environment variable
is removed immediately afterward:

```powershell
$Config = "$OperatorRoot\kabus.validation.yaml"
$Password = Read-Host "kabuStation API password" -AsSecureString
$Bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password)
try {
    $env:NLP_TRADER_KABUS_API_PASSWORD = `
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Bstr)
    uv run nlp-trader broker status --config $Config
}
finally {
    Remove-Item Env:NLP_TRADER_KABUS_API_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr)
    $Password = $null
}
```

Repeat that prompt wrapper for each authenticated command, or place a reusable wrapper in a
current-user ACL-protected script outside git. The password necessarily exists briefly in the
process environment and child-process memory, but this pattern avoids placing it literally in shell
history. An ignored `.env` is supported, but an interactive prompt is preferable; if `.env` is
used, keep it outside git, restrict its ACL, and remember that it contains plaintext.

The token returned by kabuStation remains memory-only and must not be logged or persisted. It is
accepted only as bounded visible ASCII suitable for an HTTP header; control characters, non-ASCII
text, and oversized values fail authentication before any authorized request. The provider states
that the token becomes invalid after kabuStation exits or logs out, or when another token is issued.

### Loopback is not server authentication

The transport uses numeric `127.0.0.1`, ignores proxy environment variables, refuses redirects, and
checks the connected peer address and expected port before writing credentials. These controls
prevent accidental off-host transmission; they do not cryptographically identify kabuStation.
Malformed status lines, truncated reads, and other HTTP parser failures are collapsed into sanitized
transport failures; after a mutation attempt, that means an ambiguous outcome and never a retry.
Another process running locally can bind the expected port when kabuStation is absent and receive
the API password. Keep the PC free of untrusted software, confirm kabuStation is running before an
authenticated command, and investigate unexpected listeners or connection behavior. Do not weaken
this boundary by exposing the loopback ports.

## Fixed current-user safety state

Audit, kill-switch, and lock paths are not config fields. Every validation and production config for
the current operating-system user shares the same state:

| Platform | State root |
|---|---|
| Windows | `%LOCALAPPDATA%\nlp-trader\kabus` |
| macOS | `~/Library/Application Support/nlp-trader/kabus` |
| Linux/other | `$XDG_STATE_HOME/nlp-trader/kabus`, or `~/.local/state/nlp-trader/kabus` |

The files below live in that root:

| File | Purpose |
|---|---|
| `audit.jsonl` | Append-only broker audit ledger shared across configs and environments. |
| `KILL_SWITCH` | Presence blocks every new submission for this user. |
| `operation.lock` | Stable file on which the process holds a nonblocking operating-system advisory lock. |

Changing config files cannot bypass an unresolved mutation, daily accepted-notional history, or the
kill switch within one state root. On Windows, `%LOCALAPPDATA%` is a trusted process-security input:
do not override it when running broker commands. An override would select a different safety domain.
Run `validate-config` in the same launch environment before authenticated use and confirm every
broker config prints the same expected paths.

`operation.lock` is expected to remain on disk. Never delete it as crash recovery. The lock is held
by the open file descriptor, not by file presence; the operating system releases it when the
command closes the descriptor or the process exits. If a command reports contention, find the other
running broker process and wait for or stop it safely. Deleting the stable file can create two lock
domains and defeat serialization.

## Install and validate configuration

The broker uses an independent strict config; do not add broker fields to research YAML:

```powershell
uv run nlp-trader broker validate-config --config $Config
```

This command is offline and does not read the password. Unknown fields, duplicate mapping keys,
aliases, merge keys, credential-like keys, non-lowercase YAML booleans, and type coercions are
rejected.

### Broker configuration reference

| Field | Allowed value or rule | Meaning |
|---|---|---|
| `schema_version` | `kabus-broker-v1` | Strict broker-config schema identity. |
| `provider` | `kabus` | Selects this adapter. |
| `environment` | `validation` or `production` | Selects fixed port `18081` or real-order port `18080`. |
| `enabled` | boolean | Master adapter gate. |
| `order_submission_enabled` | boolean | Gate for new submissions. Recovery status, reconciliation, and cancellation remain available when false if `enabled` is true. |
| `production_acknowledgement` | `REAL_ORDERS` or `null` | Required only when production submission is enabled; forbidden otherwise. It does not replace the per-command production confirmation. |
| `single_user_private_use` | `true` | Records the account-holder-only deployment assumption. |
| `account_type` | `2`, `4`, or `12` | kabuS `AccountType`; confirm it for the account. |
| `cash_buy_deliv_type` | `2` or `3` | Cash-buy `DelivType`; `2` checks broker-held cash and `3` checks total cash buying power, including the linked-bank facility when enabled. |
| `cash_buy_fund_type` | `"02"` or `"AA"` | Cash-buy `FundType`; quote `"02"` to preserve the leading zero. |
| `allowed_symbols` | unique uppercase symbol codes | Exact nonempty allowlist while enabled. |
| `allowed_exchanges` | unique values from `3, 5, 6, 9, 27` | Exact nonempty routing allowlist for new orders while enabled. Direct TSE `1` is intentionally unsupported. |
| `allow_market_orders` | `false` only | Reserved fail-closed setting. Market orders are not supported. |
| `max_order_quantity` | positive integer | Per-order share ceiling. |
| `max_order_notional_jpy` | positive number | Per-order limit-price notional ceiling. |
| `max_daily_order_notional_jpy` | positive number | Cumulative locally accepted-order notional ceiling for the Japan date. |
| `max_position_quantity_per_symbol` | positive integer | Maximum projected cash quantity for one symbol across cash account types. |
| `max_position_notional_jpy_per_symbol` | positive number | Maximum projected cash notional for one symbol, including pending buys. |
| `max_gross_cash_position_notional_jpy` | positive number | Maximum projected gross cash-position notional, including pending buys. |
| `max_total_unrealized_loss_jpy` | positive number | Blocks submission at or below the corresponding negative account-wide reported unrealized P/L. |
| `max_open_orders` | positive integer | Maximum currently open cash orders before a new submission. |
| `max_intent_age_seconds` | positive integer | Maximum age of `created_at`. |
| `max_future_intent_skew_seconds` | nonnegative integer | Maximum tolerated future clock or timestamp skew. |
| `max_preflight_duration_seconds` | `1` through `60` | Whole-preflight duration ceiling, rechecked immediately before transport. |
| `max_reconciliation_match_seconds` | `1` through `300` | Bounded evidence window after the later recorded attempt/unknown-outcome timestamp; cannot be shorter than the combined preflight ceiling and one transport blocking timeout. |
| `max_quote_age_seconds` | `1` through `300` | Maximum age of the applicable broker quote. |
| `max_price_deviation_bps` | nonnegative number | Maximum difference between the limit price and applicable current quote. |
| `timeout_seconds` | `0.1` through `30.0` | urllib socket/blocking-operation timeout, not a whole-request deadline. Any mutation transport failure is ambiguous, never permission to retry. |

The config additionally requires the per-order notional not to exceed the daily or per-symbol
notional limit, the per-symbol notional not to exceed the gross limit, the order quantity not to
exceed the position quantity, the preflight duration not to exceed intent age, and the reconciliation
window to cover the maximum preflight plus one transport blocking timeout. A slow response can hold
the global lock longer than `timeout_seconds`; it cannot make the adapter retry.

For a buy, account and custody settings become the provider's `AccountType`, `DelivType`, and
`FundType`. Preflight queries the symbol-specific `/wallet/cash/{symbol@exchange}` endpoint. For
`cash_buy_deliv_type: 2`, it uses `AuKCStockAccountWallet` so linked-bank cash cannot overstate the
cash available to that custody path; for `3`, it uses `StockAccountWallet`. A missing, null,
non-finite, or negative applicable value fails closed. Cash-sale constants, `SecurityType`, and
`CashMargin` are fixed by the adapter. The unrealized-loss check uses all positions returned by the
account-wide positions request and requires complete numeric `ProfitLoss` coverage. It is not a
realized or intraday-loss monitor.

New orders intentionally exclude direct TSE `Exchange=1`. The provider documents that route as a
maintenance-only exception for cash orders when SOR/TSE+ cannot be used; the adapter does not expose
that break-glass path. `reference_exchange: 1` remains required for ordinary SOR `9` and TSE+ `27`
symbol and quote lookups; the symbol-specific wallet lookup uses the actual order route (`9` or
`27`). Existing broker orders may still report exchange `1` and can be
reviewed for risk-reducing cancellation.

## Cash-order intent JSON

Order JSON is strict, bounded to 16 KiB, and independent from research schemas. Duplicate or unknown
keys, non-finite values, booleans in integer fields, an invalid root, or an oversized document are
rejected.

| Field | Rule |
|---|---|
| `schema_version` | Exactly `kabus-cash-order-v1`. |
| `client_order_id` | Fresh operator-assigned ID, 1–128 supported identifier characters. Never reuse it. |
| `strategy_id` | Human-auditable origin label, not permission to automate strategy output. |
| `created_at` | Timezone-aware ISO-8601 timestamp, normalized to UTC and freshness-checked. |
| `symbol` | Uppercase cash-equity symbol also present in `allowed_symbols`. |
| `exchange` | `3`, `5`, `6`, `9`, or `27`, also present in `allowed_exchanges`; new direct TSE `1` orders are unsupported. |
| `reference_exchange` | Same direct exchange for `3`, `5`, or `6`; exactly `1` for SOR `9` or TSE+ `27`. |
| `side` | `buy` or `sell`; a sell is cash-only and cannot exceed available confirmed holdings. |
| `quantity` | Positive integer within limits and an exact multiple of broker `TradingUnit`. |
| `order_type` | Exactly `limit`. |
| `limit_price` | Positive finite number. |
| `expire_day` | Required explicit valid `YYYYMMDD` integer; `0` and implicit provider expiry are forbidden. |

Example only—not an investment instruction:

```json
{
  "schema_version": "kabus-cash-order-v1",
  "client_order_id": "manual-20260715-0001",
  "strategy_id": "operator-reviewed",
  "created_at": "2026-07-15T01:00:00Z",
  "symbol": "9433",
  "exchange": 27,
  "reference_exchange": 1,
  "side": "buy",
  "quantity": 100,
  "order_type": "limit",
  "limit_price": 215.5,
  "expire_day": 20260716
}
```

Recreate every field for the order actually being reviewed. Save the file under the protected intent
directory, not the repository or shell history.

## Command and platform reference

| Command | Authentication | Mutation | Purpose |
|---|---:|---:|---|
| `validate-config` | no | no | Validate strict broker YAML and show fixed state paths. |
| `preview-order` | no | no | Print approval, intent, and config digests plus the effective request. |
| `status` | yes, Windows only | no broker write | Return a sanitized best-effort incident snapshot. |
| `reconcile` | yes, Windows only | local audit write | Record scoped, bounded evidence without clearing unknowns. |
| `submit-order` | yes, Windows only | **real in production** | Strictly preflight and attempt one confirmed order. |
| `preview-cancel` | yes, Windows only | no broker write | Fetch current order context and print a cancellation approval digest. |
| `cancel-order` | yes, Windows only | **real in production** | Attempt one context-confirmed cancellation. |
| `resolve-unknown` | yes, Windows only | local audit write | Resolve only from a valid scoped reconciliation and fresh evidence. |
| `engage-kill-switch` | no | local file/audit write | Block new submissions for every config and environment. |
| `release-kill-switch` | no | local file/audit write | Release the exact reviewed marker only when no mutation is unresolved. |

Run `uv run nlp-trader broker --help` and each command's `--help` for the installed revision.

### Preview and submit an order

Preview with the exact config and unchanged order file that will be submitted:

```powershell
$Order = "$IntentRoot\order-20260715-0001.json"
uv run nlp-trader broker preview-order --config $Config --order $Order
```

Review `intent`, `effective_request_payload`, `environment`, `config_digest`, and
`confirmation_digest`. The `confirmation_digest` is a SHA-256 approval envelope over:

- the adapter version;
- the complete effective config, including environment and all limits/account settings;
- the exact effective kabuS request payload; and
- the complete canonical intent.

`intent_digest` identifies only the canonical intent and is not sufficient approval. Any change to
the config, environment, adapter, mapped payload, or intent invalidates the confirmation. The digest
is not a durable authorization token and does not reserve buying power, holdings, a quote, or an
order slot.

Inside the secure password wrapper, submit the same files and exact approval digest:

```powershell
uv run nlp-trader broker submit-order `
    --config $Config `
    --order $Order `
    --confirm EXACT_CONFIRMATION_DIGEST_FROM_PREVIEW
```

Production submission additionally requires `environment: production`,
`order_submission_enabled: true`, `production_acknowledgement: REAL_ORDERS`, and the exact
per-command flag:

```powershell
--confirm-production REAL_ORDERS
```

None of these acknowledgements bypasses a limit, freshness, state, or evidence failure.

### Status and tolerant reconciliation

Use authenticated commands inside the secure password wrapper:

```powershell
uv run nlp-trader broker status --config $Config
uv run nlp-trader broker reconcile --config $Config
```

`status` and `reconcile` query each incident-information endpoint independently and return only a
sanitized summary. If one endpoint fails or returns unusable data, other available fields remain
visible and `unavailable_fields` names the missing category. A `null` or unavailable field is not a
passing safety check. Submission preflight remains strict and fails closed instead of using the
tolerant incident snapshot.

`status` does not alter broker state or append an audit event. `reconcile` appends the sanitized
summary and per-unresolved-mutation evidence. If orders are unavailable, the environment/config does
not match the attempt, or evidence cannot be parsed within the configured time window,
`evidence_valid` is false and the reconciliation cannot resolve an unknown.

### Preview and cancel one order

Cancellation is a two-step ceremony because the approval binds the order's current broker context,
the full config, adapter version, and a fresh local action ID:

```powershell
$OrderId = "BROKER_ORDER_ID"
$ActionId = "cancel-20260715-0001"
uv run nlp-trader broker preview-cancel `
    --config $Config `
    --order-id $OrderId `
    --client-action-id $ActionId
```

Review `broker_order` and copy its `confirmation_digest`. Then run:

```powershell
uv run nlp-trader broker cancel-order `
    --config $Config `
    --order-id $OrderId `
    --client-action-id $ActionId `
    --confirm EXACT_CONFIRMATION_DIGEST_FROM_PREVIEW_CANCEL
```

Add `--confirm-production REAL_ORDERS` in production. If the broker order changes between preview
and cancellation, the confirmation no longer matches. Preview and cancellation accept only a
processed top-level `State=3` order; receipt/processing states `1` and `2`, modification/cancellation
in flight `4`, and terminal state `5` are rejected. The approval context also binds the observed
`OrdType`, so an order-condition change invalidates confirmation. A duplicate action ID or second
cancellation while the first outcome is unresolved is also rejected. Cancellation remains
available when new-order submission is disabled, the kill switch is engaged, or another mutation is
unresolved.

### Engage and release the kill switch

Engage it with a specific reason:

```powershell
uv run nlp-trader broker engage-kill-switch `
    --config $Config `
    --reason "unexpected account state"
```

The marker blocks new submissions across both environments. Engagement uses the same global
operation lock as submission: if another broker operation holds that lock, engagement fails and
must be retried; once engagement succeeds, no adapter mutation overlapped it. The marker does not
cancel orders, liquidate positions, log out kabuStation, preempt a mutation already handed to the
provider before an unsuccessful engagement attempt, or stop orders entered outside this adapter.

Before release, inspect kabuStation, `status`, `reconcile`, and the audit ledger. `status` reports
`kill_switch_digest`. Release only that exact marker:

```powershell
uv run nlp-trader broker release-kill-switch `
    --config $Config `
    --confirm RELEASE_KILL_SWITCH:EXACT_MARKER_DIGEST_FROM_STATUS
```

Release is refused while any mutation is unresolved. A later marker has a different digest, so a
previous release confirmation cannot release it.

### Resolve an ambiguous mutation

Every resolution is tied to the attempt's original environment and exact config digest, a later
reconciliation sequence, the attempt sequence, and bounded broker evidence. First run `reconcile`
with the original config and inspect:

- top-level `audit_sequence` and `config_digest`;
- the matching evidence item's `attempt_sequence` and `attempt_config_digest`;
- `scope_matches`, `orders_available`, `orders_valid`, `evidence_valid`, and
  `negative_observation_complete`;
- `observed_at`, `match_window_anchor`, and `match_window_end`;
- `candidate_order_ids`; and
- `exact_match_order_ids`.

The matcher starts from the durable attempt and ends only after the configured interval following
the later of that attempt or its recorded `broker_*_unknown` outcome. This prevents a response that
outlasts one socket timeout from falling outside the negative-evidence window.

For an accepted result, the named broker ID must be the one unique exact and candidate evidence
match. Immediately before resolving, the adapter requires the recorded evidence to remain within
its freshness bound and re-reads the complete cash-order list; that fresh read must still contain
exactly one candidate and exact match for the attempted request or processed cancellation.
Use:

```powershell
uv run nlp-trader broker resolve-unknown `
    --config $Config `
    --client-order-id CLIENT_OR_ACTION_ID `
    --resolution accepted `
    --broker-order-id BROKER_ORDER_ID `
    --confirm ACCEPTED:CLIENT_OR_ACTION_ID:BROKER_ORDER_ID:ATTEMPT_SEQUENCE:RECONCILIATION_SEQUENCE:CONFIG_DIGEST
```

For an operator-asserted not-accepted result, both candidate and exact-match lists must be empty,
`negative_observation_complete` must be true after the full match window, and the kill switch must
be engaged. Resolution also requires a fresh complete cash-order read with no candidates. For an
unknown cancellation, only the original order still being in unchanged processed `State=3` with no
cancellation detail in the bounded window is negative evidence; missing, duplicate, malformed,
in-flight, terminal, or cancellation-detail evidence remains unresolved.

```powershell
uv run nlp-trader broker resolve-unknown `
    --config $Config `
    --client-order-id CLIENT_OR_ACTION_ID `
    --resolution not-accepted `
    --confirm NOT_ACCEPTED:CLIENT_OR_ACTION_ID:ATTEMPT_SEQUENCE:RECONCILIATION_SEQUENCE:CONFIG_DIGEST
```

Add `--confirm-production REAL_ORDERS` in production. Absence from one snapshot is not proof of
non-acceptance. Keep the kill switch engaged if the order endpoint is unavailable, evidence is
invalid or outside the bounded match window, candidates are present, or kabuStation and the audit
record disagree.

## Submission preflight

Before `POST /sendorder`, the adapter fails closed on:

- master and production gates, approval digest, client ID uniqueness, unresolved mutations, and the
  shared kill switch;
- intent age/future skew, explicit expiry, Japan-date rollover, and total preflight duration;
- symbol/exchange allowlists, limit-only semantics, trading unit, broker price range, current quote
  timestamp, price deviation, and exact tick alignment from the symbol's `PriceRangeGroup`;
- per-order and locally accepted daily notional;
- the custody-appropriate symbol-specific cash wallet and broker cash-equity soft limit;
- cash holdings and pending sells for a sale;
- projected per-symbol quantity/notional and gross cash notional including pending buys, plus the
  provider's `PerSymbolLimit` when it is numeric;
- open cash-order counts and the provider's concurrent per-symbol order limit; and
- complete all-product unrealized `ProfitLoss` evidence and the configured loss threshold.

`max_open_orders` counts the cash-order (`product=1`) view. Separately, the provider's approximate
five-concurrent-orders-per-symbol ceiling is checked against the all-products (`product=0`) order
view, so margin orders and different order conditions cannot be hidden from that provider limit.
Malformed state or symbol evidence in either required view fails closed.
Projected buy quantity includes same-symbol cash holdings and pending buys across all account types.
For a sale, available quantity remains restricted to the configured `account_type`; holdings in a
different account bucket cannot authorize that sale.

`PriceRangeGroup` must be one of the supported official groups `10000`, `10003`, or `10004`, and the
limit price must be an exact multiple of the tier's tick. The adapter uses decimal arithmetic and
fails closed on a missing/unknown group or an off-tick price. `PerSymbolLimit` must be present in the
symbol response: a numeric value caps projected symbol notional, while the provider's explicit
`null` means that provider-side category limit does not apply. The local configured limits always
remain in force. The cash-equity soft limit is interpreted in the provider-documented units of
JPY 10,000.

The adapter rechecks time, intent and quote freshness, expiry, Japan date, daily accepted notional,
preflight duration, and kill-switch state immediately before transport. This reduces but cannot
eliminate the gap between observation and broker processing. Fees, taxes, partial fills, price
movement, SOR, reservations, external orders, and concurrent broker state can still change the
outcome.

## Mutation and ambiguity policy

The adapter never automatically retries a mutation. Once a send or cancel attempt has been audited,
every failure to obtain the one expected successful response is treated as ambiguous, including a
timeout, connection loss, HTTP non-success status, malformed/truncated body, nonzero provider
`Result`, or unexpected returned order ID. The adapter does not treat any of those outcomes as a
definitive rejection and does not reuse the client/action ID.

An unresolved send or cancel blocks all new submissions. A second cancel for the same broker order
is also blocked while its prior cancel is unresolved. Reconcile and resolve from scoped evidence;
never resend blindly. A successful send response means the provider returned an accepted order ID,
not that the order filled. A successful cancel response likewise requires later broker-state review.

The provider documents request-rate ceilings in its [FAQ][faq]. This low-throughput, one-operation-
at-a-time adapter is not permission to approach them.

## Audit ledger integrity and privacy

`audit.jsonl` is canonical append-only JSONL with monotonic sequence numbers and a SHA-256
previous-record hash chain. It records the adapter/config/payload identities, approval digests,
sanitized preflight evidence, attempts, outcomes, reconciliation evidence, cancellation context,
kill-switch changes, and resolutions. It must never contain the password or token.

Treat it as sensitive account and trading data. Restrict the state directory to the operating user,
back it up securely, and never commit or redistribute it. A broken chain, partial final record,
duplicate sequence, or noncanonical record blocks unsafe operation. Do not edit, truncate, replace,
or reset the ledger to clear an error, unknown, or daily limit.

The chain is tamper-evident only relative to a trusted prior copy. It is not signed or externally
timestamped, and kabuStation/broker records remain authoritative. Anyone with write access to the
code, config, state directory, and its backups may be able to change both records and validation
logic.

## Production runbook

### Before enabling production

1. Re-read the current [service rules][service-rules], [portal][portal], [API reference][api-reference],
   and [error guide][errors]. Confirm that private sole-operator use still complies.
2. Harden the Windows account and PC, protect state/config/intent ACLs, and verify kabuStation owns
   the expected local listener before entering the password.
3. Exercise parsing, approvals, lock contention, kill-marker lifecycle, cancellation, ambiguity, and
   reconciliation against validation port `18081`. Do not expect realistic account values or fresh
   quote timestamps; a stale fixed timestamp should fail strict preflight.
4. Copy the config to the protected operator directory. Set `environment: production`, leave
   `order_submission_enabled: false`, use minimal allowlists and limits, and validate it.
5. With submission disabled, run production `status` and compare every available summary with the
   kabuStation UI. Treat unavailable or inconsistent evidence as a stop.
6. Exercise the shared kill switch and inspect the audit chain, stable lock file, and ACLs.
7. Only for a deliberately reviewed minimal-size real limit order, enable submission and set
   `production_acknowledgement: REAL_ORDERS`. Create a fresh intent, preview it with that exact config,
   and submit with both confirmations. Observe the result in kabuStation immediately.

### For every operation

1. Confirm Windows account, kabuStation login/environment/account, listener ownership, current
   positions, open orders, buying power, market state, and kill-switch status.
2. Use fresh client/action IDs and current timestamps. Never derive an executable intent
   automatically from research output.
3. Preview and review the full config, effective request or broker-order context, and approval
   digest immediately before the mutation.
4. Stop and engage the kill switch on unavailable, stale, surprising, or inconsistent evidence.
5. After an accepted response, verify the broker ID and complete order details in kabuStation, then
   reconcile. Continue monitoring fills and cancellations in the authoritative broker UI/history.

### On any mutation error

1. Do not retry. Engage the kill switch with a specific incident reason.
2. Preserve terminal output and the audit ledger without sharing secrets or private account data.
3. Run tolerant `status` and `reconcile`; inspect kabuStation, its API logs, and the official
   [error guide][errors].
4. Cancel a known open order only through a fresh preview-cancel ceremony if that is the intended
   risk action.
5. Resolve only with the original config and exact bounded evidence. Leave the marker engaged while
   anything remains uncertain.

### After logout, restart, or date rollover

The API token does not survive kabuStation logout/restart. Confirm the UI account and environment,
rerun `status`, and reconcile before another operation. Never delete the audit ledger or stable lock
file at a day boundary. The ledger's continuity and Japan-date accepted-notional history are safety
state.

## Explicit limitations

- This is hypothetical-research software with an explicitly requested live boundary, not financial
  advice, an investment adviser, or evidence of profitability.
- CLI confirmation is single-user friction, not independent authorization. A process or person with
  the same Windows account, password, code, config, and files can act as the operator.
- Plain HTTP loopback has no cryptographic kabuStation identity; a malicious local listener can
  impersonate it and capture the password.
- Local limits, the lock, ledger, and kill switch coordinate only this installation and operating-
  system user. They do not constrain kabuStation UI orders, another program/user, or another PC.
- The final kill-switch check cannot recall a request already handed to the transport. The marker
  does not cancel or liquidate anything automatically.
- `timeout_seconds` limits individual blocking socket operations, not total wall-clock request
  duration; slow/trickling responses can hold the global operation lock longer.
- Status and reconciliation are best-effort incident views. Provider observations can be stale,
  incomplete, or change concurrently; unavailable fields are not proof of safety or non-acceptance.
- The bounded matcher is conservative evidence, not a provider idempotency key. kabuS supplies no
  client idempotency key used by this adapter.
- There is no complete fill/partial-fill lifecycle synchronizer, tax-lot ledger, realized-P/L daily
  stop, fee/tax reconstruction, or multi-product account-wide exposure engine. Use broker records as
  authoritative.
- The unrealized-loss guard depends on complete provider `ProfitLoss` values and is not a realized,
  intraday, or guaranteed account-wide loss limit.
- The validation stub does not test real fills, production constraints, execution quality, SOR,
  production availability, fresh quote timestamps, or Windows credential/ACL behavior end to end.
- Reconciliation currently expects a submitted normal limit request (`FrontOrderType=20`) to appear
  in the order response as `OrdType=1`. The published [OpenAPI schema][openapi-schema] does not
  eliminate all ambiguity around the reported cash `CashMargin`, `DelivType`, and order-condition
  fields. Capture and compare the actual enrolled validation responses on Windows before any
  production readiness claim; treat any different mapping as a stop, not permission to loosen
  matching.
- The Windows integration must be exercised manually against the operator's own enrolled validation
  setup; automated tests use injected offline fixtures and no credentials.
- No guard proves suitability, legality, compliance, correctness, or loss prevention for an order.
  All operations remain the account holder's responsibility.

[portal]: https://kabucom.github.io/kabusapi/ptal/
[api-reference]: https://kabucom.github.io/kabusapi/reference/index.html
[openapi-schema]: https://github.com/kabucom/kabusapi/blob/master/reference/kabu_STATION_API.yaml
[faq]: https://kabucom.github.io/kabusapi/ptal/faq.html
[validation-stub]: https://kabucom.github.io/kabusapi/ptal/add-in.html
[time-zone]: https://kabucom.github.io/kabusapi/ptal/time-zone-setting.html
[errors]: https://kabucom.github.io/kabusapi/ptal/error.html
[service-rules]: https://kabu.com/pdf/Gmkpdf/service/kabustationapiuserpolicy.pdf
