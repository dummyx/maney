# Backtesting

Backtests are deterministic, hypothetical target-weight simulations. Signal generation, portfolio
construction, execution costs, position accounting, and metrics are separate modules. The engine
rejects duplicate predictions/labels and accepts one horizon per run.

Read this page when deciding whether the replay matches a research question. It documents what the
engine simulates, what it merely approximates, and what it does not model. Metric field names and
artifact locations are indexed in [Outputs](outputs.md).

## Replay and fills

Each decision occurs after a complete daily bar at the official exchange close. Candidates
are ranked by absolute model score. Positive scores request a long position; negative scores request
a short only when shorting is enabled. The portfolio layer applies eligibility and risk limits before
any return is realized.

Every replay period is an independent round trip that starts flat:

1. Enter constrained targets at the raw tradable `open` of the exact next XNYS session.
2. Apply the label return through the exact configured horizon session. That return compares the raw
   entry open and raw exit close after multiplying each by its bar's causal
   `return_adjustment_factor`.
3. Drift weights with those returns, then force every position to zero at the raw tradable horizon
   close.
4. Charge commission, spread, slippage, and impact on both entry and liquidation, plus borrow over
   `max(1, elapsed calendar days)` where applicable.

Multi-session horizons select non-overlapping decision windows. There is no position carried from
one window into the next and no exposure before the next-session entry. Initial capital converts
weight changes into participation estimates; performance is reported as normalized returns.

The recorded raw open and close prices are assumed full fills. This is not an opening/closing auction
or venue simulator: there is no order book, queue priority, partial-fill path, intraday path, latency
race, tax model, or guarantee that a target-weight fill was achievable.

## Enforced constraints

The portfolio constructor enforces:

- maximum absolute position weight
- maximum gross and absolute net exposure
- maximum gross sector exposure
- maximum absolute beta exposure
- conservative missing-risk handling: `missing_beta_fallback` replaces an unavailable 60-session
  beta, while volatility costs use the maximum of `missing_volatility_floor`, available short-window
  realized volatility, and the high-low estimator when the 20-session estimate is unavailable
- maximum daily turnover. A one-session round trip sizes its entry against
  `max_daily_turnover / (2 + same_day_exit_notional_buffer)`, then hard-checks realized entry plus
  exit turnover against the full same-day budget. A longer horizon applies the full budget
  separately on its entry and exit days
- maximum participation rate using the decision-time dollar-volume proxy for both legs. A
  one-session entry is conservatively sized against
  `max_participation_rate / (1 + same_day_exit_notional_buffer)` before both realized legs are
  checked against the full cap
- minimum price and dollar volume
- shorting permission, short availability, and hard-to-borrow permission

When a request breaches an exposure limit, weights are scaled deterministically. The participation
cap reduces an individual entry delta using information available at the decision. Replay recomputes
each leg's participation with the changing portfolio notional but deliberately reuses the decision-
time liquidity proxy; a modeled entry or exit above the cap fails rather than being silently
accepted. The configurable same-day buffer is 0.10 in the bundled configs and reserves room for an
exit notional that grows during the holding session; it does not replace the realized hard checks.
Ineligible assets are rejected with reason codes. Limits apply to constructed entry
targets. Returns can drift weights before the mandatory exit; the engine records pre-trade, target,
post-return, and post-liquidation exposures so that drift and the return to flat are visible. Period,
trade, position, rejection, risk-flag, and exposure logs are retained.

Participation clipping is rechecked against every exposure constraint because asymmetric liquidity
caps can break a previously balanced beta or net hedge. The corrected target is scaled toward cash;
if it cannot satisfy exposure, turnover, and participation limits together, construction fails
closed. Missing-beta and missing-volatility substitutions are recorded as explicit risk flags.

Entry turnover is recorded on period-start NAV. Exit turnover is recorded on both contemporaneous
pre-exit NAV and period-start NAV, and total round-trip turnover uses the period-start basis. A same-
session round trip must fit the configured daily budget in total. For a longer horizon, entry and exit
are separate execution days and each is checked against the daily limit.

## Cost model

For traded absolute weight `t`, volatility `v`, participation `p`, short exposure `s`, and holding
days `d`, the modeled return costs are:

```text
commission = t * commission_bps / 10,000
spread = t * half_spread_bps / 10,000
dynamic_slippage_bps = base_slippage_bps
                     + v * 10,000 * volatility_slippage_multiplier
                     + p * participation_slippage_bps
slippage = t * dynamic_slippage_bps / 10,000
impact_bps = v * sqrt(p) * 10,000 * market_impact_multiplier
market_impact = t * impact_bps / 10,000
borrow = abs(min(0, s)) * borrow_bps_per_year / 10,000 * d / 365
```

Commission, spread, slippage, and impact are applied independently to entry and forced exit. Borrow
is charged once from the target short exposure over the holding interval. Exit trading costs are
scaled to the post-return portfolio value. These are configurable proxies, not measured
implementation shortfall. Half-spread assumes crossing half the configured spread estimate.
Volatility and the liquidity proxy come from the decision feature/prediction row. The proxy is never
updated with future full-session volume, but it is still a daily-bar approximation rather than an
observed auction book. Missing or inaccurate market microstructure inputs make the resulting costs
unreliable.

The bundled configs are long-only and the pipeline currently marks short availability false. Borrow
cost code exists for properly supplied short data, but the sample does not validate a short strategy.

## Baselines and reports

The same replay is run for traditional-only, text-only, combined, equal-weight, momentum-only, and
no-trade prediction families. This makes incremental text comparisons cost- and constraint-consistent.

Reported diagnostics include:

- gross, total, cost-adjusted, and annualized return
- annualized volatility, Sharpe, Sortino, hit rate, and 5% tail loss
- maximum drawdown
- average turnover and holding period
- gross, net, beta, and sector exposure
- maximum participation
- minimum participation-based capacity proxy equity
- aggregate commission, spread, slippage, impact, borrow, and total cost
- trade count, position/trade logs, rejected intents, and risk flags

Every period also records decision, execution, and exit timestamps; gross/cost/net return; entry
turnover and exit turnover on both NAV bases; gross/net/beta/sector exposure; pre-trade, post-return,
and post-liquidation exposure; cost components; holding days; participation; capacity proxy;
rejects; missing-return flags; and equity. The capacity proxy is the minimum entry-or-exit equity
supported by the same decision-time liquidity proxy:

```text
entry = max_participation_rate * decision_dollar_volume / abs(entry_weight)
exit = max_participation_rate * decision_dollar_volume / exit_notional_weight
capacity_proxy_equity = min(all entry and exit values)
```

It is a screening diagnostic based on one decision-row dollar-volume value. It is not calibrated
strategy capacity and does not model how liquidity varies during entry or exit.

The replay never drops individual assets based on whether their future label exists. A partially
observed decision cross-section fails. A wholly unobserved common trailing decision date may be
omitted only when every expected label end lies beyond the available prediction boundary.

Annualization over a short sample is mechanically unstable. Always inspect period count, raw return,
drawdown, turnover, exposure, and costs alongside annualized statistics.

## Explicit limitations

- The next-open entry and horizon-close liquidation use observed raw tradable bar prices as assumed
  full fills; opening/closing auction mechanics are not simulated. Corporate-action continuity is in
  the factor-adjusted return, not a rewritten fill price.
- Open and intraday strategy decisions are unsupported. The existing next-open entry follows a prior
  close decision and does not make daily OHLC features known early.
- Venue queues, partial fills beyond the participation cap, latency, and intraday liquidity paths are
  not modeled.
- Market impact and spread are proxies rather than calibrated quote/fill models.
- Forced locate recalls and dynamic borrow availability are not modeled.
- Every supplied OHLC bar must set `corporate_action_adjusted=true`, supply a positive causal
  `return_adjustment_factor`, and show `adjustment_vintage_at <= ts`. Returns compare raw prices after
  applying their respective factors, while fills and liquidity retain raw tradable prices. Optional
  corporate-action event records add point-in-time event features but do not alter prices or factors;
  provider-specific revision and action histories remain the user's responsibility.
- Capacity beyond the configured participation and impact proxies is not established.
- A backtest does not model operational failure, broker controls, taxes, or live-trading risk.

The report set labels results hypothetical. The Markdown note includes the config hash, input hashes,
assumptions, limitations, and benchmark comparisons; `config.snapshot.json` holds the full resolved
config, and `run.final.json` adds per-run artifact hashes.

Continue with [Research protocol](research_protocol.md) for the acceptance checklist or return to the
[documentation home](README.md).
