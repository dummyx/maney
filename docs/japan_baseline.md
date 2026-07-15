# Japan Cash-Equity Baseline

`configs/japan_baseline.yaml` is a strict local-input template for hypothetical Japanese
cash-equity research. It selects the XJPX calendar and `japan_cash_equity_v1` input contract. It
does not download J-Quants data, bundle a vendor payload, connect to kabuS, or establish strategy
performance.

The template is expected to fail input validation until the repository owner supplies permitted
files under `data/local/japan/`. Keep those source files and all derived artifacts private and out
of git.

## Required local files

Prepare these paths relative to the repository root:

```text
data/local/japan/assets.csv
data/local/japan/market_bars.parquet
data/local/japan/text_items.jsonl
```

The Japanese contract is intentionally exact. Unsupported columns, missing required fields,
ambiguous security codes, and inconsistent asset/bar metadata fail instead of being guessed.

The asset file requires:

```text
asset_id,symbol,exchange,currency,name,sector,active_from,active_to,trading_unit
```

It may also contain `cik`, `figi`, `isin`, `industry`, `short_available`, and
`hard_to_borrow`. Use a stable `asset_id` for the listing spell, the exact canonical
four-character Japanese security code as `symbol`, `XJPX` as `exchange`, `JPY` as `currency`, and
an explicitly sourced positive `trading_unit`. Do not assume that every historical listing always
used the present trading unit. A current master applied retrospectively can introduce survivorship
and reference-data leakage.

The daily-bar input requires:

```text
asset_id,symbol,exchange,currency,trading_unit,session_date,ts,available_at,bar_size,open,high,low,close,volume,corporate_action_adjusted,adjustment_vintage_at,return_adjustment_factor,price_basis
```

`vwap` and `adjusted_close` are the only optional bar fields. Set `bar_size: 1d`,
`price_basis: raw_tradable`, and retain raw tradable OHLC and volume. Null OHLC rows are not valid
fills and must not be forward-filled. `corporate_action_adjusted: true` certifies that a causal
return factor and its vintage are present; it does not mean that OHLC has been rewritten.

## J-Quants V2 normalization

The official references are the [V2 daily-bar specification](https://jpx-jquants.com/en/spec/eq-bars-daily),
[listed-issue master](https://jpx-jquants.com/en/spec/eq-master),
[trading calendar](https://jpx-jquants.com/en/spec/mkt-cal),
[product categories](https://jpx-jquants.com/en/spec/eq-master/product-category), and
[adjustment-factor explanation](https://jpx-jquants.com/en/spec/eq-bars-daily/adj). Check the
[V1-to-V2 migration guide](https://jpx-jquants.com/en/spec/migration-v1-v2) before normalizing an
older export.

For a permitted V2 export, use this mapping:

| J-Quants V2 value | Local field or treatment |
|---|---|
| `Date` | `session_date`; derive `ts` from that XJPX session's official close. |
| `Code` | Exact canonical code in `symbol`; build a stable `asset_id` that distinguishes listing spells. Preserve the unmodified vendor value in bronze. |
| `O`, `H`, `L`, `C` | Raw `open`, `high`, `low`, `close`; these are the only simulated fill-price basis. |
| `Vo` | Integer `volume`. |
| `Va` | Preserve as turnover in source/staging provenance if useful. Do not relabel it as `vwap`. |
| `AdjFactor` | Input to the causal factor recurrence below, using only the vintage available at that historical decision. |
| Vendor adjusted OHLC/close fields | Optional diagnostics only; never use them as simulated fills or as hindsight historical features. |

For a domestic-stock baseline, define the universe from a point-in-time master and the official
domestic-stock product category (`ProdCat == "011"`). Supply `trading_unit` from a permitted
historical master or another authoritative point-in-time record; the [JPX trading-unit
overview](https://www.jpx.co.jp/english/equities/trading/domestic/03.html) explains the current
market convention but is not a historical-vintage dataset.

The local normalized bar also sets `exchange: XJPX` and `currency: JPY`. Do not truncate, pad, or
otherwise guess a code representation; explicitly reconcile it through the official master. A row
with null raw OHLC represents no usable daily fill observation and must be rejected or handled by
an explicit, documented universe/terminal-censoring policy.

## Close time, availability, and the next tradable decision

`ts` is the official XJPX session close, not the API response time. JPX cash trading currently runs
09:00–11:30 and 12:30–15:30 JST; the close changed from 15:00 to 15:30 on 2024-11-05. Use the
[official trading-hours page](https://www.jpx.co.jp/english/equities/trading/domestic/01.html) and
[JPX change notice](https://www.jpx.co.jp/english/corporate/news/news-releases/1030/20230920-01.html),
not a hardcoded UTC close.

J-Quants says daily OHLC is normally updated around 16:30 JST but explicitly does not guarantee
that time. Therefore:

- `available_at` is the earliest defensible timestamp at which the exact payload could have been
  used, based on actual vendor delivery/capture evidence;
- it will normally be later than `ts`, and must not be replaced with a blanket 16:30 assumption;
- `adjustment_vintage_at` records when that causal factor vintage became usable, and must satisfy
  `ts <= adjustment_vintage_at <= available_at`; and
- the feature `asof_ts` must be no earlier than every contributing row's `available_at`.

See the official [data-update schedule](https://jpx-jquants.com/en/spec/data-update). For a complete
session cross-section, the safe decision time is the latest required `available_at`. Entry is the
first official XJPX open strictly after that decision. If delivery is delayed past the next open,
the replay must roll entry to a later open; it must never backdate the information to the session
close or assume a same-close fill.

Permitted text that becomes available after the close but before the complete market cross-section
may enter that same safe decision. Text arriving later maps to the first subsequent market decision;
it is never moved backward.

Forward outcomes follow the same rule. `label_end_ts` is the official horizon close, while
`label_available_at` is the latest required bar availability for that complete exit-session
cross-section. A label delivered exactly at a later decision is excluded from that decision's
training set because walk-forward cutoffs are strict.

## Causal corporate-action factors

Keep raw OHLC for execution and build a separate factor using only adjustment events whose vintage
was available at that point in history. With an arbitrary positive initial scale, update the causal
factor in chronological order as:

```text
factor_t = factor_(t-1) / AdjFactor_t
```

Treat a no-action factor as `1.0`, and set the normalized `return_adjustment_factor` to the resulting
causal value. Returns compare `raw_price * return_adjustment_factor` at their two endpoints; the
absolute seed scale cancels. Never backfill a factor learned later across earlier decisions. Retain
the source payload and vintage evidence so the recurrence can be audited.

## Market-only starting point

The pipeline requires a text path even when the first experiment is market-only. To create a new,
empty permitted JSONL file without truncating any existing file:

```bash
mkdir -p data/local/japan
touch data/local/japan/text_items.jsonl
wc -c data/local/japan/text_items.jsonl
```

Confirm that `wc` reports zero bytes before relying on it as an empty input. If it is not empty,
inspect it or choose a new path; do not erase a source file. With no text items, interpret the
traditional, momentum-only, equal-weight, and no-trade paths as the useful starting comparisons.
Text-only and combined outputs with no text evidence do not test incremental text value.

The template is long-only and uses deliberately restrictive placeholder cost, turnover,
participation, position, and liquidity assumptions. Despite its historical field name,
`min_dollar_volume` is computed from raw price times volume in the asset currency, so its value in
this JPY-only template is yen not US dollars. Replace every proxy with a documented, point-in-time
assumption appropriate to the intended study; none is calibrated to a broker or venue.

## Validate and run

Replace the two placeholder terms references in the config, supply enough contiguous sessions for
the 60-session warm-up, 252-row minimum training history, horizon, and 20-period holdout, then run:

```bash
uv run nlp-trader validate-config --config configs/japan_baseline.yaml
uv run nlp-trader ingest-market --config configs/japan_baseline.yaml
uv run nlp-trader backtest --config configs/japan_baseline.yaml --limit 320
```

Start with a bounded date/symbol slice sized to local memory when the full universe is large. A
successful pipeline run would show only that the supplied data satisfies the implemented contract
and that the hypothetical replay completed. It would not prove profitability, capacity, or live
execution quality.

Continue with [Input data](input_data.md), [Data contracts](data_contracts.md), and
[Backtesting](backtesting.md).
