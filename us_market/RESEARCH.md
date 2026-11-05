# Independent US daily strategy research

The frozen experiment compares relative strength plus breakout, OBV breakout,
and volatility contraction plus breakout against a plain breakout control and
SPY. It never imports Taiwan strategy/pipeline modules, writes Taiwan SQL, runs
the Taiwan pipeline, deletes existing history, or submits brokerage orders.

Read `research_protocol.json` before running. The stock cohort is the dated
Massive common-stock snapshot on 2018-12-31, across XNYS/XNAS/XASE, without a
present-day survivor filter. Unmapped when-issued variants are explicitly listed
as exclusions. Later IPOs are outside this first fixed-cohort experiment.

## Acquisition and replay

Use the existing private `.env.us` credentials. Do not put credentials in commands,
artifacts, Git, or the Taiwan `.env`. Requests are limited to historical Massive
reference data, Alpaca SIP daily bars, corporate actions, and the read-only paper
trading calendar.

```sh
python -m us_market.research_cli acquire \
  --root .cache/us-market-v1/three-strategy-20261105 \
  --fetch --max-requests 2500

# Offline verification and replay: no network, credentials not required.
python -m us_market.research_cli acquire \
  --root .cache/us-market-v1/three-strategy-20261105

# Corporate-action overlay, with raw bars to verify adjusted share units.
python -m us_market.research_cli actions \
  --root .cache/us-market-v1/three-strategy-20261105 \
  --fetch --max-requests 500
python -m us_market.research_cli compare \
  --root .cache/us-market-v1/three-strategy-20261105 --cash-actions
```

Each explicit acquisition invocation has a maximum of 2,500 requests, four
workers, and a shared rate limiter. Failures stop acquisition; rerunning resumes
verified completed requests. There are no automatic network retries, substitute
feeds, current-stock-list fallbacks, or partial-success backtests. The root has
a 2 GiB cap and preserves at least 40 GiB of host free space. A changed protocol
requires a new root; existing objects are immutable.

Historical bars are requested in batches of at most 25 symbols, separately for
`split` and `all` adjustments, using `asof=-`. The action overlay also requests
raw prices for affected symbols. All continuation tokens are consumed
even when a page is shorter than 10,000 observations. Calendar membership,
positive OHLC, OHLC bounds, duplicate bars, paired-adjustment coverage and full
SPY calendar coverage are checked. Empty symbols remain in the coverage report.

Raw canonical JSON responses use gzip; typed analytical bars use Zstandard
Parquet. Request receipts link to raw object hashes, and the final bundle links
to universe, calendar and bar checksums. All data stays in the dedicated US root;
the study requires no SQL migration or new database rows. This design avoids
downloading a full-market tick history for daily strategies. It does not delete
or compact any pre-existing data.

## Interpreting results

The generated `reports/<run-id>/report.md` includes period comparisons, costs,
drawdowns, turnover, exposure, blackhorse labels and coverage limitations.
`results.json` records inputs, protocol and implementation hashes. CSV files
contain equity, fills, completed trades, open holdings, coverage and large moves.
Same-code/same-input reruns produce the same immutable artifacts.

All signals use close-known features with at least one session of lag. Indicators
never forward-fill missing prices; orders never fill without observed prices.
Pending exits remain pending across missing sessions. Portfolio marks carry the
last known price, expose stale-position days and include a final stale-holdings
zero-value sensitivity. Entry budgets use prior-close NAV and cash, without
same-day sale proceeds. Base and stress costs are hypothetical all-in 15/30 bps
per side. Every evaluation period starts with fresh cash.

This is exploratory research, with `live_qualified=false`. Current-vintage
adjusted prices are return proxies, not a corporate-action cash ledger. Daily
opens are not certified regular-session executable prices. Historical symbol
continuity, symbol reuse, rename coverage and delisting proceeds need further
reconciliation. Unknown or immature future labels are reported separately from
success/failure. Repeated 60-session labels overlap and are not independent
samples. No parameter search or live strategy promotion occurs.

The explicit `--cash-actions` comparison requires verified action inputs; missing
inputs cause an error. Only cash mergers with an explicit payable date, identity,
and matching raw/all-adjusted price conversion can settle. Cash is released no
earlier than the latest payment, process and effective date, cannot fund same-day
purchases, and incurs no hypothetical trading fee. Duplicate nearby events at
the same CUSIP/rate are coalesced; conflicting rates are quarantined. This repairs
cash accounting without changing the original strategy protocol. The overlay
policy is separately recorded in `research_actions.POLICY` and the result hash.

The provider returned no cash mergers before 2020 in this acquisition, and many
later records lack payment dates. Stock mergers, renames and missing payments
remain unresolved. Never promote the adjusted replay to a fully reconciled or
point-in-time certified backtest. The original no-settlement run is preserved;
omitting `--cash-actions` explicitly runs that diagnostic convention.

For acceptance, run `make -C us_market test`, the independent US fixture pipeline
twice, and its API on port 8001. Run repository `make test` with Taiwan DB sessions
read-only. Do not run the root Taiwan `make pipeline` for this isolated study.

Primary references:

- [Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)
- [Massive dated ticker reference](https://massive.com/docs/rest/stocks/tickers/all-tickers)
- [Alpaca market calendar](https://docs.alpaca.markets/us/reference/legacycalendar)
- [Alpaca corporate actions](https://docs.alpaca.markets/us/reference/corporateactions-1)
