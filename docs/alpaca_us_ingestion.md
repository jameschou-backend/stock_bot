# Alpaca historical SIP ingestion

The independent US landing pipeline supports Alpaca daily bars and trades. It
does not calculate POC, produce signals, train models or place orders. Taiwan
configuration, data, strategies and backtests remain on their original paths.
Existing Massive commands and snapshot identities are preserved.

## Credentials and cost

Register/sign in at [Alpaca](https://app.alpaca.markets/), select Paper Trading,
and generate an API key pair. Put the values into the existing, ignored `.env.us`:

```dotenv
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

Keep `MASSIVE_API_KEY` as it is. Do not commit secrets, modify the Taiwan `.env`,
paste credentials into chat, or pass them on the command line. `.env.us` must
have mode 600 (`chmod 600 .env.us`). The loader reads only the two Alpaca names
without variable interpolation or changing process environment. If either name
is provided in the process environment, both must be supplied there; it never
combines credentials from different sources.

Official documentation checked 2026-11-03 lists Basic as free for Paper and Live
accounts, history since 2016 and 200 historical calls/minute. Historical SIP
queries are available without a subscription when `end` is at least 15 minutes
old. Free real-time data is IEX-only, which is a different scope. Actual account
entitlement and returned data still require a live test.

- [Plans and authentication](https://docs.alpaca.markets/us/docs/about-market-data-api)
- [Historical SIP access and bar condition rules](https://docs.alpaca.markets/us/docs/market-data-faq)
- [Historical trades](https://docs.alpaca.markets/us/reference/stocktrades-1)
- [Historical bars](https://docs.alpaca.markets/us/reference/stockbars)

## Offline acceptance

```sh
make -C us_market test
make -C us_market pipeline PROVIDER=alpaca
make -C us_market pipeline PROVIDER=alpaca
make -C us_market api PROVIDER=alpaca
```

Fixtures use `.cache/us-market-v1/alpaca-acceptance`. They are explicitly marked
`synthetic_fixture`; the second pipeline run makes zero fetches. API port 8001
exposes `/health`, `/picks`, `/models` and `/jobs?limit=10`. Jobs include provider
and feed; picks/models report unimplemented stages. Original root Taiwan
`make pipeline` is deliberately excluded from this preservation task because
it writes existing data/models. Run the root full test suite using the read-only
DB configuration documented in [Massive acceptance](massive_us_ingestion.md).

## Bounded provider test

After setting credentials, this fetches one symbol/date, with a shared maximum
of 20 HTTP requests across daily bars and trades. It checks offline replay and
writes an immutable diagnostic receipt:

```sh
python -m us_market.alpaca_smoke --ticker KO --date 2026-10-29 --fetch
```

No `--fetch` means offline-only. If a symbol needs more than the request budget,
verified pages remain resumable but no complete trade snapshot/diagnostic is
published. Another explicit invocation resumes it with a new bounded budget;
it does not retry automatically. A 401/403/429 stops with a sanitized error,
without switching to IEX or Massive.

Individual datasets can be acquired explicitly:

```sh
python -m us_market.cli ingest --provider alpaca --dataset trades \
  --ticker KO --start 2026-10-29 --end 2026-10-29 --fetch --max-requests 20
```

The default real-data root is `.cache/us-market-v1/alpaca`. For the optional
existing US metadata catalog, add
`--mysql-config ~/.config/stock_bot/us-market-v1/db.json`. This accepts only
`127.0.0.1:3308/stock_bot_us` with the dedicated `us_market` account. No schema
change is required. MySQL stores manifest metadata, not trade rows. The root
directory remains needed to resolve the manifest's relative path.

## Data contract and research limits

- Only GET requests to `data.alpaca.markets/v2/stocks/{trades,bars}` are allowed.
  No brokerage endpoints are contacted. Credentials stay in authentication
  headers, with redirects/proxies disabled and sanitized errors/payloads.
- Always request `feed=sip`, `asof=-`, ascending pages and unadjusted daily bars.
  `asof=-` disables automatic symbol mapping; renamed/delisted symbols still
  need a separately verified historical universe.
- Trades cover one full New York calendar date, including extended hours when
  present. Timezone/DST conversion and the inclusive end use exact nanoseconds.
  This is not a regular-session or exchange-calendar certificate. Full-day
  acquisition waits until at least 15 minutes after the NY date has ended.
- Retain native alphabetic exchanges/conditions, tape, trade ID, timestamps,
  exact fractional shares and prices. Unknown conditions remain null. Do not
  interpret these as Massive numeric enums. Missing identity or duplicate
  identities fail rather than silently dropping prints.
- Complete pagination means every continuation token was exhausted. It does
  not prove market completeness, stable point-in-time revisions, corporate
  action consistency, correction/cancellation reconciliation or POC readiness.
- The diagnostic compares the raw sum of trade sizes with daily bar volume.
  Bar volume eligibility depends on sale conditions. Equality alone proves
  neither completeness nor a valid volume profile; differences need review.
  `poc_ready`, `strategy_ready` and `live_qualified` stay false.

## Storage bounds

Each provider query has separate identity and immutable page receipts. Raw
canonical JSON is gzip-compressed; analytical columns use Parquet/Zstd. Reads
verify checksums, retries reuse saved pages, and existing objects are never
overwritten or deleted. Each Store root defaults to a 10 GiB cap and a 40 GiB
host free-space reserve; these are per-root limits, not a combined cap across
providers or the Docker metadata volume. HTTP responses are capped at 32 MiB.
The smoke test limits calls to 20, uses 10,000 trades/page and spaces requests
at least 0.35 seconds apart. Start with selected symbols/dates; no full-market
historical backfill or scheduled ingestion is enabled.
