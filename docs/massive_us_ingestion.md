# Massive US ingestion and Taiwan baseline protection

This first phase adds an independent **data landing** path. It does not implement
a US trading strategy, retrain Taiwan models, change Taiwan configuration, enable
a schedule, or place orders. Provider data quality and strategy validity are
separate from transport/fixture acceptance.

## Isolation

- Existing Taiwan commands, configuration, `.env`, migrations, databases and
  strategy/backtest modules are unchanged.
- `us_market/` does not import `app`, `skills` or `pipelines`. US ticker validation
  is separate from the existing four-digit Taiwan validator.
- Repository-local US writes are allowed only under `.cache/us-market-v1/`.
  A new external directory is also allowed; an existing unmarked directory, a
  Taiwan directory, or a symlink is rejected. Runtime files are already covered
  by the existing `.cache/` ignore rule.
- The optional catalog is a dedicated Compose project `stock-bot-us-v1`, MySQL
  8.4, `127.0.0.1:3308/stock_bot_us`, account `us_market`, separate volume. It is
  not a new schema on the Taiwan instance. Only manifest metadata is put in SQL.
- The US HTTP server binds localhost and defaults to 8001; port 8000 is refused.
  `/picks` and `/models` explicitly report unimplemented US research stages.

## Taiwan protection

```sh
python scripts/protect_tw_baseline.py capture \
  --baseline .cache/tw-baseline-protection/<new-baseline-name>
python scripts/protect_tw_baseline.py verify \
  --baseline .cache/tw-baseline-protection/<baseline-name>
```

The JSONL inventory records hashes of existing source, configuration, tests,
data and research outputs. The receipt lists exclusions: the new US/guard paths,
bytecode, locks, temporary files and the guard's own receipts. New files are
allowed; replacing or deleting any recorded file fails verification. The `.env`
contents are never copied into a receipt. Do not commit the local inventories.

For all existing Taiwan DB rows, use a separate consistent **READ ONLY**
transaction. Only row counts, schema hashes and ordered row hashes are saved:

```sh
python scripts/fingerprint_tw_database.py \
  --output .cache/tw-baseline-protection/database-before.json
python scripts/fingerprint_tw_database.py \
  --compare .cache/tw-baseline-protection/database-before.json \
  --output .cache/tw-baseline-protection/database-after.json
```

These are integrity receipts, **not backups or filesystem write locks**. A
concurrent authorized Taiwan daily update legitimately changes a fingerprint;
investigate differences rather than overwriting the baseline or reverting data.
Frozen file equality establishes preservation, not a new replay/performance
validation. No old data is deleted by any new command.

## Offline acceptance (no API key)

Run from the repository root:

```sh
make -C us_market test
make -C us_market pipeline
make -C us_market pipeline
make -C us_market api
```

The US `pipeline` target is explicitly a **synthetic fixture** pipeline. It
exercises all five datasets and receipt publication. A second run performs zero
fetches and preserves the same manifests. It is not a live provider acceptance.
The root Taiwan `make pipeline` must not be run against the existing data for
this task, because it updates data/features/models and violates the requested
preservation scope.

```sh
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/picks
curl -fsS http://127.0.0.1:8001/models
curl -fsS 'http://127.0.0.1:8001/jobs?limit=10'
```

The repository's full `make test` is also required before pushing. Its existing
UI suite includes a DB read and local HTTP server tests. For a preservation run,
use a local-socket-enabled environment and enforce MySQL session read-only:

```sh
DB_NAME='stock_bot?init_command=SET%20SESSION%20TRANSACTION%20READ%20ONLY' \
  FINMIND_TOKEN='' OPENAI_API_KEY='' AI_ASSIST_ENABLED=false \
  PYTHONDONTWRITEBYTECODE=1 make test
```

## Dedicated MySQL metadata catalog

The image must already be installed; setup deliberately uses `--pull never`.
Dependencies are the existing project requirements plus Docker Compose.

```sh
python -m us_market.setup
```

Setup creates random credentials in `~/.config/stock_bot/us-market-v1/`, with
directory mode 700 and file mode 600. It refuses to overwrite credentials or
reuse an existing orphaned US volume. Neither passwords nor key values belong
in the repository, `.env`, terminal arguments or chat.

To restart an existing US installation:

```sh
US_MYSQL_PASSWORD_FILE="$HOME/.config/stock_bot/us-market-v1/user-password" \
US_MYSQL_ROOT_PASSWORD_FILE="$HOME/.config/stock_bot/us-market-v1/root-password" \
docker compose -f us_market/compose.yaml up -d --pull never
```

To stop it without deleting its volume, use the same environment variables with
`docker compose -f us_market/compose.yaml stop`. Never use `down -v` as cleanup.

The file manifest is authoritative. SQL mirroring is explicitly enabled with
`--mysql-config`; no implicit SQLite/MySQL fallback or automatic migration occurs.
If mirroring fails after a manifest is published, rerun offline with the same
query/revision and catalog configuration to finish the idempotent mirror.

```sh
python -m us_market.cli fixture \
  --root .cache/us-market-v1/acceptance \
  --mysql-config "$HOME/.config/stock_bot/us-market-v1/db.json"
```

## Real provider sample (requires user-supplied key)

Use a process `MASSIVE_API_KEY`, or a private mode-600 key file outside the
repository, recommended `~/.config/stock_bot/massive_api_key`. The US module never
reads `.env`. Without `--fetch`, an incomplete snapshot fails without network
access. Network access is GET-only to `https://api.massive.com`, bearer-header
authentication, no redirects, retries or fallback provider.

```sh
python -m us_market.cli ingest --dataset daily --ticker AAPL \
  --start 2026-10-01 --end 2026-10-30 --revision provider-smoke-v1 \
  --fetch --api-key-file "$HOME/.config/stock_bot/massive_api_key" \
  --mysql-config "$HOME/.config/stock_bot/us-market-v1/db.json"
```

Datasets: `daily`, `trades`, `tickers`, `splits`, `dividends`. Trades and dated
ticker lookups require a single New York date; other ranges are bounded to 367
calendar days. This stage supports explicit symbols, **not a certified complete
historical universe**. Use a new revision to intentionally refresh a snapshot;
old responses and research references stay intact.

Per invocation defaults: at most 20 network requests, 100 pages, 32 MiB per HTTP
response, 10 GiB in the US file directory and 40 GiB host free-space reserve.
The directory quota includes partial/orphaned files but excludes the separate
MySQL Docker volume; SQL contains only small manifest records. Page receipts are
committed only after source and Parquet objects exist. Incomplete pagination is
never published as complete. Resume
checks every cached object's hash. Provider pagination is not guaranteed to be
a single historical revision; this limitation remains in every manifest.

## Data contracts and storage

- Unadjusted daily bars (`adjusted=false`) and raw trades are preserved separately
  from corporate actions. No dividend adjustment or split normalization is
  silently inferred. Numeric precision is checked before Decimal128 storage.
- UTC nanosecond trade timestamps are retained; the query date is interpreted in
  `America/New_York`, including DST. The initial landing preserves the full
  provider day; it is **not regular-session/holiday/early-close certified**.
- Keep trade conditions, correction flags, provider IDs and sequence numbers.
  Distinct IDs at equal prices/times remain distinct. Duplicate provider
  identities fail the snapshot instead of silently dropping rows.
- Source responses are canonical JSON gzip (exact decimal values as strings),
  normalized analytical columns are Zstandard Parquet. Source response headers,
  credentials and query-string API keys are not retained.
- Store query/date/revision receipts once and reuse them. No whole-market tick
  download, repeated experiment input copy, pruning or legacy recompression.
- All manifests retain `live_qualified=false`, `strategy_ready=false`, and
  unverified calendar/corporate-action/universe status. Empty successful pages
  are not proof of a zero-trade market session.

Before US POC/backtesting: verify an exchange calendar, session/condition rules,
correction handling, dated listing identity, splits/dividends and source volume
reconciliation. Then implement a separate US strategy and account model. The
existing Taiwan POC's Taipei timestamps and fee/market rules remain unchanged.

## Recorded acceptance

See [2026-11-02 acceptance](massive_us_acceptance_20261102.json) for the file/DB
integrity receipts, full test result and isolated synthetic pipeline/API checks.
Real Massive authentication and subscription entitlement checks remain pending
the user-provided API key. The fixture checks do not certify real provider data.

## Official references (checked 2026-11-02)

- [Authentication](https://massive.com/docs/rest/quickstart)
- [Daily bars](https://massive.com/docs/rest/stocks/aggregates/custom-bars)
- [Trades and pagination](https://massive.com/docs/rest/stocks/trades-quotes/trades)
- [Dated ticker lookup](https://massive.com/docs/rest/stocks/tickers/all-tickers)
- [Splits](https://massive.com/docs/rest/stocks/corporate-actions/splits)
- [Dividends](https://massive.com/docs/rest/stocks/corporate-actions/dividends)
- [Adjustment policy](https://massive.com/knowledge-base/article/is-massives-stock-data-adjusted-for-splits-or-dividends)
