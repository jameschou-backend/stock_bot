# Endpoint-scoped odd-lot completion for the 2024 account

The ongoing user-authorized Volume Profile cash-account research encountered an
uncached TWSE intraday odd-lot table on 2025-02-13. The parent agent explicitly
reviewed one ordinary public request on 2026-10-03 using the repository's
endpoint-scoped recovery mechanism. This does not remove the global origin hold.

The only allowed endpoints are TWSE
`https://www.twse.com.tw/rwd/zh/afterTrading/TWTC7U` with
`date=YYYYMMDD&response=json`, and TPEx
`https://www.tpex.org.tw/www/zh-tw/afterTrading/oddQuote` with
`date=YYYY/MM/DD&response=json`. One normal probe is permitted for each market:
TWSE 2025-02-13, required by original-account stock 6558's 339-share residual
sale; TPEx 2025-06-23, required by the availability accounts stopped after
2025-06-20. The parent reviewed the TPEx probe before any helper request.
No alternate
host/path, proxy, cookies, custom User-Agent, challenge handling or redirect.
TLS verification remains enabled. Automatic retries are zero.

A successful HTTP 200 payload must pass the existing `parse_odd` query/date/title,
named-column, row-count and numeric validation and contain the required stock.
Only that exact endpoint may then supply other necessary missing dates for 24
hours, with at most 50 total attempts across both markets including probes and
failures. Each
additional date must be bound to a retained failed account's odd-lot requirement
and a precommitted positive odd-share order. Budget is persistent across runs.

Use the existing shared `.cache/official-daily-origin-dispatch` origin lock and
dispatch state, at least 3.1 seconds between request starts, and a separate study
lock. Record attempt before I/O and retain raw bytes, query, request/response
timestamps, HTTP status, hashes, recovery proof and receipt. Resume accepted
receipts without network; an existing unsuccessful or interrupted attempt is
never retried automatically. A newer shared origin/security stop invalidates
the previous proof. HTTP 401/403/428/429, redirection, or challenge/security
content stops this acquisition immediately and persists shared stop evidence.

Preserve the original global hold and all its evidence bytes. Publish only a
validated response wrapper into the existing execution cache under
`odd-{twse,tpex}-YYYY-MM-DD.json`, without replacing existing files. The wrapper links
the immutable raw response and receipt. No total-volume/ordinary-tick/TPEx or
after-hours substitution, synthetic row, zero imputation, or price clamping.
The full-v1 source snapshots and strategy rules remain unchanged. A new explicit
`--odd-fetch` runner option may request a missing date only after execution reaches
an existing positive odd-share plan. It writes that current requirement before
I/O, uses successful endpoint proof, and stops on missing/failed data. It cannot
automatically consume either initial probe. Collection is not a strategy trial;
every subsequent cash-account replay remains a trial.
