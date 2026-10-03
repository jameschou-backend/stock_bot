# 2024 Volume Profile full-account baseline preregistration

Frozen before the new account results on 2026-10-03. This is a new historical
research trial, not unseen validation or a change to the live strategy.

## Fixed comparison

- Start with an empty NT$1,000,000 account on 2024-01-02; finish on 2026-09-09.
- Run exactly two initial arms: `original` (current median50m universe, original
  priority ordering, corrected three-black exit) and `benchmark` (0050 with
  distributions reinvested using the same account engine).
- Keep at most three active stock positions; unused capital stays cash. Preserve
  the reviewed release of exited sub-lot residuals, its 5% residual-exposure
  block, and all receivables. Initial positions and receivables are empty.
- Membership is by execution date: 2023-12-29 signals scheduled for 2024-01-02
  remain eligible. Every stock entry follows its signal by one market session.
- Original ordering is descending frozen `priority`, then ascending `event_id`.
  No POC feature, outcome, eligibility preselection, or new ranking is used in
  these two arms. POC variants require their own frozen specification.
- Preserve 12% loss / 63-session time exit priority, then three completed bearish
  candles whose adjusted closes each decline. All three candles must occur from
  entry onward; exit executes on the following session. No intraday stop is added.

## Frozen implementation and inputs

- Adapt `scripts/replay_repaired_market_inputs.py` explicitly in a new standalone
  runner; do not execute arbitrary source text, change sealed modules, or select
  another engine on failure. Source SHA256:
  `199e171d5433760039ffdae1152cabe633b427b98c60e88d3c1d72fd0735ce26`.
- Input directory: `.cache/market-input-repair-20261002/inputs-v2`.
- Input manifest SHA256:
  `f62dbe32d263abb464c9f283879b7d6340b0667b6fa9e99404c6ece7e04663d6`.
- Signals SHA256:
  `f0821897f077d616494bcaafe2b2278df27666c03466758642c80784f7542513`.
- Corrected three-black module SHA256:
  `b03d2209c3d35c6fefac285d500888efa555a0222a913f01c58687f9dee8afa3`.
- Preserve historical security identity, ordinary/odd execution routing, legal
  price limits, corporate-action terms, pending share delivery, and the reviewed
  face-value/capital-reduction adapters. Reuse only query/hash-bound local caches.

## Execution and costs

- Explicit `legacy_total_research` volume policy: board capacity uses 1% of the
  smaller of daily total volume and preceding 20-session average total volume.
  This is a research proxy, not complete verified ordinary-board volume.
- Board and odd orders are separate integer-share orders. Odd capacity uses its
  separate official daily odd-volume evidence. Both retain channel daily HL2
  reference prices, legal price-limit eligibility, and existing cash constraints.
- Preserve 0.1425% commission with NT$20 minimum for each child order, rounded
  to an integer NT dollar; sell tax is 0.3% for stocks / 0.1% for 0050, floored;
  slippage is 0.45% each side, rounded upward. No fee discount is introduced.
- Lock opening cash, opening occupied slots, and the entire attempted order
  budget. Same-day sale proceeds, dividends, and unfilled budget cannot fund a
  later candidate. An unfilled attempt does not release its slot intraday.

## Evidence and failure policy

- Offline only: no FinMind, official exchange, HTTP, or DB requests. Existing
  official-source holds are retained. Missing data or corporate terms stop the
  affected arm; never fill an unknown with zero or treat it as rejection.
- Preserve each arm's source hashes, settings, daily account, integer trades,
  orders, cash ledger, corporate actions, resource/slot decisions, and independent
  accounting/execution audits. Preserve any failed prefix without calling its
  partial return a completed-period result.
- Register every attempted arm, including failed arms; do not delete or overwrite
  prior run directories. The initial target is
  `.cache/volume-profile-account-20261003/base-v1`.
- Report elapsed time, candidate counts, opening free-slot decision contexts,
  net return, maximum drawdown, yearly NAV changes, costs, and ending assets.
  These decision contexts estimate future data needs, not which outcomes to keep.
- Always retain `live_qualified=false`, `actual_fill_verified=false`,
  `complete_historical_universe=false`, and `unseen_validation=false`. No promotion
  is allowed from this baseline. Do not adjust parameters after seeing results.
