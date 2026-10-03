# Red signal candles and early volume contraction: funded account experiment

Frozen before examining returns of these variants on 2026-10-03, following the
user's instruction to require a red volume-expansion entry candle, count three
black candles from entry, and consider exiting after post-entry volume shrinks.
This is researched history, not unseen validation or live qualification.

## Fixed account and source baseline

- Empty NT$1,000,000 account, 2024-01-02 through 2026-09-09, three active stock
  slots, profits reinvested and idle cash retained. Preserve residual odd shares,
  pending rights/receivables, integer shares, channel-specific HL2 execution,
  prior/opening resource locks, partial fills and all original costs.
- Signal T is known after its close and entry is T+1. Boundary signals from
  2023-12-29 remain eligible for 2024-01-02. Do not condition an entry on the
  entry day's future closing candle or volume.
- Preserve the original 12% loss / 63-session time exits and then three black
  candles with successively lower adjusted closes. The first actually filled
  entry day E counts as the first eligible candle. All three candles must be E
  or later; the earliest three-black execution is E+3. No pre-entry candle counts.
  Zero fills do not start the clock; a new event after reentry gets a fresh clock.
- Frozen source runner `scripts/research_volume_profile_account.py` SHA256:
  `576668c79215f97fb82e3ed2aaaaead44b8cb6f08a15b6f70eae60c6fa94ed23`.
- Baseline report `.cache/volume-profile-account-20261003/full-v2/report.json`
  SHA256 `4f6daf42bcf0dc7781ebccc9a7aa47f22c581d8f27e41768e91a640a0e9552ac`.
- Input manifest `.cache/market-input-repair-20261002/inputs-v2/manifest.json`
  SHA256 `f62dbe32d263abb464c9f283879b7d6340b0667b6fa9e99404c6ece7e04663d6`.
- Adapt the runner explicitly in a new source file. Never modify sealed sources,
  execute generated source text, or patch old runner globals to add new arms.
  Retain the existing explicitly scoped corporate validator integration.

## Entry hypothesis: red signal candle

Original candidates already require signal volume at least 1.5 times the
preceding 20-session mean, a 60-session close breakout, positive 20-session stock
and relative returns, historical eligibility and the existing liquidity gates.
Do not change those thresholds or the original ranking.

The new gate additionally requires raw close(T) > raw open(T), with valid positive
OHLC observations. Black candles and doji are rejected. Unknown or invalid OHLC
is a data failure, not a false candle. Preserve event identities and ordering.
Apply this gate before POC classification and before cash/slot reservations.
If the POC availability policy restores a day's original ordering, it restores
the already red-qualified list; it must not readmit a rejected black candle.

## Exit hypothesis: early volume contraction

Fix both the threshold and window before returns, without parameter search:

- Observe the first five held market sessions E..E+4, including the entry day's
  completed candle. No intraday estimate is substituted for a daily volume.
- Two consecutive completed market sessions must each have strictly less than
  50% of the original signal-day T volume. Equality is not contraction.
- Both observed sessions must be E or later. The earliest trigger close is E+1
  and execution is E+2. The latest eligible trigger close is E+4, execution E+5.
- `dry` adds no price condition. `dry_weak` additionally requires the latest
  adjusted close to be strictly below BOTH the preceding session's adjusted
  close and the original signal T adjusted close.
- The volume denominator is fixed at the original signal, not a later maximum
  or a retrospectively selected high-volume day. Daily volume is total-session
  shares, not certified ordinary-board volume or proof of large-holder selling.
- Missing/invalid volume, zero-volume/no-trade days and unavailable required
  prices cannot be treated as contraction, forward-filled, or skipped to join
  nonconsecutive observations. Record an explicit unavailable status; this added
  exit is inactive for that decision, while the original exits continue.
- In the absence of a separately verified share-unit conversion, any corporate
  event from signal T through decision close makes this volume comparison
  unavailable. Conservatively include cash dividends in that guard and disclose
  coverage. Never adjust volume using a price adjustment factor.
- Original loss/time/three-black exits take precedence. A volume exit is added
  only to a held cohort without a previously latched exit. Once triggered, keep
  its reason and signal date through subsequent partial fills or limit failures;
  restored volume does not cancel the exit.
- Execution remains the original next-session board/odd price, capacity and
  limit model. Do not sell at the signal close or spend sale proceeds that the
  original opening-resource policy disallows.

## Eight fixed arms

| Arm | Candidate policy | Red gate | New volume exit |
|---|---|---|---|
| original | original ordering | off | none |
| benchmark | original 0050 account | not applicable | none |
| poc_base | frozen poc_priority_available | off | none |
| poc_red | frozen poc_priority_available | on | none |
| poc_dry | frozen poc_priority_available | off | dry |
| poc_red_dry | frozen poc_priority_available | on | dry |
| poc_dry_weak | frozen poc_priority_available | off | dry_weak |
| poc_red_dry_weak | frozen poc_priority_available | on | dry_weak |

The POC base is the preceding research result, not an assumed validated edge.
The prior unknown-profile policy remains explicit and is not a new fallback.
First reproduce original, benchmark and poc_base against full-v2. Account,
summary, execution audit and profile query sequence must agree before variant
results may be interpreted. Red entry, volume-only exit and their combination
are separate comparisons; no best-result-based choice of threshold is allowed.

## Data, evidence, bounded completion and reporting

- Reuse the frozen inputs and exact query-bound local caches. Offline is default.
  Explicit fetch flags may continue the previously reviewed profile provider
  and missing-execution preparer, without resetting their persistent budgets:
  4,800 profile attempts total (1,650 already reserved at registration), 100
  execution-preparation attempts total; shared FinMind protection at 5,400/hour.
- An actual precommitted positive odd-share requirement may use the existing
  successful exact-endpoint proof for necessary missing odd dates within its
  original 24-hour authorization and 50 total attempts. No new probe, alternate
  endpoint, new authorization, retry, global-hold deletion or bypass is added.
  New stop evidence, expiry, data errors or exhausted budgets remain blockers.
- Missing execution prices, legal bounds, corporate terms or necessary POC raw
  sources stop the affected arm. Do not publish an incomplete account's partial
  return as the full-period result. The volume-indicator unavailability rule
  above is a declared strategy rule and must be reported separately from missing
  execution evidence.
- Retain each entry gate decision and volume decision, including dates, original
  signal, first fill, observed quantities, fixed denominator, age, price gates,
  unavailable reasons, existing-exit priority and first latch.
- Independently rebuild new entry/exit decisions from raw observations and actual
  cohorts, not by calling the production decision function. Keep existing cash,
  corporate, integer-share, resource, fill and three-black audits.
- Register every arm, failure and rerun; retain immutable result directories and
  source/code hashes. Report net return, yearly returns, drawdown, cost, trades,
  exit-reason counts and the effect of early exits on later observed prices.
  Later-price diagnostics are outcomes only and cannot affect the rules.
- `live_qualified`, `actual_fill_verified`, `unseen_validation`, complete historical
  universe and full point-in-time publication-archive certification remain false.
