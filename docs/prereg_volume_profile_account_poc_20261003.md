# POC candidate ordering and filter in the 2024 cash account

Fixed on 2026-10-03 before new POC account outcomes. This is an extension of
`prereg_volume_profile_account_baseline_20261003.md`, whose SHA256 is
`d89cdf12062245d1c195bc8d694b5023ecf2d6df02e9cee17d298d4c02485660`.
All cash, execution, corporate-action, candidate, exit, period and fee rules are
identical to that baseline. The experiment changes only candidate selection
before opening resources are reserved. Initial cash is NT$1,000,000, up to three
active stocks, idle cash, 2024-01-02 through 2026-09-09. This is research using
the explicit legacy-total-volume / HL2 proxy, not verified executable profit.

## Four fixed POC arms

1. `poc_priority`: stable partition of known `poc_up=True` candidates before
   known `False`; preserve descending original priority and event-id ordering
   within both groups. Unknown necessary classification stops the account.
2. `poc_filter`: admit only known `poc_up=True` in original order. Unknown
   necessary classification stops the account. Empty slots remain cash.
3. `poc_priority_available`: same priority rule when necessary profiles are
   known. A permanent quality failure from the exact allowlist below restores
   the entire day's original candidate list and restarts opening reservations
   from the untouched real account. Never retain a partly selected POC prefix.
4. `poc_filter_available`: same filter rule, with the same complete-day original
   fallback for the allowlisted permanent quality failures only.

The last two arms are availability diagnostics, separate from strict POC
results. They cannot support a claim that strict POC ranking/filtering works.
Every fallback day, reason, original list and discarded selection certificate
must be retained. No choice among variants is made using their returns.

## Causal profile

For signal close T, use the 20 audited market sessions strictly before T. Use
authentic regular-board transaction-price/volume rows only, preserve repeated
executions, normalize reported lots to shares, and retain source/query hashes.
The complete window and its first/last ten sessions use common 40 equal-width
price bins. Highest share volume determines POC; ties choose the lower bin.
`poc_up` is strictly higher second-half POC than first-half POC. A flat POC is
false. Preserve the pilot algorithm including positive-volume range, exact
single-price handling and half-window comparability. No daily OHLC allocation
can substitute for missing transaction-price volume. No current/future outcome
or execution-session tick is used in candidate selection.

Use repaired raw/adjusted daily paths to guard corporate discontinuities through
T and official dated identity and daily OHLC evidence. Where exact ordinary
amount/volume is cached, require exact reconciliation under the existing audit
tolerances. Other admissible tapes retain provider-diagnostic status and cannot
be described as fully ordinary-volume verified. Existing official-source holds
remain active and are never bypassed.

## Missing evidence and bounded collection

Only these exact permanent reasons, together with `recoverable is False`, may
trigger the availability arms' complete-day fallback:

- `corporate_action_or_nonconstant_price_scale`
- `pre_signal_daily_path_invalid`
- `official_identity_missing_or_ambiguous`
- `ordinary_tape_conflict`

Unfetched raw data, empty/missing responses, quota/budget pauses, provider or
transport errors, malformed classifications, and any other failure must stop
the account. Unknown is never converted to false and a later candidate cannot
replace an unresolved necessary candidate. Fallbacks are registered policy,
never silent error recovery.

Profiles are obtained only on demand when they can affect opening reservations.
Check the pure planner first: opening occupied slots, opening cash, prior
liquidity and price, residual exposure and whole attempted-budget lock. With no
possible new reservation, do not query profiles. Priority scans candidates in
original order, reserves true candidates, and can stop once opening resources
are exhausted. False candidates can be admitted only after every remaining
potentially reservable candidate is resolved. Keep a certificate proving why
unqueried suffix candidates cannot change orders. Full-list selection and this
lazy selection must have the same reserved orders for known inputs.

Collect only query-bound missing `TaiwanStockPriceTick` stock-days through the
shared FinMind client, at most four workers, shared protection at 5400 requests
per hour, no retries. The persistent account-study budget is at most 4800
attempts including failed requests and prior smoke/preparation calls. Reuse
bound pilot/cache coordinates; never auto-adopt unbound shared gzip files.
Offline is default. `--profile-fetch` may enable this provider only; all legacy
account inputs remain local-cache-only. Missing legal limits, dividends, odd
data or corporate terms remain explicit blockers and require separate repair.

## Reporting and audit

Register all four attempted arms and every rerun/failure. Compare completed
accounts to the fresh 2024 original account and 0050; include net account return,
yearly NAV changes, maximum drawdown, costs, entries/exits, cash utilisation,
ending holdings and receivables. Preserve full daily/journal/resource evidence,
per-arm queried profiles, hash-bound source snapshots and decision certificates.
Strict incomplete accounts have no complete-period return. Report availability
fallback coverage and exact-ordinary profile coverage beside diagnostics.
No parameter search, automatic strategy promotion or current-stock advice.
`live_qualified`, `actual_fill_verified`, and `unseen_validation` remain false.
