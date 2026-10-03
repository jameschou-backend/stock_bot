# Candle / volume account: data-only cumulative budget addendum

Frozen before any acquisition under this addendum on 2026-10-03. This supplements
`docs/prereg_candle_volume_account_20261003.md` (SHA-256
`052b231af61a09c619f15069cfdc18d2d4aa1356fe0a254547c64ddf8a553d68`).
The user authorized completing all five registered comparisons and the parent
agent approved this additive data preparation. No selection, exit, accounting,
quality tolerance, date, or parameter is changed.

## Observed preparation boundary

The original `full-v1/report.json` at
`.cache/red-volume-exit-20261003/full-v1/report.json` has SHA-256
`2b33236460533746c4fab4f2e65a9dd3df04efb2ab0bd5031b61a1c4d8e37a55`.
It retains all five registered attempts:

| Arm | Completed | Final NAV (NT$) | Net return | Maximum drawdown |
| --- | --- | ---: | ---: | ---: |
| poc_red | Yes | 6,620,605.21 | +562.060521% | -26.8876618773% |
| poc_dry | Yes | 1,467,170.22 | +46.717022% | -40.7152869647% |
| poc_red_dry | No | Unknown | Unknown | Unknown |
| poc_dry_weak | No | Unknown | Unknown | Unknown |
| poc_red_dry_weak | No | Unknown | Unknown | Unknown |

The first unresolved profiles are 2501 / 2024-04-11 (2 missing days),
6535 / 2024-01-25 (17), and 8027 / 2024-01-22 (20). All 39 missing-day statuses
are `request_budget_or_quota_paused`; none has an attempt or receipt. The sealed
profile-features SHA-256 is
`df79ce27c9896e6d936f4ec1dea5c3a9e0cdd83b8d0f70ec6e237787ba728e56`.
The shared persistent ledger contains 4,800 attempts. This is exhaustion of our
chosen cumulative research budget, not evidence of a FinMind 6,000/hour service
limit or rejection. The two known returns above are disclosed prior observations;
they do not select the new cap or change any trading parameter. The engineering
purpose is to complete all five comparisons, preserving incomplete cases as
unknown rather than interpreting unavailable data as poor investment returns.

## Fixed acquisition extension

- Raise only this experiment's cumulative `TaiwanStockPriceTick` attempt cap from
  4,800 to **8,000**, including all existing attempts; at most 3,200 additional
  attempts. No command-line cap override is allowed.
- Reuse exactly `.cache/volume-profile-account-20261003/profiles-v1`, including
  `attempts`, `receipts`, raw sources and the existing `.run.lock`. Do not reset,
  rename, delete, or replace prior attempts/receipts. Prior failure receipts and
  started/orphan reservations remain non-retryable.
- Continue the existing shared FinMind client, no more than 5,400 requests/hour
  (or the lower configured limit), four workers, and `max_retries=0`. Provider
  or quota errors remain unknown. Requests paused before reservation are eligible
  for their first request; this does not authorize resending a prior attempt.
- Fetch only on the original causal candidate path, using the unchanged 20
  pre-signal sessions and ordinary-tape checks. Do not skip a missing profile or
  choose another stock because it is easier to obtain.
- The existing execution preparation cumulative cap of 100 and official odd-lot
  cumulative cap of 50, prior endpoint proofs, proof expiry, origin locks, holds,
  query boundaries and all rejection rules remain unchanged. No new probe,
  renewed proof, alternate endpoint, or official-source exception is authorized.
- Default offline. Only the parent agent launches any online replay after anchor
  verification. Exhausting 8,000 still stops the affected arm with no full-period
  return; any further preparation requires another explicit addendum.

## Additive implementation and provenance

Preserve the sealed provider `skills/volume_profile_data.py` (SHA-256
`3d87d08ed7e807c434d785520c6af7307634a3bbd2cb29ffed14015432554349`)
and runner `scripts/research_candle_volume_account.py` (SHA-256
`43be2f40492c474475aa2ece1b0dcef4064fa1190e682edb20e87e42a0a00b89`).
Use a new explicit provider subclass. After constructing the original provider
with its original 4,800 cap and exact ledger path, assign only that new instance's
maximum to 8,000. Inherit acquisition, profile calculation and quality handling
without overriding them. No monkeypatch of the sealed runner/provider globals.

A thin CLI imports the sealed runner's `run` function and injects the subclass.
Only the original eight registered arms and existing fetch/overlay flags are
accepted; all financial parameters remain inside the sealed runner. Source bytes
for this addendum, subclass, CLI and tests are stored as immutable hash-addressed
snapshots and included in the published source closure. Bind the old provider,
runner, rule source, preregistration, full-v1 report and all five case hashes.
Retain the complete prior report, failure cases and trial entries without editing
them. Each replay and failure remains separately counted in the trial registry.

## Required replay and final comparison

1. Run `original`, `benchmark`, `poc_base` offline to a new `anchors-v2` directory.
   Require exact equality of account, summary, audit and profile queries against
   the sealed VP full-v2 anchor accounts. The anchor report must bind the current
   extension source/test/addendum hashes as well as the original financial source.
2. Only after all three anchors match, run exactly the three incomplete arms
   `poc_red_dry,poc_dry_weak,poc_red_dry_weak` to a new `full-v2` directory.
   Do not rerun the two already completed arms for this supplement.
3. Compare all five originally preregistered variants using the two completed
   full-v1 cases plus the three new cases, with full provenance and failure history.
   A complete prior case must never be replaced. Repeated/failed trial counts
   remain visible. No post-result threshold choice or live qualification follows.

The original HL2 approximation, historical ordinary-volume coverage limits,
corporate fractional-cash timing, historical membership/publication timing and
in-sample research limitations remain unchanged.
