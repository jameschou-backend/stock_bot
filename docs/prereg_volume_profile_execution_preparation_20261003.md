# Bounded execution data preparation for the 2024 account

The first offline baseline stopped because 1231 dividend input was absent and
0050 legal price bounds conflicted with daily prices. Preserve both failed
trials and their prefixes. Before subsequent account runs, the user authorizes
missing execution-source preparation with unchanged trading rules.

Only exact existing `TaiwanStockDividend` or `TaiwanStockPriceLimit` queries for
one four-digit stock, 2018-01-01 through 2026-09-09, are permitted. At most 100
persistent attempts, shared FinMind gateway at no more than 5400/hour, no retry.
Record each attempt before I/O; preserve raw response, query identity, hashes,
and redacted error class. Do not overwrite existing caches. Dividend execution
copies must exactly match their query-bound source.

No official exchange endpoint requests, bypass of security holds, source-based
candidate selection, guessed corporate settlement, or price-limit clamping.
Existing source conflicts require independent evidence; downloading another
unrelated source does not silently resolve them. A missing corporate term or
other unsupported source stops the arm and retains its failure journal. All
replays including failures remain trials. Preparation does not make the
legacy-total-volume or HL2 execution proxy verified ordinary-board execution.
