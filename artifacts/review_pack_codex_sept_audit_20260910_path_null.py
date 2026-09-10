"""Path-luck null and per-event alpha check for the sealed 3-slot leader replay (2026-09-10 audit).

Reads only frozen research inputs under .cache/million-replay-signals (official adjusted closes,
raw close/volume, gated signal entries). Simplified execution (fractional shares, proportional
costs, no participation caps / dividends) reproduces the sealed control ledger to within ~60pp
(+649% vs sealed +711%), which is sufficient for the questions below:

  1. NULL1  same 458 candidates, random priority when several compete for a slot  -> path-luck band
  2. NULL2  same entry dates, random liquid stock (adv20 >= NT$50M) instead of the leader
  3. NULL3  random liquid stock on random dates
  4. per-event 63-session excess return vs 0050 with month-cluster bootstrap CI (path independent)
  5. effect of the 120-day 0050 trend gate (458 gated vs 530 ungated events)

Run:  python artifacts/review_pack_codex_sept_audit_20260910_path_null.py   (~25s, 0 API calls)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

SIG = '.cache/million-replay-signals'
rng = np.random.default_rng(20260910)
close = pd.read_parquet(f'{SIG}/close-official.parquet').set_index('date').sort_index()
rawc = pd.read_parquet(f'{SIG}/raw-close.parquet').set_index('date').sort_index()
rawv = pd.read_parquet(f'{SIG}/raw-volume.parquet').set_index('date').sort_index()
days = close.index
pos = {d: i for i, d in enumerate(days)}
P = close.to_numpy(float)
ids = list(close.columns)
col = {s: j for j, s in enumerate(ids)}
b = col['0050']
amt20 = (rawc * rawv).rolling(20, min_periods=20).mean().to_numpy(float)
sig = json.load(open(f'{SIG}/signals.json'))
ent = sig['entries']
allev = ent + [{k: v for k, v in r.items() if k != 'reason'} for r in sig['rejections'] if r['reason'] == 'trend_off']
START, END = pos[pd.Timestamp('2022-01-03')], pos[pd.Timestamp('2026-09-09')]
H, SLOTS = 63, 3
CB, CS, EB, ES = 0.005925, 0.008925, 0.005925, 0.006925  # stock buy/sell, 0050 buy/sell (fee+slippage+tax)


def simulate(events, stoploss=True, slots=SLOTS, use_cost=True, pick=None):
    byday = {}
    for e in events:
        d = pd.Timestamp(e['entry_date'])
        if d not in pos or pos[d] < START or pos[d] > END:
            continue
        s = pick(e) if pick else e['members'][0]
        if s is None or s not in col:
            continue
        byday.setdefault(pos[d], []).append((e['priority'], s))
    cb, cs, eb, es = (CB, CS, EB, ES) if use_cost else (0, 0, 0, 0)
    cash, held = 0., []
    etf = 1_000_000. * (1 - eb) / P[START, b]
    nav = np.full(END - START + 1, np.nan)
    for i in range(START, END + 1):
        keep = []
        for (j, q, ei, ep) in held:  # exits decided on prior close, executed at today's close
            px = P[i, j]
            stop = stoploss and i - 1 > ei and np.isfinite(P[i - 1, j]) and P[i - 1, j] < ep * 0.88
            if (i - ei >= H or stop) and np.isfinite(px):
                cash += q * px * (1 - cs)
            else:
                keep.append((j, q, ei, ep))
        held = keep
        if i in byday and len(held) < slots:
            navi = cash + etf * P[i, b] + sum(q * P[i, j] for j, q, _, _ in held if np.isfinite(P[i, j]))
            budget = navi / 3
            heldset = {j for j, _, _, _ in held}
            for _, s in sorted(byday[i], key=lambda t: -t[0]):
                if len(held) >= slots:
                    break
                j = col[s]
                if j in heldset or not np.isfinite(P[i, j]):
                    continue
                if cash < budget:  # sell 0050 to fund
                    sell_q = min(etf, (budget - cash) / (P[i, b] * (1 - es)))
                    cash += sell_q * P[i, b] * (1 - es)
                    etf -= sell_q
                spend = min(cash, budget)
                if spend <= 0:
                    continue
                held.append((j, spend * (1 - cb) / P[i, j], i, P[i, j]))
                heldset.add(j)
                cash -= spend
        if cash > 5000 and np.isfinite(P[i, b]):
            etf += cash * (1 - eb) / P[i, b]
            cash = 0.
        nav[i - START] = cash + etf * P[i, b] + sum(q * (P[i, j] if np.isfinite(P[i, j]) else P[ei, j]) for j, q, ei, _ in held)
    s = pd.Series(nav, index=days[START:END + 1])
    return s.iloc[-1] / 1e6 - 1, (s / s.cummax() - 1).min(), s


def yearly(s):
    out, prev = {}, s.iloc[0]
    for yr, row in s.groupby(s.index.year).agg(['last']).iterrows():
        out[yr] = f"{row['last'] / prev - 1:+.0%}"
        prev = row['last']
    return out


def random_liquid_pick(e):
    i = pos[pd.Timestamp(e['signal_date'])]
    ok = np.isfinite(amt20[i]) & (amt20[i] >= 5e7) & np.isfinite(P[i]) & np.isfinite(P[min(i + 1, len(P) - 1)])
    elig = np.flatnonzero(ok)
    return ids[rng.choice(elig[elig != b])]


def fwd(evs):
    rows = []
    for e in evs:
        i, j = pos[pd.Timestamp(e['entry_date'])], col[e['members'][0]]
        if i + H < len(P) and all(np.isfinite(x) for x in (P[i, j], P[i + H, j], P[i, b], P[i + H, b])):
            rows.append((e['entry_date'][:7], P[i + H, j] / P[i, j] - 1, P[i + H, b] / P[i, b] - 1))
    return pd.DataFrame(rows, columns=['ym', 'r', 'rb'])


def band(values, label):
    v = np.asarray(values)
    print(f'{label:58s} median {np.median(v):+7.1%}  p5 {np.percentile(v, 5):+7.1%}  p95 {np.percentile(v, 95):+7.1%}')


if __name__ == '__main__':
    N = 300
    print(f'0050 buy&hold (adj, gross) 2022-01-03..2026-09-09: {P[END, b] / P[START, b] - 1:+.1%}\n')
    for label, kw, evs in [('gated 458, loss12 (sealed control analogue)', {}, ent),
                           ('gated 458, time63 only', {'stoploss': False}, ent),
                           ('gated 458, loss12, no cost', {'use_cost': False}, ent),
                           ('ungated 530, loss12 (trend gate removed)', {}, allev)]:
        r, m, s = simulate(evs, **kw)
        print(f'{label:58s} ret {r:+7.1%}  mdd {m:6.1%}  yearly {yearly(s)}')
    print()
    band([simulate([dict(e, priority=float(rng.random())) for e in ent])[0] for _ in range(N)],
         'NULL1 gated loss12, random slot priority')
    band([simulate([dict(e, priority=float(rng.random())) for e in ent], stoploss=False)[0] for _ in range(N)],
         'NULL1 gated time63, random slot priority')
    band([simulate([dict(e, priority=float(rng.random())) for e in allev])[0] for _ in range(N)],
         'NULL1 ungated loss12, random slot priority')
    band([simulate(ent, pick=random_liquid_pick)[0] for _ in range(N)],
         'NULL2 same dates, random liquid stock instead of leader')
    rnd = []
    for _ in range(N):
        ev = []
        for e in ent:
            i = int(rng.integers(START, END - 1))
            ev.append(dict(e, signal_date=str(days[i].date()), entry_date=str(days[i + 1].date()), priority=float(rng.random())))
        rnd.append(simulate(ev, pick=random_liquid_pick)[0])
    band(rnd, 'NULL3 random liquid stock, random dates')
    print()
    for label, evs in [('gated 458', ent), ('rejected by gate (trend OFF)', [e for e in allev if e.get('trend_state') == 'OFF']), ('all 530', allev)]:
        f = fwd(evs)
        ex = f.r - f.rb
        groups = [g.values for _, g in ex.groupby(f.ym)]
        bs = np.array([np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(2000)])
        print(f'{label:30s} n={len(f):3d}  63-session excess vs 0050: mean {ex.mean():+.2%}  median {ex.median():+.2%}  '
              f'hit {(ex > 0).mean():.1%}  month-cluster bootstrap 95% CI [{np.percentile(bs, 2.5):+.2%}, {np.percentile(bs, 97.5):+.2%}]  '
              f'top-5% events share of positive excess {np.sort(ex)[-max(1, len(ex) // 20):].sum() / ex[ex > 0].sum():.0%}')
