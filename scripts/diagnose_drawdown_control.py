#!/usr/bin/env python3
"""Explain the sealed account's drawdown; no counterfactual fills are inferred."""
from pathlib import Path
from collections import defaultdict
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from skills.liquidity_diagnostics import market_breadth
from skills.account_cohort_attribution import cohort_outcomes


def pending_cash(account, day):
    paid = {(r['action_id'], r['event_id']) for r in account['corporate_actions']
            if r['kind'] == 'payment' and r['date'] <= day}
    delivered = {(r['action_id'], r['event_id']) for r in account['corporate_actions']
                 if r['kind'] == 'share_delivery' and r['date'] <= day}
    result = defaultdict(float)
    for r in account['corporate_actions']:
        if r['date'] > day:
            continue
        identity = r['action_id'], r['event_id']
        if r['kind'] == 'stock_dividend' and identity not in delivered:
            raise ValueError('Boundary contains noncash rights requiring explicit valuation')
        if r['kind'] == 'cash_dividend' and identity not in paid:
            result[r['stock_id']] += r['entitlement_value']
    return dict(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT)
    if args.output.exists():
        raise ValueError('Preserve prior diagnostic')
    base = ROOT/'.cache/liquidity-universe-20261001/final-a'
    parent = read(base/'report.json')
    cases = {}
    for arm in ('median50m', 'benchmark'):
        p = base/(arm+'.json')
        if sha(p) != parent['cases'][arm]['sha256']:
            raise ValueError('Sealed case changed')
        cases[arm] = read(p)
    a = cases['median50m']['account']
    daily = pd.DataFrame(a['daily']).set_index('date')
    benchmark = pd.DataFrame(cases['benchmark']['account']['daily']).set_index('date')
    trough = daily.drawdown.idxmin()
    peak = daily.loc[:trough].nav.idxmax()
    h = pd.DataFrame(a['holdings'])
    first, last = pending_cash(a, peak), pending_cash(a, trough)
    for day, rights in ((peak, first), (trough, last)):
        if abs(sum(rights.values())-daily.at[day, 'receivable']) > .011:
            raise ValueError('Boundary rights do not reconcile')
    names = dict(zip(h.stock_id, h.name))
    ids = set(h.loc[h.date.between(peak, trough), 'stock_id']) | set(first) | set(last)
    rows = []
    for sid in sorted(ids):
        initial = float(h.loc[h.date.eq(peak) & h.stock_id.eq(sid), 'market_value'].sum())
        final = float(h.loc[h.date.eq(trough) & h.stock_id.eq(sid), 'market_value'].sum())
        cash = sum(r['cash_change'] for r in a['cash_ledger']
                   if r.get('stock_id') == sid and peak < r['date'] <= trough)
        rights = last.get(sid, 0)-first.get(sid, 0)
        rows.append(dict(stock_id=sid, name=names[sid], pnl=final-initial+cash+rights,
                         initial_value=initial, final_value=final, cash_change=cash,
                         receivable_change=rights, peak_weight=initial/daily.at[peak, 'nav']))
    loss = daily.at[trough, 'nav']-daily.at[peak, 'nav']
    if abs(sum(r['pnl'] for r in rows)-loss) > .011:
        raise ValueError('Stock contributions do not reconcile to NAV change')
    inputs = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
    paths = [inputs/n for n in ('close-official.parquet', 'quotes-unmasked.parquet', 'eligibility.parquet')]
    for p in paths:
        if sha(p) != parent['source_sha256'][str(p.relative_to(ROOT))]:
            raise ValueError('Market diagnostic source changed')
    close = pd.read_parquet(paths[0]).set_index('date')
    close.index = pd.to_datetime(close.index)
    q = pd.read_parquet(paths[1])
    raw, volume = [q.pivot(index='date', columns='stock_id', values=k)
                   .reindex(index=close.index, columns=close.columns) for k in ('close', 'volume')]
    eligibility = pd.read_parquet(paths[2]).set_index('date').reindex(index=close.index, columns=close.columns)
    breadth = market_breadth(close, raw, volume, eligibility)
    ma120 = close['0050'].rolling(120).mean()
    window = daily.loc[peak:trough]
    entry_cohorts = [c for c in a['cohorts'] if peak < c['entry_date'] <= trough]
    outcomes = cohort_outcomes(cases['median50m'])
    entries = [{**outcomes[c['event_id']], 'later_outcome_is_retrospective': True}
               for c in entry_cohorts]
    focus = []
    for c in a['cohorts']:
        if c['stock_id'] not in ('2630', '6625') or not c['entry_date'] <= peak < (c['exit_date'] or '9999'):
            continue
        trades = [t for t in a['trades'] if t['event_id']==c['event_id'] and t['side']=='sell']
        focus.append(dict(**outcomes[c['event_id']], sales=[{k:t[k] for k in
            ('date', 'qty', 'reference_price', 'reason')} for t in trades]))
    result = dict(peak_date=peak, trough_date=trough, peak_nav=daily.at[peak, 'nav'],
        trough_nav=daily.at[trough, 'nav'], loss=loss, max_drawdown=loss/daily.at[peak, 'nav'],
        benchmark_same_window=benchmark.at[trough,'nav']/benchmark.at[peak,'nav']-1,
        contributions=sorted(rows,key=lambda r:r['pnl']), later_entries=entries, peak_winners=focus,
        average_physical_exposure=float((window.market_value/window.nav).mean()),
        benchmark_above_ma120_sessions=int(close.loc[window.index,'0050'].gt(ma120.loc[window.index]).sum()),
        window_sessions=len(window),
        breadth=[dict(date=d, **breadth.loc[d].to_dict()) for d in (peak, trough)],
        breadth_scope='reconstructed eligible cohort with positive actual volume and complete MA20',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [base/'report.json',
            base/'median50m.json',base/'benchmark.json',Path(__file__),ROOT/'skills/liquidity_diagnostics.py',
            ROOT/'skills/account_cohort_attribution.py',*paths]},
        descriptive_only=True, counterfactual_backtest=False, live_qualified=False)
    write(args.output,result)
    print({k:result[k] for k in ('peak_date','trough_date','loss','benchmark_same_window',
                               'average_physical_exposure','benchmark_above_ma120_sessions','window_sessions')})
    print('contributions',result['contributions'])


if __name__ == '__main__':
    main()
