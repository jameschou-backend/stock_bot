#!/usr/bin/env python3
"""Read-only reconstruction of the original 6488 cohort; no alternative fills."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from skills.scenario_exit_replay import ExitSignals
from skills.exit_policy import decide_exit


def review():
    source = ROOT/'.cache/midpoint-since-2025-20260928/final-a/original.json'
    published = ROOT/'artifacts/forward_simulation/peak_stop15_20260929.json'
    control = ROOT/'.cache/peak-stop15-20260929/final-a/control.json'
    adjusted = ROOT/'.cache/historical-selector-replay-20260925/final-v7/combined/close-official.parquet'
    hashes = json.loads(published.read_text())['source_sha256']
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(control) != hashes[str(control.relative_to(ROOT))]:
        raise ValueError('Published control hash changed')
    original = json.loads(source.read_text())
    account = original['account']
    if account != json.loads(control.read_text())['account']:
        raise ValueError('Original and audited control differ')
    manifest = json.loads((control.parent/'report.json').read_text())
    if digest(adjusted) != manifest['source_sha256'][str(adjusted.relative_to(ROOT))]:
        raise ValueError('Adjusted signal source changed')
    sid = '6488'
    cohorts = [c for c in account['cohorts'] if c['stock_id'] == sid]
    if len(cohorts) != 1: raise ValueError('Expected one original cohort')
    cohort = cohorts[0]; eid = cohort['event_id']
    close = pd.read_parquet(adjusted).set_index('date'); close.index = pd.to_datetime(close.index)
    signals = ExitSignals(close, close.index)
    entry = close.index.get_loc(pd.Timestamp(cohort['entry_date']))
    end = close.index.get_loc(pd.Timestamp(cohort['exit_date']))
    state = dict(entry_index=entry, entry_price=signals.price(entry,sid), peak_price=signals.price(entry,sid))
    rows = []
    for i in range(entry+1,end+1):
        context = signals.context(i,sid,state)
        rows.append(dict(signal_date=str(close.index[i-1].date()), target_date=str(close.index[i].date()),
            adjusted_close=signals.price(i-1,sid), peak=state['peak_price'],
            **context, **decide_exit(context,'loss12')))
    first = next(r for r in rows if r['exit'])
    archived = original['exit_evidence'][eid]
    for field in ('signal_date','target_date','reason','held_sessions','entry_return','peak_drawdown'):
        if first[field] != archived[field]: raise ValueError('Exit evidence mismatch: '+field)
    trades = [t for t in account['trades'] if t['event_id']==eid]
    orders = [t for t in account['orders'] if t['event_id']==eid]
    actions = [t for t in account['corporate_actions'] if t['event_id']==eid]
    holds = [h for h in account['holdings'] if h['event_id']==eid]
    peak = max(holds,key=lambda h:h['market_value'])
    buy_cost = -sum(t['cash_change'] for t in trades if t['side']=='buy')
    payments = sum(a['amount'] for a in actions if a['kind']=='payment')
    profit = sum(t['cash_change'] for t in trades)+payments
    for t in trades:
        if t['side']=='sell' and (t['date']<first['target_date'] or t['reason']!=first['reason'] or t['signal_date']!=first['signal_date']):
            raise ValueError('Sale is inconsistent with first latched exit')
    failed = [o for o in orders if o['side']=='sell' and o['channel']=='board' and not o['filled_qty']]
    for o in failed:
        if not (o['failure']=='midpoint_limit_not_crossed' and o['limit_price']>o['source_high']
                and min(o['source_volume'],o['prior_avg_volume20'])*.01>=1000):
            raise ValueError('Failure was not solely explained by unreachable sell limit')
    result = dict(cohort=cohort, original_exit=archived, daily_decisions=rows, orders=orders,
        trades=trades, corporate_actions=actions, peak_holding=peak, buy_cost=buy_cost,
        gross_entry_mean=sum(t['gross'] for t in trades if t['side']=='buy')/cohort['bought_qty'],
        entry_cost_per_share=buy_cost/cohort['bought_qty'], dividends=payments, net_profit=profit,
        net_return=profit/buy_cost, peak_marked_profit_before_exit_costs=peak['market_value']-buy_cost,
        peak_marked_profit_to_final_net_difference=peak['market_value']-buy_cost-profit,
        first_close15_diagnostic=next(r for r in rows if r['peak_drawdown']<=-.15),
        first_close15_after_july_peak_diagnostic=next(r for r in rows if r['signal_date']>=peak['date'] and r['peak_drawdown']<=-.15),
        first_weak_rule_diagnostic=next(r for r in rows if r['below_ma20_two'] and r['relative20']<0),
        frozen_original_path=True, alternative_account_simulated=False, actual_fill_verified=False,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [source,published,control,control.parent/'report.json',
            adjusted,Path(__file__),ROOT/'skills/scenario_exit_replay.py',ROOT/'skills/exit_policy.py',
            ROOT/'skills/residual_tick_replay.py',ROOT/'skills/mixed_odd_replay.py',ROOT/'skills/midpoint_replay.py']})
    return result


if __name__=='__main__':
    result=review()
    target=ROOT/'artifacts/forward_simulation/globalwafers_exit_review_20260929.json'
    target.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    target.with_suffix('.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'\n')
    print({k:result[k] for k in ('buy_cost','gross_entry_mean','net_profit','net_return','peak_marked_profit_to_final_net_difference')})
