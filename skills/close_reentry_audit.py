"""Rebuild reentry observations independently from historical price prefixes."""
from collections import Counter,defaultdict
import math
import pandas as pd


def audit_reentries(account,data):
    dates = [str(d.date()) for d in data.days]
    cohorts = {c['event_id']:c for c in account['cohorts']}
    triggers = {r['event_id']:r for r in account['close_stop_evidence'] if r['triggered']}
    close = data.features.adjusted_close
    arm = account['settings']['reentry_arm']
    original = defaultdict(list)
    for e in sorted(data.entries,key=lambda e:(-e['priority'],e['event_id'])):
        original[e['entry_date']].append(e['event_id'])
    logs = {r['event_id']:r for r in account['reentry_log']}
    counts,parents = Counter(),set()
    for row in account['reentry_screens']:
        i = dates.index(row['date']); j = i-1
        c = cohorts[row['parent']]; sid = c['stock_id']
        x = dates.index(c['exit_date'])
        if not (row['signal_date']==dates[j] and 1<=j-x<=20):
            raise ValueError('Reentry window or lag changed')
        trigger = triggers[row['parent']]
        floor = trigger['threshold']*close.at[pd.Timestamp(trigger['signal_date']),sid]/trigger['close']
        if not math.isclose(floor,row['adjusted_stop_floor'],rel_tol=1e-12):
            raise ValueError('Reentry corporate price basis changed')
        indices = [j] if arm=='reclaim' else [j-1,j]
        ok = min(indices)>x
        if len(row['observations'])!=len(indices):raise ValueError('Reentry confirmation length changed')
        for k,obs in zip(indices,row['observations']):
            p = close[sid].iloc[k]; history = close[sid].iloc[k-9:k+1]
            ma = history.mean() if len(history)==10 and history.notna().all() else float('nan')
            rs = p/close[sid].iloc[k-5]-close['0050'].iloc[k]/close['0050'].iloc[k-5]
            expected = dict(close=p,ma10=ma,excess5=rs)
            if obs['signal_date']!=dates[k]:raise ValueError('Future recovery observation')
            for key,value in expected.items():
                actual = obs[key]
                if (pd.isna(value) and actual is not None) or (pd.notna(value) and (actual is None or not math.isclose(value,actual,abs_tol=1e-12))):
                    raise ValueError('Reentry indicator differs from price prefix')
            ok = bool(ok and pd.notna(p) and pd.notna(ma) and pd.notna(rs) and p>floor and p>ma and rs>0)
        market_history = close['0050'].iloc[:j+1].dropna().tail(120)
        market = len(market_history)==120 and pd.notna(close['0050'].iloc[j]) and close['0050'].iloc[j]>market_history.mean()
        same_stock = any(e['members'][0]==sid and e['entry_date']==dates[i] for e in data.entries)
        outcome = 'same_stock_original_signal' if same_stock else 'market_off' if not market else 'not_recovered' if not ok else 'candidate'
        if outcome!=row['outcome'] or bool(market)!=row['market_on']:
            raise ValueError('Reentry eligibility differs')
    for eid,row in logs.items():
        counts[row['root']]+=1
        if counts[row['root']]>2 or counts[row['root']]!=row['attempt'] or row['parent'] in parents:
            raise ValueError('Reentry attempt cap or parent reused')
        parents.add(row['parent'])
        matching = [r for r in account['reentry_screens'] if r['parent']==row['parent'] and r['outcome']=='candidate']
        if not matching or matching[0]['date']!=row['date'] or len(matching)!=1:
            raise ValueError('Reentry is not the first eligible attempt')
        parent_sales = [t for t in account['trades'] if t['event_id']==row['parent'] and t['side']=='sell']
        if not parent_sales or max(t['date'] for t in parent_sales)>=row['signal_date']:
            raise ValueError('Reentry did not wait for completed exit')
    for q in account['reentry_queue']:
        if q['original']!=original[q['date']] or q['ordered']!=q['original']+q['reentries']:
            raise ValueError('Original candidates lost priority')
        expected = sorted([r for r in logs.values() if r['date']==q['date']],
            key=lambda r:(-r['observations'][-1]['excess5'],r['stock_id'],r['event_id']))
        if q['reentries']!=[r['event_id'] for r in expected]:raise ValueError('Reentry ranking changed')
    for t in account['trades']:
        if t['side']=='buy' and cohorts[t['event_id']].get('reentry_parent'):
            r = logs.get(t['event_id'])
            if not r or t['signal_date']!=r['signal_date'] or t['date']!=r['date']:
                raise ValueError('Reentry trade lacks T+1 candidate')
    return dict(signal_rows=len(account['reentry_screens']),attempts=len(logs),
        first_signal_checked=True,window_and_attempt_caps_checked=True,
        original_candidate_priority_checked=True,indicators_rebuilt=True)
