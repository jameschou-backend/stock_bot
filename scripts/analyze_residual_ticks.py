#!/usr/bin/env python3
"""Reconcile execution-path differences; never tune or rerun strategy rules."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.residual_ticks_ui import load, REPORT
from scripts.research_exit_scenarios import read,write,sha
from skills.execution_factorial import stock_pnl,path_comparison


def pnl(account):
    # End-of-period share rights require an actual final valuation row.
    marks={r['stock_id']:dict(price=r['price']) for r in account['holdings']
           if r['date']==account['daily'][-1]['date']}
    return stock_pnl(account,marks)


def analyze():
    report=load()
    baseline_path=ROOT/'artifacts/forward_simulation/residual_slots_20260926.json'
    baseline_publication=read(baseline_path)
    old_ref=baseline_publication['cases']['release_0']['result']
    new_ref=report['cases']['strategy_normal']['result']
    for ref in (old_ref,new_ref):
        if sha(ROOT/ref['path'])!=ref['sha256']:
            raise ValueError('Attribution account hash changed')
    old,new=(read(ROOT/ref['path'])['account'] for ref in (old_ref,new_ref))
    left,right=pnl(old),pnl(new)
    names={t['stock_id']:t['name'] for a in (old,new) for t in a['trades']}
    differences=[dict(stock_id=s,name=names.get(s,s),daily_profit=left.get(s,{}).get('profit',0.),
                      tick_profit=right.get(s,{}).get('profit',0.),
                      difference=right.get(s,{}).get('profit',0.)-left.get(s,{}).get('profit',0.))
                 for s in sorted(set(left)|set(right))]
    differences.sort(key=lambda r:r['difference'])
    total=new['daily'][-1]['nav']-old['daily'][-1]['nav']
    if not math.isclose(sum(r['difference'] for r in differences),total,abs_tol=.02,rel_tol=0):
        raise ValueError('Attribution does not reconcile to NAV difference')
    path=path_comparison(old,new)
    failures=Counter(r['failure'] for r in new['orders'] if r['side']=='buy' and r.get('failure'))
    no_fill=[]
    for entry in path['only_base']:
        rows=[r for r in new['orders'] if r['event_id']==entry['event_id'] and r['side']=='buy']
        no_fill.append(dict(entry,reasons=sorted({r['failure'] for r in rows if r.get('failure')})))
    sessions={r['date']:i for i,r in enumerate(new['daily'])}
    first_attempt,first_fill={},{}
    for row in new['orders']:
        if row['side']=='sell' and row['channel']=='board' and row['requested_qty']:
            first_attempt.setdefault(row['event_id'],row)
    for row in new['trades']:
        if row['side']=='sell':
            first_fill.setdefault(row['event_id'],row)
    exits=[]
    for eid,attempt in first_attempt.items():
        fill=first_fill.get(eid)
        exits.append(dict(event_id=eid,stock_id=attempt['stock_id'],signal_date=attempt['signal_date'],
            first_attempt=attempt['date'],first_fill=fill['date'] if fill else None,
            wait_sessions=sessions[fill['date']]-sessions[attempt['date']] if fill else None))
    result=dict(schema='residual_ticks_attribution_v1',source_sha256={str(REPORT.relative_to(ROOT)):sha(REPORT),
        old_ref['path']:old_ref['sha256'],new_ref['path']:new_ref['sha256']},
        nav_difference=total,stock_profit_differences=differences,entry_path=path,
        missed_baseline_entries=no_fill,buy_order_failures=dict(failures),sell_waits=exits,
        note='Differences combine missed entries, changed quantities, exits and compounding; not isolated causal effects.',
        live_qualified=False,unseen_validation=False)
    output=ROOT/'docs/research_residual_ticks_attribution_20260928.json'
    write(output,result)
    print(json.dumps(dict(common_entries=path['common_entry_events'],only_daily=len(path['only_base']),
        only_tick=len(path['only_other']),nav_difference=total,largest_gaps=differences[:6],
        delayed_exits=sum(r['wait_sessions'] is not None and r['wait_sessions']>0 for r in exits),
        maximum_sell_wait=max((r['wait_sessions'] or 0) for r in exits)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    analyze()
