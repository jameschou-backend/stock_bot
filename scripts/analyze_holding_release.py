#!/usr/bin/env python3
"""Describe stock-level contributions; must reconcile to full account net profit."""
from pathlib import Path
from collections import defaultdict
import json
r=Path(__file__).resolve().parents[1];b=r/'.cache/holding-release-20260929/final-a';out={}
for arm in ('control3','control5','stagnant3','stagnant5','stronger3','stronger5'):
 c=json.loads((b/(arm+'.json')).read_text());a=c['account'];f=defaultdict(float);names={v['stock_id']:v['name'] for v in a['trades']}
 for row in a['cash_ledger']:
  if row.get('stock_id'):f[row['stock_id']]+=row['cash_change']
 for row in c['summary']['final_holdings']:f[row['stock_id']]+=row['market_value']
 marks={row['stock_id']:row['price'] for row in c['summary']['final_holdings']}
 for row in c['summary']['final_receivables']:
  value=row['amount'] if row['kind']=='cash' else row['qty']*marks[row['stock_id']]+row['fraction']*(row['fractional_cash_per_share'] or 0)
  f[row['stock_id']]+=value
 assert abs(sum(f.values())-c['summary']['profit'])<.001
 out[arm]=dict(stock_contribution=sorted([dict(stock_id=s,name=names[s],net_profit=v) for s,v in f.items()],key=lambda x:-x['net_profit']),sum_positive=sum(v for v in f.values() if v>0),sum_negative=sum(v for v in f.values() if v<0),focus_buys=[dict(date=v['date'],stock_id=v['stock_id'],qty=v['qty'],price=v['gross']/v['qty'],event_id=v['event_id']) for v in a['trades'] if v['side']=='buy' and v['stock_id'] in ('2221','5386') and v['date'].startswith('2026')])
p=r/'artifacts/forward_simulation/holding_release_20260929-attribution.json';encoded=json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
if p.exists() and p.read_text()!=encoded:raise ValueError('Preserve published attribution')
p.write_text(encoded)
for k,v in out.items():print(k,'gains',v['sum_positive'],'losses',v['sum_negative'],'top',v['stock_contribution'][:3],'focus_buys',v['focus_buys'])
