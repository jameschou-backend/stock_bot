#!/usr/bin/env python3
"""Describe frozen cohort durations and blocked signals; no counterfactual returns."""
from pathlib import Path
from collections import defaultdict,Counter
import json,numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1];close=pd.read_parquet(root/'.cache/partial-risk-2019-20260929/inputs-final/close-official.parquet').set_index('date');close.index=pd.to_datetime(close.index);r20=close/close.shift(20)-1
result={};allcohorts={}
for key,base in [('three','stock-universe-2019-20260929'),('five','stock-universe-five-20260929')]:
 raw=json.loads((root/'.cache'/base/'final-a/liquid_universe.json').read_text());a=raw['account'];days={v['date']:i for i,v in enumerate(a['daily'])};tr=defaultdict(list);orders=defaultdict(list);flows=defaultdict(float);names={v['stock_id']:v['name'] for v in a['trades']}
 for v in a['trades']:tr[v['event_id']].append(v)
 for v in a['orders']:
  if v['side']=='sell':orders[v['event_id']].append(v)
 for v in a['cash_ledger']:
  if v.get('stock_id'):flows[v['stock_id']]+=v['cash_change']
 for v in raw['summary']['final_holdings']:flows[v['stock_id']]+=v['market_value']
 for v in raw['summary']['final_receivables']:
  assert v['kind']=='cash';flows[v['stock_id']]+=v['amount']
 assert abs(sum(flows.values())-raw['summary']['profit'])<.001
 records=[]
 for c in a['cohorts']:
  sales=[v for v in tr[c['event_id']] if v['side']=='sell'];ins=orders[c['event_id']];closed=c.get('exit_date') is not None
  end=c['exit_date'] or a['daily'][-1]['date'];first=min((v['date'] for v in ins),default=None);last=max((v['date'] for v in sales),default=None)
  records.append(dict(stock_id=c['stock_id'],name=c['name'],event_id=c['event_id'],entry_date=c['entry_date'],exit_date=c['exit_date'],closed=closed,observed_age=days[end]-days[c['entry_date']],first_exit_attempt=first,last_sell_fill=last,first_exit_age=days[first]-days[c['entry_date']] if first else None,last_sell_age=days[last]-days[c['entry_date']] if last else None,exit_reason=ins[0]['reason'] if ins else None))
 ages=[v['observed_age'] for v in records if v['closed']];first=[v['first_exit_age'] for v in records if v['first_exit_age'] is not None]
 focus=[];cohorts={v['event_id']:v for v in a['cohorts']}
 for target in [v for v in a['orders'] if v['side']=='buy' and v.get('stock_id') in ('2221','5386') and v['date'].startswith('2026') and v.get('failure')=='slots_full']:
  prev=target['signal_date'];rows=[]
  for h in a['holdings']:
   if h['date']!=prev:continue
   c=cohorts[h['event_id']];d=pd.Timestamp(prev);own=float(r20.at[d,h['stock_id']]);rel=own-float(r20.at[d,'0050']);age=days[target['date']]-days[c['entry_date']]
   rows.append(dict(stock_id=h['stock_id'],name=h['name'],qty=h['qty'],entry_date=c['entry_date'],decision_age=age,return20=own if np.isfinite(own) else None,relative20=rel if np.isfinite(rel) else None,stagnant20=bool(age>=20 and own<.03 and rel<0),prior_exit_attempted=any(v['date']<=prev for v in orders[h['event_id']])))
  focus.append(dict(order=target,previous_close_holdings=rows))
 result[key]=dict(closed_cohorts=len(ages),open_cohorts=len(records)-len(ages),closed_age_percentiles=dict(zip(('min','median','p90','max'),map(float,np.percentile(ages,[0,50,90,100])))),first_exit_age_percentiles=dict(zip(('min','median','p90','max'),map(float,np.percentile(first,[0,50,90,100])))),exit_reasons=Counter(v['exit_reason'] for v in records),cohorts=records,attribution=sorted([dict(stock_id=s,name=names[s],net_profit=v) for s,v in flows.items()],key=lambda v:-v['net_profit']),focus=focus)
 allcohorts[key]={c['event_id'] for c in a['cohorts']}
result['shared_entries']=len(allcohorts['three']&allcohorts['five']);result['only_three_entries']=len(allcohorts['three']-allcohorts['five']);result['only_five_entries']=len(allcohorts['five']-allcohorts['three']);result['scope']='Descriptive diagnosis of frozen baseline accounts; not counterfactual returns'
p=root/'artifacts/forward_simulation/holding_release_20260929-diagnosis.json';encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
if p.exists() and p.read_text()!=encoded:raise ValueError('Preserve prior diagnostic output')
p.write_text(encoded)
for k in ('three','five'):
 print(k,{n:v for n,v in result[k].items() if n not in ('cohorts','attribution','focus')})
 print('5386 blockers',[v for v in result[k]['focus'] if v['order']['stock_id']=='5386' and v['order']['date']=='2026-02-11'])
print('shared',result['shared_entries'],'only3',result['only_three_entries'],'only5',result['only_five_entries'])
