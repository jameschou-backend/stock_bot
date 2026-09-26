"""Merge sealed ETF periods without resetting cash or learning new parameters."""
from copy import deepcopy
import math
import pandas as pd
from skills.index_earlier_inputs import load as earlier, EarlierBenchmarkCorporate, EarlierBenchmarkFeeds
from skills.index_exposure_inputs import load as current, ROOT
from skills.replay_corporate_actions import CorporateActions, SPLIT_0050
from skills.etf_limit_audit import reconcile
from scripts.research_exit_scenarios import read, sha

BASE=ROOT/'.cache/index-continuous-20260927'
START,END='2016-01-04','2026-09-09'

def combine(left,right,keys):
    for day in set(left)&set(right):
        if any(left[day][k]!=right[day][k] for k in keys):
            raise ValueError('Overlapping raw prices differ: '+day)
    return dict(left,**right)

class ContinuousCorporate(EarlierBenchmarkCorporate):
    def reference_price(self,sid,day,prior):
        self.prepare(sid)
        # Only the already documented unit conversion is applied to planning.
        # Missing cash-dividend announcement times never become prior signals.
        return prior/4 if day==SPLIT_0050['date'] else prior

def load():
    old=earlier();new=current();refs=dict(old['sources']);refs.update(new['sources'])
    # Both inputs validate their own source manifests before any merge.
    days=old['days']+new['days']
    if days!=sorted(set(days)) or days[0]!=START or days[-1]!=END:
        raise ValueError('Continuous calendar has a gap, duplicate, or wrong boundary')
    keys=('open','close','low','high','volume')
    quotes=combine(old['quotes'],new['quotes'],keys)
    overlap=sorted(set(old['signals'])&set(new['signals']))
    if len(overlap)<244 or any(abs(old['signals'][d]/new['signals'][d]-1)>1e-6 for d in overlap):
        raise ValueError('Adjusted signal seam differs')
    # Preserve the entire earlier prefix bit-for-bit; no retrospective re-basing.
    signals={**new['signals'],**old['signals']}
    calendar=sorted(d for d in set(old['calendar'])|set(new['calendar']) if d<=END)
    if [d for d in calendar if d>=START]!=days:raise ValueError('Unaccounted market date')
    raw_path=ROOT/'.cache/million-replay-inputs/quotes.parquet'
    if refs.get(str(raw_path.relative_to(ROOT)))!=sha(raw_path):raise ValueError('Raw benchmark not sealed')
    raw=pd.read_parquet(raw_path,filters=[('stock_id','==','0050')])
    raw['date']=pd.to_datetime(raw.date).dt.strftime('%Y-%m-%d')
    recent={r['date']:{k:float(r[k]) for k in keys} for r in raw.to_dict('records') if r['date']<=END}
    historical={r['date']:{k:float(r[k]) for k in keys} for r in old['benchmark_quotes'].to_dict('records')}
    benchmark=combine(historical,recent,keys)
    known_halt={'2025-06-11','2025-06-12','2025-06-13','2025-06-16','2025-06-17'}
    if set(days)-set(benchmark)!=known_halt:raise ValueError('Unexpected 0050 quote gap')
    limit_rows={r['date']:r for r in read(ROOT/'.cache/index-earlier-20260927/sources-v1/0050-TaiwanStockPriceLimit.json')['data']}
    limit_rows.update({r['date']:r for r in read(ROOT/'.cache/million-replay-inputs/execution-feeds/limits-0050.raw.json')['data']})
    limits,limit_audit=reconcile(benchmark,limit_rows,days)
    empty=pd.DataFrame(columns=['stock_id','event_date','payload_json','source'])
    corp=CorporateActions(empty,ROOT/'.cache/million-replay-inputs/dividends',offline=True)
    corp.prepare('0050')
    actions=deepcopy(old['dividends'])+[deepcopy(a) for a in corp.loaded['0050'] if a['date']>old['days'][-1] and a['date']<=END]
    if len({a['action_id'] for a in actions})!=len(actions):raise ValueError('Duplicate corporate rights')
    recent_actions=[a for a in actions if a['date']>=new['days'][0]]
    expected=[a for a in new['benchmarks']['control']['account']['corporate_actions'] if a['kind']!='payment']
    if len(expected)!=len(recent_actions):raise ValueError('Recent corporate action coverage changed')
    for a,b in zip(recent_actions,expected):
        fields=('stock_id','date','kind','action_id','multiplier') if a['kind']=='split' else ('stock_id','date','kind','action_id','cash_per_share','pay_date')
        if any(a[k]!=b[k] for k in fields):raise ValueError('Recent rights differ from sealed source')
    return dict(days=days,calendar=calendar,quotes=quotes,signals=signals,sources=refs,
        benchmark_quotes=pd.DataFrame([dict(date=d,stock_id='0050',**q) for d,q in sorted(benchmark.items())]),
        benchmark_limits=limits,limit_audit=limit_audit,actions=actions,split=SPLIT_0050,early=old,current=new,
        quality=dict(cash_reset_count=0,initial_deposits=1,strict_data_ready=False,
            official_00631L_limits=False,official_0050_limits=False,dividend_payment_dates_independently_verified=False,
            provider_0050_limit_conflict_days=[r['date'] for r in limit_audit['provider_ohlc_conflicts']],
            planning_uses_undiscounted_prior_close=True,signal_overlap_rows=len(overlap),known_0050_halt_dates=sorted(known_halt)))
