"""Two preregistered limit widths; all plans still precede current-day ticks."""
from copy import deepcopy
from decimal import Decimal
import math

import pandas as pd

from skills.execution_stress import StressOrder
from skills.intraday_limit_replay import limit_price,match_ticks
from skills.residual_tick_replay import audit_tick_plans


def band_limit(reference,sid,side,mask):
    if mask not in range(4):raise ValueError('Unknown fixed limit policy')
    if not reference:return None
    offset=Decimal('.02') if side=='buy' and mask&1 else Decimal('-.02') if side=='sell' and mask&2 else Decimal(0)
    return limit_price(float(Decimal(str(reference))*(1+offset)),sid,side)


class BandPlanning:
    def __init__(self,*args,band_mask,**kwargs):
        if band_mask not in range(4):raise ValueError('Unknown fixed limit policy')
        self.band_mask,self.opening_tick_plans=band_mask,[]
        super().__init__(*args,**kwargs)

    def _plan(self,day,sid,side,eid,signal,qty,budget,opening_cash,failure):
        super()._plan(day,sid,side,eid,signal,qty,budget,opening_cash,failure)
        plan=self.day_plans[(eid,side)]
        self.opening_tick_plans.append(deepcopy(plan))
        plan['limit_price']=band_limit(plan['prior_reference'],sid,side,self.band_mask)
        sizing_budget=budget if self.benchmark else min(budget,self.residual_budget)
        plan['sizing_budget']=sizing_budget
        if side=='buy' and plan['planned_qty']:
            plan['planned_qty']=self._affordable(plan['planned_qty'],1000,plan['limit_price'],sizing_budget,sid)
            if not plan['planned_qty']:plan['rejection']='band_budget_below_one_lot'
        self.tick_plans[-1]=deepcopy(plan)

    def run(self):
        result=super().run()
        result['opening_tick_plans']=self.opening_tick_plans
        result['settings'].update(execution='precommitted_limit_bands_v1',band_mask=self.band_mask)
        return result


def factory(mask):
    def build(cls):
        class FixedBand(BandPlanning,cls):
            def __init__(self,*args,**kwargs):super().__init__(*args,band_mask=mask,**kwargs)
        return FixedBand
    return build


def audit_bands(account,ticks,markets,quotes,calendar,corporate):
    base=dict(account,tick_plans=account['opening_tick_plans'],orders=[],trades=[])
    audit_tick_plans(base,ticks,markets,quotes,calendar,corporate)
    before=account['opening_tick_plans'];after=account['tick_plans']
    if len(before)!=len(after):raise ValueError('Band plan count changed')
    price=quotes.pivot(index='date',columns='stock_id',values='close').reindex(calendar)
    volume=quotes.pivot(index='date',columns='stock_id',values='volume').reindex(calendar)
    adv=volume.rolling(20,min_periods=20).mean().shift(1)
    amount=(price*volume).rolling(20,min_periods=20).mean().shift(1)
    cost=StressOrder();cost.stress_slippage=account['settings']['slippage']
    plans={}
    for old,new in zip(before,after):
        expected=deepcopy(old)
        limit=band_limit(old['prior_reference'],old['stock_id'],old['side'],account['settings']['band_mask'])
        budget=new['sizing_budget']
        if not math.isfinite(budget) or not 0<=budget<=old['reserved_cash']:
            raise ValueError('Band enlarged frozen budget')
        expected.update(limit_price=limit,sizing_budget=budget)
        if old['side']=='buy' and old['planned_qty']:
            expected['planned_qty']=cost._affordable(old['planned_qty'],1000,limit,budget,old['stock_id'])
            if not expected['planned_qty']:expected['rejection']='band_budget_below_one_lot'
        if expected!=new:raise ValueError('Band plan differs from fixed pre-tick rule')
        plans[(new['date'],new['event_id'],new['side'])]=new
    matched={};seen=set()
    for row in account['orders']:
        if 'ticks_sha256' not in row:continue
        key=(row['date'],row['event_id'],row['side']);plan=plans[key]
        stockday=(row['stock_id'],row['date'])
        if stockday in seen or row['requested_qty']!=plan['planned_qty'] or row['limit_price']!=plan['limit_price']:
            raise ValueError('Band execution changed frozen plan or reused ticks')
        seen.add(stockday)
        tape,digest=ticks.get(row['stock_id'],row['date'],markets.get(stockday,markets.get(row['stock_id'])))
        day=pd.Timestamp(row['date']);sid=row['stock_id']
        if row['prior_avg_volume20']!=adv.at[day,sid] or row['prior_avg_amount20']!=amount.at[day,sid]:
            raise ValueError('Band liquidity differs from prior source window')
        rebuilt=match_ticks(tape,row['side'],plan['limit_price'],plan['planned_qty'],adv.at[day,sid],account['settings']['participation'])
        if digest!=row['ticks_sha256'] or any(row[k]!=v for k,v in rebuilt.items()):
            raise ValueError('Band tick fill did not independently reproduce')
        matched[key]=row
    fills={}
    for trade in account['trades']:
        key=(trade['date'],trade['event_id'],trade['side']);row=matched.get(key)
        if not row or trade['reference_price']!=plans[key]['limit_price']:
            raise ValueError('Band fill lacks an audited frozen plan')
        fills[key]=fills.get(key,0)+trade['qty']
        if trade['side']=='buy' and -trade['cash_change']>plans[key]['sizing_budget']+.005:
            raise ValueError('Band fill spent beyond frozen cash')
    if any(fills.get(k,0)!=r['filled_qty'] for k,r in matched.items()):raise ValueError('Band trades differ from fill log')
    return dict(precommitted_limits=True,opening_cash_reserved=True,tick_fills_rebuilt=True,
                fixed_band_rule_rebuilt=True)
