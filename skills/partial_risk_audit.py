"""Rebuild partial instructions from source history and reconciled journals."""
import math
import pandas as pd
from skills.high_return_audit import audit_high_return_resources
from skills.partial_risk import PARTIAL_REASONS


def audit_core_resources(account,engine,quotes):
    # Only liquidation attempts can release a residual slot. The original
    # independent cash/holding reconstruction still sees every actual trade.
    view = dict(account,orders=[r for r in account['orders'] if r['reason'] not in PARTIAL_REASONS])
    return audit_high_return_resources(view,engine.resource_plans,engine.slot_decisions,
        engine.board_decisions,engine.residual_days,quotes)


def audit_partial_risk(account,data,engine):
    arm = account['settings']['risk_arm']
    days = [str(d.date()) for d in data.days]; quotes = data.quotes.set_index(['date','stock_id'])
    holdings = {(r['date'],r['event_id']):r for r in account['holdings']}
    daily = {r['date']:r for r in account['daily']}
    plans = {(r['date'],r['event_id']):r for r in account['tick_plans'] if r['side']=='sell'}
    observed = {(r['date'],r['event_id']):r for r in account['risk_decisions']}
    if len(observed)!=len(account['risk_decisions']): raise ValueError('Duplicate reduction decision')
    expected_log = {}; checked = 0
    for cohort in account['cohorts']:
        eid,sid = cohort['event_id'],cohort['stock_id']
        entry,end = days.index(cohort['entry_date']),days.index(cohort['exit_date'] or data.end)
        adjusted = data.features.adjusted_close[sid]
        half_signal,pending,exit_instruction = None,None,None
        fills = [t['reference_price'] for t in account['trades'] if t['event_id']==eid and t['side']=='buy']
        highs = [(entry,max(float(quotes.loc[(data.days[entry],sid),'close']),*fills))]
        factors = {}
        for i in range(entry+1,end+1):
            date,prev = days[i],days[i-1]
            old = holdings.get((prev,eid)); qty = old['qty'] if old else 0
            for action in account['corporate_actions']:
                if action['date']!=date or action['stock_id']!=sid: continue
                if action['kind'] in ('split','capital_reduction'):
                    qty = action['qty_after']
                    if pending: pending['keep'] = math.ceil(pending['keep']*action['qty_after']/action['entitled_qty'])
                elif action['kind']=='share_delivery':
                    qty += action['qty']
                    if pending: pending['keep'] += action['qty']
            reason = None
            if not exit_instruction:
                if pd.notna(adjusted.iloc[i-1]) and pd.notna(adjusted.iloc[entry]) and adjusted.iloc[i-1]/adjusted.iloc[entry]-1<=-.12+1e-12:
                    reason = 'loss12'
                elif i-entry>=63: reason = 'time63'
                if reason: exit_instruction = dict(reason=reason,signal_date=prev)
            if not qty: continue
            if exit_instruction:
                pending = None
            else:
                triggered = False
                if 'half15' in arm and i-1>entry:
                    actions = data.events.loc[data.events.stock_id.eq(sid)&pd.to_datetime(data.events.event_date).eq(data.days[i-1])]
                    if len(actions)>1: raise ValueError('Ambiguous independent peak basis')
                    factors[i-1] = float(actions.iloc[0].ratio) if len(actions) else 1.
                    if not engine.official_halt(data.days[i-1],sid):
                        q = quotes.loc[(data.days[i-1],sid)]
                        highs.append((i-1,float(q.high)))
                        values = [price*math.prod(v for j,v in factors.items() if k<j<=i-1) for k,price in highs]
                        triggered = float(q.close)<=max(values)*.85+1e-10
                core = False
                if half_signal and days[i-2]>half_signal:
                    # Separate window means, not the engine's rolling matrix.
                    core = all(len(adjusted.iloc[j-59:j+1])==60 and adjusted.iloc[j-59:j+1].notna().all()
                               and adjusted.iloc[j]<adjusted.iloc[j-59:j+1].mean() for j in (i-2,i-1))
                if core:
                    exit_instruction = dict(reason='core_ma60_two',signal_date=prev)
                    pending = None
                    expected_log[(date,eid)] = dict(date=date,event_id=eid,stock_id=sid,
                        signal_date=prev,reason='core_ma60_two',keep=0,requested_qty=qty)
                else:
                    keep = pending['keep'] if pending else qty
                    reasons = []
                    if 'half15' in arm and triggered and half_signal is None:
                        half_signal = prev; keep = min(keep,(qty+1)//2); reasons.append('half15')
                    weight = old['market_value']/daily[date]['opening_nav'] if old else 0.
                    if 'cap40' in arm and weight>.4:
                        ref = float(quotes.loc[(data.days[i-1],sid),'close'])
                        if hasattr(engine.corporate,'reference_price'): ref = engine.corporate.reference_price(sid,date,ref)
                        capped = max(1,math.floor(daily[date]['opening_nav']/3/ref))
                        if capped<keep: keep=capped;reasons.append('cap40')
                    if reasons and keep<qty:
                        pending = dict(keep=keep,reason='reduce_'+'_'.join(reasons),signal_date=prev)
                    if pending and pending['keep']<qty:
                        expected_log[(date,eid)] = dict(date=date,event_id=eid,stock_id=sid,
                            **pending,requested_qty=qty-pending['keep'])
            instruction = exit_instruction or pending
            should_sell = instruction and (exit_instruction or pending['keep']<qty)
            plan = plans.get((date,eid))
            if bool(plan)!=bool(should_sell): raise ValueError('Missing or extra independent sale plan')
            if should_sell:
                requested = qty if exit_instruction else qty-pending['keep']
                if plan['planned_qty']!=requested or plan['signal_date']!=instruction['signal_date']:
                    raise ValueError('Reduction plan quantity or timing differs')
                rows = [r for r in account['orders'] if r['date']==date and r['event_id']==eid and r['side']=='sell']
                if not rows or any(r['reason']!=instruction['reason'] or r['signal_date']!=instruction['signal_date'] for r in rows):
                    raise ValueError('Reduction execution reason differs')
                sold = sum(t['qty'] for t in account['trades'] if t['date']==date and t['event_id']==eid and t['side']=='sell')
                if not 0<=sold<=requested: raise ValueError('Reduction exceeded instructions')
                if not exit_instruction and qty-sold<=pending['keep']: pending=None
                checked += 1
    if observed!=expected_log: raise ValueError('Partial decision ledger differs from history reconstruction')
    return dict(independent_partial_plans=checked,core_slots_preserved=True,prior_only_decisions_rebuilt=True)
