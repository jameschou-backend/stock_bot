"""Independently rebuild every addition opportunity from source prices and journals."""
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math

import pandas as pd

from skills.support_risk_audit import audit_support_risk
from skills.support_risk_source_audit import audit_candidate_binding


def original_entry_view(account):
    # Only entry-signal audits use this view. Share, cash, NAV, board and slot
    # audits MUST receive the unmodified account, including every addition.
    return dict(account, **{k:[r for r in account[k] if r.get('reason')!='pyramid_add']
                           for k in ('trades','orders')})


def cash_amount(price, qty, side, slip):
    gross = Decimal(str(price))*qty
    rounded = lambda v, rule: v.quantize(Decimal(1), rounding=rule)
    cents = lambda v: v.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    fee = max(Decimal(20), rounded(gross*Decimal('.001425'), ROUND_HALF_UP))
    sliding = rounded(gross*Decimal(str(slip)), ROUND_CEILING)
    tax = rounded(gross*Decimal('.003'), ROUND_FLOOR) if side=='sell' else Decimal(0)
    return float(cents((cents(gross) if side=='sell' else -cents(gross))-fee-sliding-tax))


def independent_capacity(price, ratio, held, nav, available, slip, maximum=None):
    budget = max(0., min(available, nav*.1, nav/5-held*price)) if price else 0.
    chosen, loss = 0, None
    if price and ratio and 0 < ratio < 1:
        for qty in range(1000, int(budget/price)//1000*1000+1, 1000):
            if maximum is not None and qty > maximum:
                break
            paid = -cash_amount(price, qty, 'buy', slip)
            risk = float(Decimal(str(held*price+paid-cash_amount(price*ratio, held+qty, 'sell', slip))).quantize(
                Decimal('.01'), rounding=ROUND_HALF_UP))
            if paid <= budget+1e-8 and risk <= nav*.02+1e-8:
                chosen, loss = qty, risk
    return dict(budget=budget, qty=chosen, planned_loss=loss)


def audit_pyramid(case, signals, quotes, prior, candidates, final_pending):
    account, mask = case['account'], case['config']['factor_mask']
    view = original_entry_view(account)
    checked = audit_support_risk(view, case['technical_entries'], case['support_decisions'],
        case['exit_decisions'], case['exit_states'], case['slot_decisions'], signals,
        'support_risk2', mask, prior)
    checked.update(audit_candidate_binding(dict(case, account=view), candidates, signals.days))
    positions = {str(d.date()):i for i,d in enumerate(signals.days)}
    cohorts = {c['event_id']:c for c in account['cohorts']}
    supports = {(r['date'],r['event_id']):r for r in case['support_decisions']}
    traces = defaultdict(list)
    for row in case['pyramid_decisions']:
        traces[row['date']].append(row)
    if any(d not in {r['date'] for r in account['daily']} for d in traces):
        raise ValueError('Addition decision outside completed account')
    q = quotes.copy();q['date'] = pd.to_datetime(q['date'])
    prices = q.pivot(index='date', columns='stock_id', values='close').reindex(signals.days)
    volumes = q.pivot(index='date', columns='stock_id', values='volume').reindex(signals.days)
    amount20 = (prices*volumes).rolling(20, min_periods=20).mean().shift(1)
    units, rights, pending, done = defaultdict(int), {}, {}, set()
    opening_cash = account['settings']['initial_cash']
    matched_plans, matched_trades = set(), set()
    cents = lambda v: float(Decimal(str(v)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    for daily in account['daily']:
        day, nav = daily['date'], daily['opening_nav']
        for action in account['corporate_actions']:
            if action['date'] != day: continue
            sid = action['stock_id']
            if action['kind']=='split': units[sid] = action['qty_after']
            elif action['kind']=='stock_dividend':
                rights[action['action_id']] = (action['event_id'], action['whole_new_shares'])
            elif action['kind']=='share_delivery':
                units[sid] += action['qty'];rights.pop(action['action_id'])
        trades = [t for t in account['trades'] if t['date']==day]
        additions = [t for t in trades if t['reason']=='pyramid_add']
        ordinary = [t for t in trades if t['reason']!='pyramid_add']
        if additions and ordinary and min(t['sequence'] for t in additions)<max(t['sequence'] for t in ordinary):
            raise ValueError('Addition consumed resources before original orders')
        for trade in ordinary:
            units[trade['stock_id']] += trade['qty']*(1 if trade['side']=='buy' else -1)
        plans = [p for p in case['resource_plans'] if p['date']==day]
        original_plans = [p for p in plans if p.get('kind')!='pyramid_add']
        add_plans = [p for p in plans if p.get('kind')=='pyramid_add']
        if plans != original_plans+add_plans:
            raise ValueError('Addition resource reservations must follow original orders')
        cash = cents(opening_cash+sum(r['cash_change'] for r in account['cash_ledger']
            if r['date']==day and r['kind'] not in ('buy','sell','initial_deposit'))+sum(t['cash_change'] for t in ordinary))
        remaining = cents(opening_cash+sum(t['cash_change'] for t in ordinary if t['side']=='buy'))
        locked = original_plans[-1]['locked_after'] if original_plans else 0.
        present = {eid for eid,c in cohorts.items() if c['entry_date']<day
                   and (c['exit_date'] is None or c['exit_date']>day)}
        ids = sorted((present|set(pending))-done)
        if [r['event_id'] for r in traces[day]] != ids:
            raise ValueError('Missing, duplicated or reordered addition opportunity')
        for row, eid in zip(traces[day], ids):
            sid = cohorts[eid]['stock_id'];instruction = pending.get(eid)
            wanted = dict(date=day, stock_id=sid, event_id=eid,
                          pending_before=instruction.copy() if instruction else None, filled_qty=0)
            trace = supports.get((day,eid))
            if eid not in present or not units[sid] or (trace and trace['trigger_reason']):
                wanted['status'] = 'cancelled_exit_or_no_physical_shares';pending.pop(eid,None)
            elif units[sid]%1000 or any(e==eid and qty for e,qty in rights.values()):
                wanted['status'] = 'cancelled_unsettled_or_nonboard_shares';pending.pop(eid,None)
            else:
                context = signals.technical_context(positions[day],sid)
                entry_price = signals.price(positions[cohorts[eid]['entry_date']],sid)
                amount = float(amount20.at[pd.Timestamp(day),sid])
                strong = (context['adjusted_close'] is not None and entry_price is not None
                    and context['adjusted_close']>=entry_price*1.1-1e-12 and context['breakout20'] is True)
                if instruction is None and not strong:
                    wanted['status'] = 'no_strong_signal'
                elif not math.isfinite(amount) or amount < 50_000_000:
                    wanted['status'] = 'cancelled_prior_liquidity';pending.pop(eid,None)
                else:
                    price = prior(pd.Timestamp(day),sid)
                    available = max(0.,cents(min(cash,remaining)-locked))
                    slip = .009 if mask&1 else .0045
                    if instruction is None:
                        ratio = max(entry_price*.88,trace['floor_after'])/context['adjusted_close']
                        capacity = independent_capacity(price,ratio,units[sid],nav,available,slip)
                        wanted.update(context=context,capacity=capacity)
                        if capacity['qty']:
                            instruction = dict(signal_date=context['signal_date'],created_date=day,
                                target_index=positions[day]+bool(mask&2),stop_ratio=ratio,original_qty=capacity['qty'])
                            pending[eid] = instruction
                            wanted['created_instruction'] = instruction.copy()
                        else: wanted['status'] = 'no_capacity'
                    if instruction:
                        if positions[day]<instruction['target_index']:
                            wanted['status'] = 'waiting_extra_entry_delay'
                        else:
                            cap = independent_capacity(price,instruction['stop_ratio'],units[sid],nav,available,
                                                       slip,instruction['original_qty'])
                            wanted['execution_capacity'] = cap
                            fills = [t for t in additions if t['event_id']==eid]
                            filled = sum(t['qty'] for t in fills)
                            if filled > cap['qty'] or any(t['signal_date']!=instruction['signal_date'] or
                                    t['side']!='buy' or t['stock_id']!=sid for t in fills):
                                raise ValueError('Addition violates its frozen signal or quantity cap')
                            if cap['qty']:
                                selected = [p for p in add_plans if p['event_id']==eid]
                                if len(selected)!=1: raise ValueError('Addition lacks one resource reservation')
                                plan = selected[0];spent = cents(-sum(t['cash_change'] for t in fills))
                                expected = dict(date=day,signal_date=instruction['signal_date'],stock_id=sid,
                                    event_id=eid,opening_cash=opening_cash,available_before=available,
                                    budget=cap['budget'],planned_qty=cap['qty'],locked_unused_before=locked,
                                    kind='pyramid_add',spent=spent,locked_after=cents(locked+cap['budget']-spent),filled_qty=filled)
                                if any(plan[k]!=v for k,v in expected.items()) or sid not in plan['occupied_before']:
                                    raise ValueError('Addition reused cash or changed its capital reservation')
                                matched_plans.add((day,eid));locked = plan['locked_after']
                                cash = cents(cash-spent);remaining = cents(remaining-spent)
                            elif fills: raise ValueError('Zero-capacity addition filled')
                            units[sid] += filled
                            matched_trades.update(t['sequence'] for t in fills)
                            wanted.update(filled_qty=filled,status='filled' if filled else 'unfilled')
                            if filled: done.add(eid)
                            pending.pop(eid)
            if row != wanted:
                raise ValueError('Addition opportunity did not reconstruct: '+day+' '+eid)
        opening_cash = daily['cash']
    actual_plans = [(p['date'],p['event_id']) for p in case['resource_plans'] if p.get('kind')=='pyramid_add']
    if (len(actual_plans)!=len(matched_plans) or set(actual_plans)!=matched_plans or pending!=final_pending
            or matched_trades!={t['sequence'] for t in account['trades'] if t['reason']=='pyramid_add'}):
        raise ValueError('Unmatched addition plan, trade or final pending instruction')
    checked.update(addition_opportunities_rebuilt=True,addition_cash_reservations_rebuilt=True,
        addition_delay_and_quantity_rebuilt=True,one_addition_per_cohort=True,
        added_cohorts=len(done),addition_trade_count=len(matched_trades))
    return checked
