"""Rebuild technical decisions and quantities independently of execution hooks."""
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP

import pandas as pd

from skills.exit_mechanism_replay import audit_exits
from skills.exit_policy import decide_exit
from skills.scenario_exit_replay import _finite


def independent_loss(price, stop, qty, slippage, commission):
    if not qty:
        return 0.
    buy, sell = Decimal(str(price))*qty, Decimal(str(stop))*qty
    whole = lambda v, rule: v.quantize(Decimal('1'), rounding=rule)
    fee = lambda v: max(Decimal(20), whole(v*Decimal(str(commission)), ROUND_HALF_UP))
    slip = lambda v: whole(v*Decimal(str(slippage)), ROUND_CEILING)
    tax = whole(sell*Decimal('.003'), ROUND_FLOOR)
    cents = lambda v: v.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    return float(cents(cents(buy)+fee(buy)+slip(buy)-cents(sell)+fee(sell)+slip(sell)+tax))


def audit_support_risk(account, entries, supports, decisions, states, slots,
                       signals, mode, mask, prior_reference):
    dates = [str(day.date()) for day in signals.days]
    positions = {day:i for i,day in enumerate(dates)}
    use_support = mode in ('support20', 'support_risk2')
    use_risk = mode in ('risk2', 'support_risk2')
    if mode == 'control':
        if entries or supports:
            raise ValueError('Control must not have technical interventions')
        return audit_exits(account, decisions, states, signals.adjusted_close, signals.days, 'loss12', mask)
    daily = {row['date']:row for row in account['daily']}
    expected_entries = [(r['date'],r['event_id']) for r in slots]
    if [(r['date'],r['event_id']) for r in entries] != expected_entries or len(set(expected_entries)) != len(entries):
        raise ValueError('Technical entry records must cover every sizing attempt')
    fills = {(r['date'],r['event_id']) for r in account['trades'] if r['side']=='buy'}
    if not fills.issubset(set(expected_entries)):
        raise ValueError('Buy lacks original-signal sizing evidence')
    for row in entries:
        day, sid = row['date'], row['stock_id']
        original = positions.get(row['signal_date'])
        if original is None or original >= positions[day]:
            raise ValueError('Future or unknown original entry signal')
        context = signals.technical_context(original+1, sid)
        price = prior_reference(pd.Timestamp(day), sid)
        nav = daily[day]['opening_nav']
        requested = row['requested_qty']
        if type(requested) is not int or requested < 0:
            raise ValueError('Requested quantity is invalid')
        ratio, stop, loss, why = None, None, None, 'original_quantity'
        allowed = requested
        slip = .009 if mask & 1 else .0045
        if use_support and not context['risk_available']:
            allowed, why = 0, 'missing_or_nonpositive_support_distance'
        elif use_risk:
            if context['adjusted_close'] is None or not price:
                allowed, why = 0, 'missing_planned_risk_price'
            else:
                ratio = max(.88, context['support20']/context['adjusted_close']) if use_support else .88
                stop = price*ratio
                permitted = [q for q in range(0,requested//1000*1000+1,1000)
                    if independent_loss(price,stop,q,slip,account['settings']['commission']) <= nav*.02+1e-8]
                allowed = max(permitted)
                loss = independent_loss(price,stop,allowed,slip,account['settings']['commission'])
                why = 'risk_quantity_cap' if allowed < requested//1000*1000 else 'original_board_quantity'
        traded = [t for t in account['trades'] if (t['date'],t['event_id'])==(day,row['event_id']) and t['side']=='buy']
        filled = sum(t['qty'] for t in traded)
        wanted = dict(date=day, stock_id=sid, event_id=row['event_id'], signal_date=row['signal_date'],
            context=context, reference_price=price, opening_nav=nav, requested_qty=requested,
            allowed_qty=allowed, planned_stop_ratio=ratio, planned_stop_price=stop, planned_loss=loss,
            risk_budget=nav*.02, slippage=slip, reason=why, filled_qty=filled)
        if row != wanted or filled > allowed:
            raise ValueError('Original signal, fee-aware risk cap or filled quantity did not reconstruct')
    checked = dict(original_entry_signals_rebuilt=True, planned_quantity_caps_rebuilt=True,
                   scenario_fees_rebuilt=True, all_buy_quantities_checked=True)
    if not use_support:
        if supports:
            raise ValueError('Risk-only arm cannot change support exits')
        checked.update(audit_exits(account,decisions,states,signals.adjusted_close,signals.days,'loss12',mask))
        return checked
    cohorts = {r['event_id']:r for r in account['cohorts']}
    expected = {(d,c['event_id']) for d in daily for c in cohorts.values()
                if c['entry_date']<d and (c['exit_date'] is None or d<=c['exit_date'])}
    if (len(decisions)!=len(expected) or {(r['date'],r['event_id']) for r in decisions}!=expected
            or [(r['date'],r['event_id']) for r in supports] != [(r['date'],r['event_id']) for r in decisions]
            or decisions != sorted(decisions,key=lambda r:(r['date'],list(cohorts).index(r['event_id'])))):
        raise ValueError('Support decisions must cover all active cohort days in order')
    rebuilt, floors, delayed = {}, {}, set()
    for row, trace in zip(decisions,supports):
        day, identity, sid = row['date'], row['event_id'], row['stock_id']
        i = positions[day]
        cohort = cohorts[identity]
        if sid != cohort['stock_id']:
            raise ValueError('Support cohort stock differs')
        original = signals.technical_context(positions[cohort['signal_date']]+1,sid)
        if identity not in rebuilt:
            entry = positions[cohort['entry_date']]
            price = signals.price(entry,sid)
            rebuilt[identity] = dict(stock_id=sid,event_id=identity,entry_index=entry,
                entry_price=price,peak_price=price,trigger_reason=None,
                signal_date=None,target_date=None,target_index=None)
            if not original['risk_available']:
                raise ValueError('Support entry has no positive original support distance')
            floors[identity] = original['support20']
        state = rebuilt[identity]
        context = signals.context(i,sid,state)
        technical = signals.technical_context(i,sid)
        before = floors[identity]
        after = max(before,technical['support20']) if technical['support_available'] else before
        floors[identity] = after
        price = technical['adjusted_close']
        broken = price is not None and price<after
        previously = bool(state['trigger_reason'])
        if previously:
            decision = dict(exit=True,reason=state['trigger_reason'],phase='exiting',extend=False)
        else:
            decision = decide_exit(context,'loss12')
            if decision['reason']!='loss12' and broken:
                decision = dict(exit=True,reason='support20',phase='exiting',extend=False)
            if decision['exit']:
                state.update(trigger_reason=decision['reason'],signal_date=dates[i-1],target_date=day,target_index=i)
        wanted = dict(date=day,signal_date=dates[i-1],stock_id=sid,event_id=identity,mode='loss12',
            **context,**decision,signal_close=signals.price(i-1,sid),
            ma20=_finite(signals.ma20.iloc[i-1][sid]),market_state=signals.trend.state.iloc[i-1],
            entry_price=state['entry_price'],peak_price=state['peak_price'],
            first_signal_date=state['signal_date'],target_date=state['target_date'])
        if row!=wanted:
            raise ValueError('Support exit did not rebuild from strictly prior observations')
        if mask&4 and state['target_index'] is not None and identity not in delayed:
            state['original_target_date'] = state['target_date']
            state['target_index'] += 1
            state['target_date'] = dates[state['target_index']] if state['target_index']<len(dates) else None
            delayed.add(identity)
        wanted_trace = dict(date=day,event_id=identity,stock_id=sid,signal_date=technical['signal_date'],
            initial_signal_date=original['signal_date'],support_available=technical['support_available'],
            observed_support=technical['support20'],floor_before=before,floor_after=after,
            signal_close=price,broken=broken,previously_latched=previously,
            trigger_reason=state['trigger_reason'],first_signal_date=state['signal_date'],target_date=state['target_date'])
        if trace!=wanted_trace:
            raise ValueError('Support ratchet, missing history or once-only delay did not reconstruct')
    if rebuilt!=states:
        raise ValueError('Final support exit state differs from rebuilt history')
    for trade in account['trades']:
        if trade['side']!='sell':
            continue
        state = rebuilt.get(trade['event_id'])
        if (not state or state['target_index'] is None or positions[trade['date']]<state['target_index']
                or trade['reason']!=state['trigger_reason'] or trade['signal_date']!=state['signal_date']):
            raise ValueError('Sale violates its latched technical instruction')
    checked.update(support_ratchets_rebuilt=True, all_active_cohort_days_checked=True,
        exit_delay_rebuilt=True, no_sale_before_instruction=True,
        trigger_reasons=dict(Counter(s['trigger_reason'] for s in rebuilt.values() if s['trigger_reason'])))
    return checked
