"""Reconstruct ETF cash, units, planning inputs and order lineage independently."""
from collections import defaultdict
from decimal import Decimal,ROUND_HALF_UP,ROUND_FLOOR,ROUND_CEILING
from copy import deepcopy
import math

def audit_index(value,data):
    account=value['account'];window=value['config']['window'];mask=value['config']['factor_mask']
    slip=Decimal('.009' if mask&1 else '.0045');cash=Decimal(1000000);qty=0;peak=cash;mark=Decimal(0)
    def fail(condition,message):
        if not condition:raise ValueError(message)
    def same(a,b,message):fail(math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-6),message)
    def charge(price,quantity,side):
        gross=Decimal(str(price))*quantity
        fee=max(Decimal(20),(gross*Decimal('.001425')).quantize(Decimal(1),rounding=ROUND_HALF_UP))
        tax=(gross*Decimal('.001')).quantize(Decimal(1),rounding=ROUND_FLOOR) if side=='sell' else Decimal(0)
        friction=(gross*slip).quantize(Decimal(1),rounding=ROUND_CEILING)
        return gross,fee,tax,friction
    def budget_qty(maximum,price,budget):
        quantity=maximum
        while quantity:
            gross,fee,_,friction=charge(price,quantity,'buy')
            if gross+fee+friction<=Decimal(str(budget))+Decimal('.00000001'):break
            quantity-=1000
        return quantity
    days=data['days'];calendar=data['calendar'];quotes=data['quotes'];signals=data['signals']
    fail([r['date'] for r in account['daily']]==days,'Incomplete daily account')
    fail([r['date'] for r in value['decisions']]==days,'Missing decision opportunities')
    plans={r['order_id']:r for r in value['plans']}
    fail(len(plans)==len(value['plans']) and sorted(plans)==list(range(1,len(plans)+1)),'Duplicate/missing plan')
    by_trade=defaultdict(list);by_order=defaultdict(list);by_hold=defaultdict(list);by_action=defaultdict(list)
    for row in account['trades']:by_trade[row['date']].append(row)
    for row in account['orders']:by_order[row['date']].append(row)
    for row in account['holdings']:by_hold[row['date']].append(row)
    for row in account['corporate_actions']:by_action[row['date']].append(row)
    fail(all(set(rows)<=set(days) for rows in (by_trade,by_order,by_hold,by_action)),'Journal outside interval')
    ledger=[dict(date=days[0],kind='initial_deposit',cash_change=1000000.,cash_after=1000000.)]
    pending=None;latched=False;retry=False;previous_trend=False;previous_nav=Decimal(1000000);mark_date=None
    created=[];executed=0
    for i,(day,daily,decision) in enumerate(zip(days,account['daily'],value['decisions'])):
        if day=='2026-03-31':
            rows=by_action[day];fail(len(rows)==1,'Missing split journal');action=rows[0]
            fail(action['known_date']=='2026-03-30' and action['multiplier']==22,'Wrong split terms')
            fail(action['old_qty']==qty and action['new_qty']==qty*22,'Split shares mismatch')
            same(action['old_mark'],mark,'Split old mark');same(action['new_mark'],mark/22,'Split new mark')
            fail(action['pending_before']==pending,'Pending split source differs')
            qty*=22;mark/=22
            if pending:
                pending['qty']*=22;pending['reference_price']/=22
                pending['split_adjustments'].append(dict(date=day,multiplier=22))
            fail(action['pending_after']==pending,'Pending split conversion differs')
        else:fail(not by_action[day],'Unannounced corporate action')
        position=calendar.index(day);prior_days=calendar[:position];signal=prior_days[-1]
        history=[d for d in sorted(signals) if d<day]
        history=history[-window:] if window else history[-1:]
        fail(len(history)==(window or 1),'Signal warmup short')
        close=signals[history[-1]];sma=math.fsum(signals[d] for d in history)/len(history)
        trend=window==0 or close>sma
        ref_date=next(d for d in reversed(prior_days) if d in quotes)
        reference=quotes[ref_date]['close']/(22 if ref_date<'2026-03-31'<=day else 1)
        vols=[quotes.get(d,{}).get('volume',0)*(22 if d<'2026-03-31'<=day else 1) for d in prior_days[-20:]]
        adv=math.fsum(vols)/20 if len(vols)==20 else None
        ctx=dict(signal_date=signal,signal_price_date=history[-1],signal_close=close,sma=sma,
            signal_first_observation=history[0],observations=len(history),trend=trend,
            reference_date=ref_date,reference_price=reference,prior_avg_volume20=adv,
            month_first=day[:7]!=signal[:7])
        fail(decision['context']==ctx,'Prior signal/liquidity inputs differ')
        same(decision['opening_cash'],cash,'Opening cash');fail(decision['opening_qty']==qty,'Opening shares')
        fail(decision['pending_before']==pending and decision['exit_latched_before']==latched,'Order state changed')
        cancelled=None
        if pending and pending['side']=='buy' and not trend:cancelled=pending;pending=None;retry=False
        if not trend and qty and not latched:
            latched=True
            if pending:cancelled=pending;pending=None
        fail(decision.get('cancelled_order')==cancelled,'Missing/incorrect cancellation')
        should_plan=pending is None and (latched or (trend and (i==0 or qty==0 or ctx['month_first'] or not previous_trend or retry)))
        if should_plan:
            plan=decision.get('created_plan');fail(plan is not None,'Missing eligible rebalance plan')
            created.append(plan);nav=float(cash)+qty*reference
            if latched:side,quantity,reason='sell',qty,'trend_off'
            elif .75*nav-qty*reference>=0:
                budget=min(.75*nav-qty*reference,float(cash));side='buy';reason='initial_or_monthly_or_reentry'
                maximum=int((.75*nav-qty*reference)/reference)//1000*1000
                quantity=budget_qty(maximum,reference,budget)
            else:
                side='sell';reason='initial_or_monthly_or_reentry'
                quantity=qty-int(.75*nav/reference)//1000*1000
            fail(plan['side']==side and plan['qty']==quantity and plan['original_qty']==quantity and plan['reason']==reason,'Planned sizing differs')
            fail(plan['date']==day and plan['signal_date']==signal and plan['context']==ctx and not plan['split_adjustments'],'Plan data leakage')
            same(plan['reference_price'],reference,'Plan reference');same(plan['original_reference_price'],reference,'Original reference')
            same(plan['prior_nav'],nav,'Plan NAV');same(plan['available_cash'],cash,'Plan cash')
            fail(plan['prior_qty']==qty and plan['due_index']==i+(bool(mask&2) if side=='buy' else bool(mask&4)),'Plan delay/units')
            pending=deepcopy(plan) if quantity else None;retry=False
        else:fail('created_plan' not in decision,'Unscheduled plan')
        rows=by_order[day];trades=by_trade[day]
        if pending and i>=pending['due_index']:
            executed+=1;fail(len(rows)==1 and len(trades)<=1,'Order/fill cardinality differs')
            order=rows[0];fail(decision['executing_instruction']==pending and decision['execution']==order,'Execution lineage differs')
            fail(order['order_id']==pending['order_id'] and order['requested_qty']==pending['qty'] and order['signal_date']==pending['signal_date']<day,'Future signal or inflated pending units')
            fail(order['side']==pending['side'] and order['reason']==pending['reason'],'Order semantics changed')
            quote=quotes.get(day);failure=None;cap=0;filled=0
            if not quote or not quote['volume']:failure='missing_or_zero_quote_volume'
            elif quote['high']==quote['low']:failure='single_price_session'
            elif not quote.get('lower') or not quote.get('upper'):failure='missing_price_limits'
            elif order['side']=='buy' and quote['close']>=quote['upper']-1e-8:failure='at_upper_limit'
            elif order['side']=='sell' and quote['close']<=quote['lower']+1e-8:failure='at_lower_limit'
            elif not adv:failure='missing_adv20'
            else:
                cap=int(min(adv,quote['volume'])*.01)//1000*1000
                maximum=min(order['requested_qty'],cap)
                filled=budget_qty(maximum,quote['close'],cash) if order['side']=='buy' else min(maximum,qty)
                if filled<order['requested_qty']:failure='partial_capacity_or_cash' if filled else 'capacity_or_cash_zero'
            fail(order['capacity_qty']==cap and order['filled_qty']==filled and order['failure']==failure,'Execution rejection/capacity/cash mismatch')
            fail(decision['status']==('filled' if filled else 'unfilled'),'Decision execution status')
            fail(len(trades)==int(filled>0),'Missing/spurious fill')
            if filled:
                trade=trades[0];fail(trade['qty']==filled and filled%1000==0 and trade['stock_id']=='00631L','Trade identity/units')
                fail(trade['side']==order['side'] and trade['signal_date']==order['signal_date'] and trade['order_id']==order['order_id'],'Fill detached from order')
                same(trade['reference_price'],quote['close'],'Fill price differs from raw source')
                gross,fee,tax,friction=charge(quote['close'],filled,trade['side']);total=fee+tax+friction
                change=(-gross if trade['side']=='buy' else gross)-total
                for key,amount in dict(gross=gross,commission=fee,tax=tax,slippage=friction,total_cost=total,cash_change=change).items():same(trade[key],amount,'Fee/cash mismatch '+key)
                cash+=change;qty+=filled if trade['side']=='buy' else -filled
                same(trade['cash_after'],cash,'Trade cash balance');fail(trade['remaining_shares']==qty,'Trade shares')
                ledger.append(dict(date=day,kind=trade['side'],cash_change=float(change),cash_after=float(cash),stock_id='00631L',order_id=trade['order_id']))
            retry=filled<pending['qty'];pending=None
            if latched and qty==0:latched=False;retry=False
        else:
            fail(not rows and not trades,'Trade on a waiting/unscheduled day')
            fail(decision['status']==('waiting_extra_delay' if pending else 'hold'),'Waiting/hold status')
        if day in quotes:mark=Decimal(str(quotes[day]['close']));mark_date=day
        nav=cash+qty*mark;peak=max(peak,nav)
        fail(cash>=0 and qty>=0,'Negative cash or units')
        for key,amount in dict(nav=nav,cash=cash,market_value=qty*mark,opening_nav=previous_nav,
            total_return=nav/1000000-1,daily_return=nav/previous_nav-1,drawdown=nav/peak-1).items():same(daily[key],amount,'Daily journal differs '+key)
        fail(daily['receivable']==0 and daily['stale_holdings']==int(qty>0 and mark_date!=day),'Valuation stale/receivable mismatch')
        held=by_hold[day];fail(len(held)==int(qty>0),'Missing holding valuation')
        if qty:
            fail(held[0]['qty']==qty and held[0]['mark_date']==mark_date,'Holding quantity/date')
            same(held[0]['price'],mark,'Holding price');same(held[0]['market_value'],qty*mark,'Holding value')
        same(decision['closing_cash'],cash,'Decision closing cash');fail(decision['closing_qty']==qty,'Decision closing shares')
        fail(decision['pending_after']==pending and decision['exit_latched_after']==latched,'Closing instruction state')
        previous_nav=nav;previous_trend=trend
    fail(created==value['plans'],'Plan journal includes missing/extra decisions')
    fail(ledger==account['cash_ledger'],'Cash ledger differs')
    fail(value['pending']==pending,'Final pending state')
    return dict(cash_and_shares_rebuilt=True,fees_and_etf_tax_rebuilt=True,all_opportunities_rebuilt=True,
        prior_signals_only=True,split_and_pending_units_reconciled=True,examined_days=len(days),
        examined_orders=executed,observed_intraday_fills=False,official_historical_limits=False)
