"""Residual resource audit for mixed channels; sealed board-only source unchanged."""
import math
import pandas as pd
from skills.execution_resources import audit_resources
from skills.residual_slot_replay import residual_values, RESIDUAL_CAP

def audit_mixed_resources(account, plans, decisions, board, snapshots, quotes):
    checked = audit_resources(account, plans, opening_cash_only=True, lock_slots=True, lock_unused=True)
    dates = [r['date'] for r in account['daily']]
    if [r['date'] for r in snapshots] != dates:
        raise ValueError('Residual snapshots must cover every market day')
    holdings, pending, attempted_exits = {}, {}, set()
    source = quotes[['date', 'stock_id', 'close']].copy()
    source['date'] = pd.to_datetime(source['date']).dt.strftime('%Y-%m-%d')
    if source.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate valuation source')
    source = iter(source.sort_values('date').itertuples(index=False))
    quote = next(source, None)
    prior_quotes = {}
    for day, snapshot in zip(dates, snapshots):
        # Outstanding stock rights continue to be marked after physical shares
        # have sold. Rebuild those prices from frozen quotes, strictly before
        # this session, rather than a now-absent physical holding row.
        while quote is not None and quote.date < day:
            if math.isfinite(float(quote.close)) and quote.close > 0:
                prior_quotes[quote.stock_id] = dict(price=float(quote.close), date=quote.date)
            quote = next(source, None)
        # Rebuild opening shares and undelivered rights from the completed
        # journals; engine snapshots are never accepted as opening evidence.
        open_cohorts = {c['stock_id']: c for c in account['cohorts'] if c['entry_date'] < day
                        and (c['exit_date'] is None or c['exit_date'] >= day)}
        opening = {sid: dict(qty=holdings.get(sid, {}).get('qty', 0), event_id=c['event_id'])
                   for sid, c in open_cohorts.items()}
        marks = {sid: dict(price=h['price'], date=h['mark_date']) for sid,h in holdings.items()}
        # A zero holding can retain a last mark while awaiting late stock rights.
        for sid in opening:
            if sid not in marks:
                previous = [h for h in account['holdings'] if h['stock_id']==sid and h['date']<day]
                if previous:
                    marks[sid] = dict(price=previous[-1]['price'], date=previous[-1]['mark_date'])
                if sid in prior_quotes and (sid not in marks or prior_quotes[sid]['date'] >= marks[sid]['date']):
                    marks[sid] = prior_quotes[sid]
        wanted = residual_values(opening, list(pending.values()), marks, attempted_exits)
        value = sum(r['value'] for r in wanted.values())
        nav = next(r['opening_nav'] for r in account['daily'] if r['date']==day)
        active = set(opening)-set(wanted)
        if (snapshot['released'] != wanted or snapshot['opening_active'] != sorted(active)
                or abs(snapshot['residual_value']-value)>.01
                or snapshot['block_new_buys'] != (value > nav*RESIDUAL_CAP)
                or abs(snapshot['new_position_budget']-(nav-value)/5)>.01):
            raise ValueError('Residual opening classification differs from journals: '+day
                             + ' expected=' + repr(wanted) + ' observed=' + repr(snapshot))
        for row in account['corporate_actions']:
            if row['date'] != day:
                continue
            if row['kind'] == 'stock_dividend':
                pending[row['action_id']] = dict(kind='shares', stock_id=row['stock_id'], qty=row['whole_new_shares'])
            elif row['kind'] == 'share_delivery':
                pending.pop(row['action_id'])
        day_decisions = [r for r in decisions if r['date']==day]
        attempts, failed = set(), set()
        # Remaining active members after sales; released positions stay in the
        # account. Opening members remain locked even if sold during the day.
        current = set(opening)
        for sid in list(current):
            qty = opening[sid]['qty']
            for row in account['corporate_actions']:
                if row['date']==day and row['stock_id']==sid:
                    if row['kind']=='split': qty=row['qty_after']
                    elif row['kind']=='share_delivery': qty += row['qty']
            qty -= sum(t['qty'] for t in account['trades'] if t['date']==day and t['stock_id']==sid and t['side']=='sell')
            rights = sum(p['qty'] for p in pending.values() if p['stock_id']==sid)
            if (qty == 0 and rights == 0) or (sid in wanted and qty+rights < 1000):
                current.remove(sid)
        for record in day_decisions:
            expected = current | active | attempts
            if (record['occupied_before'] != sorted(expected) or record['held_before'] != sorted(current)
                    or record['attempts_before'] != sorted(attempts) or record['unfilled_before'] != sorted(failed)):
                raise ValueError('Residual slot membership did not reconstruct: '+day)
            buys = [t for t in account['trades'] if t['date']==day and t['event_id']==record['event_id'] and t['side']=='buy']
            spent = sum(-t['cash_change'] for t in buys)
            if spent > snapshot['new_position_budget']+.02 or (buys and snapshot['block_new_buys']):
                raise ValueError('Residual risk budget breached')
            if record['filled_qty'] != sum(t['qty'] for t in buys):
                raise ValueError('Residual entry differs from fills')
            if record['attempted']:
                if len(expected)>=5 or record['stock_id'] in expected:
                    raise ValueError('Residual entry exceeded active slots')
                attempts.add(record['stock_id'])
                if buys: current.add(record['stock_id'])
                else: failed.add(record['stock_id'])
        attempted_exits |= {r['event_id'] for r in account['orders'] if r['date']==day and r['side']=='sell'}
        holdings = {r['stock_id']:r for r in account['holdings'] if r['date']==day}
    checked.update(residual_classification_rebuilt=True, residual_assets_retained=True,
                   residual_risk_budget_rebuilt=True, active_slots_rebuilt=True)
    return checked


def audit_mixed_execution(account,ticks,odd_feeds,markets,quotes,calendar,corporate,feeds):
    from copy import deepcopy
    from skills.mixed_odd_replay import sized_quantity,odd_match
    from skills.execution_stress import StressOrder
    from skills.residual_tick_replay import audit_tick_plans
    from skills.opening_entry_replay import opening_match
    from skills.intraday_limit_replay import match_ticks
    base=dict(account,tick_plans=account['base_tick_plans'],orders=[],trades=[])
    audit_tick_plans(base,ticks,markets,quotes,calendar,corporate)
    cost=StressOrder();cost.stress_slippage=account['settings']['slippage']
    price=quotes.pivot(index='date',columns='stock_id',values='open').reindex(calendar)
    volume=quotes.pivot(index='date',columns='stock_id',values='volume').reindex(calendar)
    adv=volume.rolling(20,min_periods=20).mean().shift(1)
    close=quotes.pivot(index='date',columns='stock_id',values='close').reindex(calendar)
    amount=(close*volume).rolling(20,min_periods=20).mean().shift(1)
    plans={}
    if len(account['base_tick_plans'])!=len(account['tick_plans']):raise ValueError('Mixed plan count differs')
    for old,p in zip(account['base_tick_plans'],account['tick_plans']):
        expected=deepcopy(old);budget=p['sizing_budget'];sid=p['stock_id']
        if not math.isfinite(budget) or not 0<=budget<=old['reserved_cash']:raise ValueError('Mixed budget grew')
        expected.update(odd_limit=None,sizing_budget=budget,order_time='08:59:00',odd_order_time='09:00:00',odd_expires_at='13:30:00')
        eligible=(p['side']=='sell' or (old['reserved_cash']>0 and old['prior_reference'] and old['rejection'] in (None,'cash_below_one_lot_or_missing_prior')))
        if eligible:
            limits=feeds.get_limits(sid)[p['date']]
            expected['odd_limit']=limits['upper'] if p['side']=='buy' else limits['lower']
            if p['side']=='buy':
                maximum=max(0,math.floor((budget-40)/(old['prior_reference']*(1+.001425+.0045))))
                n=sized_quantity(maximum,limits['upper'],budget,cost,sid)
                expected.update(planned_qty=n,limit_price=limits['upper'],rejection=None if n else 'mixed_budget_below_one_share')
            else:
                if p['planned_qty']//1000*1000!=old['planned_qty']:raise ValueError('Mixed sell added unavailable board shares')
                expected['planned_qty']=p['planned_qty']
        expected.update(board_qty=expected['planned_qty']//1000*1000,odd_qty=expected['planned_qty']%1000)
        if p!=expected:raise ValueError('Mixed plan differs from prior-only sizing')
        plans[(p['date'],p['event_id'],p['side'])]=p
    fills={};seen=set();orders={}
    for row in account['orders']:
        if row['channel'] not in ('board','odd') or not row['requested_qty']:continue
        key=(row['date'],row['event_id'],row['side']);p=plans[key];channel=row['channel'];child=(*key,channel)
        if child in seen:raise ValueError('Mixed channel volume reused')
        seen.add(child);orders[child]=row
        n=p[channel+'_qty'];sid=row['stock_id'];day=pd.Timestamp(row['date'])
        limit=p['limit_price'] if channel=='board' else p['odd_limit']
        if row['prior_avg_amount20']!=amount.at[day,sid] or row['prior_avg_volume20']!=adv.at[day,sid]:
            raise ValueError('Mixed prior-only liquidity changed')
        if n!=row['requested_qty'] or limit!=row['limit_price']:raise ValueError('Mixed child changed frozen plan')
        if channel=='odd':
            source=odd_feeds.get_odd(row['date'],sid,markets.get((sid,row['date']),markets.get(sid)))
            rebuilt=odd_match(source,row['side'],limit,n,account['settings']['participation'])
        else:
            tape,digest=ticks.get(sid,row['date'],markets.get((sid,row['date']),markets.get(sid)))
            if digest!=row['ticks_sha256'] or row['prior_avg_volume20']!=adv.at[day,sid]:raise ValueError('Board source or prior volume changed')
            limits=feeds.get_limits(sid)[row['date']]
            if row['side']=='buy':rebuilt=opening_match(tape,price.at[day,sid],limit,n,adv.at[day,sid],account['settings']['participation'])
            elif not limits['lower']<=limit<=limits['upper']:
                rebuilt=dict(filled_qty=0,reference_price=limit,failure='precommitted_limit_outside_legal_range')
            else:rebuilt=match_ticks(tape,'sell',limit,n,adv.at[day,sid],account['settings']['participation'])
        if any(row[k]!=v for k,v in rebuilt.items() if k!='failure'):raise ValueError('Mixed execution did not reproduce')
    for t in account['trades']:
        child=(t['date'],t['event_id'],t['side'],t['channel']);row=orders.get(child)
        if not row or t['reference_price']!=row['reference_price']:raise ValueError('Mixed fill has no price evidence')
        paid=cost._costs(t['reference_price'],t['qty'],t['side'],t['stock_id'])
        if any(t[k]!=v for k,v in paid.items()):raise ValueError('Mixed per-channel costs differ')
        fills[child]=fills.get(child,0)+t['qty']
    if any(fills.get(k,0)!=r['filled_qty'] for k,r in orders.items()):raise ValueError('Mixed child fills disagree')
    if (account['settings'].get('odd_tick_verified') is not False or account['settings'].get('live_qualified') is not False
            or account['settings'].get('odd_execution_evidence')!='daily_envelope_estimate'):raise ValueError('Estimated odds mislabeled as verified')
    return dict(precommitted_mixed_plans=True,channel_fills_rebuilt=True,independent_odd_prices=True,
                separate_channel_costs=True,daily_odd_estimate_only=True)
