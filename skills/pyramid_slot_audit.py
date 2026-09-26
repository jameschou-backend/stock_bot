"""Residual audit derivative: additions retain a held slot, new entries do not.

The remaining journal reconstruction is copied from residual_slot_replay;
the sealed original is unchanged. Full cash/share audits always see ALL trades.
"""
from collections import defaultdict
import math
import pandas as pd
from skills.residual_slot_replay import residual_values, RESIDUAL_CAP
from skills.execution_resources import audit_resources
from skills.board_only_verified_replay import audit_verified_board_only


def audit_pyramid_slots(account, plans, decisions, board, snapshots, quotes):
    checked = audit_resources(account, plans, opening_cash_only=True, lock_slots=False, lock_unused=True)
    for plan in plans:
        if plan.get('filled_qty') and plan.get('kind') != 'pyramid_add':
            if plan['stock_id'] in plan['occupied_before'] or len(plan['occupied_before']) >= 5:
                raise ValueError('Original entry reused a locked slot')
    checked.update(audit_verified_board_only(account, board, plans))
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
