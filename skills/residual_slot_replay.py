"""Release exited sub-lot remnants without removing any account assets."""
from collections import defaultdict
from copy import deepcopy
import math
import pandas as pd

from skills.execution_factorial import FactorialReplay
from skills.slot_reuse_replay import SlotReuseReplay
from skills.residual_account_loop import ResidualAccountLoop
from skills.million_replay import money
from skills.execution_resources import audit_resources
from skills.board_only_verified_replay import audit_verified_board_only

POLICIES = ('keep', 'release')
RESIDUAL_CAP = .05


def residual_values(holdings, rights, marks, exited):
    pending = defaultdict(int)
    for row in rights:
        if row['kind'] == 'shares':
            pending[row['stock_id']] += row['qty']
    released = {}
    for sid, holding in holdings.items():
        if sid == '0050' or holding['event_id'] not in exited:
            continue
        total = holding['qty'] + pending[sid]
        if not 0 < total < 1000:
            continue
        mark = marks.get(sid)
        if not mark or not math.isfinite(mark['price']) or mark['price'] <= 0:
            raise ValueError('Residual position has no prior valuation: ' + sid)
        released[sid] = dict(qty=holding['qty'], pending_qty=pending[sid],
            prior_price=mark['price'], prior_mark_date=mark['date'],
            value=total*mark['price'], event_id=holding['event_id'])
    return released


class ResidualSlotDecisions(SlotReuseReplay):
    def corporate_day(self, day):
        if self.residual_policy == 'keep':
            return super().corporate_day(day)
        exited = {r['event_id'] for r in self.orders if r['side'] == 'sell' and r['date'] < str(day.date())}
        released = residual_values(self.holdings, self.receivables, self.marks, exited)
        if any(r['prior_mark_date'] >= str(day.date()) for r in released.values()):
            raise ValueError('Residual sizing used a current or future price')
        value = sum(r['value'] for r in released.values())
        self.residual_budget = max(0., (self.previous_nav-value)/self.slots)
        self.residual_block = value > self.previous_nav * RESIDUAL_CAP
        self.released_residuals = set(released)
        opening = set(self.holdings) - {'0050'} - self.released_residuals
        self.residual_days.append(dict(date=str(day.date()), opening_nav=self.previous_nav,
            released=deepcopy(released), residual_value=value, new_position_budget=self.residual_budget,
            block_new_buys=self.residual_block, opening_active=sorted(opening)))
        income = super().corporate_day(day)
        # Parent wrappers apply today's rights and exits; release eligibility
        # above was fixed beforehand, and cannot be enlarged intraday.
        self.opening_members = opening
        self.occupied = self.entry_slot_members()
        return income

    def entry_slot_members(self):
        current = set(self.holdings) - {'0050'}
        if self.residual_policy == 'keep':
            return current
        pending = defaultdict(int)
        for row in self.receivables:
            if row['kind'] == 'shares':
                pending[row['stock_id']] += row['qty']
        excluded = {s for s in self.released_residuals if s in self.holdings
                    and self.holdings[s]['qty'] + pending[s] < 1000}
        return current - excluded

    def _affordable(self, qty, step, price, cash, sid):
        if self.residual_spend_left is not None:
            cash = min(cash, self.residual_spend_left)
        return super()._affordable(qty, step, price, cash, sid)

    def cash_move(self, day, kind, change, **extra):
        if kind == 'buy' and self.residual_spend_left is not None:
            if -change > self.residual_spend_left + .005:
                raise ValueError('Buy exceeded residual-adjusted position budget')
            self.residual_spend_left = money(self.residual_spend_left + change)
        return super().cash_move(day, kind, change, **extra)

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if self.residual_policy == 'keep' or side != 'buy' or sid == '0050':
            return super().order(day, sid, side, qty, reason, event_id, signal_date)
        if sid not in self.holdings or self.holdings[sid]['qty'] != 0:
            raise ValueError('New entry requires its provisional zero holding')
        current = self.entry_slot_members() - {sid}
        occupied = current | self.opening_members | self.attempted_members
        self.occupied = occupied
        record = dict(date=str(day.date()), stock_id=sid, event_id=event_id, signal_date=signal_date,
            opening_members=sorted(self.opening_members), held_before=sorted(current),
            attempts_before=sorted(self.attempted_members), unfilled_before=sorted(self.failed_members),
            occupied_before=sorted(occupied), new_position_budget=self.residual_budget,
            released_residuals=sorted(self.released_residuals), blocker_categories=[])
        before = len(self.resource_plans)
        if self.residual_block:
            self.orders.append(dict(date=str(day.date()), stock_id=sid, name=self.names.get(sid,sid),
                event_id=event_id, signal_date=signal_date, side='buy', channel='event',
                requested_qty=qty, filled_qty=0, reason=reason, failure='residual_exposure_cap'))
            filled, blocked = 0, 'residual_exposure_cap'
        else:
            self.residual_spend_left = self.residual_budget
            try:
                # Replace only SlotReuseReplay's membership calculation. All
                # downstream cash, reservation and execution layers still run.
                filled = super(SlotReuseReplay, self).order(day,sid,side,qty,reason,event_id,signal_date)
            finally:
                self.residual_spend_left = None
            plan = self.resource_plans[-1] if len(self.resource_plans) > before else None
            blocked = plan.get('failure') if plan else None
        attempted = blocked is None
        record.update(filled_qty=filled, attempted=attempted, failure=blocked)
        if blocked == 'resource_slots_locked':
            if self.opening_members-current:
                record['blocker_categories'].append('opening_slot_not_released')
            if self.failed_members-current:
                record['blocker_categories'].append('unfilled_attempt_not_released')
        if attempted:
            self.attempted_members.add(sid)
            if not filled:
                self.failed_members.add(sid)
        self.slot_decisions.append(record)
        return filled


class ResidualSlotReplay(FactorialReplay, ResidualSlotDecisions, ResidualAccountLoop):
    def __init__(self, *args, residual_policy, **kwargs):
        if residual_policy not in POLICIES:
            raise ValueError('Unknown residual slot policy')
        self.residual_policy = residual_policy
        self.residual_days = []
        self.released_residuals = set()
        self.residual_spend_left = None
        super().__init__(*args, **kwargs)

    def run(self):
        result = super().run()
        if self.residual_policy == 'release':
            result['settings'].update(residual_slot_policy='exited_sub_lot_next_session_v1',
                                      residual_exposure_cap=RESIDUAL_CAP)
        return result


def audit_residual_slots(account, plans, decisions, board, snapshots, quotes):
    checked = audit_resources(account, plans, opening_cash_only=True, lock_slots=True, lock_unused=True)
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
