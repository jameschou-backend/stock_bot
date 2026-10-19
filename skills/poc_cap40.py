"""One preregistered concentration reduction on the unchanged POC account."""
from copy import deepcopy
import math

import pandas as pd

from skills.partial_risk import CoreSlots

VARIANT = 'cap40'
REASON = 'reduce_cap40'


class Cap40:
    def __init__(self, *args, **kwargs):
        self.cap_pending, self.cap_due, self.cap_records = {}, {}, []
        super().__init__(*args, **kwargs)

    def corporate_day(self, day):
        # Temporary sale eligibility must never change the original holding clock.
        for h in self.holdings.values():
            if h['event_id'] in self.cap_due:
                h['due_index'] = self.cap_due[h['event_id']]
        self.cap_due = {}
        first_action = len(self.actions)
        income = super().corporate_day(day)
        index = self.positions[day]
        signal = str(self.days[index-1].date())
        previous = {h['event_id']: h for h in self.holding_rows if h['date'] == signal}
        for sid, h in self.holdings.items():
            eid = h['event_id']
            state = self.exit_states.get(eid)
            if not h['qty'] or state is None:
                continue
            pending = self.cap_pending.get(eid)
            for action in self.actions[first_action:]:
                if action.get('event_id') != eid or not pending:
                    continue
                if action['kind'] in ('split', 'capital_reduction', 'share_exchange'):
                    pending['keep'] = math.ceil(pending['keep']*action['qty_after']/action['entitled_qty'])
                elif action['kind'] == 'share_delivery':
                    pending['keep'] += action['qty']
            old = previous.get(eid)
            weight = old['market_value']/self.previous_nav if old else 0.
            reference = self.prior(day, sid) if weight > .4 and not state['trigger_reason'] else None
            before = deepcopy(pending)
            triggered = False
            if state['trigger_reason']:
                self.cap_pending.pop(eid, None)
                pending = None
            elif weight > .4:
                if not reference or not math.isfinite(reference):
                    raise ValueError('Cap40 requires an evidenced prior opening reference')
                target = max(1, math.floor(self.previous_nav/3/reference))
                if target < (pending['keep'] if pending else h['qty']):
                    pending = dict(keep=target, signal_date=signal)
                    self.cap_pending[eid] = pending
                    triggered = True
            requested = max(0, h['qty']-pending['keep']) if pending else 0
            if requested:
                self.cap_due[eid] = h['due_index']
                h['due_index'] = index
                self._plan(day, sid, 'sell', eid, pending['signal_date'],
                           requested//1000*1000, 0., self.opening_limit, None)
                plan = self.day_plans[(eid, 'sell')]
                plan.update(planned_qty=requested, board_qty=requested//1000*1000,
                            odd_qty=requested%1000)
                self.tick_plans[-1] = deepcopy(plan)
            self.cap_records.append(dict(date=str(day.date()), signal_date=signal,
                stock_id=sid, event_id=eid, opening_qty=h['qty'], prior_nav=self.previous_nav,
                prior_weight=weight, reference=reference, original_reason=state['trigger_reason'],
                before=before, after=deepcopy(pending), requested_qty=requested, triggered=triggered))
        return income

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        pending = self.cap_pending.get(event_id)
        partial = side == 'sell' and pending and not self.exit_states[event_id]['trigger_reason']
        if partial:
            reason, signal_date = REASON, pending['signal_date']
            qty = min(qty, max(0, self.holdings[sid]['qty']-pending['keep']))
        filled = super().order(day, sid, side, qty, reason, event_id, signal_date)
        if partial and self.holdings[sid]['qty'] <= pending['keep']:
            self.cap_pending.pop(event_id)
        return filled

    def run(self):
        account = super().run()
        account['settings']['concentration_control'] = dict(variant=VARIANT, trigger=.4,
            target=1/3, execution_lag=1, partial_core_occupies_slot=True,
            original_exits_first=True, denominator='prior_full_nav')
        account['cap40_records'] = self.cap_records
        def reference(sid, date):
            values = self.fields['close'][sid].loc[:pd.Timestamp(date)].iloc[:-1]
            values = values[values.gt(0) & values.map(math.isfinite)]
            if values.empty:
                raise ValueError('Missing independent cap40 reference')
            return self.corporate.reference_price(sid, date, float(values.iloc[-1]))
        account['cap40_audit'] = audit_cap40(account, self.exit_signals.adjusted_close, reference)
        return account


def audit_cap40(account, adjusted, reference=None):
    """Rebuild decisions from daily journals, corporate units and original exits.

    Execution references in sell plans are independently source-checked by the
    unchanged range auditor. The run also checks non-order references directly.
    """
    days = list(adjusted.index)
    indices = {str(d.date()): i for i, d in enumerate(days)}
    daily = {r['date']: r for r in account['daily']}
    held = {(r['date'], r['event_id']): r for r in account['holdings']}
    cohorts = {r['event_id']: r for r in account['cohorts']}
    plans = {(r['date'], r['event_id']): r for r in account['tick_plans'] if r['side']=='sell'}
    black = {(r['date'], r['event_id']): r['trigger'] for r in account.get('black_log', [])}
    observed = {(r['date'], r['event_id']): r for r in account['cap40_records']}
    if len(observed) != len(account['cap40_records']):
        raise ValueError('Duplicate cap40 decisions')
    pending, original, checked, triggers = {}, {}, set(), []
    for date, day in daily.items():
        i = indices[date]
        previous = str(days[i-1].date())
        for eid, cohort in cohorts.items():
            if cohort['entry_date'] >= date or (cohort['exit_date'] and cohort['exit_date'] < date):
                continue
            old = held.get((previous, eid))
            qty = old['qty'] if old else 0
            p = pending.get(eid)
            for a in account['corporate_actions']:
                if a['date'] != date or a.get('event_id') != eid:
                    continue
                if a['kind'] in ('split', 'capital_reduction', 'share_exchange'):
                    qty = a['qty_after']
                    if p: p['keep'] = math.ceil(p['keep']*a['qty_after']/a['entitled_qty'])
                elif a['kind'] == 'share_delivery':
                    qty += a['qty']
                    if p: p['keep'] += a['qty']
            if not qty:
                continue
            row = observed.get((date,eid))
            if row is None:
                raise ValueError('Missing cap40 decision')
            checked.add((date,eid))
            entry = indices[cohort['entry_date']]
            entry_price, close = adjusted[cohort['stock_id']].iloc[[entry,i-1]]
            reason = original.get(eid)
            if not reason:
                if pd.notna(entry_price) and pd.notna(close) and close/entry_price-1 <= -.12+1e-12:
                    reason = 'loss12'
                elif i-entry >= 63:
                    reason = 'time63'
                elif black.get((date,eid)):
                    reason = 'three_black'
                if reason: original[eid] = reason
            weight = old['market_value']/day['opening_nav'] if old else 0.
            if (row['opening_qty'] != qty or row['prior_nav'] != day['opening_nav']
                    or row['prior_weight'] != weight or row['original_reason'] != reason
                    or row['signal_date'] != previous or row['before'] != p):
                raise ValueError('Cap40 opening journal or original-exit priority differs')
            new = False
            if reason:
                p = None
            elif weight > .4:
                ref = row['reference']
                if reference and ref != reference(row['stock_id'],date):
                    raise ValueError('Cap40 source reference differs')
                if not isinstance(ref,(int,float)) or not math.isfinite(ref) or ref <= 0:
                    raise ValueError('Invalid cap40 prior reference')
                target = max(1, int(day['opening_nav']/(3*ref)))
                if target < (p['keep'] if p else qty):
                    p = dict(keep=target, signal_date=previous)
                    new = True
                    triggers.append(date)
            request = max(0,qty-p['keep']) if p else 0
            if row['after'] != p or row['triggered'] != new or row['requested_qty'] != request:
                raise ValueError('Cap40 target, trigger or pending quantity differs')
            trades = [r for r in account['trades'] if r['date']==date and r['event_id']==eid and r['side']=='sell']
            if request:
                plan = plans.get((date,eid))
                if (not plan or plan['planned_qty'] != request or plan['signal_date'] != p['signal_date']
                        or (row['reference'] is not None and plan['prior_reference'] != row['reference'])):
                    raise ValueError('Cap40 sale plan differs')
                if any(t['reason'] != REASON or t['signal_date'] != p['signal_date'] for t in trades):
                    raise ValueError('Cap40 execution differs')
                sold = sum(t['qty'] for t in trades)
                if sold > request or qty-sold < p['keep']:
                    raise ValueError('Cap40 sold retained shares')
                if sold == request: p = None
            elif any(t['reason']==REASON for t in trades):
                raise ValueError('Cap40 sale lacks a prior instruction')
            if p: pending[eid] = p
            else: pending.pop(eid,None)
    if checked != set(observed):
        raise ValueError('Extra cap40 decisions')
    return dict(passed=True, decisions_checked=len(checked), triggers=len(triggers),
                first_trigger=min(triggers,default=None), prior_only_decisions_rebuilt=True,
                original_exits_preserved=True, retained_core_occupies_slot=True)


def install(runner, variant, metadata):
    if variant != VARIANT:
        raise ValueError('Only preregistered cap40 may run')
    factory = runner.engine_types
    def engines(*args, **kwargs):
        stock, benchmark = factory(*args, **kwargs)
        return type('Cap40'+stock.__name__, (Cap40, stock, CoreSlots), {}), benchmark
    runner.engine_types = engines
    resource_audit = runner.audit_face_resources
    def audit(account, *args, **kwargs):
        # A reduction is not a liquidation attempt. All cash/units/trades remain
        # in the independent corporate and execution audits.
        view = dict(account, orders=[r for r in account['orders'] if r['reason'] != REASON])
        return resource_audit(view, *args, **kwargs)
    runner.audit_face_resources = audit
    writer = runner.write
    def write(path, value):
        if 'account' in value or str(path).endswith('/report.json'):
            value['risk_controls'] = metadata
        return writer(path,value)
    runner.write = write
