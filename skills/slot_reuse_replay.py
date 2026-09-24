"""Split released opening positions from unfilled same-day buy attempts."""
from collections import defaultdict

from skills.execution_resources import ResourceCapacityReplay, audit_resources


class SlotReuseReplay(ResourceCapacityReplay):
    def __init__(self, *args, lock_opening_slots=False, lock_failed_slots=False, **kwargs):
        if 'lock_slots' in kwargs:
            raise ValueError('Use the two explicit slot policies')
        self.lock_opening_slots = lock_opening_slots
        self.lock_failed_slots = lock_failed_slots
        self.slot_decisions = []
        super().__init__(*args, lock_slots=lock_opening_slots or lock_failed_slots, **kwargs)

    def corporate_day(self, day):
        self.opening_members = {sid for sid in self.holdings if sid != '0050'}
        self.attempted_members = set()
        self.failed_members = set()
        return super().corporate_day(day)

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if side != 'buy' or sid == '0050':
            return super().order(day, sid, side, qty, reason, event_id, signal_date)
        # The sealed daily loop has already created this order's provisional
        # zero holding. Every other holding, including stock rights, occupies.
        if sid not in self.holdings or self.holdings[sid]['qty'] != 0:
            raise ValueError('New entry requires its provisional zero holding')
        current = {s for s in self.holdings if s not in ('0050', sid)}
        occupied = set(current)
        if self.lock_opening_slots:
            occupied |= self.opening_members
        if self.lock_failed_slots:
            occupied |= self.attempted_members
        self.occupied = occupied
        record = dict(date=str(day.date()), stock_id=sid, event_id=event_id,
            signal_date=signal_date, opening_members=sorted(self.opening_members),
            held_before=sorted(current), attempts_before=sorted(self.attempted_members),
            unfilled_before=sorted(self.failed_members), occupied_before=sorted(occupied))
        before = len(self.resource_plans)
        filled = super().order(day, sid, side, qty, reason, event_id, signal_date)
        plan = self.resource_plans[-1] if len(self.resource_plans) > before else None
        blocked = plan.get('failure') if plan else None
        attempted = blocked is None
        record.update(filled_qty=filled, attempted=attempted, failure=blocked)
        record['blocker_categories'] = []
        if blocked == 'resource_slots_locked':
            if self.lock_opening_slots and self.opening_members-current:
                record['blocker_categories'].append('opening_slot_not_released')
            if self.lock_failed_slots and self.failed_members-current:
                record['blocker_categories'].append('unfilled_attempt_not_released')
        if attempted:
            self.attempted_members.add(sid)
            if not filled:
                self.failed_members.add(sid)
        self.slot_decisions.append(record)
        return filled


def audit_slots(account, plans, decisions, *, lock_opening_slots, lock_failed_slots,
                opening_cash_only, lock_unused):
    checked = audit_resources(account, plans, opening_cash_only=opening_cash_only,
        lock_unused=lock_unused, lock_slots=lock_opening_slots or lock_failed_slots)
    byday = defaultdict(list)
    for row in decisions:
        byday[row['date']].append(row)
    planned_keys = {(r['date'], r['stock_id'], r['event_id']) for r in decisions}
    for trade in account['trades']:
        if trade['side']=='buy' and trade['stock_id']!='0050' and (trade['date'],trade['stock_id'],trade['event_id']) not in planned_keys:
            raise ValueError('Buy has no slot decision')
    for day, rows in byday.items():
        opening = {c['stock_id'] for c in account['cohorts'] if c['entry_date'] < day
            and (c['exit_date'] is None or c['exit_date'] >= day)}
        closed = {c['stock_id'] for c in account['cohorts'] if c['entry_date'] < day and c['exit_date'] == day}
        current, attempted, failed = opening-closed, set(), set()
        for row in rows:
            expected = current | (opening if lock_opening_slots else set()) | (attempted if lock_failed_slots else set())
            for key, want in [('opening_members',opening),('held_before',current),('attempts_before',attempted),
                              ('unfilled_before',failed),('occupied_before',expected)]:
                if row[key] != sorted(want):
                    raise ValueError('Slot ledger reconstruction differs: '+key)
            filled = sum(t['qty'] for t in account['trades'] if t['date']==day and t['stock_id']==row['stock_id']
                         and t['event_id']==row['event_id'] and t['side']=='buy')
            if row['filled_qty'] != filled or (filled and not row['attempted']):
                raise ValueError('Slot decision does not reconcile to fills')
            if row['attempted'] and (row['stock_id'] in expected or len(expected) >= account['settings']['slots']):
                raise ValueError('Attempt exceeded slot policy')
            if row['attempted']:
                attempted.add(row['stock_id'])
                if filled:
                    current.add(row['stock_id'])
                else:
                    failed.add(row['stock_id'])
    checked['slot_membership_independently_reconstructed'] = True
    return checked
