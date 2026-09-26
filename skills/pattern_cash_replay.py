"""One fixed original-signal pattern filter over the verified cash accounts."""
import pandas as pd

from skills.support_risk_replay import SupportRiskReplay


class PatternCashReplay(SupportRiskReplay):
    def __init__(self, *args, pattern_filter, **kwargs):
        if type(pattern_filter) is not bool:
            raise ValueError('Pattern filter must be explicitly on or off')
        super().__init__(*args, **kwargs)
        self.pattern_filter = pattern_filter
        self.pattern_entries = []

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if not self.pattern_filter or side != 'buy' or sid == '0050':
            return super().order(day,sid,side,qty,reason,event_id,signal_date)
        context = self.original_context(signal_date,day,sid)
        available = context['pattern_available']
        passed = available and context['pattern_pass'] is True
        allowed = qty if passed else 0
        why = 'pattern_pass' if passed else ('pattern_rejected' if available else 'pattern_unavailable')
        filled = super().order(day,sid,side,allowed,reason,event_id,signal_date)
        self.pattern_entries.append(dict(date=str(day.date()),event_id=event_id,stock_id=sid,
            signal_date=signal_date,context=context,requested_qty=qty,allowed_qty=allowed,
            filled_qty=filled,reason=why))
        return filled

    def run(self):
        account = super().run()
        if self.pattern_filter:
            account['settings']['entry_pattern'] = 'original_signal_contraction_breakout_volume_v1'
        return account


def audit_pattern_entries(account, rows, slots, signals, candidates, mask):
    source = {row['event_id']:row for row in candidates}
    if len(source) != len(candidates):
        raise ValueError('Pattern input candidates are not unique')
    keys = [(r['date'],r['event_id']) for r in slots]
    if [(r['date'],r['event_id']) for r in rows] != keys or len(set(keys)) != len(keys):
        raise ValueError('Every sizing attempt requires one pattern record')
    for row,slot in zip(rows,slots):
        event = source.get(row['event_id'])
        if event is None:
            raise ValueError('Pattern record has an unknown original candidate')
        sid = event['members'][0]
        i = signals.days.get_loc(pd.Timestamp(event['signal_date']))+1
        context = signals.technical_context(i,sid)
        date = signals.days[signals.days.get_loc(pd.Timestamp(event['entry_date']))+bool(mask&2)]
        passed = context['pattern_available'] and context['pattern_pass'] is True
        qty = row['requested_qty']
        if type(qty) is not int or qty < 0:
            raise ValueError('Pattern requested shares must be nonnegative integers')
        allowed = qty if passed else 0
        why = 'pattern_pass' if passed else ('pattern_rejected' if context['pattern_available'] else 'pattern_unavailable')
        trades = [t for t in account['trades'] if t['side']=='buy'
                  and (t['date'],t['event_id'])==(row['date'],row['event_id'])]
        filled = sum(t['qty'] for t in trades)
        wanted = dict(date=str(date.date()),event_id=event['event_id'],stock_id=sid,
            signal_date=event['signal_date'],context=context,requested_qty=qty,
            allowed_qty=allowed,filled_qty=filled,reason=why)
        if (row != wanted or filled > allowed or (not passed and slot['attempted'])
                or slot['signal_date'] != event['signal_date'] or slot['stock_id'] != sid):
            raise ValueError('Pattern eligibility, original timing or rejected-entry resources differ')
    fills = {(t['date'],t['event_id']) for t in account['trades'] if t['side']=='buy'}
    if not fills.issubset(set(keys)):
        raise ValueError('Buy lacks a verified pattern decision')
    return dict(original_pattern_signals_rebuilt=True, missing_patterns_blocked=True,
                rejected_patterns_do_not_reserve_slots=True,
                pattern_passed_attempts=sum(r['reason']=='pattern_pass' for r in rows),
                pattern_rejected_attempts=sum(r['reason']=='pattern_rejected' for r in rows),
                pattern_missing_attempts=sum(r['reason']=='pattern_unavailable' for r in rows))
