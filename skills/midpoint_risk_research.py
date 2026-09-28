"""Frozen daily-close risk and one-time re-entry experiments for the HL2 account."""
from copy import deepcopy
import math
import pandas as pd
from skills.midpoint_replay import MidpointStock

ARMS = ('control', 'protect', 'market', 'combined', 'reentry', 'trail_only', 'weak_only', 'protect_reentry')


def market_states(close):
    """Close-date observations; the account consumes only the previous row."""
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    ret5 = close / close.shift(5) - 1
    above = close.gt(ma20) & close.shift(1).gt(ma20.shift(1))
    below = close.lt(ma20) & close.shift(1).lt(ma20.shift(1))
    rising = ma20.gt(ma20.shift(5))
    state, rows = 0, []
    for i, day in enumerate(close.index):
        valid = all(pd.notna(x) and math.isfinite(float(x)) for x in
                    (close.iloc[i], ma20.iloc[i], ma60.iloc[i], ret5.iloc[i], ma20.shift(5).iloc[i]))
        shock = valid and ret5.iloc[i] <= -.05
        weak = valid and close.iloc[i] < ma60.iloc[i] and not rising.iloc[i]
        if not valid or shock or weak or (state == 1 and below.iloc[i]):
            state, why = 0, 'missing' if not valid else 'shock' if shock else 'weak_trend'
        elif state == 0:
            state, why = (1, 'recovery_probe') if above.iloc[i] and ret5.iloc[i] > 0 else (0, 'await_recovery')
        elif state == 1 and close.iloc[i] > ma60.iloc[i] and rising.iloc[i] and above.iloc[i]:
            state, why = 3, 'trend_recovered'
        else:
            why = 'retain'
        rows.append(dict(date=str(day.date()), slots=state, reason=why,
                         close=float(close.iloc[i]) if pd.notna(close.iloc[i]) else None))
    return pd.DataFrame(rows).set_index('date')


def protective_exit(context, kind='both'):
    if kind not in ('both','trail','weak'):
        raise ValueError('Unknown protection kind')
    if kind != 'weak' and context['peak_return'] is not None and context['peak_drawdown'] is not None:
        if context['peak_return'] >= .20 and context['peak_drawdown'] <= -.12:
            return 'profit_trail12'
    if kind != 'trail' and context['below_ma20_two'] and context['relative20'] is not None and context['relative20'] < 0:
        return 'weak_ma20'
    return None


class RiskResearchReplay(MidpointStock):
    def __init__(self, *args, risk_arm, **kwargs):
        if risk_arm not in ARMS:
            raise ValueError('Unknown risk arm')
        super().__init__(*args, **kwargs)
        self.risk_arm = risk_arm
        self.market_schedule = market_states(self.exit_signals.adjusted_close['0050'])
        self.risk_log, self.entry_gate_log, self.reentry_log = [], [], []
        self.reentered_roots = set()
        self.reentry_high = self.exit_signals.adjusted_close.rolling(10, min_periods=10).max().shift(1)

    def corporate_day(self, day):
        i = self.positions[day]
        prior = self.days[i-1]
        signal = str(prior.date())
        market = self.market_schedule.loc[signal].to_dict()
        gated = self.risk_arm in ('market', 'combined', 'reentry')
        candidates = list(self.events.get(day, []))
        reentry_allowed = market['slots'] if gated else self.exit_signals.trend.state.at[prior]=='ON'
        if self.risk_arm in ('reentry','protect_reentry') and reentry_allowed:
            for c in self.cohorts:
                root, sid = c['event_id'], c['stock_id']
                if root in self.reentered_roots or c.get('reentry_root') or not c.get('exit_date') or sid in self.holdings:
                    continue
                exit_i = self.positions[pd.Timestamp(c['exit_date'])]
                if not 5 <= i-1-exit_i <= 63:
                    continue
                p = self.exit_signals.price(i-1, sid)
                high = self.reentry_high.at[prior, sid]
                ma = self.exit_signals.ma20.at[prior, sid]
                relative = self.exit_signals.relative20.at[prior, sid]
                volumes = self.fields['volume'][sid]
                history = volumes.iloc[i-21:i-1]
                avg = history.mean() if len(history)==20 and history.notna().all() and history.ge(0).all() else float('nan')
                volume = volumes.iloc[i-1]
                if not (p and pd.notna(high) and p > high and p > ma and relative > 0
                        and avg > 0 and volume >= avg*1.5):
                    continue
                event = deepcopy(c)
                event.update(event_id=root+'-reentry-'+signal, signal_date=signal,
                    entry_date=str(day.date()), priority=float(relative), reentry_root=root,
                    selection_reason='After completed exit and 5-session cooldown: 10-close breakout, above MA20, excess20>0, volume>=1.5x')
                for key in ('exit_date','bought_qty','due_date','due_index'):
                    event.pop(key, None)
                candidates.append(event)
                self.source_events[event['event_id']] = deepcopy(event)
                self.reentry_log.append(dict(date=str(day.date()), signal_date=signal, root=root,
                    event_id=event['event_id'], stock_id=sid, signal_close=p, prior_high10=float(high),
                    relative20=float(relative), volume_ratio=float(volume/avg), exit_date=c['exit_date']))
                # One fresh attempt per completed original cohort, including a rejected order.
                self.reentered_roots.add(root)
        candidates.sort(key=lambda e: (-e['priority'], e['event_id']))
        if gated:
            occupied = len([h for h in self.holdings.values() if h['qty']])
            free = max(0, int(market['slots'])-occupied)
            permitted = candidates[:free]
            self.entry_gate_log.append(dict(date=str(day.date()), signal_date=signal,
                slots=int(market['slots']), occupied=occupied, reason=market['reason'],
                accepted=[e['event_id'] for e in permitted], rejected=[e['event_id'] for e in candidates[free:]]))
            candidates = permitted
        self.events[day] = candidates
        income = super().corporate_day(day)
        if self.risk_arm == 'control':
            return income
        # Parent first applies the original hard stop and 63-session deadline.
        # New exits cannot cancel or postpone an already-latched instruction.
        for sid, holding in self.holdings.items():
            if not holding['qty']:
                continue
            eid = holding['event_id']
            state = self.exit_states.get(eid)
            if not state or state['trigger_reason']:
                continue
            context = self.exit_signals.context(i, sid, state)
            reason = 'market_defensive' if gated and market['slots']==0 else None
            if not reason and self.risk_arm in ('protect','combined','reentry','trail_only','weak_only','protect_reentry'):
                kind={'trail_only':'trail','weak_only':'weak'}.get(self.risk_arm,'both')
                reason = protective_exit(context,kind)
            self.risk_log.append(dict(date=str(day.date()), signal_date=signal, stock_id=sid,
                event_id=eid, reason=reason, context=context, market_slots=int(market['slots'])))
            if not reason:
                continue
            state.update(trigger_reason=reason, signal_date=signal, target_date=str(day.date()), target_index=i)
            holding['due_index'] = i
            self._plan(day, sid, 'sell', eid, signal, holding['qty']//1000*1000, 0.,
                       self.opening_limit, None)
        return income

    def run(self):
        result = super().run()
        if self.risk_arm != 'control':
            result['settings']['risk_research_arm'] = self.risk_arm
        return result
