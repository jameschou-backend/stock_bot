"""Reuse audited exit policies with current cash, identity and execution rules."""
from collections import Counter

from skills.exit_policy import MODES, decide_exit
from skills.residual_slot_replay import ResidualSlotReplay
from skills.scenario_exit_replay import ExitSignals, _finite


class ExitMechanismReplay(ResidualSlotReplay):
    def __init__(self, *args, exit_mode, **kwargs):
        if exit_mode not in MODES:
            raise ValueError('Unknown preregistered exit mechanism')
        super().__init__(*args, residual_policy='release', **kwargs)
        # Ancestor allocation constructors enforce loss12 while initializing.
        # After construction the inherited causal policy dispatch accepts all
        # seven existing modes. No event, budget or account loop is replaced.
        self.mode = exit_mode

    def run(self):
        result = super().run()
        if self.mode != 'loss12':
            result['settings']['exit_mechanism'] = self.mode
        return result


def audit_exits(account, decisions, states, close, calendar, mode, mask):
    """Rebuild every active cohort's prior-close decision and latched due date."""
    signals = ExitSignals(close, calendar)
    dates = [str(d.date()) for d in signals.days]
    positions = {d:i for i,d in enumerate(dates)}
    cohorts = {r['event_id']:r for r in account['cohorts']}
    expected = {(d, c['event_id']) for d in (r['date'] for r in account['daily'])
                for c in cohorts.values() if c['entry_date'] < d
                and (c['exit_date'] is None or d <= c['exit_date'])}
    if len(decisions) != len(expected) or {(r['date'],r['event_id']) for r in decisions} != expected:
        raise ValueError('Exit decisions do not cover every active cohort day')
    if decisions != sorted(decisions, key=lambda r:(r['date'], list(cohorts).index(r['event_id']))):
        raise ValueError('Exit decisions must follow account chronology')
    rebuilt, delayed = {}, set()
    for row in decisions:
        day, identity, sid = row['date'], row['event_id'], row['stock_id']
        i = positions[day]
        if sid != cohorts[identity]['stock_id'] or i < 1:
            raise ValueError('Exit decision has invalid stock or day')
        if identity not in rebuilt:
            entry = positions[cohorts[identity]['entry_date']]
            price = signals.price(entry, sid)
            rebuilt[identity] = dict(stock_id=sid, event_id=identity, entry_index=entry,
                entry_price=price, peak_price=price, trigger_reason=None,
                signal_date=None, target_date=None, target_index=None)
        state = rebuilt[identity]
        context = signals.context(i, sid, state)
        if state['trigger_reason']:
            decision = dict(exit=True,reason=state['trigger_reason'],phase='exiting',extend=False)
        else:
            decision = decide_exit(context, mode)
            if decision['exit']:
                state.update(trigger_reason=decision['reason'],signal_date=dates[i-1],
                             target_date=day,target_index=i)
        wanted = dict(date=day, signal_date=dates[i-1], stock_id=sid,event_id=identity,
            mode=mode, **context, **decision, signal_close=signals.price(i-1,sid),
            ma20=_finite(signals.ma20.iloc[i-1][sid]), market_state=signals.trend.state.iloc[i-1],
            entry_price=state['entry_price'],peak_price=state['peak_price'],
            first_signal_date=state['signal_date'],target_date=state['target_date'])
        if row != wanted:
            raise ValueError('Exit policy did not reconstruct from prior observations')
        if mask & 4 and state['target_index'] is not None and identity not in delayed:
            state['original_target_date'] = state['target_date']
            state['target_index'] += 1
            j = state['target_index']
            state['target_date'] = dates[j] if j < len(dates) else None
            delayed.add(identity)
    if states != rebuilt:
        raise ValueError('Latched exit state differs from rebuilt decisions')
    for trade in account['trades']:
        if trade['side'] != 'sell':
            continue
        state = rebuilt.get(trade['event_id'])
        if (not state or state['target_index'] is None or positions[trade['date']] < state['target_index']
                or trade['signal_date'] != state['signal_date'] or trade['reason'] != state['trigger_reason']):
            raise ValueError('Sale precedes or contradicts its latched instruction')
    return dict(exit_decisions_rebuilt=True, all_active_cohort_days_checked=True,
                exit_delay_rebuilt=True, no_sale_before_instruction=True,
                trigger_reasons=dict(Counter(s['trigger_reason'] for s in rebuilt.values() if s['trigger_reason'])))
