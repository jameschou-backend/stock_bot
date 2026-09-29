"""Confirm a 15% drawdown at close; submit the exit next market session."""
import math
import pandas as pd

from skills.midpoint_exit_replay import MidpointExitReplay
from skills.replay_market_feeds import ReplayDataUnavailable


def update_peak(peak, high, close, ratio=1.):
    if any(not math.isfinite(v) or v <= 0 for v in (peak, high, close, ratio)) or close > high:
        raise ReplayDataUnavailable('Invalid close-stop prices or corporate adjustment')
    peak = max(peak * ratio, high)
    return peak, close <= peak * .85 + 1e-10


class CloseConfirmedExit(MidpointExitReplay):
    """Keep baseline loss12/time63; replace only the intraday peak trigger.

    Entry HL2 has no timestamp: exclude its daily high. Initialize from actual
    modeled entry prices and the entry close, then use full held-session highs.
    Every observation is from the previous session, never the execution day.
    """
    def __init__(self, *args, stop_events, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_events = stop_events
        self.close_peaks, self.close_evidence = {}, []

    def corporate_day(self, day):
        income = super().corporate_day(day)
        index = self.positions[day]
        signal_day = self.days[index-1]
        signal = str(signal_day.date())
        for eid, state in self.exit_states.items():
            sid = state['stock_id']
            holding = self.holdings.get(sid)
            owns = holding and holding['event_id'] == eid and holding['qty'] > 0
            rights = any(r.get('event_id') == eid and r.get('qty', 0) > 0 for r in self.receivables)
            if state['trigger_reason'] or not (owns or rights):
                continue
            entry = self.days[state['entry_index']]
            if eid not in self.close_peaks:
                fills = [t['reference_price'] for t in self.trades if t['event_id']==eid and t['side']=='buy']
                entry_close = self.raw(entry,sid)
                if not fills or entry_close is None or not math.isfinite(entry_close) or entry_close <= 0:
                    raise ReplayDataUnavailable('Missing close-stop entry evidence')
                self.close_peaks[eid] = max(entry_close,*fills)
            if signal_day <= entry:
                continue
            peak = self.close_peaks[eid]
            actions = self.stop_events.loc[self.stop_events.stock_id.eq(sid)
                & pd.to_datetime(self.stop_events.event_date).eq(signal_day)]
            if len(actions)>1:
                raise ReplayDataUnavailable('Multiple close-stop peak adjustments require review')
            ratio = float(actions.iloc[0].ratio) if len(actions) else 1.
            if not math.isfinite(ratio) or ratio <= 0:
                raise ReplayDataUnavailable('Invalid close-stop corporate adjustment')
            if self.official_halt(signal_day,sid):
                self.close_peaks[eid] = peak*ratio
                continue
            high, close = self.raw(signal_day,sid,'high'), self.raw(signal_day,sid)
            if high is None or close is None:
                raise ReplayDataUnavailable('Missing close-stop bar')
            peak, triggered = update_peak(peak,high,close,ratio)
            self.close_peaks[eid] = peak
            self.close_evidence.append(dict(event_id=eid,stock_id=sid,signal_date=signal,
                target_date=str(day.date()),peak=peak,close=close,threshold=peak*.85,triggered=triggered))
            if triggered:
                state.update(trigger_reason='close_confirmed_peak15',signal_date=signal,
                    target_date=str(day.date()),target_index=index)
                if owns:
                    holding['due_index'] = index
                    self._plan(day,sid,'sell',eid,signal,holding['qty']//1000*1000,
                        0.,self.opening_limit,None)
        return income

    def run(self):
        result = super().run()
        result['close_stop_evidence'] = self.close_evidence
        result['settings'].update(exit_mechanism='loss12_or_time63_or_close_confirmed_peak15',
            stop_drawdown=.15,stop_signal='prior_close_vs_post_entry_held_high',
            stop_activation='session_after_entry',entry_day_high_used=False,
            intraday_stop=False,exit_execution='next_session_channel_HL2_proxy')
        return result
