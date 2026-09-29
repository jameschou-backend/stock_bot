"""Preregistered partial exits; all decisions precede the execution session."""
from copy import deepcopy
import math
import pandas as pd
from skills.close_confirmed_exit import update_peak
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.residual_slot_replay import ResidualSlotDecisions, residual_values, RESIDUAL_CAP
from skills.replay_market_feeds import ReplayDataUnavailable

ARMS = ('original', 'full15', 'half15', 'cap40', 'half15_cap40')
PARTIAL_REASONS = ('reduce_half15', 'reduce_cap40', 'reduce_half15_cap40')


class CoreSlots(ResidualSlotDecisions):
    """An intentionally retained odd-lot core is not an unfilled liquidation."""
    def corporate_day(self, day):
        if self.residual_policy != 'release':
            raise ValueError('Partial risk research requires the frozen residual release policy')
        exited = {r['event_id'] for r in self.orders if r['side']=='sell'
                  and r['reason'] not in PARTIAL_REASONS and r['date'] < str(day.date())}
        released = residual_values(self.holdings,self.receivables,self.marks,exited)
        if any(r['prior_mark_date'] >= str(day.date()) for r in released.values()):
            raise ValueError('Core sizing used future marks')
        value = sum(r['value'] for r in released.values())
        self.residual_budget = max(0.,(self.previous_nav-value)/self.slots)
        self.residual_block = value > self.previous_nav*RESIDUAL_CAP
        self.released_residuals = set(released)
        opening = set(self.holdings)-{'0050'}-self.released_residuals
        self.residual_days.append(dict(date=str(day.date()),opening_nav=self.previous_nav,
            released=deepcopy(released),residual_value=value,new_position_budget=self.residual_budget,
            block_new_buys=self.residual_block,opening_active=sorted(opening)))
        income = super(ResidualSlotDecisions,self).corporate_day(day)
        self.opening_members = opening
        self.occupied = self.entry_slot_members()
        return income


class PartialRisk(MidpointExitReplay, CoreSlots):
    def __init__(self,*args,stop_events,risk_arm,**kwargs):
        if risk_arm not in ARMS[2:]: raise ValueError('Unknown partial risk arm')
        super().__init__(*args,**kwargs)
        self.risk_arm,self.stop_events = risk_arm,stop_events
        self.peaks,self.half_dates,self.pending,self.temporary_due = {},{},{},{}
        self.risk_decisions = []
        c = self.exit_signals.adjusted_close
        self.ma60 = c.rolling(60,min_periods=60).mean()

    def corporate_day(self,day):
        # The normal account loop executes due holdings. Restore temporary
        # partial deadlines before the parent's hard-exit planning runs.
        for sid,due in self.temporary_due.items():
            if sid in self.holdings: self.holdings[sid]['due_index'] = due
        self.temporary_due = {}
        action_start = len(self.actions)
        income = super().corporate_day(day)
        i = self.positions[day]; previous = self.days[i-1]; signal = str(previous.date())
        for eid,state in self.exit_states.items():
            sid = state['stock_id']; h = self.holdings.get(sid)
            if not h or h['event_id']!=eid or not h['qty']: continue
            pending = self.pending.get(eid)
            if pending:
                for a in self.actions[action_start:]:
                    if a['stock_id'] != sid: continue
                    if a['kind'] in ('split','capital_reduction'):
                        pending['keep'] = math.ceil(pending['keep']*a['qty_after']/a['entitled_qty'])
                    elif a['kind']=='share_delivery': pending['keep'] += a['qty']
            if state['trigger_reason']:
                self.pending.pop(eid,None)
                continue
            entry = self.days[state['entry_index']]
            half = 'half15' in self.risk_arm
            peak_trigger = False
            if half:
                if eid not in self.peaks:
                    fills = [t['reference_price'] for t in self.trades if t['event_id']==eid and t['side']=='buy']
                    self.peaks[eid] = max(self.raw(entry,sid),*fills)
                if previous > entry:
                    actions = self.stop_events.loc[self.stop_events.stock_id.eq(sid)
                        & pd.to_datetime(self.stop_events.event_date).eq(previous)]
                    if len(actions)>1: raise ReplayDataUnavailable('Multiple peak adjustments')
                    ratio = float(actions.iloc[0].ratio) if len(actions) else 1.
                    if not math.isfinite(ratio) or ratio<=0: raise ReplayDataUnavailable('Invalid peak ratio')
                    if self.official_halt(previous,sid): self.peaks[eid] *= ratio
                    else:
                        self.peaks[eid],peak_trigger = update_peak(self.peaks[eid],
                            self.raw(previous,sid,'high'),self.raw(previous,sid),ratio)
            # A core exit requires two closes strictly after the half signal.
            core_exit = False
            if eid in self.half_dates and i>=2 and str(self.days[i-2].date())>self.half_dates[eid]:
                c = self.exit_signals.adjusted_close[sid].iloc[i-2:i]
                m = self.ma60[sid].iloc[i-2:i]
                core_exit = bool(c.notna().all() and m.notna().all() and c.lt(m).all())
            if core_exit:
                self.pending.pop(eid,None)
                state.update(trigger_reason='core_ma60_two',signal_date=signal,
                    target_date=str(day.date()),target_index=i)
                h['due_index'] = i
                self._plan(day,sid,'sell',eid,signal,h['qty']//1000*1000,0.,self.opening_limit,None)
                self.risk_decisions.append(dict(date=str(day.date()),event_id=eid,stock_id=sid,
                    signal_date=signal,reason='core_ma60_two',keep=0,requested_qty=h['qty']))
                continue
            keep = pending['keep'] if pending else h['qty']
            reasons = []
            if half and peak_trigger and eid not in self.half_dates:
                self.half_dates[eid] = signal
                keep = min(keep,math.ceil(h['qty']/2))
                reasons.append('half15')
            reference = self.prior(day,sid)
            # Prior closing physical holdings, before today's corporate actions.
            prior_h = next((r for r in reversed(self.holding_rows) if r['date']==signal and r['event_id']==eid),None)
            weight = prior_h['market_value']/self.previous_nav if prior_h else 0.
            if 'cap40' in self.risk_arm and weight>.40:
                if not reference or not math.isfinite(reference): raise ReplayDataUnavailable('Missing cap reference')
                capped = max(1,math.floor(self.previous_nav/3/reference))
                if capped < keep:
                    keep = capped
                    reasons.append('cap40')
            if reasons and keep < h['qty']:
                pending = dict(keep=keep,signal_date=signal,reason='reduce_'+'_'.join(reasons))
                self.pending[eid] = pending
            if pending and pending['keep'] < h['qty']:
                qty = h['qty']-pending['keep']
                self.temporary_due[sid] = h['due_index']; h['due_index'] = i
                self._plan(day,sid,'sell',eid,pending['signal_date'],qty//1000*1000,0.,self.opening_limit,None)
                p = self.day_plans[(eid,'sell')]
                p.update(planned_qty=qty,board_qty=qty//1000*1000,odd_qty=qty%1000)
                self.tick_plans[-1] = deepcopy(p)
                self.risk_decisions.append(dict(date=str(day.date()),event_id=eid,stock_id=sid,
                    signal_date=pending['signal_date'],reason=pending['reason'],keep=pending['keep'],requested_qty=qty))
        return income

    def order(self,day,sid,side,qty,reason,event_id,signal_date=None):
        partial = side=='sell' and event_id in self.pending and not self.exit_states[event_id]['trigger_reason']
        if partial:
            instruction = self.pending[event_id]
            reason,signal_date = instruction['reason'],instruction['signal_date']
        filled = super().order(day,sid,side,qty,reason,event_id,signal_date)
        if partial and self.holdings[sid]['qty'] <= instruction['keep']:
            self.pending.pop(event_id)
        return filled

    def run(self):
        result = super().run()
        result['risk_decisions'] = self.risk_decisions
        result['settings'].update(risk_arm=self.risk_arm,partial_core_occupies_slot=True,
            cap_trigger=.4,cap_target=1/3,core_trend_window=60,half_drawdown=.15)
        return result
