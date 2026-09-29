"""Overnight-armed peak stop: board tape exits, odd lots on the next session."""
from copy import deepcopy
import math
import pandas as pd

from skills.intraday_peak_exit import IntradayPeakExit
from skills.midpoint_exit_replay import MidpointExitReplay, legal_exit_floor
from skills.midpoint_replay import midpoint_match
from skills.million_replay import money
from skills.replay_market_feeds import ReplayDataUnavailable


class PortfolioIntradayExit(MidpointExitReplay):
    def __init__(self, *args, stop_events, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_events = stop_events
        self.intraday_peaks, self.intraday_exits = {}, {}
        self.intraday_evidence, self.intraday_today = [], {}

    def corporate_day(self, day):
        income = super().corporate_day(day)
        # All buy plans are frozen by the parent BEFORE inspecting today's tape.
        buys = deepcopy({k:v for k,v in self.day_plans.items() if k[1]=='buy'})
        self.intraday_today = {}
        for sid, holding in self.holdings.items():
            if not holding['qty']:
                continue
            eid = holding['event_id']
            state = self.exit_states[eid]
            pending = self.intraday_exits.get(eid)
            if state['trigger_reason'] and not pending:
                continue  # Prior-close baseline instruction takes precedence.
            date = str(day.date())
            if eid not in self.intraday_peaks:
                buys_for_position = [t for t in self.trades if t['event_id']==eid and t['side']=='buy']
                entry = pd.Timestamp(buys_for_position[0]['date'])
                if entry >= day:
                    raise ValueError('Entry-session intraday stop is not modeled')
                self.intraday_peaks[eid] = max(float(self.raw(entry,sid)),
                    *[t['reference_price'] for t in buys_for_position])
            peak = self.intraday_peaks[eid]
            actions = self.stop_events.loc[self.stop_events.stock_id.eq(sid)
                & pd.to_datetime(self.stop_events.event_date).eq(day)]
            if len(actions)>1:
                raise ReplayDataUnavailable('Multiple same-day peak adjustments need review')
            for action in actions.itertuples():
                ratio = float(action.ratio)
                if not math.isfinite(ratio) or ratio <= 0:
                    raise ReplayDataUnavailable('Unknown corporate peak adjustment')
                peak *= ratio
            self.intraday_peaks[eid] = peak
            if self.official_halt(day, sid):
                self.intraday_evidence.append(dict(date=date,stock_id=sid,event_id=eid,
                    prior_peak=peak,status='official_full_session_halt'))
                continue
            identity = self.identity(day, sid)
            if identity['status'] != 'identified' or identity['category'] != '股票':
                raise ReplayDataUnavailable(f'Intraday identity unavailable: {sid} {date}')
            self.markets[sid] = identity['market'].upper()
            values = [self.raw(day,sid,k) for k in ('open','high','low','close','volume')]
            if any(v is None or not math.isfinite(v) or v <= 0 for v in values):
                raise ReplayDataUnavailable(f'Unknown intraday bar: {sid} {date}')
            opening, high, low, close, volume = values
            if not low <= min(opening,close) <= max(opening,close) <= high:
                raise ReplayDataUnavailable('Invalid intraday OHLC')
            if not pending and low > max(peak,high)*.85+1e-10:
                self.intraday_peaks[eid] = max(peak,high)
                self.intraday_evidence.append(dict(date=date,stock_id=sid,event_id=eid,
                    prior_peak=peak,high=high,low=low,status='no_crossing_under_any_bar_order'))
                continue
            self.require_prior_inputs(day,sid)
            limits = self.feeds.get_limits(sid).get(date)
            legal_exit_floor(limits)
            if not limits['lower'] <= low <= high <= limits['upper']:
                raise ReplayDataUnavailable(f'Intraday bar/limit conflict: {sid} {date}')
            board_qty = holding['qty']//1000*1000
            trace = None
            if board_qty or not pending:
                tape, digest = self.ticks.get(sid,date,self.markets[sid])
                if (tape.empty or tape.price.max()>high+1e-6 or tape.price.min()<low-1e-6
                        or tape.shares.sum()>volume*1.01):
                    raise ReplayDataUnavailable(f'Intraday tape/daily conflict: {sid} {date}')
                valid = tape.loc[tape.time.ge(pd.Timedelta('09:00:00'))
                    & tape.time.lt(pd.Timedelta('13:25:00')) & tape.shares.gt(0)]
                if valid.empty:
                    raise ReplayDataUnavailable(f'No intraday tape observations: {sid} {date}')
                basis = sid+':raw:'+date
                previous = str(self.days[self.positions[day]-1].date())
                watcher = IntradayPeakExit(stock_id=sid,qty=board_qty or 1000,venue='board',
                    peak=peak,asof=previous+'T13:30:00+08:00',basis=basis,
                    source='Frozen entry price/close, subsequent held highs and official action ratios')
                if pending:
                    watcher.trigger = deepcopy(pending)
                watcher.start_session(date=date,known_at=date+'T08:59:00+08:00',
                    lower=limits['lower'],upper=limits['upper'],prior_adv_shares=float(self.volume20.at[day,sid]),
                    basis=basis,source='Frozen dated official limits / prior ADV20')
                for time, group in valid.groupby('time',sort=True):
                    watcher.observe(at=(pd.Timestamp(date,tz='Asia/Taipei')+time).isoformat(),
                        stock_id=sid,venue='board',basis=basis,
                        prints=[dict(price=float(r.price),shares=int(r.shares)) for r in group.itertuples()])
                    if not board_qty and watcher.trigger:
                        break  # Signal-only probe cannot fabricate a board fill.
                trace = watcher.report()
                trigger = trace['trigger']
                self.intraday_evidence.append(dict(date=date,stock_id=sid,event_id=eid,
                    prior_peak=peak,high=high,low=low,board_position_qty=board_qty,
                    status='triggered' if trigger else 'tape_no_crossing',
                    tape_sha256=digest,trace=trace))
                if trigger and not pending:
                    self.intraday_exits[eid] = deepcopy(trigger)
                    state.update(trigger_reason='intraday_peak_stop15', signal_date=date,
                        target_date=date,target_index=self.positions[day])
                    pending = trigger
                self.intraday_peaks[eid] = max(peak,high)
            if pending:
                holding['due_index'] = self.positions[day]
                self.intraday_today[eid] = dict(trace=trace,limits=limits,
                    trigger=pending,board_qty=board_qty)
        if buys != {k:v for k,v in self.day_plans.items() if k[1]=='buy'}:
            raise ValueError('Intraday observations changed frozen buy plans')
        return income

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if side != 'sell' or event_id not in self.intraday_exits:
            return super().order(day,sid,side,qty,reason,event_id,signal_date)
        date = str(day.date())
        if self.official_halt(day,sid):
            self.orders.append(dict(date=date,stock_id=sid,event_id=event_id,side='sell',
                channel='event',requested_qty=qty,filled_qty=0,reason='intraday_peak_stop15',
                failure='official_full_session_halt'))
            return 0
        if sid in self.tick_attempts:
            raise ValueError('Stock/day execution volume reused')
        self.tick_attempts.add(sid)
        info = self.intraday_today[event_id]
        first_date = info['trigger']['at'][:10]
        common = dict(date=date,stock_id=sid,name=self.names.get(sid,sid),event_id=event_id,
            side='sell',signal_date=first_date,reason='intraday_peak_stop15',
            limit_price=info['limits']['lower'],prior_avg_volume20=float(self.volume20.at[day,sid]),
            prior_avg_amount20=float(self.amount20.at[day,sid]),participation_limit=.01)
        total = 0
        for fill in (info['trace']['fills'] if info['trace'] and info['board_qty'] else []):
            n = fill['qty']
            row = dict(common,channel='board',requested_qty=info['board_qty'],
                capacity_qty=total+n,filled_qty=n,reference_price=fill['price'],
                order_time=info['trace']['orders'][0]['created_at'],fill_time=fill['at'],
                execution_evidence='post_trigger_tape_participation_proxy',failure=None)
            self._record_intraday(day,sid,event_id,row,n,fill['price'])
            total += n
        if info['board_qty'] > total:
            self.orders.append(dict(common,channel='board',requested_qty=info['board_qty']-total,
                filled_qty=0,execution_evidence='post_trigger_tape_participation_proxy',
                failure='pending_post_trigger_capacity'))
        odd_qty = self.holdings[sid]['qty']%1000
        if odd_qty and date > first_date:
            odd = self.odd_feeds.get_odd(date,sid,self.markets[sid])
            if odd is None:
                raise ReplayDataUnavailable(f'Missing delayed odd exit: {sid} {date}')
            row = dict(common,channel='odd',requested_qty=odd_qty,order_time='09:00:00',
                execution_evidence='next_session_odd_HL2_proxy')
            row.update(midpoint_match(odd['odd_high'],odd['odd_low'],odd['odd_shares'],
                float(self.volume20.at[day,sid]),odd_qty,'sell',info['limits']['lower'],
                info['limits']['lower'],info['limits']['upper'],'odd'))
            n = row['filled_qty']
            if n:
                self._record_intraday(day,sid,event_id,row,n,row['reference_price'])
                total += n
            else:
                self.orders.append(row)
        elif odd_qty:
            self.orders.append(dict(common,channel='odd',requested_qty=odd_qty,filled_qty=0,
                failure='deferred_to_next_session_no_odd_tape'))
        return total

    def _record_intraday(self,day,sid,eid,row,qty,price):
        paid = self._costs(price,qty,'sell',sid)
        if qty > self.holdings[sid]['qty'] or money(self.cash+paid['cash_change'])<0:
            raise ValueError('Invalid intraday sale balance')
        self.cash_move(day,'sell',paid['cash_change'],stock_id=sid,event_id=eid,channel=row['channel'])
        self.holdings[sid]['qty'] -= qty
        mark = self.raw(day,sid)
        self.marks[sid] = dict(price=mark,date=str(day.date()))
        self.used[(sid,row['channel'])] += qty
        self.day_cost += paid['total_cost']
        self.day_basis += qty*(price-mark)
        trade = dict(row,**paid,qty=qty,cash_after=self.cash,
            remaining_shares=self.holdings[sid]['qty'],sequence=len(self.trades)+1)
        trade.pop('filled_qty'); trade.pop('failure',None)
        self.trades.append(trade); self.orders.append(row)

    def run(self):
        result = super().run()
        result['intraday_evidence'] = self.intraday_evidence
        result['settings'].update(intraday_stop=True,stop_drawdown=.15,
            stop_activation='session_after_entry',entry_day_high_used=False,
            exit_mechanism='loss12_or_time63_or_intraday_peak15',
            sell_price_policy='baseline_HL2_or_stop_board_post_trigger_ticks_odd_next_session_HL2',
            exact_all_venue_intraday=False)
        return result
