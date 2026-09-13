"""Causal, reserved-cash board-lot diagnostic; historical books remain sealed."""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import math

import numpy as np
import pandas as pd

from skills.cash_risk_replay import CashRiskReplay
from skills.execution_stress import StressOrder, audit_stress
from skills.million_replay import Replay, money
from skills.replay_market_feeds import ReplayDataUnavailable


def limit_price(price, sid, side):
    if side not in ('buy', 'sell') or not math.isfinite(price) or price <= 0:
        raise ValueError('Invalid limit price or side')
    if sid == '0050':
        tick = '.01' if price < 50 else '.05'
    else:
        tick = next(t for ceiling, t in [(10,'.01'), (50,'.05'), (100,'.1'),
                                        (500,'.5'), (1000,'1'), (math.inf,'5')] if price < ceiling)
    unit = Decimal(tick)
    return float((Decimal(str(price))/unit).to_integral_value(
        rounding=ROUND_FLOOR if side == 'buy' else ROUND_CEILING)*unit)


def normalize_ticks(frame, sid, day, market):
    """Preserve equal timestamps and equal rows; source has no unique trade ID."""
    required = {'date', 'stock_id', 'deal_price', 'volume', 'Time', 'TickType'}
    if market not in ('TWSE', 'TPEX') or frame.empty or not required.issubset(frame):
        raise ReplayDataUnavailable(f'Missing board tick schema/market: {sid} {day}')
    if not frame.date.eq(day).all() or not frame.stock_id.eq(sid).all():
        raise ReplayDataUnavailable(f'Tick query identity mismatch: {sid} {day}')
    if not frame.Time.astype(str).str.fullmatch(r'\d{2}:\d{2}:\d{2}(\.\d{1,6})?').all():
        raise ReplayDataUnavailable('Invalid tick time')
    times = pd.to_timedelta(frame.Time)
    prices = pd.to_numeric(frame.deal_price, errors='coerce')
    lots = pd.to_numeric(frame.volume, errors='coerce')
    if (not np.isfinite(prices).all() or not prices.gt(0).all()
            or not np.isfinite(lots).all() or not lots.ge(0).all()
            or not lots.mod(1).eq(0).all() or lots.gt(2**40).any()
            or times.lt(pd.Timedelta(0)).any() or times.ge(pd.Timedelta(days=1)).any()):
        raise ReplayDataUnavailable('Invalid tick price, volume or time')
    result = pd.DataFrame({'time': times, 'price': prices.astype(float),
                           'shares': lots.astype('int64')*1000})
    return result.sort_values('time', kind='stable').reset_index(drop=True)


def match_ticks(ticks, side, limit, qty, adv, participation):
    """Only post-order trade-through volume; same-price queue gets zero credit."""
    if side not in ('buy', 'sell') or type(qty) is not int or qty < 0 or qty % 1000:
        raise ValueError('Invalid board order')
    if not all(math.isfinite(v) and v > 0 for v in (limit, adv, participation)) or participation > 1:
        raise ValueError('Invalid execution assumptions')
    clock = ticks.time
    during = clock.gt(pd.Timedelta('09:01:00')) & clock.lt(pd.Timedelta('13:25:00'))
    price_ok = ticks.price.lt(limit) if side == 'buy' else ticks.price.gt(limit)
    eligible = ticks.loc[during & price_ok]
    cumulative = eligible.shares.cumsum()
    volume = int(cumulative.iloc[-1]) if len(cumulative) else 0
    cap = int(min(volume, adv)*participation)//1000*1000
    fill = min(qty, cap)
    last = None
    if fill:
        # Decimal threshold avoids floating rounding moving a fill one tick early.
        threshold = int((Decimal(fill)/Decimal(str(participation))).to_integral_value(rounding=ROUND_CEILING))
        at = eligible.loc[cumulative.ge(threshold)].iloc[0]
        last = str(at.time).split('days ')[-1]
    return dict(filled_qty=fill, capacity_qty=cap, eligible_shares=volume,
                last_fill_time=last, participation_limit=participation)


class IntradayOrders:
    def setup_intraday(self, ticks, ranking, participation, friction):
        self.ticks, self.ranking = ticks, ranking
        self.intraday_participation = participation
        self.stress_slippage = friction
        self.plans, self.reserved, self.preplanned = [], {}, {}
        self.attempted_today = set()

    def prior(self, day, sid):
        index = self.positions[day]
        if not index or not self.raw(self.days[index-1], sid):
            return None
        return super().prior(day, sid)

    def corporate_day(self, day):
        # All entry slots and budgets are fixed before any today's observations.
        self.attempted_today = set()
        previous_cash = self.cash
        opening_members = set(self.holdings)
        occupied = sum(s != '0050' for s in opening_members)
        income = super().corporate_day(day)
        cash = min(previous_cash, self.cash)
        self.reserved, self.preplanned = {}, {}
        previous_day = str(self.days[self.positions[day]-1].date())
        candidates = list(self.events.get(day, []))
        if self.ranking == 'capacity':
            candidates.sort(key=lambda e: (-float(self.amount20.at[day,e['members'][0]]),
                                           e['members'][0], e['event_id']))
        if self.benchmark:
            candidates = [dict(members=['0050'], event_id='benchmark', signal_date=previous_day)]
        chosen = []
        for event in candidates:
            sid, identity = event['members'][0], event['event_id']
            price = self.prior(day, sid)
            valid = self.benchmark or (occupied < self.slots and sid not in opening_members
                and math.isfinite(float(self.amount20.at[day,sid])) and self.amount20.at[day,sid] >= 50e6)
            budget = min(cash, self.previous_nav / (1 if self.benchmark else self.slots))
            limit = limit_price(price, sid, 'buy') if price else None
            qty = self._affordable(int(budget/limit)//1000*1000, 1000, limit, budget, sid) if valid and limit else 0
            plan = dict(date=str(day.date()), signal_date=event['signal_date'], stock_id=sid,
                        event_id=identity, limit_price=limit, planned_qty=qty, reserved_cash=0.)
            if qty:
                reserve = -self._costs(limit, qty, 'buy', sid)['cash_change']
                plan['reserved_cash'] = reserve
                self.reserved[identity], self.preplanned[identity] = reserve, plan
                cash = money(cash-reserve)
                occupied += 1
                opening_members.add(sid)
                chosen.append(event)
            else:
                plan['rejection'] = 'opening_slots_liquidity_cash_or_one_lot'
            self.plans.append(plan)
        self.events[day] = [] if self.benchmark else chosen
        return income

    def buy_etf(self, day, reason):
        if not self.benchmark or '0050' in self.attempted_today:
            return
        plan = self.preplanned.get('benchmark')
        if not plan:
            return
        self.corporate.prepare('0050')
        self.holdings.setdefault('0050', dict(qty=0,event_id='benchmark',due_index=10**9))
        self.order(day,'0050','buy',plan['planned_qty'],reason,'benchmark',plan['signal_date'])

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if sid in self.attempted_today:
            raise ValueError('Same stock/day tick volume cannot fund multiple orders')
        self.attempted_today.add(sid)
        if side == 'buy':
            plan = self.preplanned[event_id]
            qty, limit, signal_date = plan['planned_qty'], plan['limit_price'], plan['signal_date']
        else:
            state = self.exit_states.get(event_id)
            if not state or not state['trigger_reason']:
                raise ValueError('Exit must be latched before order day')
            reason, signal_date = state['trigger_reason'], state['signal_date']
            price = self.prior(day,sid)
            limit = limit_price(price,sid,side) if price else None
        day_text = str(day.date())
        if not signal_date or pd.Timestamp(signal_date) >= day:
            raise ValueError('Order signal must precede execution day')
        # Distribution-created odd shares stay in the account, visibly unexecuted.
        if qty % 1000:
            self.orders.append(dict(date=day_text,stock_id=sid,side=side,channel='odd',
                requested_qty=qty%1000,filled_qty=0,event_id=event_id,signal_date=signal_date,
                reason=reason,failure='historical_odd_tick_unavailable'))
        qty = qty//1000*1000
        if not qty:
            return 0
        adv = float(self.volume20.at[day,sid])
        amount = float(self.amount20.at[day,sid])
        row = dict(date=day_text,stock_id=sid,name=self.names.get(sid,sid),side=side,
            channel='board',requested_qty=qty,filled_qty=0,reason=reason,event_id=event_id,
            signal_date=signal_date,limit_price=limit,reference_price=limit,
            order_time='09:01:00',failure=None,
            prior_avg_volume20=adv if math.isfinite(adv) else None,
            prior_avg_amount20=amount if math.isfinite(amount) else None,
            day_volume=self.raw(day,sid,'volume'))
        if not limit or not math.isfinite(adv) or adv <= 0:
            row['failure'] = 'missing_previous_price_or_adv'
            self.orders.append(row)
            return 0
        frame, digest = self.ticks.get(sid,day_text,self.markets[sid])
        row['ticks_sha256'] = digest
        high, low, volume = self.raw(day,sid,'high'), self.raw(day,sid,'low'), self.raw(day,sid,'volume')
        if (not high or not low or not volume or frame.price.max() > high+1e-6
                or frame.price.min() < low-1e-6 or int(frame.shares.sum()) > volume*1.01):
            raise ReplayDataUnavailable(f'Tick/daily price or unit conflict: {sid} {day_text}')
        row['tick_daily_volume_ratio'] = float(frame.shares.sum()/volume)
        row.update(match_ticks(frame,side,limit,qty,adv,self.intraday_participation))
        filled = row['filled_qty']
        if filled:
            paid = self._costs(limit,filled,side,sid)
            if side == 'buy' and -paid['cash_change'] > self.reserved[event_id]+.01:
                raise ValueError('Fill exceeds precommitted cash')
            if self.cash+paid['cash_change'] < 0:
                raise ValueError('Execution would overdraw the account')
            self.cash_move(day,side,paid['cash_change'],stock_id=sid,event_id=event_id,channel='board')
            self.holdings[sid]['qty'] += filled if side == 'buy' else -filled
            mark = self.raw(day,sid)
            self.marks[sid] = dict(price=mark,date=day_text)
            self.day_cost += paid['total_cost']
            self.day_basis += filled*(mark-limit)*(1 if side == 'buy' else -1)
            trade = dict(row,**paid,qty=filled,cash_after=self.cash,
                remaining_shares=self.holdings[sid]['qty'],sequence=len(self.trades)+1)
            trade.pop('filled_qty'); trade.pop('failure')
            self.trades.append(trade)
        if filled < qty:
            row['failure'] = 'partial_trade_through_capacity' if filled else 'no_trade_through_capacity'
        self.orders.append(row)
        return filled

    def run(self):
        account = super().run()
        account['plans'] = self.plans
        account['settings'].update(execution='board_tick_trade_through',
            ranking=self.ranking,participation=self.intraday_participation,
            slippage=self.stress_slippage,odd_tick_verified=False,live_qualified=False)
        return account


class IntradayReplay(IntradayOrders, CashRiskReplay):
    def __init__(self,*args,ticks,ranking='original',participation=.01,friction=.0045,**kwargs):
        super().__init__(*args,stress_mode='control',**kwargs)
        self.setup_intraday(ticks,ranking,participation,friction)


class IntradayBenchmark(IntradayOrders, StressOrder, Replay):
    def __init__(self,*args,ticks,participation=.01,friction=.0045,**kwargs):
        super().__init__(*args,benchmark=True,**kwargs)
        self._setup_stress('control')
        self.setup_intraday(ticks,'benchmark',participation,friction)


def audit_intraday(account, ticks, markets):
    checked = audit_stress(account)
    plans = {(p['date'],p['event_id']):p for p in account['plans'] if p['planned_qty']}
    for day in account['daily']:
        reserve = sum(p['reserved_cash'] for p in account['plans'] if p['date']==day['date'])
        index = account['daily'].index(day)
        prior_cash = account['daily'][index-1]['cash'] if index else account['settings']['initial_cash']
        if reserve > prior_cash+.01:
            raise ValueError('Planned orders use later cash')
    seen = set()
    for row in account['orders']:
        if row['channel'] != 'board' or 'ticks_sha256' not in row:
            continue
        identity = (row['date'],row['stock_id'])
        if identity in seen:
            raise ValueError('Tick tape reused within account/day')
        seen.add(identity)
        if row['signal_date'] >= row['date']:
            raise ValueError('Noncausal order')
        if row['side'] == 'buy':
            plan = plans[(row['date'],row['event_id'])]
            if (row['requested_qty'],row['limit_price']) != (plan['planned_qty'],plan['limit_price']):
                raise ValueError('Order differs from precommitted plan')
        frame,digest = ticks.get(row['stock_id'],row['date'],markets[row['stock_id']])
        if digest != row['ticks_sha256']:
            raise ValueError('Execution source drift')
        # Independent volume and first threshold crossing calculation.
        cumulative, last = 0, None
        threshold = Decimal(row['filled_qty'])/Decimal(str(row['participation_limit']))
        for tick in frame.itertuples():
            eligible = (pd.Timedelta('09:01:00') < tick.time < pd.Timedelta('13:25:00')
                and (tick.price < row['limit_price'] if row['side']=='buy' else tick.price > row['limit_price']))
            if eligible:
                cumulative += tick.shares
                if row['filled_qty'] and last is None and cumulative >= threshold:
                    last = str(tick.time).split('days ')[-1]
        capacity = int(min(cumulative,row['prior_avg_volume20'])*row['participation_limit'])//1000*1000
        if (cumulative != row['eligible_shares'] or row['filled_qty'] != min(row['requested_qty'],capacity)
                or row['last_fill_time'] != last):
            raise ValueError('Tick fill failed independent audit')
    checked.update(precommitted_cash=True,tick_volume_recomputed=True,causal_orders=True)
    return checked
