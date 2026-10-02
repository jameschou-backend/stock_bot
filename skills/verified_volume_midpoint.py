"""HL2 research execution with a separate, exact ordinary-volume input.

Missing ordinary volume blocks that board child visibly. It is never replaced
by total daily volume, an observed provider tape, or an inferred zero. Signals,
ranking and precommitted sizing keep their original total-volume inputs.
"""
import math

import numpy as np
import pandas as pd

from skills.midpoint_replay import MidpointOrders, midpoint_match
from skills.million_replay import money
from skills.replay_market_feeds import ReplayDataUnavailable


def _finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


class VerifiedVolumeMidpointOrders(MidpointOrders):
    """Mixin before a midpoint replay; outer historical-era wrappers stay first.

    ordinary_volumes: {TWSE/TPEX: date-indexed, four-digit-stock-column frames}.
    The caller binds official evidence hashes before supplying these matrices.
    Resolver(day, stock_id) returns a market string or a dated identity mapping.
    When omitted, the replay's existing ``identity(day, stock_id)`` is required
    in strict mode. Each of the 20 preceding market days resolves independently,
    so a venue transfer cannot accidentally use a stock's current venue history.
    """
    def __init__(self, *args, ordinary_volumes=None, ordinary_market_resolver=None,
                 volume_policy='strict', **kwargs):
        if volume_policy not in ('strict', 'legacy_total_research'):
            raise ValueError('Unknown ordinary-volume policy')
        if volume_policy == 'strict' and ordinary_volumes is None:
            raise ValueError('Strict execution requires explicit ordinary-volume matrices')
        if ordinary_market_resolver is not None and not callable(ordinary_market_resolver):
            raise ValueError('Dated ordinary-market resolver must be callable')
        self.volume_policy = volume_policy
        self.ordinary_market_resolver = ordinary_market_resolver
        self.ordinary_volumes = {}
        for name, source in (ordinary_volumes or {}).items():
            market = name.upper()
            if market not in ('TWSE', 'TPEX') or market in self.ordinary_volumes:
                raise ValueError('Invalid or duplicate ordinary market')
            frame = source.copy(deep=True)
            frame.index = pd.DatetimeIndex(frame.index)
            if (frame.index.tz is not None or frame.index.has_duplicates
                    or not frame.index.is_monotonic_increasing
                    or not frame.index.equals(frame.index.normalize())
                    or frame.columns.has_duplicates
                    or any(not isinstance(s, str) or len(s) != 4 or not s.isdigit() for s in frame.columns)):
                raise ValueError('Invalid ordinary-volume matrix axes')
            if any(pd.api.types.is_bool_dtype(frame[c]) for c in frame):
                raise ValueError('Boolean ordinary volume is invalid')
            frame = frame.astype(float)
            values = frame.to_numpy()
            known = ~np.isnan(values)
            if (not np.isfinite(values[known]).all() or (values[known] < 0).any()
                    or (values[known] > 2**53-1).any() or (np.mod(values[known], 1) != 0).any()):
                raise ValueError('Ordinary shares must be nonnegative integers or NaN')
            self.ordinary_volumes[market] = frame
        self.volume_evidence_blocks = []
        super().__init__(*args, **kwargs)
        if volume_policy == 'strict' and ordinary_market_resolver is None and not callable(getattr(self, 'identity', None)):
            raise ValueError('Strict ordinary volume requires a dated market resolver')

    def _ordinary_market(self, day, sid):
        resolver = self.ordinary_market_resolver or getattr(self, 'identity', None)
        value = resolver(day, sid)
        if isinstance(value, dict):
            if value.get('status') not in ('identified', 'official_trading_suspension'):
                return None
            value = value.get('market')
        return value.upper() if isinstance(value, str) and value.upper() in ('TWSE', 'TPEX') else None

    def ordinary_capacity_inputs(self, day, sid):
        index = self.positions[day]
        window = list(self.days[max(0, index-20):index+1])
        values, gaps, markets = [], [], []
        for d in window:
            market = self._ordinary_market(d, sid)
            frame = self.ordinary_volumes.get(market)
            value = None if frame is None or d not in frame.index or sid not in frame else _finite(frame.at[d, sid])
            values.append(value); markets.append(market)
            if value is None:
                gaps.append(dict(date=str(d.date()), market=market, stock_id=sid,
                                 reason='dated_market_unknown' if market is None else 'ordinary_volume_missing'))
        if len(window) != 21:
            gaps.append(dict(date=str(day.date()), market=markets[-1], stock_id=sid,
                             reason='ordinary_history_less_than_20_market_days'))
        return dict(market=markets[-1], current=values[-1],
                    prior_average=sum(values[:-1])/20 if len(window) == 21 and all(v is not None for v in values[:-1]) else None,
                    prior_dates=[str(d.date()) for d in window[:-1]], gaps=gaps)

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        # Keep the frozen midpoint accounting semantics; only board capacity
        # receives a new input. In particular, do not replace self.fields or
        # self.volume20 temporarily: odd orders and sizing use their own scopes.
        if side == 'sell' and reason == 'scheduled_exit' and hasattr(self, 'exit_states'):
            state = self.exit_states.get(event_id)
            if not state or not state['trigger_reason']:
                raise ValueError('Unlatched midpoint exit')
            reason, signal_date = state['trigger_reason'], state['signal_date']
        p = self.day_plans[(event_id, side)]
        if p['stock_id'] != sid or p['signal_date'] != signal_date or sid in self.tick_attempts:
            raise ValueError('Midpoint identity changed or stock/day reused')
        self.tick_attempts.add(sid)
        if hasattr(self, 'identity'):
            identity = self.identity(day, sid)
            if identity['status'] not in ('identified', 'official_trading_suspension'):
                raise ReplayDataUnavailable('Midpoint dated identity unavailable')
            if identity['status'] == 'identified' and identity['category'] != ('ETF' if sid == '0050' else '股票'):
                raise ReplayDataUnavailable('Midpoint security is not eligible')
            self.markets[sid] = identity['market'].upper()
        if self.official_halt(day, sid):
            self.orders.append(dict(date=str(day.date()), stock_id=sid, event_id=event_id, signal_date=signal_date,
                side=side, channel='event', requested_qty=p['planned_qty'], filled_qty=0, reason=reason,
                failure='official_full_session_halt'))
            return 0
        if qty < p['planned_qty']:
            raise ValueError('Post-plan midpoint sizing shrank')
        if p['planned_qty']:
            self.require_prior_inputs(day, sid)
        total = 0
        for channel, requested in (('board', p['board_qty']), ('odd', p['odd_qty'])):
            if not requested:
                continue
            date = str(day.date())
            adv = float(self.volume20.at[day, sid])
            limit = p['limit_price'] if channel == 'board' else p['odd_limit']
            limits = self.feeds.get_limits(sid).get(date)
            if not limits:
                raise ReplayDataUnavailable('Missing midpoint legal limits')
            ordinary = None
            if channel == 'board':
                high, low = (float(self.fields[k].at[day, sid]) for k in ('high', 'low'))
                volume = float(self.fields['volume'].at[day, sid])
                total_volume = volume
                scope = 'all_daily_sessions_research_proxy'
                capacity_adv = adv
                if self.volume_policy == 'strict':
                    ordinary = self.ordinary_capacity_inputs(day, sid)
                    if ordinary['market'] is not None and ordinary['market'] != self.markets[sid]:
                        raise ReplayDataUnavailable('Execution and ordinary evidence dated markets disagree')
                    volume, capacity_adv = ordinary['current'], ordinary['prior_average']
                    scope = 'ordinary_session'
            else:
                odd = self.odd_feeds.get_odd(date, sid, self.markets[sid])
                if odd is None:
                    raise ReplayDataUnavailable('Missing midpoint odd row')
                high, low, volume = (odd.get(k) for k in ('odd_high', 'odd_low', 'odd_shares'))
                total_volume, capacity_adv, scope = None, adv, 'independent_odd_session'
            row = dict(date=date, stock_id=sid, name=self.names.get(sid, sid), side=side, event_id=event_id,
                signal_date=signal_date, reason=reason, channel=channel, requested_qty=requested, limit_price=limit,
                prior_avg_amount20=float(self.amount20.at[day, sid]), prior_avg_volume20=adv,
                capacity_prior_ordinary_volume20=capacity_adv if channel == 'board' and self.volume_policy == 'strict' else None,
                source_total_volume=_finite(total_volume), volume_scope=scope, volume_policy=self.volume_policy,
                ordinary_capacity_verified=bool(channel == 'board' and ordinary is not None and not ordinary['gaps']),
                order_time='08:59:00' if channel == 'board' else '09:00:00', expires_at='13:30:00',
                source_high=_finite(high), source_low=_finite(low), source_volume=_finite(volume), participation_limit=.01,
                execution_evidence='daily_high_low_midpoint_proxy')
            if ordinary is not None and ordinary['gaps']:
                row.update(filled_qty=0, capacity_qty=0, reference_price=None,
                           failure='ordinary_volume_evidence_missing', ordinary_volume_gaps=ordinary['gaps'])
                row['source_high'], row['source_low'] = _finite(high), _finite(low)
                self.volume_evidence_blocks.append(dict(date=date, stock_id=sid, side=side, event_id=event_id,
                    requested_qty=requested, missing=ordinary['gaps'], reason=row['failure']))
                self.orders.append(row)
                continue
            if ordinary is not None and capacity_adv == 0:
                row.update(reference_price=None, capacity_qty=0, filled_qty=0, failure='official_ordinary_history_zero')
            else:
                row.update(midpoint_match(high, low, volume, capacity_adv, requested, side, limit,
                                          limits['lower'], limits['upper'], channel))
            filled, price = row['filled_qty'], row['reference_price']
            if filled:
                paid = self._costs(price, filled, side, sid)
                if money(self.cash+paid['cash_change']) < 0:
                    row.update(filled_qty=0, failure='proceeds_below_costs_insufficient_cash'); filled = 0
                else:
                    self.cash_move(day, side, paid['cash_change'], stock_id=sid, event_id=event_id, channel=channel)
                    self.holdings[sid]['qty'] += filled if side == 'buy' else -filled
                    mark = self.raw(day, sid); self.marks[sid] = dict(price=mark, date=date)
                    self.used[(sid, channel)] += filled; self.day_cost += paid['total_cost']
                    self.day_basis += filled*(mark-price)*(1 if side == 'buy' else -1)
                    trade = dict(row, **paid, qty=filled, cash_after=self.cash, remaining_shares=self.holdings[sid]['qty'],
                                 day_participation=filled/volume, sequence=len(self.trades)+1)
                    trade.pop('filled_qty'); trade.pop('failure', None); self.trades.append(trade)
            total += filled; self.orders.append(row)
        return total

    def run(self):
        result = super().run()
        board = [r for r in result['orders'] if r['channel'] == 'board' and r['requested_qty'] > 0]
        missing_keys = {(g['market'], g['date'], b['stock_id']) for b in self.volume_evidence_blocks for g in b['missing']}
        result['ordinary_volume_evidence'] = dict(policy=self.volume_policy,
            requested_board_children=len(board), verified_board_children=sum(bool(r.get('ordinary_capacity_verified')) for r in board),
            blocked_board_children=len(self.volume_evidence_blocks), blocks=self.volume_evidence_blocks,
            missing_stock_days=[dict(market=m, date=d, stock_id=s) for m,d,s in sorted(missing_keys, key=str)],
            all_requested_board_capacity_observed=bool(board) and self.volume_policy == 'strict'
                and all(r.get('ordinary_capacity_verified') is True for r in board),
            source_hashes_must_be_bound_by_caller=True, actual_fill_verified=False, live_qualified=False)
        last_day = result['daily'][-1]['date'] if result['daily'] else None
        final_holdings = [dict(r) for r in result['holdings'] if r['date'] == last_day and r['qty'] > 0]
        result['ending_inventory'] = dict(valuation='mark_to_market', positions=final_holdings,
            automatically_liquidated=False, remaining_positions=len(final_holdings),
            stale_positions=sum(bool(r.get('stale')) for r in final_holdings))
        result['settings'].update(execution='ordinary_volume_midpoint_research_v1', volume_policy=self.volume_policy,
            board_capacity=('min(official_ordinary_day_volume, prior_20_market_days_ordinary_mean)*0.01'
                if self.volume_policy == 'strict' else 'legacy_total_daily_volume_research_proxy'),
            missing_ordinary_policy='block_board_child_record_gap_preserve_holdings',
            sizing_and_signal_volume='original_total_daily_volume_unchanged',
            ending_inventory_valuation='mark_to_market', forced_end_liquidation=False,
            actual_fill_verified=False, live_qualified=False)
        return result
