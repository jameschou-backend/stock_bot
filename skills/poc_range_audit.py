"""Independently rebuild both daily range proxies, plans and exclusions.

No matcher is used to certify itself. Source checks retain the existing quality
contract; daily prices and capacity never certify an intraday execution clock.
"""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import math
from types import SimpleNamespace

import pandas as pd

from skills.historical_universe_completion import resolve_completion
from skills.million_replay import costs, money
from skills.mixed_odd_replay import sized_quantity
from skills.poc_executable_replay import finite, validate_tape
from skills.poc_gap_audit import GAP_POLICY, _key, _one, _near, audit_gap_records
from skills.poc_intraday_execution import validate_intraday_odd
from skills.poc_range_execution import MODEL, OPEN, END, ODD_OPEN, ODD_END
from skills.replay_market_feeds import ReplayDataUnavailable


def _fractions(buy, sell):
    if (not finite(buy) or not finite(sell) or (buy, sell) not in ((.5, .5), (.7, .3))):
        raise ValueError('Unregistered range audit fractions')


def _proxy_metadata(row, *, volume, high, low, proxy, fraction, capacity, board):
    false_fields = ('actual_fill_verified', 'intraday_tick_verified', 'odd_tick_verified',
        'board_tick_verified', 'price_level_volume_verified', 'within_window_execution_verified', 'live_qualified')
    if (row.get('participation_limit') != .01 or type(row.get('source_volume')) is not int
            or row.get('source_volume') != volume or row.get('daily_capacity_qty') != capacity
            or type(row.get('filled_qty')) is not int or type(row.get('capacity_qty')) is not int
            or row.get('used_shares') != 0 or row.get('source_high') != high or row.get('source_low') != low
            or row.get('proxy_price') != proxy or row.get('price_fraction') != fraction
            or row.get('actual_fill_time') is not None or row.get('last_fill_time') is not None
            or row.get('execution_evidence') != ('ordinary_daily_range_fraction_proxy' if board else 'intraday_odd_daily_range_fraction_proxy')
            or row.get('volume_scope') != ('ordinary_session' if board else 'intraday_odd_session')
            or row.get('evidence_status') != ('parent_quality_checked_regular_tape' if board else 'official_intraday_daily_table')
            or row.get('source_hash_verification_required_by_caller') is not True
            or any(row.get(field) is not False for field in false_fields)):
        raise ValueError('Range proxy metadata invents or changes execution evidence')


def audit_proxy_cash(account):
    """Fund every purchase before any sale when odd execution time is unknown."""
    for index, day in enumerate(account['daily']):
        cash = account['daily'][index-1]['cash'] if index else account['settings']['initial_cash']
        cash = money(cash+sum(r['cash_change'] for r in account['cash_ledger']
            if r['date'] == day['date'] and r['kind'] not in ('initial_deposit', 'buy', 'sell')))
        rows = [t for t in account['trades'] if t['date'] == day['date']]
        for trade in sorted(rows, key=lambda t: t['side'] == 'sell'):
            cash = money(cash+trade['cash_change'])
            if cash < 0:
                raise ValueError('Unknown intraday timing cannot borrow same-day sale proceeds')
        _near(cash, day['cash'], 'independent session cash')
    return dict(session_cash_verified=True, all_purchases_funded_before_sales=True,
                ledger_sequence_is_execution_clock=False, intraday_sequence_verified=False)


def audit_range_children(account, ticks, odds, routes, quotes, days, corp, feeds, *, verified_halts=(), buy_fraction=.5, sell_fraction=.5):
    """Independently reconstruct committed limits, daily proxies and cash."""
    _fractions(buy_fraction, sell_fraction)
    calendar = pd.DatetimeIndex(days)
    prior = {str(calendar[i].date()): str(calendar[i-1].date()) for i in range(1, len(calendar))}
    source = quotes.copy(); source['date'] = pd.to_datetime(source.date)
    if source.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate audit quote')
    closes = source.pivot(index='date', columns='stock_id', values='close').reindex(calendar)
    volumes = source.pivot(index='date', columns='stock_id', values='volume').reindex(calendar)
    adv = volumes.rolling(20, min_periods=20).mean().shift(1)
    amounts = (closes*volumes).rolling(20, min_periods=20).mean().shift(1)
    opening_cash = {r['date']: account['daily'][i-1]['cash'] if i else account['settings']['initial_cash']
                    for i, r in enumerate(account['daily'])}
    plans, budgets, spent = {}, {}, {}
    for p in account['tick_plans']:
        key = p['date'], p['event_id'], p['side']
        if key in plans or prior.get(p['date']) != p['reference_date'] or p['signal_date'] > p['reference_date']:
            raise ValueError('Noncausal or duplicate execution plan')
        if p['side'] == 'buy' and p['signal_date'] != p['reference_date']:
            raise ValueError('Buy delayed past next session')
        if (p['order_time'], p['expires_at'], p['odd_order_time'], p['odd_expires_at']) != (OPEN, END, ODD_OPEN, ODD_END):
            raise ValueError('Execution plan window differs')
        if (any(type(p[k]) is not int or p[k] < 0 for k in ('planned_qty', 'board_qty', 'odd_qty'))
                or p['board_qty'] % 1000 or p['odd_qty'] >= 1000
                or p['planned_qty'] != p['board_qty']+p['odd_qty']):
            raise ValueError('Execution plan quantity differs')
        sid, day = p['stock_id'], pd.Timestamp(p['date'])
        raw = float(closes.at[pd.Timestamp(p['reference_date']), sid])
        if math.isfinite(raw) and raw > 0:
            reference = corp.reference_price(sid, p['date'], raw)
            _near(p['prior_reference'], reference, 'prior reference', 1e-8)
        elif p['planned_qty']:
            raise ValueError('Order sized without previous-session reference')
        if p['planned_qty']:
            limits = feeds.get_limits(sid).get(p['date'])
            if not limits:
                raise ValueError('Plan has no legal limits')
            bound = limits['upper' if p['side'] == 'buy' else 'lower']
            if p['limit_price'] != bound or p['odd_limit'] != bound:
                raise ValueError('Plan was not fixed at legal marketable limit')
            if p['side'] == 'buy':
                maximum = max(0, math.floor((p['sizing_budget']-40)/(p['prior_reference']*(1+.001425+.0045))))
                class Cost:
                    _costs = staticmethod(costs)
                if p['planned_qty'] != sized_quantity(maximum, bound, p['sizing_budget'], Cost(), sid):
                    raise ValueError('Plan size differs from opening budget')
        plans[key] = p
        budgets[p['date']] = budgets.get(p['date'], 0.)+p['reserved_cash']
        if budgets[p['date']] > opening_cash[p['date']]+.011:
            raise ValueError('Plans use unconfirmed same-day proceeds')
    required = {(key, channel) for key, p in plans.items()
                for channel in ('board', 'odd') if p[channel+'_qty'] > 0}
    verified, seen, halted = {}, set(), set()
    for row in account['orders']:
        if row['channel'] == 'event':
            key = row['date'], row['event_id'], row['side']
            p = plans.get(key)
            if not p or row['filled_qty'] or row['stock_id'] != p['stock_id'] or row['signal_date'] != p['signal_date']:
                raise ValueError('Unexpected non-tradable event fill')
            if not p['planned_qty']:
                if not row.get('failure'):
                    raise ValueError('Rejected zero plan lacks a reason')
                continue
            matching = []
            for h in verified_halts:
                s = h.get('source_row', [])
                full = (h.get('kind') == 'trading_suspension' or
                    h.get('kind') == 'information_halt' and len(s) == 7 and s[4] in ('8:00','08:00') and s[6] in ('8:00','08:00'))
                market = routes.get((p['stock_id'], row['date']), routes.get(p['stock_id']))
                if (full and h['stock_id'] == p['stock_id'] and h.get('start') and h.get('end')
                    and h['start'] <= row['date'] < h['end'] and h['market'].upper() == market.upper()):
                    matching.append(h)
            if (key in halted or row['failure'] != 'official_full_session_halt'
                or row['requested_qty'] != p['planned_qty'] or not matching):
                raise ValueError('Nonzero plan lacks independently verified halt evidence')
            halted.add(key)
            continue
        day, sid = row['date'], row['stock_id']
        key = day, row['event_id'], row['side']
        identity = day, sid, row['channel']
        if identity in seen:
            raise ValueError('Execution tape reused')
        seen.add(identity); p = plans[key]
        channel = row['channel']
        if channel not in ('board', 'odd') or p['stock_id'] != sid or p['signal_date'] != row['signal_date']:
            raise ValueError('Execution differs from plan identity')
        limit = p['limit_price' if channel == 'board' else 'odd_limit']
        if row['limit_price'] != limit or row['requested_qty'] != p[channel+'_qty']:
            raise ValueError('Execution differs from planned price/quantity')
        if (row['order_time'], row['expires_at']) != ((OPEN, END) if channel == 'board' else (ODD_OPEN, ODD_END)):
            raise ValueError('Execution timing differs')
        _near(row['prior_avg_volume20'], float(adv.at[pd.Timestamp(day), sid]), 'prior ADV', 1e-7)
        _near(row['prior_avg_amount20'], float(amounts.at[pd.Timestamp(day), sid]), 'prior turnover', 1e-5)
        market = routes.get((sid, day), routes.get(sid))
        fraction = buy_fraction if row['side'] == 'buy' else sell_fraction
        if channel == 'board':
            tape, digest = ticks.get(sid, day, market)
            if digest != row['ticks_sha256']:
                raise ValueError('Execution tick hash differs')
            quote = _one(source.loc[source.date.eq(pd.Timestamp(day)) & source.stock_id.eq(sid)].to_dict('records'), 'day quote')
            evidence = ticks.audit_day(sid, day, market, tape, {k: quote[k] for k in ('open', 'high', 'low', 'close', 'volume')})
            if row.get('tape_evidence') != evidence:
                raise ValueError('Ordinary source audit evidence differs')
            bounds = feeds.get_limits(sid).get(day)
            validate_tape(tape)
            if (not bounds or tape.price.lt(bounds['lower']-1e-8).any()
                    or tape.price.gt(bounds['upper']+1e-8).any()):
                raise ValueError('Ordinary source outside official legal limits')
            # Independently sum only normal-session positive prints, including
            # delayed close. No per-price liquidity or order timestamp is inferred.
            regular = [t for t in tape.itertuples(index=False)
                       if pd.Timedelta('09:00:00') <= t.time < pd.Timedelta('13:34:00') and t.shares > 0]
            volume = sum(int(t.shares) for t in regular)
            high = max(float(t.price) for t in regular) if volume else None
            low = min(float(t.price) for t in regular) if volume else None
            proxy = float(Decimal(str(low))+Decimal(str(fraction))*(Decimal(str(high))-Decimal(str(low)))) if volume else None
            if row['prior_avg_volume20'] <= 0:
                raise ValueError('Invalid prior ADV for range capacity')
            cap_adv = int(Decimal(str(row['prior_avg_volume20']))/100000)*1000
            day_capacity = volume//100000*1000
            cap = min(day_capacity, cap_adv)
            crosses = proxy is not None and (proxy < limit if row['side'] == 'buy' else proxy > limit)
            capacity = cap if crosses else 0
            shares = min(p['board_qty'], capacity)
            price, last = (proxy if shares else None), None
            if (row.get('allocations') != [] or row.get('prior_adv_capacity_qty') != cap_adv
                    or row.get('regular_daily_capacity_qty') != day_capacity
                    or row.get('source_window') != '09:00:00<=time<13:34:00_including_delayed_close'):
                raise ValueError('Ordinary daily capacity or timing claims differ')
            _proxy_metadata(row, volume=volume, high=high, low=low, proxy=proxy,
                            fraction=fraction, capacity=cap, board=True)
        else:
            raw = odds.get_odd(day, sid, market)
            if (not isinstance(raw, dict) or raw.get('after_hours') is not False
                    or raw.get('volume_scope') != 'intraday_odd_session'
                    or raw.get('volume_unit') != 'shares' or raw.get('price_unit') != 'TWD_per_share'
                    or raw.get('evidence_status') != 'official_intraday_daily_table'
                    or raw.get('intraday_tick_verified') is not False or raw.get('actual_fill_verified') is not False
                    or raw.get('auction_time') is not None
                    or raw.get('source_date') != day or not isinstance(raw.get('market'), str)
                    or raw['market'].upper() != market.upper()
                    or type(raw.get('odd_shares')) is not int or not 0 <= raw['odd_shares'] <= 2**53-1):
                raise ValueError('Odd proxy lacks a verified intraday daily source')
            volume, high, low = raw['odd_shares'], raw.get('odd_high'), raw.get('odd_low')
            if volume:
                if not finite(high) or not finite(low) or not 0 < low <= high:
                    raise ValueError('Odd proxy has invalid daily bounds')
                bounds = feeds.get_limits(sid).get(day)
                if not bounds or not bounds['lower']-1e-8 <= low <= high <= bounds['upper']+1e-8:
                    raise ValueError('Odd proxy daily high/low exceed official legal range')
                proxy = float(Decimal(str(low))+Decimal(str(fraction))*(Decimal(str(high))-Decimal(str(low))))
            else:
                if high is not None or low is not None:
                    raise ValueError('Zero-volume odd proxy has fabricated prices')
                proxy = None
            crosses = proxy is not None and (proxy < limit if row['side'] == 'buy' else proxy > limit)
            capacity = volume//100 if crosses else 0
            shares = min(p['odd_qty'], capacity)
            price, last = (proxy if shares else None), None
            _proxy_metadata(row, volume=volume, high=high, low=low, proxy=proxy,
                            fraction=fraction, capacity=volume//100, board=False)
        if row['filled_qty'] != shares or row['capacity_qty'] != capacity or row['last_fill_time'] != last:
            raise ValueError('Execution share/clock/capacity audit differs')
        failure = (('zero_regular_session_volume' if channel == 'board' else 'official_zero_intraday_odd_volume') if not volume
                   else 'range_limit_not_crossed' if not crosses
                   else 'range_capacity_zero' if not shares
                   else 'partial_range_capacity' if shares < row['requested_qty'] else None)
        if row.get('failure') != failure:
            raise ValueError('Range fill outcome differs from source capacity or limit')
        if shares:
            _near(row['reference_price'], price, 'fill price', 1e-8)
        elif row['reference_price'] is not None:
            raise ValueError('Unfilled order has a fabricated fill price')
        verified[(key, channel)] = row
    halted_children = {item for item in required if item[0] in halted}
    if set(verified) & halted_children or set(verified) | halted_children != required:
        raise ValueError('Nonzero planned child order missing, unexpected, or both halted and executed')
    fills = {}
    for trade in account['trades']:
        key = trade['date'], trade['event_id'], trade['side']
        row = verified[(key, trade['channel'])]
        if ((key, trade['channel']) in fills or type(trade['qty']) is not int or trade['qty'] <= 0
                or trade['qty'] != row['filled_qty']
                or trade['stock_id'] != row['stock_id'] or trade['signal_date'] != row['signal_date']):
            raise ValueError('Trade quantity differs from order evidence')
        if trade['channel'] in ('board', 'odd'):
            for field in ('execution_evidence', 'last_fill_time', 'actual_fill_time', 'actual_fill_verified',
                          'intraday_tick_verified', 'odd_tick_verified', 'within_window_execution_verified',
                          'price_level_volume_verified', 'source_high', 'source_low', 'source_volume', 'proxy_price',
                          'board_tick_verified', 'price_fraction', 'participation_limit', 'volume_scope',
                          'source_hash_verification_required_by_caller', 'live_qualified'):
                if trade.get(field) != row.get(field):
                    raise ValueError('Trade evidence differs from its daily range proxy order')
        fills[(key, trade['channel'])] = trade['qty']
        _near(trade['reference_price'], row['reference_price'], 'trade reference', 1e-8)
        expected = costs(row['reference_price'], trade['qty'], trade['side'], trade['stock_id'])
        for name, value in expected.items():
            _near(trade[name], value, 'trade '+name)
        if trade['side'] == 'buy':
            spent[key] = money(spent.get(key, 0.)-trade['cash_change'])
            if spent[key] > plans[key]['reserved_cash']+.011:
                raise ValueError('Combined channel spend exceeds planned budget')
    if any(fills.get(k, 0) != row['filled_qty'] for k, row in verified.items()):
        raise ValueError('Filled order omitted from trade ledger')
    return dict(**audit_proxy_cash(account), preplanned_limits_verified=True, opening_cash_only=True,
        all_planned_children_reconciled=len(required), verified_halt_children=len(halted_children),
        ordinary_daily_proxies_rebuilt=sum(r['channel'] == 'board' for r in account['orders']),
        chronological_board_allocations_rebuilt=0, board_tick_verified=False,
        intraday_daily_proxies_rebuilt=sum(r['channel'] == 'odd' for r in account['orders']),
        intraday_odd_tick_verified=False, proxy_not_tick=True,
        fills_reconciled=len(fills), actual_fill_verified=False, live_qualified=False)


def _rebuild_failure(gap, ticks, odds, routes, source, calendar, corp, feeds, identity_report, panels, verified_halts):
    p = gap['original_plan']; day, sid, side, _ = _key(gap)
    market = routes.get((sid, day), routes.get(sid))
    stage = 'execution_context'
    try:
        if identity_report is not None:
            identity = resolve_completion(identity_report, sid, day)
            if identity['status'] not in ('identified', 'official_trading_suspension'):
                raise ReplayDataUnavailable('Execution requires identified dated security')
            if identity['status'] == 'identified' and identity['category'] != ('ETF' if sid == '0050' else '股票'):
                raise ReplayDataUnavailable('Execution security category differs')
            market = identity['market'].upper()
        if not market or market != gap.get('market'):
            raise ValueError('Gap market lacks independently dated identity')
        # These are prior inputs, reconstructed from the original daily matrix.
        close, adv_panel, amount_panel = panels
        raw = float(close.at[pd.Timestamp(p['reference_date']), sid])
        prior = corp.reference_price(sid, day, raw) if finite(raw) and raw > 0 else None
        adv = float(adv_panel.at[pd.Timestamp(day), sid])
        amount = float(amount_panel.at[pd.Timestamp(day), sid])
        known_prior_halt = any(h['stock_id'] == sid and h['start'] <= p['reference_date'] < h['end']
                              for h in verified_halts)
        unknown = [name for name, v in [('prior_price', prior), ('adv20', adv), ('amount20', amount)]
                   if (v is None or not math.isfinite(v)) and not (name == 'prior_price' and known_prior_halt)]
        if unknown:
            raise ReplayDataUnavailable(f'Unknown execution inputs: {sid} {day} {",".join(unknown)}')
        for channel in ('board', 'odd'):
            requested = p[channel+'_qty']
            if not requested:
                continue
            stage = channel
            bounds = feeds.get_limits(sid).get(day)
            limit = p['limit_price' if channel == 'board' else 'odd_limit']
            if not bounds or not finite(limit) or not 0 < bounds['lower'] <= limit <= bounds['upper']:
                raise ReplayDataUnavailable('Order lacks a valid preknown legal limit')
            stage = channel
            if channel == 'board':
                tape, _ = ticks.get(sid, day, market)
                quote = _one(source.loc[source.date.eq(pd.Timestamp(day)) & source.stock_id.eq(sid)].to_dict('records'), 'gap day quote')
                ticks.audit_day(sid, day, market, tape, {k: quote[k] for k in ('open', 'high', 'low', 'close', 'volume')})
                if tape.price.lt(bounds['lower']-1e-8).any() or tape.price.gt(bounds['upper']+1e-8).any():
                    raise ReplayDataUnavailable('Tape prices outside official legal range')
                validate_tape(tape)
                if not finite(adv) or adv <= 0:
                    raise ValueError('Invalid prior ADV for range capacity')
            else:
                odd = odds.get_odd(day, sid, market)
                validate_intraday_odd(odd)
                if (odd.get('source_date') != day or not isinstance(odd.get('market'), str)
                        or odd['market'].upper() != market.upper()):
                    raise ReplayDataUnavailable('Intraday odd source date or market differs from the order')
                if odd['odd_shares'] and not (bounds['lower']-1e-8 <= odd['odd_low']
                                              <= odd['odd_high'] <= bounds['upper']+1e-8):
                    raise ReplayDataUnavailable('Intraday odd daily range outside official legal limits')
    except ReplayDataUnavailable as exc:
        if str(exc) != gap['failure_reason'] or stage != gap['failure_stage']:
            raise ValueError('Gap failure cannot be independently reproduced') from exc
        return dict(date=day, stock_id=sid, side=side, event_id=p['event_id'],
                    stage=stage, reason=str(exc), excluded_children=deepcopy(gap['excluded_children']))
    raise ValueError('Gap sources are usable; exclusion is unsupported')


def audit_range_execution(account, ticks, odds, routes, quotes, days, corp, feeds,
                        *, verified_halts=(), identity_report=None, buy_fraction=.5, sell_fraction=.5):
    _fractions(buy_fraction, sell_fraction)
    audit = audit_gap_records(account)
    gaps = account.get('data_gap_exclusions', [])
    settings = account['settings']
    if (settings.get('data_gap_policy') != GAP_POLICY
            or settings.get('posthoc_data_exclusion') is not True
            or settings.get('excluded_event_count') != len(gaps)
            or settings.get('execution') != MODEL or settings.get('odd_participation') != .01
            or settings.get('board_participation') != .01
            or settings.get('buy_fraction') != buy_fraction or settings.get('sell_fraction') != sell_fraction
            or any(settings.get(field) is not False for field in ('intraday_tick_verified', 'odd_tick_verified', 'board_tick_verified', 'live_qualified',
                       'actual_fill_verified', 'within_window_execution_verified', 'price_level_volume_verified'))):
        raise ValueError('Data-gap experiment is not identified in settings')
    calendar = pd.DatetimeIndex(days)
    prior = {str(calendar[i].date()): str(calendar[i-1].date()) for i in range(1, len(calendar))}
    source = quotes.copy(); source['date'] = pd.to_datetime(source.date)
    if source.duplicated(['date', 'stock_id']).any():
        raise ValueError('Gap audit has duplicate daily quotes')
    opening = {r['date']: account['daily'][i-1]['cash'] if i else settings['initial_cash']
               for i, r in enumerate(account['daily'])}
    seen, totals = set(), Counter()
    checked_limits, unknown_limits, unknown_references = 0, 0, 0
    gap_keys = {_key(g) for g in gaps}
    for p in account['tick_plans']:
        key = p['date'], p['event_id'], p['side']
        if key in seen or not finite(p['reserved_cash']) or p['reserved_cash'] < 0:
            raise ValueError('Full original planned reservation is invalid or duplicated')
        seen.add(key); totals[p['date']] += p['reserved_cash']
        if totals[p['date']] > opening[p['date']]+.011:
            raise ValueError('Full original plans reuse gap budget or same-day proceeds')
        if _key(p) not in gap_keys:
            continue  # The sealed auditor checks every nonexcluded plan below.
        if (prior.get(p['date']) != p['reference_date'] or p['signal_date'] > p['reference_date']
                or p['side'] == 'buy' and p['signal_date'] != p['reference_date']):
            raise ValueError('Gap plan has noncausal signal/reference timing')
        if (p['order_time'], p['expires_at'], p['odd_order_time'], p['odd_expires_at']) != (OPEN, END, ODD_OPEN, ODD_END):
            raise ValueError('Gap plan execution window differs')
        if (any(type(p[k]) is not int or p[k] < 0 for k in ('planned_qty', 'board_qty', 'odd_qty'))
                or p['board_qty'] % 1000 or p['odd_qty'] >= 1000
                or p['planned_qty'] != p['board_qty']+p['odd_qty']):
            raise ValueError('Gap plan quantity differs')
        sid, day = p['stock_id'], p['date']
        rows = source.loc[source.date.eq(pd.Timestamp(p['reference_date'])) & source.stock_id.eq(sid)]
        if len(rows) == 1 and finite(float(rows.iloc[0]['close'])) and rows.iloc[0]['close'] > 0:
            try:
                reference = corp.reference_price(sid, day, float(rows.iloc[0]['close']))
            except ReplayDataUnavailable:
                # Independent replay below must reproduce this precise source
                # failure; a missing input is not assigned a synthetic price.
                unknown_references += 1
            else:
                _near(p['prior_reference'], reference, 'prior reference', 1e-8)
        else:
            unknown_references += 1
        try:
            bounds = feeds.get_limits(sid).get(day)
        except ReplayDataUnavailable:
            bounds = None
        if bounds:
            checked_limits += 1
            bound = bounds['upper' if p['side'] == 'buy' else 'lower']
            if p['limit_price'] != bound or p['odd_limit'] != bound:
                raise ValueError('Gap plan legal limit changed')
            if p['side'] == 'buy':
                maximum = max(0, math.floor((p['sizing_budget']-40)/(p['prior_reference']*1.005925)))
                expected = sized_quantity(maximum, bound, p['sizing_budget'], SimpleNamespace(_costs=costs), sid)
                if expected != p['planned_qty']:
                    raise ValueError('Gap plan was not sized from its original budget')
        else:
            unknown_limits += 1
    if gaps:
        closes = source.pivot(index='date', columns='stock_id', values='close').reindex(calendar)
        volumes = source.pivot(index='date', columns='stock_id', values='volume').reindex(calendar)
        panels = closes, volumes.rolling(20, min_periods=20).mean().shift(1), (closes*volumes).rolling(20, min_periods=20).mean().shift(1)
    else:
        panels = None
    failures = [_rebuild_failure(g, ticks, odds, routes, source, calendar, corp, feeds, identity_report, panels, verified_halts) for g in gaps]
    projected = deepcopy(account)
    projected['tick_plans'] = [p for p in projected['tick_plans'] if _key(p) not in gap_keys]
    projected['orders'] = [r for r in projected['orders'] if _key(r) not in gap_keys]
    good = audit_range_children(projected, ticks, odds, routes, quotes, days, corp, feeds,
        verified_halts=verified_halts, buy_fraction=buy_fraction, sell_fraction=sell_fraction)
    count = good.pop('all_planned_children_reconciled')
    good['nonexcluded_preplanned_limits_verified'] = good.pop('preplanned_limits_verified')
    audit['all_original_planned_children_source_data_present'] = audit['all_original_planned_children_execution_evidence_complete']
    # A complete daily quote is still not proof of an executable intraday path.
    audit['all_original_planned_children_execution_evidence_complete'] = False
    audit.update(gap_source_failures_independently_rebuilt=True, gap_failure_ledger=failures,
        gap_failure_types=dict(Counter(g['failure_reason'].split(':', 1)[0] for g in gaps)),
        gap_plans_with_verified_legal_limits=checked_limits, gap_plans_without_verified_legal_limits=unknown_limits,
        gap_plans_without_verified_prior_reference=unknown_references,
        all_original_preplanned_limits_verified=not unknown_limits,
        nonexcluded_planned_children_reconciled=count,
        original_planned_children=count+audit['data_gap_children'])
    return dict(**good, **audit)
