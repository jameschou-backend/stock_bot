"""Independent checks for the disclosed, whole-order data-gap experiment.

An exclusion is not a halt or an executed trade.  Original plans and their
reservations remain in the account; only the separately audited exclusions are
removed from the projection passed to the sealed execution auditor.
"""
from collections import Counter
from copy import deepcopy
import math
from types import SimpleNamespace

import pandas as pd

from skills.historical_universe_completion import resolve_completion
from skills.million_replay import costs, money
from skills.mixed_odd_replay import sized_quantity
from skills.poc_executable_odd import match_after_hours
from skills.poc_executable_replay import (
    END, ODD_END, ODD_OPEN, OPEN, audit_executable, finite, match_board_prints,
)
from skills.replay_market_feeds import ReplayDataUnavailable


GAP_POLICY = 'user_authorized_whole_order_data_gap_exclusion_v1'
GAP_REASON = 'user_authorized_data_gap_exclusion'


def _key(row):
    return row.get('date'), row.get('stock_id'), row.get('side'), row.get('event_id')


def _one(rows, label):
    if len(rows) != 1:
        raise ValueError('Gap audit requires one '+label)
    return rows[0]


def _near(a, b, label, tolerance=.011):
    if not finite(a) or not finite(b) or not math.isclose(a, b, rel_tol=0, abs_tol=tolerance):
        raise ValueError('Gap audit differs: '+label)


def _journal(value):
    return value.get('partial_journal', value.get('account', value))


def audit_gap_records(value, *, require_complete_days=True):
    """Check zero fills, unchanged assets and reservations, including partial runs.

    This helper deliberately does not certify the data-source failure.  The
    complete audit independently re-reads the frozen sources below.
    """
    account = _journal(value)
    gaps = account.get('data_gap_exclusions', [])
    seen = set()
    day_set = {r['date'] for r in account['daily']}
    for gap in gaps:
        key = _key(gap)
        if key in seen or key[2] not in ('buy', 'sell'):
            raise ValueError('Duplicate or invalid data-gap exclusion')
        seen.add(key)
        plan = _one([p for p in account['tick_plans'] if _key(p) == key], 'original gap plan')
        children = [dict(channel=c, requested_qty=plan[c+'_qty'])
                    for c in ('board', 'odd') if plan[c+'_qty']]
        if (gap.get('policy') != GAP_POLICY or gap.get('original_plan') != plan
                or not plan['planned_qty'] or gap.get('signal_date') != plan['signal_date']
                or gap.get('excluded_children') != children
                or gap.get('excluded_channels') != [r['channel'] for r in children]
                or gap.get('filled_qty') != 0 or gap.get('failure_class') != 'ReplayDataUnavailable'
                or not isinstance(gap.get('failure_reason'), str) or not gap['failure_reason']
                or gap.get('posthoc_data_exclusion') is not True
                or gap.get('source_verification_required_by_caller') is not True
                or gap.get('actual_fill_verified') is not False or gap.get('live_qualified') is not False
                or gap.get('retry_sell') is not (plan['side'] == 'sell')):
            raise ValueError('Gap metadata or original quantity changed')
        rows = [r for r in account['orders'] if _key(r) == key]
        order = _one(rows, 'unfilled gap order')
        if (order.get('channel') != 'event' or order.get('filled_qty') != 0
                or order.get('requested_qty') != plan['planned_qty']
                or order.get('failure') != GAP_REASON or order.get('exclusion_policy') != GAP_POLICY
                or order.get('signal_date') != plan['signal_date']
                or order.get('reason') != gap.get('reason')
                or order.get('failure_reason') != gap['failure_reason']
                or order.get('failure_stage') != gap.get('failure_stage')
                or order.get('excluded_children') != children
                or order.get('actual_fill_verified') is not False or order.get('live_qualified') is not False
                or any(order.get(name) is not None for name in ('reference_price', 'auction_price'))):
            raise ValueError('Gap order was changed, executed, or disguised as a halt')
        for row in account['trades']:
            if row['date'] == key[0] and row['stock_id'] == key[1]:
                raise ValueError('Gap stock-day contains a hidden fill')
        for row in account['cash_ledger']:
            if (row.get('date') == key[0] and row.get('stock_id') == key[1]
                    and row.get('kind') in ('buy', 'sell')):
                raise ValueError('Gap stock-day contains a hidden trade cash change')
        _near(gap['cash_before'], gap['cash_after'], 'gap cash rollback', 0)
        if (type(gap['holding_qty_before']) is not int or type(gap['holding_qty_after']) is not int
                or gap['holding_qty_before'] < 0
                or gap['holding_qty_before'] != gap['holding_qty_after']):
            raise ValueError('Gap order changed held shares')
        for name, journal_name in (('cash_ledger_length_before', 'cash_ledger'),
                                   ('trade_count_before', 'trades'), ('order_count_before', 'orders')):
            count = gap.get(name)
            if type(count) is not int or not 0 <= count <= len(account[journal_name]):
                raise ValueError('Gap journal anchor missing or invalid: '+name)
        if (gap['order_count_before'] >= len(account['orders'])
                or account['orders'][gap['order_count_before']] != order):
            raise ValueError('Gap order journal anchor changed')
        earlier_orders = account['orders'][:gap['order_count_before']]
        fills_before = sum(bool(r.get('filled_qty')) for r in earlier_orders)
        if fills_before != gap['trade_count_before']:
            raise ValueError('Gap lost a prior fill or retained a rolled-back fill')
        ledger_prefix = account['cash_ledger'][:gap['cash_ledger_length_before']]
        if not ledger_prefix:
            raise ValueError('Gap lacks original deposit and cash journal')
        _near(gap['cash_before'], ledger_prefix[-1]['cash_after'], 'gap cash journal anchor')
        if sum(r['kind'] in ('buy', 'sell') for r in ledger_prefix) != gap['trade_count_before']:
            raise ValueError('Gap trade cash journal anchor differs')

        completed = key[0] in day_set
        if require_complete_days and not completed:
            raise ValueError('Gap lacks a completed valuation day')
        if plan['side'] == 'buy':
            if gap['holding_qty_before'] or any(r.get('event_id') == key[3] for r in account['cohorts']):
                raise ValueError('Excluded buy created a cohort or erased an existing holding')
            if any(r.get('date') == key[0] and r.get('stock_id') == key[1] for r in account['holdings']):
                raise ValueError('Excluded buy created a holding')
            if key[1] != '0050':
                _audit_buy_reservation(account, gap, plan)
        else:
            if gap['holding_qty_before'] < plan['planned_qty']:
                raise ValueError('Skipped sell exceeded held shares')
            if completed:
                holding = _one([r for r in account['holdings']
                                if r['date'] == key[0] and r['stock_id'] == key[1]], 'retained sell holding')
                if holding['qty'] != gap['holding_qty_before'] or holding.get('event_id') != key[3]:
                    raise ValueError('Skipped sell erased or changed its holding')
            state = gap.get('exit_state_before')
            if (key[1] != '0050' and (not state or state != gap.get('exit_state_after')
                    or state.get('trigger_reason') != gap['reason']
                    or state.get('signal_date') != gap['signal_date']
                    or not state.get('target_date') or state['target_date'] > key[0])):
                raise ValueError('Skipped sell lost its original exit latch')
            if completed:
                later_days = sorted(d for d in day_set if d > key[0])
                if later_days:
                    next_day = later_days[0]
                    attempts = [r for r in account['orders'] if r.get('date') == next_day
                                and r.get('event_id') == key[3] and r.get('side') == 'sell']
                    if not attempts or any(r.get('signal_date') != gap['signal_date']
                                           or r.get('reason') != gap['reason'] for r in attempts):
                        raise ValueError('Skipped sell was not retried with the original exit signal')
    tagged = {_key(r) for r in account['orders'] if r.get('failure') == GAP_REASON
              or r.get('exclusion_policy') == GAP_POLICY}
    if tagged != seen:
        raise ValueError('Gap order and exclusion ledgers differ')
    return dict(data_gap_plans=len(gaps), data_gap_children=sum(len(r['excluded_children']) for r in gaps),
        data_gap_buys=sum(r['side'] == 'buy' for r in gaps), data_gap_sells=sum(r['side'] == 'sell' for r in gaps),
        gap_zero_fill_and_resource_ledger_verified=True,
        gap_source_failures_independently_rebuilt=False, posthoc_data_exclusion=True,
        all_original_planned_children_execution_evidence_complete=not gaps)


def _audit_buy_reservation(account, gap, plan):
    day, sid, _, eid = _key(gap)
    resource = _one([r for r in account['resource_plans']
                     if r.get('date') == day and r.get('event_id') == eid], 'gap buy reservation')
    if (resource.get('stock_id') != sid or resource.get('signal_date') != plan['signal_date']
            or resource.get('spent') != 0 or resource.get('filled_qty') != 0 or resource.get('failure')
            or resource.get('budget') != plan['reserved_cash']
            or resource.get('opening_cash') != plan['opening_cash']
            or resource.get('planned_qty', 0) < plan['planned_qty']
            or resource.get('locked_after') != money(resource['locked_unused_before']+resource['budget'])):
        raise ValueError('Gap buy released or altered its daily budget')
    resources = [r for r in account['resource_plans'] if r['date'] == day]
    for before, after in zip(resources, resources[1:]):
        if before['locked_after'] != after['locked_unused_before']:
            raise ValueError('Gap daily budget was reused')
    decisions = [r for r in account['slot_decisions'] if r['date'] == day]
    slot = _one([r for r in decisions if r['event_id'] == eid], 'gap slot attempt')
    if (slot.get('stock_id') != sid or slot.get('signal_date') != plan['signal_date']
            or slot.get('attempted') is not True or slot.get('filled_qty') != 0 or slot.get('failure')):
        raise ValueError('Gap buy released its daily slot')
    for later in decisions[decisions.index(slot)+1:]:
        if any(sid not in later.get(name, []) for name in ('attempts_before', 'unfilled_before', 'occupied_before')):
            raise ValueError('Gap buy allowed a same-day replacement slot')


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
                match_board_prints(tape, side, limit, requested, adv)
            else:
                odd = odds.get_odd(day, sid, market)
                if odd is None:
                    raise ReplayDataUnavailable('Missing after-hours auction evidence')
                result = match_after_hours(odd, side, limit, requested)
                price = result['auction_price']
                if price is not None and not bounds['lower']-1e-8 <= price <= bounds['upper']+1e-8:
                    raise ReplayDataUnavailable('Odd auction price outside official legal range')
    except ReplayDataUnavailable as exc:
        if str(exc) != gap['failure_reason'] or stage != gap['failure_stage']:
            raise ValueError('Gap failure cannot be independently reproduced') from exc
        return dict(date=day, stock_id=sid, side=side, event_id=p['event_id'],
                    stage=stage, reason=str(exc), excluded_children=deepcopy(gap['excluded_children']))
    raise ValueError('Gap sources are usable; exclusion is unsupported')


def audit_gap_execution(account, ticks, odds, routes, quotes, days, corp, feeds,
                        *, verified_halts=(), identity_report=None):
    audit = audit_gap_records(account)
    gaps = account.get('data_gap_exclusions', [])
    settings = account['settings']
    if (settings.get('data_gap_policy') != GAP_POLICY
            or settings.get('posthoc_data_exclusion') is not True
            or settings.get('excluded_event_count') != len(gaps)):
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
    good = audit_executable(projected, ticks, odds, routes, quotes, days, corp, feeds, verified_halts=verified_halts)
    count = good.pop('all_planned_children_reconciled')
    good['nonexcluded_preplanned_limits_verified'] = good.pop('preplanned_limits_verified')
    audit.update(gap_source_failures_independently_rebuilt=True, gap_failure_ledger=failures,
        gap_failure_types=dict(Counter(g['failure_reason'].split(':', 1)[0] for g in gaps)),
        gap_plans_with_verified_legal_limits=checked_limits, gap_plans_without_verified_legal_limits=unknown_limits,
        gap_plans_without_verified_prior_reference=unknown_references,
        all_original_preplanned_limits_verified=not unknown_limits,
        nonexcluded_planned_children_reconciled=count,
        original_planned_children=count+audit['data_gap_children'])
    return dict(**good, **audit)
