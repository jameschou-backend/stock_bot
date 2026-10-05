"""First day of a consecutive red-candidate run, before portfolio selection.

The supplied candidate registry and market calendar must be the complete,
hash-bound research inputs. A day with no candidate is not a red signal; an
unknown candidate candle is an error, never an artificial break in a run.
POC remains a later ordering rule. Holdings, fills and free slots are not inputs.
"""
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import json
import math

import numpy as np
import pandas as pd

from skills.candle_volume_rules import EntryCandleUnavailable


POLICY = 'first_consecutive_red_candidate_only_v1'
SCHEMA = 'poc_first_signal_decision_v1'


def _iso(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('First-signal dates must be ISO market dates')
    return value


def _inputs(entries, signals, start, end):
    start, end = _iso(start), _iso(end)
    if start > end:
        raise ValueError('First-signal research dates are reversed')
    days = pd.DatetimeIndex(signals.days)
    if (not len(days) or days.hasnans or not days.is_unique
            or not days.is_monotonic_increasing or days.tz is not None
            or not days.equals(days.normalize())):
        raise ValueError('First-signal market calendar must be unique and ordered')
    positions = {day: i for i, day in enumerate(days)}
    if signals.positions != positions:
        raise ValueError('First-signal market calendar positions differ')
    calendar = [str(day.date()) for day in days]
    index = {day: i for i, day in enumerate(calendar)}
    if start not in index or end not in index:
        raise ValueError('First-signal research boundary missing from market calendar')
    for frame in [signals.adjusted, *signals.raw.values()]:
        if (not frame.index.equals(days) or not frame.columns.is_unique
                or not frame.columns.equals(signals.adjusted.columns)):
            raise ValueError('First-signal candles and market calendar differ')
    rows = deepcopy(list(entries))
    seen, observations = set(), set()
    previous_date = None
    for event in rows:
        if not isinstance(event, Mapping):
            raise ValueError('First-signal candidate must be a mapping')
        eid, members = event.get('event_id'), event.get('members')
        if (not isinstance(eid, str) or not eid or eid in seen
                or not isinstance(members, list) or len(members) != 1
                or not isinstance(members[0], str)):
            raise ValueError('First-signal candidate identity is invalid or duplicated')
        signal, entry = _iso(event.get('signal_date')), _iso(event.get('entry_date'))
        if signal not in index or entry not in index:
            raise ValueError('First-signal candidate missing from market calendar')
        i = index[signal]
        if i == 0:
            raise ValueError('First-signal history needs the preceding market session')
        if index[entry] != i + 1:
            raise ValueError('First-signal entry must be the next market session')
        if previous_date is not None and signal < previous_date:
            raise ValueError('First-signal candidates must be chronological')
        previous_date = signal
        key = (members[0], signal)
        if key in observations:
            raise ValueError('First-signal stock/session candidate is duplicated')
        if signals.entries.get(eid) != event:
            raise ValueError('First-signal candidate differs from candle registry')
        if members[0] not in signals.adjusted.columns:
            raise ValueError('First-signal stock is missing candle data')
        seen.add(eid)
        observations.add(key)
    if seen != set(signals.entries):
        raise ValueError('First-signal requires the complete candle candidate registry')
    return rows, calendar, index


def _decision(event, gate, previous, previous_gate, previous_session, run, start):
    red = gate['passed']
    previous_red = bool(previous_gate and previous_gate['passed'])
    first = red and not previous_red
    return dict(schema=SCHEMA, policy=POLICY, event_id=event['event_id'],
        stock_id=event['members'][0], signal_date=event['signal_date'],
        entry_date=event['entry_date'], research_start=start,
        previous_session=previous_session,
        previous_event_id=previous['event_id'] if previous else None,
        previous_red_qualified=previous_red,
        previous_red_status=previous_gate['status'] if previous_gate else 'no_raw_signal',
        signal_red_qualified=red, red_status=gate['status'],
        run_start=run[0] if red else None,
        run_start_event_id=run[1] if red else None,
        run_sessions=run[2] if red else 0,
        passed=first, status='first_red_signal' if first else
            'red_signal_continuation' if red else 'signal_not_red',
        portfolio_state_used=False, execution_results_used=False,
        uses_entry_day_candle=False)


def first_signal_entries(entries, candle_signals, start, end):
    """Return preserved entries, the unchanged scoped red gate, and first gate.

    Warmup includes every registered candidate before ``start``. Later candles
    after ``end`` are never read. Outside-period entries remain unchanged and
    in their original order. A failed first-day order cannot reset this gate.
    """
    rows, calendar, index = _inputs(entries, candle_signals, start, end)
    history = [e for e in rows if e['entry_date'] <= end]
    _, red = candle_signals.filter_entries(history)
    gates = {d['event_id']: d for d in red}
    events = {(e['members'][0], e['signal_date']): e for e in history}
    runs, decisions = {}, []
    for event in history:
        sid, signal = event['members'][0], event['signal_date']
        previous_session = calendar[index[signal] - 1]
        previous = events.get((sid, previous_session))
        previous_gate = gates[previous['event_id']] if previous else None
        gate = gates[event['event_id']]
        run = (signal, event['event_id'], 1)
        if gate['passed'] and previous_gate and previous_gate['passed']:
            prior = runs[(sid, previous_session)]
            run = (prior[0], prior[1], prior[2] + 1)
        if gate['passed']:
            runs[(sid, signal)] = run
        if start <= event['entry_date'] <= end:
            decisions.append(_decision(event, gate, previous, previous_gate,
                                       previous_session, run, start))
    allowed = {d['event_id'] for d in decisions if d['passed']}
    selected = [e for e in rows if not start <= e['entry_date'] <= end
                or e['event_id'] in allowed]
    return selected, [d for d in red if start <= d['entry_date'] <= end], decisions


def _scalar_red(event, signals):
    """Audit oracle: do not call the production red gate or first-run selector."""
    sid, signal = event['members'][0], pd.Timestamp(event['signal_date'])
    values = [signals.raw[k].at[signal, sid] for k in ('open', 'high', 'low', 'close')]
    known = all(not isinstance(v, (bool, np.bool_)) and pd.notna(v)
                and math.isfinite(float(v)) and float(v) > 0 for v in values)
    if not known:
        raise EntryCandleUnavailable('Unknown or impossible signal-day OHLC: '+event['event_id'], [])
    opened, high, low, closed = map(float, values)
    if not low <= opened <= high or not low <= closed <= high:
        raise EntryCandleUnavailable('Unknown or impossible signal-day OHLC: '+event['event_id'], [])
    return dict(passed=closed > opened,
                status='red' if closed > opened else 'black' if closed < opened else 'doji')


def audit_first_signal(account, decisions, *, entries, candle_signals, start, end):
    """Rebuild every decision from complete sources and check all entry journals.

    Sources are mandatory: trusting ``passed`` from a saved ledger would not
    certify a first signal. Financial reconciliation remains the cash auditor's
    responsibility; this checks signal identity and the one-day entry window.
    """
    rows, calendar, index = _inputs(entries, candle_signals, start, end)
    history = [e for e in rows if e['entry_date'] <= end]
    events = {(e['members'][0], e['signal_date']): e for e in history}
    gates = {e['event_id']: _scalar_red(e, candle_signals) for e in history}
    rebuilt = []
    for event in history:
        if not start <= event['entry_date'] <= end:
            continue
        sid, signal = event['members'][0], event['signal_date']
        previous_session = calendar[index[signal] - 1]
        previous = events.get((sid, previous_session))
        previous_gate = gates[previous['event_id']] if previous else None
        gate = gates[event['event_id']]
        run = (signal, event['event_id'], 1)
        if gate['passed']:
            j = index[signal] - 1
            while j >= 0:
                earlier = events.get((sid, calendar[j]))
                if earlier is None or not gates[earlier['event_id']]['passed']:
                    break
                run = (earlier['signal_date'], earlier['event_id'], run[2] + 1)
                j -= 1
        rebuilt.append(_decision(event, gate, previous, previous_gate,
                                 previous_session, run, start))
    # JSON comparison distinguishes forged integer flags (1/0) from booleans.
    if (not isinstance(decisions, list)
            or json.dumps(decisions, sort_keys=True, allow_nan=False)
            != json.dumps(rebuilt, sort_keys=True, allow_nan=False)):
        raise ValueError('First-signal decision ledger differs from source reconstruction')
    allowed = {d['event_id']: d for d in rebuilt if d['passed']}
    checked = Counter()

    def check(row, journal, cohort=False):
        decision = allowed.get(row.get('event_id'))
        entry = row.get('entry_date') if cohort else row.get('date')
        if (decision is None or row.get('stock_id') != decision['stock_id']
                or row.get('signal_date') != decision['signal_date']
                or entry != decision['entry_date']):
            raise ValueError('Entry is not an authorized first-day signal: '+journal)
        checked[journal] += 1

    cohorts = account['cohorts']
    funded = set()
    for row in cohorts:
        check(row, 'cohorts', cohort=True)
        if row['event_id'] in funded:
            raise ValueError('First-signal account has duplicate funded cohorts')
        funded.add(row['event_id'])
    bought = set()
    for row in account['trades']:
        if row['side'] == 'buy':
            check(row, 'buy_trades')
            if row.get('qty', 0) <= 0:
                raise ValueError('First-signal buy trade must have positive shares')
            bought.add(row['event_id'])
    if funded != bought:
        raise ValueError('First-signal cohorts and funded buy events differ')
    for name in ('orders', 'tick_plans', 'base_tick_plans'):
        for row in account.get(name, []):
            if row['side'] == 'buy':
                check(row, name)
    for name in ('resource_plans', 'slot_decisions'):
        for row in account.get(name, []):
            check(row, name)
    return dict(policy=POLICY, decisions_rebuilt=len(rebuilt),
        decision_reconstruction_verified=True, first_red_signals=len(allowed),
        statuses=dict(Counter(d['status'] for d in rebuilt)), journals_checked=dict(checked),
        full_candidate_registry_checked=True, uses_entry_day_candle=False,
        resets_on_order_failure=False, resets_on_portfolio_capacity=False,
        poc_is_separate_ranking=True)
