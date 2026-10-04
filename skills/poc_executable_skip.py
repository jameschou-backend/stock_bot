"""One authorized, post-hoc data exclusion; the sealed strict engine is unchanged.

The caller must verify and bind STRICT_PARTIAL_PATH's bytes before supplying the
decoded case to the audit. These pure helpers do not claim to verify file hashes.
The original positive plan stays in the account, including its locked resources.
"""
from copy import deepcopy
import math

import pandas as pd

from skills.poc_executable_replay import ExecutableOrders, audit_executable, finite
from skills.million_replay import costs, money
from skills.mixed_odd_replay import sized_quantity


SKIP_POLICY = 'user_authorized_3230_20241016_buy_data_conflict_v1'
SKIP_REASON = 'user_authorized_data_conflict_exclusion'
STRICT_PARTIAL_PATH = '.cache/poc-executable-20261004/online-v2/poc_red_executable.json'
STRICT_PARTIAL_SHA256 = 'fc45bd29bc55ba25621bce078c246562c0a212845fc5e3e109f5a99c92508b14'
STRICT_FAILURE = 'Executable board tape quality conflict: 3230 2024-10-16 official_ordinary_aggregate_conflict'
SKIP_DATE, SKIP_STOCK, SKIP_EVENT = '2024-10-16', '3230', 'liquid_universe-2024-10-15-3230'
SKIP_KEY = (SKIP_DATE, SKIP_STOCK, 'buy', SKIP_EVENT)
_PLAN = dict(board_qty=4000, date=SKIP_DATE, event_id=SKIP_EVENT,
    expires_at='13:25:00', limit_price=69.4, odd_expires_at='14:30:00', odd_limit=69.4,
    odd_order_time='13:40:00', odd_qty=958, opening_cash=395371.51,
    order_time='09:01:00', planned_qty=4958, prior_reference=63.1,
    reference_date='2024-10-15', rejection=None, reserved_cash=346859.0366666667,
    side='buy', signal_date='2024-10-15', sizing_budget=346154.37, stock_id=SKIP_STOCK)


def expected_plan():
    return deepcopy(_PLAN)


def _key(row):
    return row.get('date'), row.get('stock_id'), row.get('side'), row.get('event_id')


def _same_plan(plan):
    if plan != _PLAN or any(type(plan[k]) is not int for k in ('planned_qty', 'board_qty', 'odd_qty')):
        raise ValueError('Explicit exclusion differs from the sealed positive plan')


def _order():
    return dict(date=SKIP_DATE, stock_id=SKIP_STOCK, side='buy', event_id=SKIP_EVENT,
        signal_date=_PLAN['signal_date'], reason='leader_entry', channel='event',
        requested_qty=_PLAN['planned_qty'], filled_qty=0, failure=SKIP_REASON,
        exclusion_policy=SKIP_POLICY, execution_evidence='not_evaluated_explicit_data_exclusion',
        actual_fill_verified=False, live_qualified=False)


def _record():
    return dict(policy=SKIP_POLICY, date=SKIP_DATE, stock_id=SKIP_STOCK, side='buy',
        event_id=SKIP_EVENT, signal_date=_PLAN['signal_date'], original_plan=expected_plan(),
        excluded_channels=['board', 'odd'], filled_qty=0, posthoc_data_exclusion=True,
        source_reference=dict(path=STRICT_PARTIAL_PATH, sha256=STRICT_PARTIAL_SHA256),
        source_hash_verification_required_by_caller=True, actual_fill_verified=False,
        live_qualified=False)


class ExplicitConflictSkipOrders(ExecutableOrders):
    def __init__(self, *args, **kwargs):
        self.explicit_exclusions = []
        super().__init__(*args, **kwargs)

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        key = str(day.date()), sid, side, event_id
        if key != SKIP_KEY:
            return super()._execute_order(day, sid, side, qty, reason, event_id, signal_date)
        plan = self.day_plans.get((event_id, side))
        _same_plan(plan)
        matching = [p for p in self.tick_plans if _key(p) == SKIP_KEY]
        if (matching != [plan] or signal_date != plan['signal_date'] or reason != 'leader_entry'
                or type(qty) is not int or qty < plan['planned_qty']):
            raise ValueError('Explicit exclusion order differs from its committed plan')
        if self.explicit_exclusions or sid in self.tick_attempts:
            raise ValueError('Explicit exclusion was already applied')
        if self.holdings.get(sid, {}).get('qty') != 0:
            raise ValueError('Explicit exclusion must not remove an existing holding')
        self.tick_attempts.add(sid)
        self.orders.append(_order())
        self.explicit_exclusions.append(_record())
        # Outer resource/slot wrappers record an attempted zero fill and keep
        # the full daily reservation. No plan, cash, holdings or tape is changed.
        return 0

    def run(self):
        account = super().run()
        account['explicit_exclusions'] = deepcopy(self.explicit_exclusions)
        account['settings'].update(explicit_data_exclusion_policy=SKIP_POLICY,
            posthoc_data_exclusion=True, excluded_event_count=len(self.explicit_exclusions),
            actual_fill_verified=False, live_qualified=False, unseen_validation=False)
        volume = account['ordinary_volume_evidence']
        volume.update(excluded_positive_board_children=len(self.explicit_exclusions),
            all_original_requested_board_capacity_observed=(not self.explicit_exclusions
                and volume['all_requested_board_capacity_observed']))
        return account


def _journal(value):
    if 'partial_journal' in value:
        return value['partial_journal']
    if 'account' in value:
        return value['account']
    return value


def verify_strict_prefix(account, strict_partial):
    """Compare the actual prior strict journals, without asserting provenance."""
    if (strict_partial.get('completed') is not False or strict_partial.get('reason') != STRICT_FAILURE
            or strict_partial.get('last_date') != '2024-10-15'
            or strict_partial.get('completed_sessions') != 188):
        raise ValueError('Explicit exclusion requires the original strict failure case')
    old, new = _journal(strict_partial), _journal(account)
    if len(old.get('daily', [])) != 188 or old['daily'][-1]['date'] != '2024-10-15':
        raise ValueError('Strict failure has a different complete-day boundary')
    plans = [p for p in old.get('tick_plans', []) if _key(p) == SKIP_KEY]
    if plans != [_PLAN]:
        raise ValueError('Strict failure lacks the original excluded plan')
    checked = {}
    # Cohort exit dates and current receivables are mutable final state. Compare
    # append-only journals instead, including earlier orders on the failure day.
    for key in ('daily', 'trades', 'orders', 'cash_ledger', 'corporate_actions', 'holdings',
                'resource_plans', 'selection_decisions', 'tick_plans'):
        if key not in old or key not in new or new[key][:len(old[key])] != old[key]:
            raise ValueError('Pre-exclusion strict journal changed: '+key)
        checked[key] = len(old[key])
    return dict(strict_prefix_compared=True, completed_sessions=188,
        last_complete_date='2024-10-15', compared_rows=checked,
        source_hash_verification_required_by_caller=True)


def _one(rows, label):
    if len(rows) != 1:
        raise ValueError('Explicit exclusion needs exactly one '+label)
    return rows[0]


def audit_explicit_exclusion(account, strict_partial):
    """Check the completed excluded attempt, even in a later partial journal.

    This checks no subsequent fill prices and cannot certify the full account.
    """
    account = _journal(account)
    prefix = verify_strict_prefix(account, strict_partial)
    if account.get('explicit_exclusions') != [_record()]:
        raise ValueError('Explicit exclusion list was missing, expanded or changed')
    plan = _one([p for p in account['tick_plans'] if _key(p) == SKIP_KEY], 'positive plan')
    _same_plan(plan)
    excluded = [r for r in account['orders'] if r.get('event_id') == SKIP_EVENT]
    if excluded != [_order()]:
        raise ValueError('Explicit exclusion order was changed or executed')
    for label in ('trades', 'cash_ledger', 'holdings', 'cohorts'):
        if any(r.get('event_id') == SKIP_EVENT for r in account[label]):
            raise ValueError('Excluded event changed '+label)
    if any(r.get('date') == SKIP_DATE and r.get('stock_id') == SKIP_STOCK
           for r in account['trades']):
        raise ValueError('Excluded stock-day has a hidden fill')
    if any(r.get('date') == SKIP_DATE and r.get('stock_id') == SKIP_STOCK
           and r.get('kind') in ('buy', 'sell') for r in account['cash_ledger']):
        raise ValueError('Excluded stock-day has a hidden cash change')

    resource = _one([p for p in account['resource_plans']
                     if p.get('date') == SKIP_DATE and p.get('event_id') == SKIP_EVENT], 'resource reservation')
    if (resource.get('stock_id') != SKIP_STOCK or resource.get('signal_date') != _PLAN['signal_date']
            or resource.get('spent') != 0 or resource.get('filled_qty') != 0 or resource.get('failure')
            or resource.get('budget') != plan['reserved_cash']
            or resource.get('opening_cash') != plan['opening_cash']
            or resource.get('planned_qty', 0) < plan['planned_qty']
            or resource.get('locked_after') != money(resource['locked_unused_before']+resource['budget'])):
        raise ValueError('Explicit exclusion released or altered the daily budget')
    decisions = [r for r in account['slot_decisions'] if r['date'] == SKIP_DATE]
    slot = _one([r for r in decisions if r['event_id'] == SKIP_EVENT], 'slot attempt')
    if (slot.get('stock_id') != SKIP_STOCK or slot.get('signal_date') != _PLAN['signal_date']
            or slot.get('attempted') is not True or slot.get('filled_qty') != 0 or slot.get('failure')):
        raise ValueError('Explicit exclusion released the daily slot')
    for later in decisions[decisions.index(slot)+1:]:
        if any(SKIP_STOCK not in later.get(key, [])
               for key in ('attempts_before', 'unfilled_before', 'occupied_before')):
            raise ValueError('Explicit exclusion funded a same-day replacement slot')
    return dict(explicit_exclusion_resource_lock_verified=True, strict_prefix=prefix,
        explicit_excluded_plans=1, explicit_excluded_children=2, posthoc_data_exclusion=True,
        source_hash_verification_required_by_caller=True,
        all_original_planned_children_execution_evidence_complete=False)


def audit_executable_skip(account, ticks, odds, routes, quotes, days, corp, feeds,
                          *, strict_partial, **kwargs):
    """Audit the exclusion and full resources, then strict-audit other orders."""
    exclusion = audit_explicit_exclusion(account, strict_partial)
    if (account['settings'].get('explicit_data_exclusion_policy') != SKIP_POLICY
            or account['settings'].get('posthoc_data_exclusion') is not True
            or account['settings'].get('excluded_event_count') != 1):
        raise ValueError('Explicit exclusion is not identified in account settings')
    plan = _one([p for p in account['tick_plans'] if _key(p) == SKIP_KEY], 'positive plan')

    calendar = pd.DatetimeIndex(days)
    index = calendar.get_loc(pd.Timestamp(SKIP_DATE))
    if not index or str(calendar[index-1].date()) != plan['reference_date']:
        raise ValueError('Excluded order is not T+1')
    prior = quotes.loc[pd.to_datetime(quotes['date']).eq(plan['reference_date'])
                       & quotes.stock_id.eq(SKIP_STOCK)]
    if len(prior) != 1 or corp.reference_price(SKIP_STOCK, SKIP_DATE, float(prior.iloc[0]['close'])) != plan['prior_reference']:
        raise ValueError('Excluded plan prior reference changed')
    limits = feeds.get_limits(SKIP_STOCK).get(SKIP_DATE)
    if not limits or limits['upper'] != plan['limit_price']:
        raise ValueError('Excluded plan official limit changed')
    maximum = math.floor((plan['sizing_budget']-40)/(plan['prior_reference']*(1+.001425+.0045)))
    class Cost:
        _costs = staticmethod(costs)
    if sized_quantity(maximum, limits['upper'], plan['sizing_budget'], Cost(), SKIP_STOCK) != plan['planned_qty']:
        raise ValueError('Excluded plan quantity changed')

    opening = {r['date']: account['daily'][i-1]['cash'] if i else account['settings']['initial_cash']
               for i, r in enumerate(account['daily'])}
    totals = {}
    seen = set()
    for p in account['tick_plans']:
        identity = p['date'], p['event_id'], p['side']
        if identity in seen or not finite(p['reserved_cash']) or p['reserved_cash'] < 0:
            raise ValueError('Full planned reservation is invalid or duplicated')
        seen.add(identity)
        totals[p['date']] = totals.get(p['date'], 0.)+p['reserved_cash']
        if totals[p['date']] > opening[p['date']]+.011:
            raise ValueError('Full plans borrow same-day proceeds or exclusion budget')

    projected = deepcopy(account)
    projected['tick_plans'] = [p for p in projected['tick_plans'] if _key(p) != SKIP_KEY]
    projected['orders'] = [r for r in projected['orders'] if r.get('event_id') != SKIP_EVENT]
    audit = audit_executable(projected, ticks, odds, routes, quotes, days, corp, feeds, **kwargs)
    verified_children = audit.pop('all_planned_children_reconciled')
    return dict(**audit, **exclusion, nonexcluded_planned_children_reconciled=verified_children,
        original_planned_children=verified_children+2)
