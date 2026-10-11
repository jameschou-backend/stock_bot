from copy import deepcopy
import pytest

from skills.all_entry_resource_rejections import resource_auditor, verified_rejections
from skills.all_entry_audit_index import private_auditor
from test_poc_executable_replay import audit_case  # noqa: F401
from test_poc_gap_audit import gap_case  # noqa: F401
from test_poc_range_audit import range_case, range_gap_case  # noqa: F401


@pytest.fixture
def blocked_case(range_case):
    a = range_case[0]
    p = a['tick_plans'][0]
    p['opening_cash'] = 1_000_000.
    a['settings']['slots'] = 3
    a['cohorts'] = []
    a['trades'] = []
    a['daily'][0]['cash'] = 1_000_000.
    identity = {k:p[k] for k in ('date', 'event_id', 'stock_id', 'signal_date')}
    a['orders'] = [dict(identity, side='buy', channel='event', requested_qty=670,
                        filled_qty=0, failure='resource_slots_locked')]
    a['resource_plans'] = [dict(identity, planned_qty=670, spent=0., budget=80000.,
        locked_after=270000., locked_unused_before=270000., opening_cash=p['opening_cash'],
        occupied_before=['2002', '3645', '3702'], failure='resource_slots_locked')]
    a['slot_decisions'] = [dict(identity, attempted=False, filled_qty=0,
        occupied_before=['2002', '3645', '3702'], held_before=['2002', '3702'],
        opening_members=['2002', '3702'], attempts_before=['3645'], failure='resource_slots_locked')]
    return range_case


def test_rejected_positive_plan_is_not_an_unverified_halt_and_does_not_need_tape(blocked_case):
    before = deepcopy(blocked_case[0])
    with pytest.raises(ValueError, match='verified halt evidence'):
        private_auditor().audit_range_execution(*blocked_case)
    result = resource_auditor()(*blocked_case)
    assert result['resource_slot_rejections_cross_checked'] == 1
    assert result['resource_rejected_children_not_submitted'] == 2
    assert result['original_planned_children'] == 2
    assert result['fills_reconciled'] == 0
    assert blocked_case[0] == before


@pytest.mark.parametrize('change', ['slot_available', 'cash', 'quantity', 'duplicate', 'fill',
    'signal', 'resource_missing', 'occupied', 'attempt', 'cohort', 'plan_size', 'limit', 'reservation'])
def test_fabricated_rejection_or_invalid_original_plan_still_fails(blocked_case, change):
    a = blocked_case[0]
    if change == 'slot_available': a['settings']['slots'] = 4
    elif change == 'cash': a['resource_plans'][0]['spent'] = 1.
    elif change == 'quantity': a['orders'][0]['requested_qty'] = 671
    elif change == 'duplicate': a['orders'].append(deepcopy(a['orders'][0]))
    elif change == 'fill': a['trades'].append(dict(a['orders'][0], qty=1))
    elif change == 'signal': a['slot_decisions'][0]['signal_date'] = a['orders'][0]['date']
    elif change == 'resource_missing': a['resource_plans'] = []
    elif change == 'occupied': a['slot_decisions'][0]['occupied_before'] = ['2002', '3702']
    elif change == 'attempt': a['slot_decisions'][0]['attempted'] = True
    elif change == 'cohort': a['cohorts'].append(dict(event_id='a'))
    elif change == 'plan_size': a['tick_plans'][0]['planned_qty'] += 1
    elif change == 'limit': a['tick_plans'][0]['limit_price'] = 109.
    else: a['tick_plans'][0]['reserved_cash'] = 1_000_001.
    with pytest.raises(ValueError): resource_auditor()(*blocked_case)


@pytest.mark.parametrize('fixture', ['range_case', 'range_gap_case'])
def test_accounts_without_resource_rejections_keep_every_original_assertion(request, fixture):
    args = request.getfixturevalue(fixture)
    before = private_auditor().audit_range_execution(*args)
    after = resource_auditor()(*args)
    for k in before: assert after[k] == before[k]
    assert after['resource_slot_rejections_cross_checked'] == 0
