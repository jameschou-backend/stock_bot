from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.poc_risk_controls import buy_cap, volatility, audit_controls
from test_poc_executable_replay import audit_case  # noqa: F401
from test_poc_range_audit import range_case  # noqa: F401


def test_chase_cap_at_tick_boundaries_and_legal_edges():
    assert buy_cap(49, '2330', dict(lower=44.1, upper=53.9)) == 50.4
    assert buy_cap(100, '2330', dict(lower=95, upper=102)) == 102
    assert buy_cap(100, '2330', dict(lower=104, upper=110)) == 104


def test_volatility_uses_only_twenty_completed_returns_without_filling():
    days = pd.bdate_range('2024-01-01', periods=25)
    values = np.cumprod(1+np.array([0., .01, -.02, .03, -.01]*5))*100
    frame = pd.DataFrame({'2330': values}, index=days)
    sigma = volatility(frame, '2330', days[20])
    frame.loc[days[21]:, '2330'] = 999999
    assert volatility(frame, '2330', days[20]) == sigma
    frame.loc[days[10], '2330'] = np.nan
    assert volatility(frame, '2330', days[20]) is None
    assert volatility(frame.iloc[:20], '2330', days[19]) is None


def fixture():
    before = dict(date='2024-02-01', signal_date='2024-01-31', event_id='event', side='buy',
        stock_id='2330', prior_reference=100., limit_price=110., odd_limit=110.,
        planned_qty=500, board_qty=0, odd_qty=500, reserved_cash=60000., sizing_budget=60000.,
        risk_control_applied=False)
    after = dict(before, limit_price=103., odd_limit=103., risk_control_applied=True)
    return dict(settings=dict(risk_control_variant='chase3'), tick_plans=[after],
        risk_control_records=[dict(before=before, after=deepcopy(after), bounds=dict(lower=90., upper=110.))])


def test_independent_control_audit_rejects_quantity_or_reservation_changes():
    account = fixture()
    assert audit_controls(account, None)['plans_checked'] == 1
    for field in ('planned_qty', 'reserved_cash', 'sizing_budget'):
        corrupt = deepcopy(account)
        corrupt['tick_plans'][0][field] += 1
        corrupt['risk_control_records'][0]['after'][field] += 1
        with pytest.raises(ValueError, match='unrelated plan'):
            audit_controls(corrupt, None)
    account['risk_control_records'] = []
    with pytest.raises(ValueError, match='Missing risk-control'):
        audit_controls(account, None)


def test_volatility_oracle_rejects_oversizing_even_when_cash_is_available():
    from skills.mixed_odd_replay import sized_quantity
    from skills.top3_validation import cost_function
    days = pd.bdate_range(end='2024-01-31', periods=21)
    frame = pd.DataFrame({'2330': 100*np.cumprod([1.]+[1.04,.97]*10)}, index=days)
    sigma = volatility(frame, '2330', '2024-01-31')
    account = fixture()
    account['settings'].update(risk_control_variant='vol20', top3_cost_mode='baseline')
    r = account['risk_control_records'][0]
    budget = r['before']['sizing_budget']*.30/sigma
    maximum = int((budget-40)/(100*1.005925))
    n = sized_quantity(maximum,110.,budget,SimpleNamespace(_costs=cost_function('baseline')),'2330')
    r['annual_volatility'] = sigma
    r['after'] = dict(r['before'], sizing_budget=budget, planned_qty=n, board_qty=n//1000*1000,
                      odd_qty=n%1000, rejection=None, risk_control_applied=True)
    account['tick_plans'] = [deepcopy(r['after'])]
    assert audit_controls(account,frame)['plans_checked']==1
    assert n<r['before']['planned_qty'] and budget<r['before']['reserved_cash']
    for p in (r['after'],account['tick_plans'][0]):
        p['planned_qty']+=1;p['odd_qty']+=1
    with pytest.raises(ValueError,match='unrelated plan'):
        audit_controls(account,frame)


def test_chase_auditor_accepts_new_limit_but_keeps_cash_and_fill_checks(range_case):
    from skills.all_entry_audit_index import private_auditor
    from skills.poc_risk_controls import install
    from skills import poc_range_audit
    original = private_auditor().audit_range_execution
    # Corporate adapters close over this function; installing the control must
    # also replace the captured function, not only its module attribute.
    def corporate_wrapper(fn):
        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapped
    runner = SimpleNamespace(engine_types=lambda *a, **k: (object, object),
        _all_entry_indexed_audit=corporate_wrapper(original), write=lambda *a: None)
    install(runner, 'chase3', {})
    account = range_case[0]
    for plan in account['tick_plans']:
        plan.update(limit_price=103., odd_limit=103.)
    for row in account['orders']+account['trades']:
        row['limit_price'] = 103.
    with pytest.raises(ValueError):
        poc_range_audit.audit_range_execution(*range_case)
    assert runner._all_entry_indexed_audit(*range_case)['fills_reconciled'] == 2
    account['daily'][0]['cash'] += 1
    with pytest.raises(ValueError):
        runner._all_entry_indexed_audit(*range_case)
