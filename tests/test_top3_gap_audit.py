from copy import deepcopy
from datetime import date, timedelta
import pytest

from skills.poc_gap_audit import audit_gap_records
from skills.top3_gap_audit import private_gap_audit
from test_poc_gap_audit import gap_case  # noqa: F401
from test_poc_executable_replay import audit_case  # noqa: F401


@pytest.fixture
def retained(gap_case):
    def convert(value):
        if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
        if isinstance(value,list):return [convert(v) for v in value]
        if value=='2330':return '0050'
        if value=='a':return 'benchmark'
        return value
    account=convert(gap_case[0])
    gap=account['data_gap_exclusions'][0]
    gap['holding_qty_before']=gap['holding_qty_after']=7656
    prior=(date.fromisoformat(gap['date'])-timedelta(days=1)).isoformat()
    account['daily'].insert(0,dict(date=prior))
    account['holdings']=[dict(date=day,stock_id='0050',event_id='benchmark',qty=7656)
                         for day in [prior,gap['date']]]
    return account


def test_retained_benchmark_gap_checks_adjacent_holdings_without_changing_original(retained):
    before=deepcopy(retained)
    with pytest.raises(ValueError,match='Excluded buy created'):
        audit_gap_records(retained)
    result=private_gap_audit()(retained)
    assert result['data_gap_buys']==1
    assert not result['gap_source_failures_independently_rebuilt']
    assert retained==before
    with pytest.raises(ValueError,match='Excluded buy created'):
        audit_gap_records(retained)


@pytest.mark.parametrize('change',['prior_qty','current_qty','both_gap_qty','missing_holding',
    'new_cohort','corporate_action','hidden_trade','cash_change','different_event'])
def test_retained_benchmark_gap_cannot_hide_share_or_cash_changes(retained,change):
    gap=retained['data_gap_exclusions'][0]
    if change=='prior_qty':retained['holdings'][0]['qty']+=1
    elif change=='current_qty':retained['holdings'][1]['qty']+=1
    elif change=='both_gap_qty':gap['holding_qty_before']=gap['holding_qty_after']=7657
    elif change=='missing_holding':retained['holdings'].pop()
    elif change=='new_cohort':retained['cohorts'].append(dict(event_id='benchmark'))
    elif change=='corporate_action':retained['corporate_actions'].append(dict(date=gap['date'],stock_id='0050'))
    elif change=='hidden_trade':retained['trades'].append(dict(date=gap['date'],stock_id='0050'))
    elif change=='cash_change':gap['cash_after']-=1
    elif change=='different_event':retained['holdings'][0]['event_id']='different'
    with pytest.raises(ValueError):private_gap_audit()(retained)


def test_new_equity_gap_still_uses_original_checks(gap_case):
    account=gap_case[0]
    assert private_gap_audit()(account)==audit_gap_records(account)
    account['data_gap_exclusions'][0]['holding_qty_before']=100
    account['data_gap_exclusions'][0]['holding_qty_after']=100
    with pytest.raises(ValueError,match='Excluded buy created'):
        private_gap_audit()(account)
