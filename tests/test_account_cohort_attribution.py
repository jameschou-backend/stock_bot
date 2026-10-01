from copy import deepcopy
import pytest

from skills.account_cohort_attribution import cohort_outcomes


def case():
    right = dict(kind='cash', event_id='e', stock_id='1101', amount=10.)
    return dict(account=dict(cohorts=[dict(event_id='e', stock_id='1101', name='A',
        entry_date='2024-01-02', exit_date=None)], corporate_actions=[dict(action_id='div',
        event_id='e', stock_id='1101')], cash_ledger=[dict(kind='initial_deposit', cash_change=1000.),
        dict(kind='buy', event_id='e', stock_id='1101', cash_change=-100.),
        dict(kind='dividend_payment', action_id='div', stock_id='1101', cash_change=5.)],
        receivables=[right]), summary=dict(initial_cash=1000., cash=905., market_value=120.,
        receivable=10., profit=35., final_holdings=[dict(event_id='e', stock_id='1101', market_value=120.)],
        final_receivables=[right]))


def test_unpaid_dividend_is_profit_but_not_spendable_cash():
    result = cohort_outcomes(case())['e']
    assert result['pnl'] == 35 and result['cash_net'] == -95
    assert result['receivable'] == 10 and result['return_on_cost'] == .35
    assert not result['closed'] and not result['settled']


def test_unknown_share_valuation_and_missing_rights_are_rejected():
    value = case()
    value['account']['receivables'][0]['kind'] = 'shares'
    with pytest.raises(ValueError, match='explicit valuation'):
        cohort_outcomes(value)
    value = case()
    value['summary']['final_receivables'] = []
    with pytest.raises(ValueError, match='journal differs'):
        cohort_outcomes(value)


def test_ambiguous_dividend_attribution_and_wrong_stock_fail_closed():
    value = case()
    value['account']['corporate_actions'].append(dict(action_id='div', event_id='other', stock_id='1101'))
    with pytest.raises(ValueError, match='unique cohort'):
        cohort_outcomes(value)
    value = deepcopy(case())
    value['summary']['final_holdings'][0]['stock_id'] = '1102'
    with pytest.raises(ValueError, match='Holding cohort/stock'):
        cohort_outcomes(value)
