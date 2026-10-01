import pytest

from scripts.diagnose_liquidity_account import cohort_outcomes


def case():
    return dict(account=dict(
        cohorts=[dict(event_id='a', stock_id='1101', name='甲', entry_date='2020-01-02', exit_date='2020-02-03'),
                 dict(event_id='b', stock_id='1102', name='乙', entry_date='2020-01-03')],
        corporate_actions=[dict(action_id='dividend', event_id='a', kind='payment'),
            dict(action_id='stock', event_id='a', kind='share_delivery', stock_id='1101',
                 date='2020-02-10', fraction=.5)],
        cash_ledger=[dict(kind='initial_deposit', cash_change=1000),
            dict(kind='buy', event_id='a', cash_change=-100),
            dict(kind='buy', event_id='b', cash_change=-100),
            dict(kind='sell', event_id='a', cash_change=90),
            dict(kind='dividend_payment', action_id='dividend', cash_change=20),
            dict(kind='fractional_share_payment', stock_id='1101', date='2020-02-10', cash_change=5)]),
        summary=dict(initial_cash=1000, cash=915, profit=5, final_receivables=[],
            final_holdings=[dict(event_id='b', market_value=90)]))


def test_dividends_and_fractional_cash_are_counted_once_and_open_mark_is_not_a_closed_win():
    outcomes = cohort_outcomes(case())
    assert outcomes['a']['pnl'] == 15 and outcomes['a']['return_on_cost'] == .15
    assert outcomes['a']['closed']
    assert outcomes['b']['pnl'] == -10 and not outcomes['b']['closed']


def test_unallocated_cash_or_unvalued_rights_are_not_silently_dropped():
    value = case()
    value['summary']['final_receivables'] = [dict(kind='cash', amount=10)]
    with pytest.raises(ValueError, match='Outstanding rights'):
        cohort_outcomes(value)
    value = case()
    value['account']['cash_ledger'].append(dict(kind='unknown_income', cash_change=1))
    with pytest.raises(ValueError, match='unique cohort'):
        cohort_outcomes(value)
