from copy import deepcopy
import pytest

from scripts.analyze_drawdown_controls import outcomes_with_pending_shares


def test_unpaid_share_rights_are_valued_but_never_close_or_mutate_the_account():
    right = dict(kind='shares', stock_id='6669', event_id='wave', qty=2,
                 fraction=.5, fractional_cash_per_share=0, pay_date=None,
                 tradable=False, delivery_status='pending_unannounced')
    account = dict(cohorts=[dict(event_id='wave', stock_id='6669', name='example',
                                entry_date='2026-08-03', exit_date='2026-09-03')],
                   corporate_actions=[], receivables=[right],
                   cash_ledger=[dict(kind='initial_deposit', cash_change=1000),
                                dict(kind='buy', cash_change=-300, event_id='wave', stock_id='6669'),
                                dict(kind='sell', cash_change=120, event_id='wave', stock_id='6669')])
    summary = dict(final_holdings=[], final_receivables=[right], receivable=220,
                   market_value=0, initial_cash=1000, cash=820, profit=40)
    case = dict(account=account, summary=summary)
    before = deepcopy(case)
    row = outcomes_with_pending_shares(case, {'6669':110})['wave']
    assert row['pnl']==40 and row['pending_share_value']==220
    assert not row['closed'] and not row['settled']
    assert case==before
    with pytest.raises(ValueError, match='Receivables do not reconcile'):
        outcomes_with_pending_shares(case, {'6669':100})
    with pytest.raises(ValueError, match='Unsupported'):
        outcomes_with_pending_shares(case, {})
    case['account']['receivables'][0]['tradable']=True
    with pytest.raises(ValueError, match='Unsupported'):
        outcomes_with_pending_shares(case, {'6669':110})
