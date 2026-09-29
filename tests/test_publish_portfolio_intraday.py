import pytest
from scripts.publish_portfolio_intraday import cohorts


def test_explicit_zero_fractional_movement_has_no_pnl_but_nonzero_needs_identity():
    result=dict(account=dict(corporate_actions=[],cohorts=[],trades=[],
        cash_ledger=[dict(kind='fractional_share_payment',cash_change=0)]),
        summary=dict(final_holdings=[],final_receivables=[],profit=0))
    assert cohorts(result)==[]
    result['account']['cash_ledger'][0]['cash_change']=1
    with pytest.raises(ValueError,match='attribution'):cohorts(result)
