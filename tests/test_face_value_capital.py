from copy import deepcopy
from types import SimpleNamespace
import pytest
from skills.face_value_capital import FaceValueCapitalActions
from skills.million_replay import UnresolvedAction


def setup_case():
    action=dict(action_id='6244-2020-09-28',stock_id='6244',date='2020-09-28',kind='capital_reduction',
        multiplier=.6569357943,cash_per_share=0,pay_date='2020-09-28',known_date='2020-07-20',
        fractional_policy='face_value_gross_receivable',fractional_face_value=10,
        fractional_reference_date='2020-07-20',fractional_cash_pay_date=None,
        fractional_cash_rounding='floor_ntd',cash_rounding='floor_ntd',evidence_files=['issuer.html'])
    provider=SimpleNamespace(on_date=lambda sid,day:[action])
    account=SimpleNamespace(raw=lambda day,sid:40.,holdings={'6244':dict(qty=1001,event_id='entry')},
                            marks={'6244':dict(price=25.)},receivables=[],actions=[])
    return action,account,FaceValueCapitalActions(provider,account)


def test_exchange_integer_shares_and_unspendable_fraction():
    action,account,adapter=setup_case()
    assert adapter.on_date('6244','2020-09-28')==[]
    assert account.holdings['6244']['qty']==657
    assert account.receivables[0]['amount']==5
    assert account.receivables[0]['pay_date'] is None
    assert account.receivables[0]['net_amount_verified'] is False
    assert account.actions[0]['fractional_reference_price']==10
    assert account.marks['6244']['price']==pytest.approx(25/.6569357943)
    with pytest.raises(UnresolvedAction):adapter.on_date('6244','2020-09-28')


@pytest.mark.parametrize('field,value', [('fractional_face_value',0),('fractional_face_value',True),
    ('fractional_face_value',float('nan')),('fractional_cash_pay_date','2020-09-28'),
    ('known_date','2020-09-28'),('evidence_files',[]),('multiplier',1.1)])
def test_invalid_terms_do_not_mutate_account(field,value):
    action,account,adapter=setup_case();action[field]=value
    before=deepcopy(account.holdings)
    with pytest.raises(UnresolvedAction):adapter.on_date('6244','2020-09-28')
    assert account.holdings==before and account.actions==[] and account.receivables==[]


def test_unrelated_action_retains_legacy_behavior():
    action,account,adapter=setup_case()
    action.clear();action.update(kind='cash_dividend',action_id='cash',cash_per_share=1)
    assert adapter.on_date('6244','2020-09-28')==[action]
    assert account.holdings['6244']['qty']==1001
