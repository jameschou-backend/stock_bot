from datetime import date
from types import SimpleNamespace
import pytest
from app import trading_risk_preference as risk
from scripts.prepare_readiness_actions import strict_parse


def test_official_parser_cannot_silently_omit_a_row_or_duplicate_event():
    event=SimpleNamespace(stock_id='1560',event_date=date(2026,7,16),source='ex_rights')
    with pytest.raises(ValueError,match='omitted'):
        strict_parse('twse_ex_rights',lambda p:[event],{'data':[['a'],['bad']]},date(2026,7,1),date(2026,7,31))
    with pytest.raises(ValueError,match='duplicate'):
        strict_parse('twse_ex_rights',lambda p:[event,event],{'data':[['a'],['a']]},date(2026,7,1),date(2026,7,31))


def test_risk_preference_preserves_sealed_simulation_and_scales_with_profit():
    p=risk.load()
    assert p['initial_capital_twd']=='1000000' and p['maximum_loss_fraction']=='0.50'
    assert not p['broker_orders_authorized'] and not p['applied_to_existing_simulation']
    r=risk.assess([1000000,2000000,1000001])
    assert r['status']=='below_threshold' and r['trigger_nav']=='1000000.00'
    assert risk.assess([1000000,2000000,1000000])['status']=='review_required'
    assert risk.assess([1000000,490000,900000])['status']=='review_required'
    assert risk.assess([1000000,0])['status']=='review_required'


@pytest.mark.parametrize('values',[[],[None],[1000000,None],['NaN'],[-1]])
def test_unknown_values_do_not_pass_risk_check(values):
    assert risk.assess(values)['status']=='unknown'


def test_unadjusted_deposit_does_not_change_risk_high_water_mark():
    assert risk.assess([1000000,2000000],external_cashflows=True)['status']=='unknown'
