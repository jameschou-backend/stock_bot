from copy import deepcopy
import pytest
from scripts.audit_market_boundary import parse_presence


def payload():
    return dict(date='20220103',stat='ok',tables=[dict(title='上櫃股票行情',
        fields=['代號','名稱'],data=[['1258','其祥'],['006201','富櫃50']],totalCount=2)])


def test_presence_does_not_invent_listing_date_or_category_and_filters_non_stock_codes():
    result=parse_presence(payload(),'TPEx','2022-01-03')
    assert set(result)=={'1258'} and result['1258']['listing_date'] is None
    assert result['1258']['category']=='unconfirmed'


@pytest.mark.parametrize('change',[
    lambda p:p.update(date='20220104'),lambda p:p.update(stat='error'),
    lambda p:p['tables'][0].update(totalCount=1),
    lambda p:p['tables'].append(deepcopy(p['tables'][0])),
    lambda p:p['tables'][0]['data'][0].append('extra'),
    lambda p:p['tables'][0]['data'].__setitem__(1,['1258','duplicate'])])
def test_invalid_or_ambiguous_roster_fails_closed(change):
    p=payload();change(p)
    with pytest.raises(ValueError):parse_presence(p,'TPEx','2022-01-03')


def test_twse_selects_daily_security_table_not_index_table():
    p=dict(date='20220103',stat='OK',tables=[dict(title='指數',fields=['指數'],data=[['index']]),
        dict(title='111年01月03日 每日收盤行情',fields=['證券代號','證券名稱'],data=[['9188','精熙-DR']])])
    assert parse_presence(p,'TWSE','2022-01-03')['9188']['category']=='unconfirmed'
