import pytest
from scripts.audit_market_identity import parse_isin,resolve_on


def page(category='股票',market='上市',day='2024/05/15'):
    return (f'本國上市證券 最近更新日期:2026/09/14 上市日 CFICode'
        f'<table><tr><td>{category}</td></tr><tr><td>6423　億而得</td><td>TW0006423007</td>'
        f'<td>{day}</td><td>{market}</td><td>半導體</td><td>ESVUFR</td><td></td></tr></table>').encode()


def test_security_category_does_not_disappear_into_market_label():
    row=parse_isin(page('創新板','上市臺灣創新板'),'TWSE')[0]
    assert row['category']=='創新板' and row['start']=='2024-05-15'
    with pytest.raises(ValueError,match='market'):parse_isin(page(market='上櫃'),'TWSE')
    with pytest.raises(ValueError):parse_isin(page(day='2024/13/01'),'TWSE')


def test_transfer_boundary_and_unobserved_future():
    rows=[dict(stock_id='6423',market='TWSE',start='2024-05-15',end='2026-01-22',category='創新板'),
          dict(stock_id='6423',market='TPEx',start='2026-01-22',end=None,category='股票',snapshot_date='2026-09-14')]
    assert resolve_on(rows,'6423','2026-01-21')['market']=='TWSE'
    assert resolve_on(rows,'6423','2026-01-22')['market']=='TPEx'
    assert resolve_on(rows,'6423','2024-01-01')['status']=='outside_verified_intervals'
    assert resolve_on(rows,'6423','2026-09-15')['status']=='outside_verified_intervals'


def test_unknown_listing_date_is_never_replaced_by_first_price_date():
    rows=[dict(stock_id='5820',market='TPEx',start=None,end='2022-11-11',category='unconfirmed')]
    assert resolve_on(rows,'5820','2022-01-04')['status']=='unknown'
    assert resolve_on(rows,'5820','2022-11-11')['status']=='outside_verified_intervals'


def test_overlapping_evidence_fails_instead_of_choosing_a_market():
    rows=[dict(stock_id='6423',market=m,start='2024-01-01',end=None,category='股票') for m in ['TWSE','TPEx']]
    with pytest.raises(ValueError,match='Overlapping'):resolve_on(rows,'6423','2024-01-02')
