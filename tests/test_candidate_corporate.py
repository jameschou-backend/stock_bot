from copy import deepcopy
import pytest
from skills.candidate_corporate import complete_cash_dividends


def fixture():
    rows = [dict(action_id='1815-missing', stock_id='1815', date='2026-09-09', kind='unresolved_cash_dividend'),
            dict(action_id='1815-stock', stock_id='1815', date='2026-09-09', kind='stock_dividend')]
    supplement = dict(stock_id='1815', date='2026-09-09', pay_date='2026-10-13',
                      cash_per_share=.50001709, cash_rounding='floor_ntd',
                      announcement_date='2026-08-24', source='issuer announcement')
    return rows, supplement


def test_missing_cash_is_replaced_without_early_payment_or_stock_mutation():
    rows, s = fixture(); original = deepcopy(rows)
    result = complete_cash_dividends(rows, '1815', [s])
    assert rows == original
    assert next(r for r in result if r['kind']=='stock_dividend') == rows[1]
    cash = next(r for r in result if r['kind']=='cash_dividend')
    assert cash['pay_date']=='2026-10-13'
    assert cash['cash_rounding']=='floor_ntd'
    assert not any(r['kind']=='unresolved_cash_dividend' for r in result)
    assert complete_cash_dividends(result,'1815',[s]) == result


def test_supplement_cannot_overwrite_disagreement_or_invent_an_event():
    rows, s = fixture()
    with pytest.raises(ValueError, match='explicit unresolved'):
        complete_cash_dividends([], '1815', [s])
    rows[0].update(kind='cash_dividend',cash_per_share=.6,pay_date=s['pay_date'])
    with pytest.raises(ValueError, match='conflicts'):
        complete_cash_dividends(rows,'1815',[s])


def test_supplement_rejects_future_announcement_and_early_payment():
    rows, s = fixture()
    for bad in (dict(announcement_date='2026-09-10'), dict(pay_date='2026-09-08')):
        with pytest.raises(ValueError, match='Invalid'):
            complete_cash_dividends(rows,'1815',[s|bad])
