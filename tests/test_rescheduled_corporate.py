import pandas as pd
import pytest

from skills.million_replay import UnresolvedAction
from skills.rescheduled_corporate import CorporateCalendar


def calendar():
    return CorporateCalendar(pd.to_datetime(['2024-07-23', '2024-07-26', '2024-07-29']),
        [dict(stock_id='3293', scheduled_date='2024-07-24', effective_date='2024-07-26',
              source='reviewed exchange table', reference_price=715., lower=644., upper=786.,
              expected_finmind_limits=dict(lower=1320., upper=1610.))])


def test_rights_move_to_resumption_but_payment_and_identity_stay_unchanged():
    c = calendar()
    original = [dict(action_id='3293-2024-07-24-stock', date='2024-07-24',
                     kind='stock_dividend', shares_per_share=1., pay_date='2024-08-28')]
    rows = c.actions('3293', original)
    assert rows[0]['date'] == '2024-07-26'
    assert rows[0]['scheduled_ex_date'] == original[0]['date'] == '2024-07-24'
    assert rows[0]['pay_date'] == '2024-08-28' and rows[0]['action_id'] == original[0]['action_id']
    assert c.actions('3293', rows) == rows
    c.guard('3293', '2024-07-26', rows)
    assert c.opening_reference('3293', '2024-07-23') is None
    assert c.opening_reference('3293', '2024-07-26') == 715.


def test_unreviewed_holiday_event_is_not_silently_lost():
    c = calendar()
    rows = c.actions('1101', [dict(action_id='cash', date='2024-07-25', kind='cash_dividend')])
    with pytest.raises(UnresolvedAction, match='Unreviewed non-session'):
        c.guard('1101', '2024-07-26', rows)


def test_legal_limits_require_the_exact_reviewed_original():
    c = calendar()
    source = {'2024-07-26': dict(lower=1320., upper=1610.)}
    assert c.limits('3293', source)['2024-07-26'] == dict(lower=644., upper=786.)
    assert source['2024-07-26']['lower'] == 1320.
    assert c.limits('1101', source) == source
    with pytest.raises(ValueError, match='reviewed source'):
        c.limits('3293', {})


def test_closed_day_cash_and_share_rights_survive_sale_until_delivery():
    from skills.million_replay import Replay
    c = calendar()
    rows = c.actions('3293', [
        dict(stock_id='3293', action_id='cash', date='2024-07-24',
             kind='cash_dividend', cash_per_share=35., pay_date='2024-08-28'),
        dict(stock_id='3293', action_id='stock', date='2024-07-24',
             kind='stock_dividend', shares_per_share=1., pay_date='2024-08-28',
             fractional_cash_per_share=None)])

    class Corporate:
        def on_date(self, sid, day):
            return [r for r in rows if r['date'] == day]

    engine = Replay.__new__(Replay)
    engine.corporate = Corporate()
    engine.cash = 1000.
    engine.cash_ledger = []
    engine.holdings = {'3293': dict(qty=100, event_id='signal', due_index=100)}
    engine.marks = {'3293': dict(price=786.)}
    engine.fields = {'close': pd.DataFrame({'3293': [786.]}, index=pd.to_datetime(['2024-07-26']))}
    engine.receivables, engine.actions = [], []
    engine.cohorts = [dict(event_id='signal', due_index=100)]
    engine.corporate_day(pd.Timestamp('2024-07-26'))
    assert engine.cash == 1000. and engine.holdings['3293']['qty'] == 100
    assert engine.receivable_value() == 3500+100*786
    engine.holdings.clear()  # Sale cannot destroy vested cash or undelivered shares.
    engine.corporate_day(pd.Timestamp('2024-08-27'))
    assert engine.cash == 1000. and len(engine.receivables) == 2
    engine.corporate_day(pd.Timestamp('2024-08-28'))
    assert engine.cash == 4500. and engine.holdings['3293']['qty'] == 100
    assert engine.receivables == []
