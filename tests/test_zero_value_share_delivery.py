from copy import deepcopy
import pandas as pd

from skills.zero_value_share_delivery import ZeroValueShareDelivery


def test_zero_stock_delivery_preserves_actual_exit_date_and_remaining_journals():
    right = dict(stock_id='6409', event_id='e', action_id='a', kind='shares',
                 qty=0, fraction=.05, fractional_cash_per_share=0., pay_date='2019-10-25')

    class AccountCore:
        def __init__(self):
            self.receivables = [deepcopy(right)]
            self.cohorts = [dict(event_id='e', exit_date='2019-10-01')]
            self.holdings, self.actions, self.movements = {}, [], []

        def corporate_day(self, day):
            # Reproduce the old zero-delivery side effect: an empty holding is
            # recreated and cleanup replaces a past exit with the delivery day.
            for r in self.receivables:
                if pd.Timestamp(r['pay_date']) <= day:
                    self.holdings[r['stock_id']] = {'qty': r['qty']}
                    self.cohorts[0]['exit_date'] = str(day.date())
            return 0.

        def cash_move(self, *args, **kwargs):
            self.movements.append((args, kwargs))

    class Account(ZeroValueShareDelivery, AccountCore):
        pass

    old = AccountCore(); old.corporate_day(pd.Timestamp('2019-10-25'))
    assert old.cohorts[0]['exit_date'] == '2019-10-25'
    fixed = Account()
    fixed.corporate_day(pd.Timestamp('2019-10-24'))
    assert len(fixed.receivables) == 1 and not fixed.actions
    fixed.corporate_day(pd.Timestamp('2019-10-25'))
    assert fixed.cohorts[0]['exit_date'] == '2019-10-01' and not fixed.holdings
    assert not fixed.receivables and len(fixed.actions) == len(fixed.movements) == 1
    assert fixed.actions[0]['kind'] == 'share_delivery' and fixed.actions[0]['qty'] == 0
    fixed.corporate_day(pd.Timestamp('2019-10-28'))
    assert len(fixed.actions) == 1


def test_real_new_shares_are_still_delivered_by_the_existing_account():
    class Core:
        def __init__(self):
            self.receivables = [dict(kind='shares', qty=1, fractional_cash_per_share=0., pay_date='2020-01-02')]
        def corporate_day(self, day):
            assert len(self.receivables) == 1
            return 456.
    class Account(ZeroValueShareDelivery, Core):
        pass
    assert Account().corporate_day(pd.Timestamp('2020-01-02')) == 456.
