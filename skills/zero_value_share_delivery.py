"""Record zero-value stock deliveries without reopening a closed holding."""
from copy import deepcopy
import pandas as pd


class ZeroValueShareDelivery:
    def __init__(self, *args, **kwargs):
        self.zero_value_deliveries = []
        super().__init__(*args, **kwargs)

    def corporate_day(self, day):
        due = [r for r in self.receivables if r['kind'] == 'shares'
               and r.get('qty') == 0 and r.get('fractional_cash_per_share') == 0
               and r.get('pay_date') and pd.Timestamp(r['pay_date']) <= day]
        for right in due:
            cohort = next(c for c in self.cohorts if c['event_id'] == right['event_id'])
            # There are no shares or cash to deliver, but retain the original
            # zero payment/delivery journals for entitlement reconciliation.
            if right['fraction']:
                self.cash_move(day, 'fractional_share_payment', 0., stock_id=right['stock_id'])
            self.actions.append(dict(date=str(day.date()), kind='share_delivery',
                                     **{k: v for k, v in right.items() if k != 'kind'}))
            self.zero_value_deliveries.append(dict(date=str(day.date()), event_id=right['event_id'],
                stock_id=right['stock_id'], action_id=right['action_id'],
                previous_exit_date=cohort['exit_date'], whole_shares=0, fractional_cash=0.))
            self.receivables.remove(right)
        return super().corporate_day(day)

    def run(self):
        account = super().run()
        if self.zero_value_deliveries:
            account['zero_value_delivery_log'] = deepcopy(self.zero_value_deliveries)
        return account
