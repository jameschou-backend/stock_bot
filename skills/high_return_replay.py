"""Revalidate archived three-stock ordering policies on mixed execution.

Only ordering and empty-account slot count change. Sealed source modules and
their evidence hashes remain intact; C/S/U constraints stay enabled throughout.
"""
from skills.mixed_odd_replay import factory
from skills.strict_tick_inputs import StrictResidualReplay


ARMS = ('original', 'capacity', 'capacity_vol')
ARCHIVE_MAPPING = {
    'budget_lock_1193': 'capacity',
    'capacity_vol_1142': 'capacity_vol',
    'capacity_1111': 'capacity',
    'cash_541': 'original',
}


class HighReturnReplay(factory(StrictResidualReplay)):
    def __init__(self, *args, ordering, position_count=3, **kwargs):
        if ordering not in ARMS or type(position_count) is not int or position_count not in (3, 5):
            raise ValueError('Use a preregistered ordering and slot count')
        super().__init__(*args, **kwargs)
        if self.holdings or self.trades or self.daily:
            raise ValueError('Policy must be fixed before any account activity')
        # FiveAxisReplay already computes causal amount20 / vol20 and orders
        # before TickPlanning reserves orders. Original retains source order.
        self.arm = 'control' if ordering == 'original' else ordering
        self.slots = position_count
        self.ordering = ordering

    def run(self):
        result = super().run()
        result['settings'].update(research_policy='high_return_revalidation_v1',
                                  ordering=self.ordering, idle_capital='cash')
        return result
