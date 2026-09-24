"""Preregistered position-count contrast without intraday resource reuse."""
from skills.slot_reuse_replay import SlotReuseReplay


class ConservativeDiversification(SlotReuseReplay):
    def __init__(self,*args,position_count=3,**kwargs):
        if type(position_count) is not int or position_count not in (3,5):
            raise ValueError('Only the preregistered 3/5 position contrast is supported')
        super().__init__(*args,opening_cash_only=True,lock_unused=True,
            lock_opening_slots=True,lock_failed_slots=True,**kwargs)
        # The sealed capacity arm initializes an empty account with three slots.
        # Slot-dependent sizing and occupancy occur only in the subsequent run.
        if self.holdings:
            raise ValueError('Position-count contrast requires an empty opening account')
        self.slots=position_count
