"""Equivalent prior mark for the reviewed consecutive-session tick planner.

TickPlanning already refuses an absent immediately preceding close. When that
close exists, Replay's search for the last positive finite close necessarily
returns that same value. Retain both checks and the dated corporate reference
adjustment, without scanning the entire earlier Series for every candidate.
"""
from skills.million_replay import Replay
from skills.residual_tick_replay import TickPlanning


class ConsecutivePrior:
    def corporate_day(self, day):
        if self.positions[day] % 100 == 0:
            print('account progress', str(day.date()), flush=True)
        return super().corporate_day(day)

    def prior(self, day, sid):
        index = self.positions[day]
        if not index:
            return None
        value = self.raw(self.days[index - 1], sid)
        if not value:
            return None
        if hasattr(self.corporate, 'reference_price'):
            value = self.corporate.reference_price(sid, str(day.date()), value)
        return value


def equivalent_type(base):
    methods = [c.__dict__['prior'] for c in base.__mro__ if 'prior' in c.__dict__]
    if methods != [TickPlanning.prior, Replay.prior] or base.raw is not Replay.raw:
        raise ValueError('New prior-price chain requires renewed equivalence review')
    return type('ConsecutivePrior' + base.__name__, (ConsecutivePrior, base), {})


def install_prior_lookup(runner):
    original = runner.engine_types
    def engine_types(*args, **kwargs):
        return tuple(equivalent_type(cls) for cls in original(*args, **kwargs))
    runner.engine_types = engine_types
