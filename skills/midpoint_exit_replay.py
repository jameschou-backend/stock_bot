"""HL2 exits with a legal floor instead of a prior-close sell-price hurdle."""
from copy import deepcopy
import math
from skills.midpoint_replay import MidpointStock
from skills.replay_market_feeds import ReplayDataUnavailable


def legal_exit_floor(limits):
    if not limits or any(isinstance(limits.get(k),bool) or not isinstance(limits.get(k),(int,float))
                         or not math.isfinite(limits[k]) for k in ('lower','upper')):
        raise ReplayDataUnavailable('Missing finite official exit price limits')
    if not 0 < limits['lower'] <= limits['upper']:
        raise ReplayDataUnavailable('Invalid official exit price limits')
    return float(limits['lower'])


class MidpointExitReplay(MidpointStock):
    def _plan(self, day, sid, side, eid, signal, qty, budget, opening_cash, failure):
        super()._plan(day,sid,side,eid,signal,qty,budget,opening_cash,failure)
        if side != 'sell':
            return
        plan = self.day_plans[(eid,side)]
        # No current high, low, close or volume is read to place the exit.
        plan['limit_price'] = legal_exit_floor(self.feeds.get_limits(sid).get(str(day.date())))
        self.tick_plans[-1] = deepcopy(plan)

    def run(self):
        result = super().run()
        result['settings'].update(sell_limit_policy='official_daily_lower_limit',
            sell_price_policy='channel_daily_high_low_midpoint',intraday_stop=False)
        return result
