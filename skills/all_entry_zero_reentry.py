"""Keep pre-open zero buys rejected when their old holding sells that day.

The account loop can revisit a sold residual as a new candidate. Its committed
buy was already rejected before the open, so it is a zero event rather than a
second attempt against the tape. Positive orders and duplicate events retain
the original strict reuse checks.
"""
from skills.poc_gap_execution import gap_key

POLICY = 'precommitted_zero_buy_after_sale_rejection_v1'


class ZeroBuyAfterSale:
    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if side != 'buy' or sid not in self.tick_attempts:
            return super()._execute_order(day,sid,side,qty,reason,event_id,signal_date)
        stamp = str(day.date())
        key = stamp,sid,side,event_id
        plan = self.day_plans.get((event_id,side))
        sells = [r for r in self.orders if r['date']==stamp and r['stock_id']==sid and r['side']=='sell']
        if (not plan or gap_key(plan)!=key or type(qty) is not int or qty<0
                or any(type(plan.get(k)) is not int or plan[k]!=0 for k in ('planned_qty','board_qty','odd_qty'))
                or plan.get('reserved_cash')!=0 or plan.get('sizing_budget')!=0
                or plan.get('rejection')!='opening_slots_locked'
                or not isinstance(signal_date,str) or not signal_date<stamp
                or plan.get('signal_date')!=signal_date or plan.get('reference_date')!=signal_date
                or [p for p in self.tick_plans if gap_key(p)==key]!=[plan]
                or not sells or any(r['event_id']==event_id for r in sells)
                or any(gap_key(r)==key for r in self.orders)):
            # No positive re-entry, malformed zero plan, or duplicate rejection
            # is authorized by this narrow correction.
            return super()._execute_order(day,sid,side,qty,reason,event_id,signal_date)
        self.orders.append(dict(date=stamp,stock_id=sid,side=side,event_id=event_id,
            signal_date=signal_date,reason=reason,channel='event',requested_qty=0,filled_qty=0,
            failure=plan['rejection']))
        return 0


def install_zero_reentry(runner):
    original = runner.engine_types
    def engine_types(*args, **kwargs):
        return tuple(type('ZeroBuyAfterSale'+cls.__name__,(ZeroBuyAfterSale,cls),{})
                     for cls in original(*args,**kwargs))
    runner.engine_types = engine_types
