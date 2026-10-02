"""Every signal is a separate proportional unit, not a funded trading account."""
from dataclasses import dataclass
import math

import numpy as np

from skills.independent_signals import SignalPath, net_unit_return
from skills.exit_policy import decide_exit


@dataclass(frozen=True)
class ThreeBlackPath(SignalPath):
    opened: np.ndarray

    def validate(self):
        super().validate()
        if len(self.opened) != len(self.days):
            raise ValueError('Open prices must share the path calendar')


def black_at_close(path, entry, index):
    if index < entry + 2:
        return False
    for k in range(index - 2, index + 1):
        opened, closed, volume = path.opened[k], path.raw_close[k], path.volume[k]
        current, previous = path.close[k], path.close[k-1]
        if not all(math.isfinite(v) and v > 0 for v in (opened, closed, volume, current, previous)):
            return False
        if not (closed < opened and current < previous
                and not math.isclose(current, previous, rel_tol=1e-12, abs_tol=1e-12)):
            return False
    return True


def path_issue(path, start, end):
    sl = slice(start, end + 1)
    c, other = path.close[sl], path.other[sl]
    raw, high, low, opened, vol = [v[sl] for v in
        (path.raw_close, path.high, path.low, path.opened, path.volume)]
    if not all((np.isfinite(v) & (v > 0)).all() for v in (c, other, raw, high, low, opened)):
        return 'missing_or_invalid_price'
    if not path.eligible[sl].all():
        return 'historical_identity_or_eligibility'
    if not ((low <= raw) & (raw <= high) & (low <= opened) & (opened <= high)).all():
        return 'raw_ohlc_conflict'
    if not (np.isfinite(vol) & (vol >= 0)).all():
        return 'missing_or_invalid_volume'
    if not vol[0] > 0:
        return 'no_volume_on_assumed_entry'
    a, b = c[1:] / c[:-1] - 1, other[1:] / other[:-1] - 1
    if ((abs(a) > .20) | (abs(b) > .20) | (abs(a-b) > .005)).any():
        return 'daily_adjustment_conflict'
    if abs(c[-1]/c[0] - other[-1]/other[0]) > .02:
        return 'cumulative_adjustment_conflict'
    return None


def observe(path, entry_index):
    """Completed close decision -> next market-session HL2, with original priority."""
    path.validate()
    if type(entry_index) is not int or not 1 <= entry_index < len(path.days):
        raise ValueError('Entry needs a preceding signal session')
    start, last = entry_index, len(path.days)-1
    day = lambda i: str(path.days[i].date())
    result = dict(status='unresolved', outcome='unknown', exit_reason=None,
        exit_trigger_date=None, exit_date=None, entry_price=None, exit_price=None,
        mark_price=None, gross_return=None, net_return=None, unrealized_net_return=None,
        holding_days=None, holding_days_inclusive=None, calendar_days=None,
        mfe=None, mfe_date=None, peak_close_return=None, peak_close_date=None,
        confirmed_high_return=None, confirmed_high_date=None, data_issue=None)
    trigger, reason, end = None, None, last
    anchor = path.close[start]
    for j in range(start, min(start+62, last)+1):
        current = path.close[j]
        valid = bool(np.isfinite(current) and current > 0 and np.isfinite(anchor) and anchor > 0)
        decision = decide_exit(dict(held_sessions=j+1-start, has_signal=valid,
            entry_return=float(current/anchor-1) if valid else None,
            peak_return=None, peak_drawdown=None, relative20=None,
            below_ma20_two=False, market_off_two=False, strong_trend=False), 'loss12')
        reason = decision['reason'] if decision['exit'] else (
            'three_black' if black_at_close(path, start, j) else None)
        if reason:
            trigger, end = j, min(j+1, last)
            break
    closed = trigger is not None and trigger < last
    issue = path_issue(path, start, end)
    if issue is None and closed and not path.volume[end] > 0:
        issue = 'no_volume_on_assumed_exit'
    # A missing pre-entry close can hide the first black-candle comparison.
    if issue is None and not (np.isfinite(path.close[start-1]) and path.close[start-1] > 0):
        issue = 'missing_pre_entry_close'
    result.update(observed_end_date=day(end), data_issue=issue)
    if issue:
        return result
    entry = float((path.high[start]+path.low[start])/2)
    adjusted_entry = entry * path.close[start]/path.raw_close[start]
    raw_end = float((path.high[end]+path.low[end])/2) if closed else float(path.raw_close[end])
    adjusted_end = raw_end * path.close[end]/path.raw_close[end]
    gross, net = float(adjusted_end/adjusted_entry-1), float(net_unit_return(adjusted_end/adjusted_entry))
    result.update(status='closed' if closed else 'pending_exit' if trigger is not None else 'open',
        outcome=('profit' if net > 1e-12 else 'loss' if net < -1e-12 else 'flat') if closed else 'unrealized',
        entry_price=entry, exit_price=raw_end if closed else None,
        mark_price=None if closed else raw_end,
        adjusted_entry_price=float(adjusted_entry), adjusted_end_price=float(adjusted_end),
        exit_trigger_date=day(trigger) if trigger is not None else None,
        exit_date=day(end) if closed else None, exit_reason=reason,
        gross_return=gross, net_return=net if closed else None,
        unrealized_net_return=None if closed else net,
        holding_days=end-start, holding_days_inclusive=end-start+1,
        calendar_days=int((path.days[end]-path.days[start]).days),
        entry_volume=float(path.volume[start]), exit_volume=float(path.volume[end]) if closed else None,
        entry_single_price=bool(path.high[start] == path.low[start]),
        exit_single_price=bool(closed and path.high[end] == path.low[end]),
        stop_anchor_adjusted_close=float(anchor))
    def peak(values, lo, hi, key, date_key):
        if hi <= lo:
            return
        section = values[lo:hi]/adjusted_entry-1
        index = int(np.argmax(section))
        result.update({key:float(section[index]), date_key:day(lo+index)})
    with np.errstate(divide='ignore', invalid='ignore'):
        adjusted_high = path.high*path.close/path.raw_close
    # Includes entry/exit days, so this is only a daily-data upper bound.
    peak(adjusted_high, start, end+1, 'mfe', 'mfe_date')
    peak(path.close, start, end if closed else end+1, 'peak_close_return', 'peak_close_date')
    peak(adjusted_high, start+1, end if closed else end+1,
         'confirmed_high_return', 'confirmed_high_date')
    return result


def summarize(rows):
    closed = [r for r in rows if r['status'] == 'closed']
    winners = [r for r in closed if r['outcome'] == 'profit']
    losers = [r for r in closed if r['outcome'] == 'loss']
    returns = [r['net_return'] for r in closed]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in returns):
        raise ValueError('Every closed row needs a finite realized return')
    return dict(total=len(rows), stocks=len({r['stock_id'] for r in rows}), closed=len(closed),
        win=len(winners), loss=len(losers), flat=len(closed)-len(winners)-len(losers),
        win_rate=len(winners)/len(closed) if closed else None,
        mean_net_return=float(np.mean(returns)) if returns else None,
        median_net_return=float(np.median(returns)) if returns else None,
        mean_holding_days=float(np.mean([r['holding_days'] for r in closed])) if closed else None,
        **{key:sum(r['status'] == key for r in rows) for key in ('open','pending_exit','unresolved','not_entered')},
        exits={key:sum(r['exit_reason'] == key for r in closed) for key in ('loss12','time63','three_black')})
