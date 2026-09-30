from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from skills.independent_signals import SignalPath, observe, net_unit_return


def path(n=75):
    p = np.full(n, 100.)
    return SignalPath(pd.bdate_range('2024-01-01', periods=n), p.copy(), p.copy(),
                      np.ones(n, dtype=bool), p.copy(), p + 1, p - 1, p * 10000)


def test_time_exit_is_next_session_and_can_lose_after_costs():
    p = path()
    r = observe(p, 1)
    assert r['exit_signal_date'] == str(p.days[63].date())
    assert r['exit_date'] == str(p.days[64].date())
    assert r['reason'] == 'time63' and r['outcome'] == 'loss'
    assert r['gross_return'] == 0 and r['net_return'] < 0


def test_stop_is_close_trigger_next_day_with_overnight_gap():
    p = path()
    for v in (p.close, p.other, p.raw_close):
        v[4] = 88
        v[5:] = 80
    p.high[:] = p.raw_close + 1
    p.low[:] = p.raw_close - 1
    r = observe(p, 1)
    assert r['reason'] == 'loss12'
    assert r['exit_signal_date'] == str(p.days[4].date())
    assert r['exit_date'] == str(p.days[5].date())
    assert r['gross_return'] == pytest.approx(-.2)


def test_intraday_low_does_not_trigger_and_anchor_is_entry_close():
    p = path(8)
    p.low[2] = 80
    assert observe(p, 1)['status'] == 'open'
    p.high[1] = 130  # HL2 114.5; later 100 close is >12% below fill, not anchor.
    assert observe(p, 1)['status'] == 'open'


def test_last_close_stop_is_pending_not_realized():
    p = path(6)
    for v in (p.close, p.other, p.raw_close):
        v[-1] = 88
    p.high[-1], p.low[-1] = 89, 87
    r = observe(p, 1)
    assert r['status'] == 'pending_exit' and r['exit_date'] is None
    assert r['net_return'] is None and r['unrealized_net_return'] < 0


def test_future_after_exit_cannot_change_result():
    p = path()
    r = observe(p, 1)
    for v in (p.close, p.other, p.raw_close, p.high, p.low):
        v[65:] = np.nan
    assert observe(p, 1) == r
    # High AFTER the intraday exit must not inflate experienced close peak.
    p.close[64] = p.other[64] = p.raw_close[64] = 110
    p.high[64] = 110
    assert observe(p, 1)['peak_close_return'] == 0


def test_missing_path_cannot_be_reported_as_profit_or_loss():
    p = path()
    p.close[3] = np.nan
    r = observe(p, 1)
    assert r['status'] == 'unknown' and r['reason'] is None
    assert 'net_return' not in r and r['exit_date'] is None


def test_split_uses_adjusted_ratio_not_raw_price_crash():
    p = path()
    p.raw_close[20:] /= 2
    p.high[20:] /= 2
    p.low[20:] /= 2
    r = observe(p, 1)
    assert r['reason'] == 'time63' and r['raw_exit_price'] == 50
    assert r['gross_return'] == pytest.approx(0)


def test_source_conflict_and_zero_fill_volume_are_explicit():
    p = path()
    p.other[4] = 95
    assert observe(p, 1)['issue'] == 'daily_adjustment_conflict'
    p = path()
    p.volume[1] = 0
    assert observe(p, 1)['issue'] == 'no_volume_on_assumed_fill'
    p = path()
    p.eligible[20] = False
    assert observe(p, 1)['issue'] == 'historical_identity_or_eligibility'


def test_bad_inputs_fail_loudly():
    with pytest.raises(ValueError):
        observe(replace(path(), high=np.ones(3)), 1)
    with pytest.raises(ValueError):
        observe(path(), 0)
    with pytest.raises(ValueError):
        observe(path(), True)
    with pytest.raises(ValueError, match='Eligibility cannot be unknown'):
        observe(replace(path(), eligible=np.full(75, np.nan)), 1)
    with pytest.raises(ValueError, match='explicit booleans'):
        observe(replace(path(), eligible=np.ones(75)), 1)
    with pytest.raises(ValueError):
        net_unit_return(float('nan'))
