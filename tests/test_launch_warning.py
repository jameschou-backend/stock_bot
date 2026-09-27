import numpy as np
import pandas as pd
import pytest

from skills.launch_warning import observations, peer_features, decisions, outcomes, fee_diagnostic


def fixture():
    days = pd.bdate_range('2022-01-03', periods=110)
    ids = ['0050', *[str(2300+i) for i in range(8)]]
    c = pd.DataFrame(100., index=days, columns=ids); raw = c.copy(); v = c*0+1_000_000
    q = dict(open=c-1, high=c+1, low=c-2, close=c.copy(), volume=v.copy())
    companies = pd.DataFrame([dict(stock_id=sid, listed_date='2000-01-01', industry='01') for sid in ids[1:]])
    events = pd.DataFrame([dict(event_id='a', stock_id='2300', signal_date=str(days[40].date()))])
    a = (c, raw, v, q, companies, events)
    set_bar(a, 40, close=104., opening=100., low=99., volume=3_000_000)
    for j in range(41, 110): set_bar(a, j, close=104., opening=103., low=102.)
    return a


def set_bar(a, pos, *, close, opening=None, low=None, high=None, volume=1_000_000):
    c, raw, v, q, _, _ = a; day = c.index[pos]
    opening = close-1 if opening is None else opening
    vals = dict(open=opening, high=max(close, opening)+1 if high is None else high,
        low=min(close, opening)-1 if low is None else low, close=close, volume=volume)
    c.at[day, '2300'] = raw.at[day, '2300'] = close; v.at[day, '2300'] = volume
    for k,val in vals.items(): q[k].at[day, '2300'] = val


def observe(a):
    c, raw, v, q, companies, events = a
    return observations(events, c, raw, v, q, companies)


def decide(a, obs=None):
    return decisions(a[-1], observe(a) if obs is None else obs, a[0])


def test_volume_drop_from_burst_alone_is_not_a_failure():
    a = fixture(); obs = observe(a)
    # One million is a third of launch volume, but normal pre-launch volume.
    assert obs.volume_ratio_normal.eq(1).all()
    assert not obs.dry_weak.any() and not obs.heavy_red.any()
    assert decide(a).state.eq('holding').all()


def test_dry_weak_is_after_two_observed_days_and_exit_is_next_day():
    a = fixture()
    set_bar(a, 41, close=103.5, opening=104., low=98.5, volume=500_000)
    set_bar(a, 42, close=103., opening=104., low=98., volume=500_000)
    obs = observe(a); assert not obs.iloc[0].dry_weak and obs.iloc[1].dry_weak
    d = decide(a, obs); r = d[(d.arm=='dry_weak') & (d.horizon==20)].iloc[0]
    assert r.signal_date == str(a[0].index[42].date())
    assert r.exit_date == str(a[0].index[43].date())
    assert r.delayed_exit_date == str(a[0].index[44].date())


def test_warning_must_execute_before_breach_to_count_as_avoiding_it():
    a = fixture()
    set_bar(a, 41, close=103.5, opening=104., low=98.5, volume=500_000)
    set_bar(a, 42, close=103., opening=104., low=98., volume=500_000)
    set_bar(a, 43, close=99.)
    d = decide(a); r = outcomes(a[-1], d, a[0], a[0])
    row = r[(r.arm=='dry_weak') & (r.horizon==20)].iloc[0]
    assert row.warning_before_failure and not row.exit_before_failure
    assert row.exit_return == pytest.approx(99/103.5-1)


def test_heavy_black_candle_can_warn_before_origin_and_uses_normal_volume():
    a = fixture(); set_bar(a, 41, close=101., opening=104., low=100., high=105., volume=1_600_000)
    obs = observe(a)
    assert obs.iloc[0].heavy_red and obs.iloc[0]['close'] > obs.iloc[0].origin
    # This is still LESS volume than the launch; the normal baseline matters.
    assert obs.iloc[0].volume_ratio_normal == pytest.approx(1.6)


def test_missing_prior_confirmation_cannot_be_backfilled_from_later_warning():
    a = fixture(); a[3]['volume'].iloc[41, 1] = np.nan
    set_bar(a, 42, close=101., opening=104., low=100., high=105., volume=2_000_000)
    r = decide(a); r = r[(r.arm=='heavy_red') & (r.horizon==20)].iloc[0]
    assert r.state == 'unknown' and r.signal_date == str(a[0].index[41].date())


def test_known_warning_overrides_other_unknown_but_does_not_guess_all_unknown():
    a = fixture(); obs = observe(a)
    obs.loc[0, 'rotation_weak'] = pd.NA; obs.loc[0, 'heavy_red'] = True
    r = decide(a, obs).query("arm == 'combined' and horizon == 20").iloc[0]
    assert r.reason == 'heavy_red'
    obs.loc[0, 'heavy_red'] = False
    r = decide(a, obs).query("arm == 'combined' and horizon == 20").iloc[0]
    assert r.state == 'unknown'


def test_peer_statistics_exclude_target_and_cover_missing_or_future_listings():
    a = fixture(); c, raw, v, _, companies, _ = a
    base = peer_features(c, raw, v, companies, {'2300'})['2300']
    c.loc[:, '2300'] *= 4; raw.loc[:, '2300'] *= 4
    changed = peer_features(c, raw, v, companies, {'2300'})['2300']
    for k in ('peer_return5', 'amount_ratio', 'above20'):
        pd.testing.assert_series_equal(base[k], changed[k])
    raw.iloc[-10:, 2:5] = np.nan
    assert not peer_features(c, raw, v, companies, {'2300'})['2300'].iloc[-1].known
    future = companies.copy(); future.loc[future.stock_id.isin(['2306','2307']), 'listed_date'] = '2030-01-01'
    reference = peer_features(a[0], a[1], a[2], future, {'2300'})['2300']
    assert reference.peer_count.eq(5).all()


def test_leave_one_out_median_matches_direct_reference_with_ties_and_nans():
    a = fixture(); c, raw, v, _, companies, _ = a
    for i, sid in enumerate(c.columns[1:]): c[sid] *= np.linspace(1, 1+i/10, len(c))
    c.iloc[35:38, 3] = np.nan
    all_values = peer_features(c, raw, v, companies, set(companies.stock_id))
    ret = (c/c.shift(5)-1).where(c.notna().rolling(6).sum().eq(6))
    for sid in companies.stock_id:
        peer_ids = [x for x in companies.stock_id if x != sid]
        for pos in (30, 36, 39, 70):
            assert all_values[sid].iloc[pos].peer_return5 == pytest.approx(ret.iloc[pos][peer_ids].median())


def test_future_bars_do_not_cancel_a_latched_exit():
    a = fixture(); set_bar(a, 41, close=101., opening=104., low=100., high=105., volume=2_000_000)
    before = decide(a).query("arm == 'heavy_red'").reset_index(drop=True)
    for f in (a[0], a[1], a[2], *a[3].values()): f.iloc[45:] *= 7
    after = decide(a).query("arm == 'heavy_red'").reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_warning_monitor_expires_after_ten_days_but_origin_stop_remains():
    a = fixture(); set_bar(a, 51, close=101., opening=104., low=100., high=105., volume=2_000_000)
    assert decide(a).query("arm == 'heavy_red'").state.eq('holding').all()
    set_bar(a, 52, close=99.)
    assert decide(a).query("arm == 'heavy_red'").reason.eq('origin').all()


def test_fee_diagnostic_charges_both_sides_and_stock_tax_not_etf_tax():
    expected = 1.1*(1-.001425-.003-.0045)/(1+.001425+.0045)-1
    assert fee_diagnostic(.1, sid='2300') == pytest.approx(expected)
    assert fee_diagnostic(.1, sid='0050') > fee_diagnostic(.1, sid='2300')
    assert fee_diagnostic(.1, sid='2300', slip_multiplier=2) < expected
