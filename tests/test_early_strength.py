import numpy as np
import pandas as pd
import pytest

from skills.early_strength import event_features, decisions, orders, outcomes, summarize


def fixture():
    days = pd.bdate_range('2022-01-03', periods=100)
    c = pd.DataFrame({'0050': 100., '2313': 100.}, index=days)
    c.iloc[25:30, 1] = 99.
    c.iloc[30:, 1] = 104.
    v = c*0+1_000_000; v.iloc[30, 1] = 3_000_000
    q = dict(open=c-1, high=c+1, low=c-2, close=c.copy(), volume=v.copy())
    f = dict(eligible=c.gt(0), shares=v.rolling(20).mean(), adv=(c*v).rolling(20).mean())
    e = pd.DataFrame([dict(event_id='test', stock_id='2313', signal_date=str(days[30].date()), named_case=False)])
    return days, c, c.copy(), v, q, f, e


def choose(values):
    _, c, raw, v, q, f, e = values
    return decisions(event_features(e, c), f, c, raw, v, q)


def set_bar(values, pos, *, close=None, volume=None, low=None):
    days, c, raw, v, q, _, _ = values
    d = days[pos]
    if close is not None:
        c.at[d, '2313'] = raw.at[d, '2313'] = close
        for k, val in [('close', close), ('open', close-1), ('high', close+1), ('low', close-2)]:
            q[k].at[d, '2313'] = val
    if volume is not None:
        v.at[d, '2313'] = q['volume'].at[d, '2313'] = volume
    if low is not None:
        q['low'].at[d, '2313'] = low


def test_prior_strength_excludes_launch_gain_and_requires_full_benchmark_window():
    a = fixture(); days, c, _, _, _, _, e = a
    c.iloc[19, 1] = 103.; c.iloc[24, 1] = 99.; c.iloc[29, 1] = 100.
    before = event_features(e, c)
    assert bool(before.iloc[0].turn)
    c.iloc[30:, 1] *= 10
    pd.testing.assert_frame_equal(before, event_features(e, c))
    c.iloc[23, 0] = np.nan
    unknown = event_features(e, c).iloc[0]
    assert not unknown.strength_known and pd.isna(unknown.turn)


def test_later_missing_benchmark_does_not_change_earlier_feature_schema():
    a = fixture(); days, c, _, _, _, _, ev = a
    future = ev.copy(); future['event_id'] = 'later'; future['signal_date'] = str(days[80].date())
    c.iloc[75, 0] = np.nan
    full = event_features(pd.concat([ev, future], ignore_index=True), c)
    past = event_features(ev, c.iloc[:60])
    pd.testing.assert_frame_equal(full.iloc[:1].reset_index(drop=True), past)


def test_third_day_confirmation_never_backdates_order():
    a = fixture(); choice = choose(a).set_index('arm'); days = a[0]
    assert choice.loc['all_first', 'entry_date'] == str(days[31].date())
    assert choice.loc['all_confirm3', 'entry_signal_date'] == str(days[33].date())
    assert choice.loc['all_confirm3', 'entry_date'] == str(days[34].date())
    assert choice.loc['all_retest5', 'entry_date'] == str(days[32].date())
    submitted = orders(choice.reset_index(), a[5])
    assert submitted['all_confirm3'][0]['feature_cutoff_date'] == str(days[33].date())


@pytest.mark.parametrize('timing', ['confirm3', 'retest5'])
def test_failure_or_missing_day_cannot_be_skipped_for_later_success(timing):
    a = fixture(); set_bar(a, 31, close=98.)
    assert choose(a).set_index('arm').loc['all_'+timing, 'state'] == 'support_failed'
    a = fixture(); a[4]['low'].iloc[31, 1] = np.nan
    row = choose(a).set_index('arm').loc['all_'+timing]
    assert row.state == 'unknown_confirmation_data' and row.entry_date is None


def test_retest_uses_adjusted_ohlc_and_keeps_order_after_later_failure():
    a = fixture(); days, c, raw, v, q, f, e = a
    # A split changes raw prices but not adjusted economics.
    raw.iloc[31:, 1] /= 2
    for key in ('open', 'high', 'low', 'close'): q[key].iloc[31:, 1] /= 2
    first = choose(a).set_index('arm').loc['all_retest5']
    assert first.entry_date == str(days[32].date())
    set_bar(a, 35, close=90.)
    later = choose(a).set_index('arm').loc['all_retest5']
    pd.testing.assert_series_equal(first, later)


def test_retest_uses_first_match_not_best_later_price():
    a = fixture(); set_bar(a, 31, close=109., low=107.)
    set_bar(a, 32, close=105., low=104.)
    set_bar(a, 33, close=104., low=103.)
    row = choose(a).set_index('arm').loc['all_retest5']
    assert row.entry_signal_date == str(a[0][32].date())
    assert row.entry_date == str(a[0][33].date())


def test_unmatured_confirmation_and_ineligible_are_different():
    a = fixture(); days, c, raw, v, q, f, e = a
    r = decisions(event_features(e, c), f, c, raw, v, q, end=str(days[32].date())).set_index('arm')
    assert r.loc['all_confirm3', 'state'] == 'unmatured_confirmation'
    f['eligible'].iloc[31, 1] = False
    assert choose(a).set_index('arm').loc['all_confirm3', 'state'] == 'ineligible'


def test_outcomes_use_actual_next_day_price_common_end_and_matched_benchmark():
    a = fixture(); ev = event_features(a[-1], a[1]); choice = choose(a)
    # Entry T+4 = 34, common end T+21 = 51. These prices never alter frozen decisions.
    a[1].iloc[34:52, 1] = np.linspace(105., 120., 18)
    a[1].iloc[34:52, 0] = np.linspace(101., 110., 18)
    r = outcomes(ev, choice, a[1], a[1])
    r = r[(r.arm == 'all_confirm3') & (r.horizon == 20)].iloc[0]
    assert r.reference_return == pytest.approx(120/105-1)
    assert r.matched_benchmark_return == pytest.approx(110/101-1)
    assert r.baseline_return == pytest.approx(120/104-1)
    assert r.benchmark_return == pytest.approx(110/100-1)


def test_missing_decision_is_not_cash_but_known_nontrigger_is_cash():
    a = fixture(); ev = event_features(a[-1], a[1]); ch = choose(a)
    ch.loc[ch.arm.eq('all_confirm3'), 'state'] = 'unknown_confirmation_data'
    ch.loc[ch.arm.eq('all_retest5'), 'state'] = 'not_triggered'
    r = outcomes(ev, ch, a[1], a[1])
    assert r.loc[r.arm.eq('all_confirm3'), 'opportunity_return'].isna().all()
    assert r.loc[r.arm.eq('all_retest5'), 'opportunity_return'].eq(0).all()


def test_filtered_surges_are_counted_as_missed_not_erased():
    a = fixture(); ev = event_features(a[-1], a[1]); ch = choose(a)
    ch.loc[ch.arm.eq('turn_first'), ['state', 'selected']] = ['filtered', False]
    a[1].iloc[31:, 1] = np.linspace(104., 250., 69)
    r = outcomes(ev, ch, a[1], a[1]); s = summarize(r)
    row = s[(s.arm == 'turn_first') & (s.horizon == 20)].iloc[0]
    assert row.baseline_surges == 1 and row.missed_baseline_surges == 1
    assert row.triggered == 0 and row.opportunity_mean == 0


def test_named_pharma_is_excluded_from_accounts_and_main_statistics():
    a = fixture(); days, c, raw, v, q, f, ev = a
    for frame in (c, raw, v, *q.values(), *f.values()): frame.rename(columns={'2313':'6446'}, inplace=True)
    ev.stock_id = '6446'; enriched = event_features(ev, c)
    assert enriched.named_case.all()
    choice = decisions(enriched, f, c, raw, v, q)
    assert not any(orders(choice, f).values())
    assert summarize(outcomes(enriched, choice, c, c)).empty
