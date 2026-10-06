"""End-to-end cohort invariants using the real entry and outcome modules."""
import numpy as np
import pandas as pd
import pytest

from scripts.study_conditional_entries import study
from skills.conditional_entries import CONDITION_IDS


def fixture():
    days = pd.bdate_range('2023-01-02', periods=480)
    rows = [dict(date=day, stock_id=sid, open=99., high=101., low=98., close=100.,
                 adjusted_close=100., volume=6_000_000., amount=600_000_000.,
                 quality=True, eligible=True)
            for day in days for sid in ('2330', '0050')]
    bars = pd.DataFrame(rows)
    def put(i, o, h, l, c):
        mask = bars.date.eq(days[i]) & bars.stock_id.eq('2330')
        bars.loc[mask, ['open', 'high', 'low', 'close', 'adjusted_close', 'amount']] = [o,h,l,c,c,c*6_000_000]
    put(401, 120, 123, 119, 121)  # original candidate entry at next open, not signal close
    put(410, 101, 102, 99, 100)  # original breakout ledger row, but black K fails original_red
    put(412, 101, 112, 100, 110)
    put(413, 111, 116, 110, 114)  # independent FVG day, absent from original ledger
    signals = [dict(signal_date=str(days[i].date()), stock_id='2330') for i in (400,401,405,410)]
    provenance = dict(original_candidates_complete=True, source_end=str(days[-1].date()),
        original_signal_start=str(days[0].date()), original_signal_end=str(days[-1].date()))
    return bars, days, signals, provenance


def run(bars, days, signals, provenance, end=None):
    return study(bars, days, original_signals=signals, provenance=provenance,
                 start=str(days[399].date()), end=end or str(days[-1].date()))


def coordinates(frame):
    return set(map(tuple, frame[['signal_date', 'stock_id']].drop_duplicates().to_numpy()))


def test_true_runner_keeps_original_first_signal_dates_and_never_generates_new_candidates():
    bars, days, signals, provenance = fixture()
    report, events, conditions = run(bars, days, signals, provenance)
    expected = {(str(days[i].date()), '2330') for i in (400,405)}
    assert report['candidate_count'] == 2 and report['candidate_stocks'] == 1
    assert coordinates(events) == coordinates(conditions) == expected
    assert (str(days[401].date()), '2330') not in expected  # consecutive original match isn't new
    assert (str(days[410].date()), '2330') not in coordinates(events)  # black K is rejected
    assert (str(days[413].date()), '2330') not in coordinates(events)  # FVG never creates entry date
    assert len(events) == 2*3 and set(events.horizon) == {5,20,60}
    assert len(conditions) == 2*7 and set(conditions.filter_id) == set(CONDITION_IDS)
    assert events.groupby(['signal_date','stock_id']).size().eq(3).all()
    assert conditions.groupby(['signal_date','stock_id']).size().eq(7).all()
    for name in ('summary', 'common_pool_summary'):
        rows = [row for row in report[name] if row['year'] == 'all']
        assert len(rows) == 7*3
        assert all(row['all_candidates'] == 2 for row in rows)
    baseline = [row for row in report['baseline_summary'] if row['year'] == 'all']
    assert len(baseline) == 3 and all(row['events'] == 2 for row in baseline)
    first = events[(events.signal_date == str(days[400].date())) & events.horizon.eq(5)].iloc[0]
    assert first.entry_date == str(days[401].date())
    assert first.exit_date == str(days[405].date())
    assert first.gross_return == pytest.approx(100/120-1)


def test_true_runner_future_prices_can_change_outcomes_but_not_candidates_or_conditions():
    bars, days, signals, provenance = fixture()
    _, full_events, full_conditions = run(bars, days, signals, provenance)
    _, prefix_events, prefix_conditions = run(bars, days, signals, provenance, str(days[414].date()))
    pd.testing.assert_frame_equal(full_conditions, prefix_conditions)
    assert coordinates(full_events) == coordinates(prefix_events)
    mature5 = prefix_events[prefix_events.horizon.eq(5)].reset_index(drop=True)
    pd.testing.assert_frame_equal(mature5, full_events[full_events.horizon.eq(5)].reset_index(drop=True))
    future = bars.date.gt(days[414]) & bars.stock_id.eq('2330')
    bars.loc[future, ['open','high','low','close','adjusted_close','amount']] = [150,202,140,200,200,1_200_000_000]
    _, changed_events, changed_conditions = run(bars, days, signals, provenance)
    pd.testing.assert_frame_equal(full_conditions, changed_conditions)
    assert coordinates(full_events) == coordinates(changed_events)
    assert not np.allclose(full_events.net_return, changed_events.net_return)


@pytest.mark.parametrize('complete', [False, None])
def test_incomplete_original_ledger_is_rejected_before_research(complete):
    bars, days, signals, provenance = fixture()
    provenance['original_candidates_complete'] = complete
    with pytest.raises(ValueError, match='complete hash-bound original candidate ledger'):
        run(bars, days, signals, provenance)


def test_requested_end_beyond_frozen_source_is_rejected():
    bars, days, signals, provenance = fixture()
    provenance['source_end'] = str(days[-2].date())
    with pytest.raises(ValueError, match='frozen source end'):
        run(bars, days, signals, provenance)


def test_original_first_requires_known_prior_ledger_day():
    bars, days, signals, provenance = fixture()
    provenance['original_signal_start'] = str(days[400].date())
    report, events, conditions = run(bars, days, signals, provenance)
    assert report['candidate_count'] == 1
    assert coordinates(events) == coordinates(conditions) == {(str(days[405].date()), '2330')}


def test_empty_original_cohort_still_reports_all_fixed_filters_and_horizons():
    bars, days, _, provenance = fixture()
    report, events, conditions = run(bars, days, [], provenance)
    assert report['candidate_count'] == report['candidate_stocks'] == 0
    assert events.empty and conditions.empty
    assert set(report['filters']) == set(CONDITION_IDS)
    for key in ('summary', 'common_pool_summary'):
        all_rows = [row for row in report[key] if row['year'] == 'all']
        assert len(all_rows) == 7*3
        assert {(row['filter_id'], row['horizon']) for row in all_rows} == {
            (identifier, horizon) for identifier in CONDITION_IDS for horizon in (5,20,60)}
        assert all(row['all_candidates'] == 0 for row in all_rows)
        assert all(row['kept_mean_net'] is None for row in all_rows)
        assert all(row['baseline_profit_win_rate'] is None for row in all_rows)
    baseline = [row for row in report['baseline_summary'] if row['year'] == 'all']
    assert len(baseline) == 3 and all(row['events'] == 0 for row in baseline)
    assert all(row['mean_net_return'] is None for row in baseline)
