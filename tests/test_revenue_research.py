import numpy as np
import pandas as pd
import pytest

from skills.revenue_research import revenue_features, revenue_scores
from skills.rule_research import simulate


def revenue():
    months = pd.date_range('2019-01-01', periods=30, freq='MS')
    return pd.DataFrame({'stock_id': '2330', 'trading_date': months,
                         'revenue_current_month': np.arange(1, 31)*100.})


def features(rows, lag=45):
    return revenue_features(rows, pd.date_range('2020-01-01', '2021-10-01'), pd.Index(['2330']), lag)


def test_revenue_requires_exact_calendar_months_and_availability_lag():
    rows = revenue()
    day = pd.Timestamp('2020-04-01') + pd.Timedelta(days=45)
    result = features(rows)['growth']
    # Feb-Apr provider dates: (1400+1500+1600)/(200+300+400)-1 = 4.
    assert result.at[day, '2330'] == pytest.approx(4.)
    assert result.at[day-pd.Timedelta(days=1), '2330'] != pytest.approx(4.)
    missing = rows[~rows.trading_date.eq('2019-03-01')]
    assert pd.isna(features(missing)['growth'].at[day, '2330'])


def test_missing_latest_month_expires_signal_instead_of_forward_filling_old_value():
    rows = revenue()
    rows = rows[~rows.trading_date.eq('2021-05-01')]
    result = features(rows)['growth']
    assert pd.notna(result.at['2021-06-14', '2330'])
    assert pd.isna(result.at['2021-06-15', '2330'])
    assert pd.isna(result.at['2021-10-01', '2330'])  # No indefinite stale tail.


def test_future_revenue_cannot_change_prior_signals_and_negative_values_are_invalid():
    rows = revenue()
    before = features(rows)
    rows.loc[rows.trading_date.ge('2021-01-01'), 'revenue_current_month'] = -1
    after = features(rows)
    for name in before:
        pd.testing.assert_frame_equal(before[name].loc[:'2021-02-14'], after[name].loc[:'2021-02-14'])
    assert after['growth'].loc['2021-02-15':].isna().all().all()


def test_longer_lag_defers_same_information_and_duplicate_periods_fail():
    rows = revenue()
    early, late = features(rows, 45)['growth'], features(rows, 60)['growth']
    assert early.at['2020-05-16', '2330'] == late.at['2020-05-31', '2330']
    with pytest.raises(ValueError, match='unique'):
        features(pd.concat([rows, rows.iloc[:1]]))


def test_revenue_scores_are_executable_only_next_session_and_never_select_benchmark():
    days = pd.bdate_range('2020-01-01', periods=160)
    close = pd.DataFrame({'0050': np.arange(160)+100., '2330': np.arange(160)+100.}, index=days)
    matrix = close*0+1.
    values = revenue_scores({'adj_close': close, 'raw_close': close, 'raw_volume': matrix*1e6},
                            matrix, matrix, {'growth': matrix*.3, 'acceleration': matrix*.1})
    assert all(s['0050'].isna().all() for s in values.values())
    start = days[130]
    score = matrix*np.nan
    score.loc[start, '2330'] = 1  # Too late for first day; next monthly signal is missing.
    run = simulate(close, close.notna(), score, start=start)
    assert not run.trades
