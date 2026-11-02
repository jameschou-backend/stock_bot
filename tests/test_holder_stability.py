"""Causal holding features and executable-time boundaries for the new study."""
import numpy as np
import pandas as pd
import pytest

from skills.holding_validation import TIERS
from skills.holder_stability import (
    COSTS, aggregate_600, holder_changes, with_holders, observations,
    event_outcomes, net_return, matched_pairs, summarize,
)


def raw_week(day, transfer=0):
    amounts = [1_000_000] * 15
    amounts[0] -= transfer
    amounts[-1] += transfer
    rows = [dict(stock_id='2492', date=day, HoldingSharesLevel=tier,
                 people=1, unit=amount, percent=round(amount / sum(amounts) * 100, 2))
            for tier, amount in zip(TIERS, amounts)]
    return pd.DataFrame(rows + [dict(stock_id='2492', date=day,
        HoldingSharesLevel='total', people=15, unit=sum(amounts), percent=100)])


def history(transfer=0, n=8):
    return pd.concat([aggregate_600(raw_week(str(day.date()), i * transfer), day, {'2492'})
        for i, day in enumerate(pd.date_range('2026-01-02', periods=n, freq='7D'))], ignore_index=True)


def queries(dates):
    return pd.DataFrame(dict(stock_id=['2492'] * len(dates), signal_date=dates,
                             price_setup=True, price_confirmed=True))


def market(n=220):
    days = pd.bdate_range('2025-01-01', periods=n)
    base = pd.DataFrame(100., index=days, columns=['0050', '2492'])
    f = {key: base.copy() for key in ('c', 'close', 'open', 'l')}
    f.update(valid=base.eq(100), eligible=base.eq(100), a=base * 1_000_000)
    companies = pd.DataFrame([dict(stock_id=sid, name=sid, industry='test', listed_date='2000-01-01')
                              for sid in base])
    return f, days, companies


def event(days, i=1, stop=90.):
    date = str(days[i].date())
    return pd.DataFrame([dict(event_id=date + ':2492', signal_date=date,
                             stock_id='2492', stop_close5=stop)])


def test_exact_600001_boundary_and_total_validation():
    raw = raw_week('2026-01-02')
    result = aggregate_600(raw, '2026-01-02', {'2492', '2308'}).set_index('stock_id')
    assert result.loc['2492', 'large_units'] == 3_000_000
    assert result.loc['2492', 'large_pct'] == .2
    assert not result.loc['2308', 'valid']
    assert pd.isna(result.loc['2308', 'large_units'])
    # The 400001-600000 tier (including exactly 600 lots) is excluded.
    lower = raw.index[raw.HoldingSharesLevel.eq('400001-600000')][0]
    raw.loc[lower, 'unit'] += 500_000
    raw.loc[0, 'unit'] -= 500_000
    raw.loc[raw.HoldingSharesLevel.ne('total'), 'percent'] = (
        raw.loc[raw.HoldingSharesLevel.ne('total'), 'unit'] / 15_000_000 * 100).round(2)
    assert aggregate_600(raw, '2026-01-02', {'2492'}).iloc[0].large_units == 3_000_000
    raw.loc[raw.HoldingSharesLevel.eq('total'), 'unit'] += 100
    assert not aggregate_600(raw, '2026-01-02', {'2492'}).iloc[0].valid


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'percent'])
def test_damaged_tiers_are_unknown(damage):
    raw = raw_week('2026-01-02')
    if damage == 'duplicate':
        raw = pd.concat([raw, raw.iloc[[-2]]])
    elif damage == 'missing':
        raw = raw.drop(raw.index[-2])
    else:
        raw.loc[0, 'percent'] = 70
    row = aggregate_600(raw, '2026-01-02', {'2492'}).iloc[0]
    assert not row.valid and pd.isna(row.large_units)


def test_four_week_share_count_and_percentage_point_thresholds():
    q = queries(['2026-02-07'])
    assert with_holders(q, holder_changes(history()), 8).iloc[0].stable_confirmed
    assert with_holders(q, holder_changes(history(-25_000)), 8).iloc[0].decreasing_confirmed
    # 40,000 shares: less than 0.5 percentage points, but more than 1% of big holdings.
    mixed = with_holders(q, holder_changes(history(-10_000)), 8).iloc[0]
    assert mixed.holder_state == 'mixed' and mixed.drawdown and not mixed.stable


def test_release_lag_staleness_and_unreleased_mutation():
    weekly = history()
    q = queries(['2026-02-06', '2026-02-07', '2026-05-01'])
    full = with_holders(q, holder_changes(weekly), 8)
    assert full.holder_state.tolist() == ['unknown', 'stable', 'unknown']
    assert not full.iloc[0].drawdown and not full.iloc[2].drawdown
    assert not with_holders(q.iloc[[1]], holder_changes(weekly), 15).iloc[0].chip_known
    unavailable = weekly.date + pd.Timedelta(days=8) > pd.Timestamp('2026-02-07')
    mutated = weekly.copy()
    mutated.loc[unavailable, ['large_units', 'large_pct']] = 999999.
    mutated.loc[unavailable, 'valid'] = False
    for changed in (weekly[~unavailable], mutated):
        actual = with_holders(q.iloc[:2], holder_changes(changed), 8)
        pd.testing.assert_frame_equal(full.iloc[:2].reset_index(drop=True), actual)


def test_missing_intermediate_week_and_cumulative_inventory_change():
    weekly = history(n=5)
    weekly.loc[2, 'valid'] = False
    assert not holder_changes(weekly).iloc[-1].change_known
    weekly = history(n=5)
    weekly['total_units'] *= 1.004 ** np.arange(5)
    result = holder_changes(weekly).iloc[-1]
    assert not result.change_known
    assert result.change_reason == 'four_week_inventory_denominator_change'


def test_price_condition_uses_prior_five_closes_and_full_history():
    f, days, companies = market()
    f['c'].iloc[-10:, 1] = 70
    f['c'].iloc[-1, 1] = 75
    end = str(days[-1].date())
    row = observations(f, days, companies, start=end, end=end).iloc[0]
    assert row.price_setup and row.price_confirmed
    assert row.prior_high5 == 70 and row.stop_close5 == 70 and row.drawdown60 == -.25
    f['c'].iloc[-1, 1] = 70
    assert not observations(f, days, companies, start=end, end=end).iloc[0].price_confirmed
    f['valid'].iloc[-30, 1] = False
    row = observations(f, days, companies, start=end, end=end).iloc[0]
    assert not row.price_known and not row.price_setup


def test_next_open_entry_and_costs_not_signal_close():
    f, days, _ = market(70)
    f['c'].iloc[1, 1] = 50  # Signal close cannot be an entry price.
    f['open'].iloc[2, 1] = 110
    f['c'].iloc[6, 1] = 120
    results = event_outcomes(f, days, event(days)).set_index('rule')
    row = results.loc['5']
    assert row.entry_date == str(days[2].date()) and row.exit_date == str(days[6].date())
    assert row.gross_return == pytest.approx(120 / 110 - 1)
    assert row.net_return == pytest.approx(net_return(110, 120, .003, .0045))
    assert row.stress_net_return < row.net_return
    assert row.holding_sessions == 5


def test_close_stop_exits_next_open_and_excludes_exit_day_low():
    f, days, _ = market(8)
    f['c'].iloc[3, 1] = 89
    f['l'].iloc[3, 1] = 88
    f['open'].iloc[4, 1] = 80
    f['l'].iloc[4, 1] = 1  # Occurs after our exit; not our MAE.
    row = event_outcomes(f, days, event(days)).set_index('rule').loc['stop20']
    assert row.status == 'evaluated'  # Early stop matures even without 20 future sessions.
    assert row.exit_reason == 'close_stop_next_open'
    assert row.exit_date == str(days[4].date())
    assert row.gross_return == pytest.approx(-.2) and row.mae == pytest.approx(-.2)
    f['open'].iloc[2, 1] = 80
    assert event_outcomes(f, days, event(days)).set_index('rule').loc['stop20'].status == 'entry_below_stop'


def test_time_exit_and_missing_path_are_not_compressed():
    f, days, _ = market(70)
    row = event_outcomes(f, days, event(days)).set_index('rule').loc['stop20']
    assert row.holding_sessions == 20 and row.exit_date == str(days[22].date())
    f['valid'].iloc[4, 1] = False
    rows = event_outcomes(f, days, event(days)).set_index('rule')
    assert rows.loc['5'].status == 'stock_path_missing' and pd.isna(rows.loc['5'].net_return)
    f['valid'].iloc[4, 1] = True
    f['eligible'].iloc[4, 0] = False
    assert event_outcomes(f, days, event(days)).set_index('rule').loc['20'].status == 'benchmark_path_missing'
    assert event_outcomes(f, days, event(days, i=68)).status.eq('immature').all()


def test_matching_excludes_named_cases_and_uses_pre_entry_features():
    rows = pd.DataFrame([dict(event_id=sid, stock_id=sid, holder_lag=8,
        signal_date='2026-01-02', industry='same', price_confirmed=False,
        drawdown=True, named_case=sid == '6442', holder_state=state,
        drawdown60=dd, amount20=100_000_000, future_return=999)
        for sid, state, dd in [('2492', 'stable', -.3), ('2308', 'decreasing', -.31),
                               ('2330', 'decreasing', -.5), ('6442', 'stable', -.3)]])
    pairs = matched_pairs(rows)
    assert len(pairs) == 1 and pairs.iloc[0].control_id == '2308'
    rows['future_return'] = [-999, -999, 999, 0]
    pd.testing.assert_frame_equal(pairs, matched_pairs(rows))
    rows.loc[rows.stock_id.eq('2308'), 'industry'] = 'different'
    assert matched_pairs(rows).iloc[0].control_id == '2330'


def test_aggregate_rejects_duplicate_event_lag():
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(pd.DataFrame(dict(event_id=['x', 'x'], holder_lag=[8, 8])), pd.DataFrame())
