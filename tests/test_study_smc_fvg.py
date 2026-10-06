"""End-to-end study wiring with real causal SMC and paired outcome functions."""
import json

import pandas as pd
import pytest

from scripts.study_smc_fvg import study
from skills.smc_research import STRATEGY_IDS


def prepared_fixture(with_setups=True):
    # A deliberately synthetic business-day calendar spanning all report years.
    days = pd.bdate_range('2023-09-01', '2026-03-31')
    ids = ['0050', '2330', '2454']
    f = {name: pd.DataFrame(value, index=days, columns=ids) for name, value in {
        'c': 100., 'h': 101., 'l': 98., 'open': 99., 'close': 100.,
        'volume': 1_000_000., 'valid': True, 'eligible': True,
    }.items()}

    def candle(stock, i, opening, high, low, close):
        for field, value in dict(open=opening, h=high, l=low, c=close, close=close).items():
            f[field].at[days[i], stock] = value

    if with_setups:
        anchors = [days.get_loc(pd.Timestamp(date)) for date in (
            '2024-01-15', '2025-02-17', '2026-02-02', '2026-03-31')]
        for anchor in anchors:
            for stock in ids[1:]:
                candle(stock, anchor-1, 101, 112, 100, 110)
                candle(stock, anchor, 111, 116, 110, 114)
                f['volume'].at[days[anchor], stock] = 2_000_000.
                if anchor+1 < len(days):
                    candle(stock, anchor+1, 109, 116, 108, 115)
        candle('0050', anchors[0], 119, 121, 118, 120)  # Known above MA60.
        candle('0050', anchors[1], 81, 82, 79, 80)  # Known below MA60.
        f['valid'].at[days[anchors[2]-3], '0050'] = False  # Unknown context, valid future path.
        f['valid'].at[days[anchors[0]+3], '2454'] = False  # Incomplete stock path after entry.
        f['valid'].at[days[anchors[1]+3], '0050'] = False  # Incomplete benchmark path.
    return f, days, ids


@pytest.fixture(scope='module')
def result():
    f, days, ids = prepared_fixture()
    return study(f, days, ids, start='2024-01-02', end=str(days[-1].date()))


def test_all_eight_rules_and_three_horizons_retained_even_when_no_events(result):
    report, events, paired, setups = result
    aggregate = [r for r in report['summary'] if r['year'] == 'all']
    assert len(aggregate) == 8*3
    assert {(r['strategy_id'], r['horizon']) for r in aggregate} == {
        (strategy, horizon) for strategy in STRATEGY_IDS for horizon in (5, 20, 60)}
    assert any(r['events'] == 0 for r in aggregate)
    assert len(report['summary']) == 8*3*4  # all +2024 +2025 +2026
    assert len(report['market_context']) == 8*3*3
    assert events.stock_id.ne('0050').all()
    assert not events.empty and not paired.empty and setups['events']
    json.dumps(report, allow_nan=False)


def test_year_and_market_context_groups_conserve_events_maturity_and_means(result):
    report, events, _, _ = result
    count_fields = ['events', 'evaluated', 'immature', 'stock_path_missing', 'benchmark_path_missing']
    observed_states = set(events.status)
    assert {'evaluated', 'immature', 'stock_path_missing', 'benchmark_path_missing'} <= observed_states
    assert set(events.market_context) == {'above_ma60', 'at_or_below_ma60', 'unknown'}
    for total in [r for r in report['summary'] if r['year'] == 'all']:
        key = (total['strategy_id'], total['horizon'])
        annual = [r for r in report['summary'] if (r['strategy_id'], r['horizon']) == key and r['year'] != 'all']
        contexts = [r for r in report['market_context'] if (r['strategy_id'], r['horizon']) == key]
        raw = events[events.strategy_id.eq(key[0]) & events.horizon.eq(key[1])]
        assert total['events'] == len(raw)
        assert sum(total[field] for field in count_fields[1:]) == total['events']
        for partitions in (annual, contexts):
            for field in count_fields:
                assert sum(r[field] for r in partitions) == total[field]
            if total['evaluated']:
                weighted_mean = sum(r['evaluated'] * r['mean_net_return'] for r in partitions if r['evaluated']) / total['evaluated']
                assert weighted_mean == pytest.approx(total['mean_net_return'])
        if total['evaluated']:
            assert total['mean_net_return'] == pytest.approx(raw.loc[raw.status.eq('evaluated'), 'net_return'].mean())


def test_all_entries_follow_signal_and_paired_exits_remain_anchored_to_formation(result):
    report, events, paired, setups = result
    _, days, _ = prepared_fixture()
    coordinates = {str(day.date()): i for i, day in enumerate(days)}
    for row in events.itertuples():
        if row.entry_date is not None:
            assert coordinates[row.entry_date] == coordinates[row.signal_date]+1
        if row.exit_date is not None:
            assert coordinates[row.exit_date] == coordinates[row.signal_date]+row.horizon
    for row in paired.itertuples():
        if row.direct_entry_date is not None:
            assert coordinates[row.direct_entry_date] == coordinates[row.formation_date]+1
        if row.wait_entry_date is not None:
            assert coordinates[row.wait_entry_date] == coordinates[row.retest_date]+1
            assert coordinates[row.wait_entry_date] > coordinates[row.direct_entry_date]
        if row.exit_date is not None:
            assert coordinates[row.exit_date] == coordinates[row.formation_date]+row.horizon
    # Last-session formations survive in reports without fictitious next-day fills.
    final = events[events.signal_date.eq(str(days[-1].date()))]
    assert not final.empty and final.status.eq('immature').all()
    assert final.entry_date.isna().all() and final.exit_date.isna().all()
    pending = paired[paired.status.eq('pending')]
    assert not pending.empty and pending.outcome_status.eq('immature').all()
    assert pending.wait_net.isna().all()
    assert all(r['live_qualified'] is False for r in report['paired_summary'])


def test_paired_cohorts_preserve_each_formation_not_only_retest_events(result):
    report, _, paired, setups = result
    formations = [s for s in setups['setups'] if s['kind'] == 'fvg' and s['setup_date'] >= '2024-01-02']
    assert len(paired) == 2*len(formations)
    assert paired.groupby('setup_id').horizon.apply(set).map(lambda horizons: horizons == {20, 60}).all()
    for total in [r for r in report['paired_summary'] if r['year'] == 'all']:
        annual = [r for r in report['paired_summary'] if r['horizon'] == total['horizon'] and r['year'] != 'all']
        for field in ['formations', 'evaluated', 'immature', 'missing', 'pending', 'wait_entered', 'cash_nonentries']:
            assert sum(r[field] for r in annual) == total[field]
        assert total['evaluated']+total['immature']+total['missing']+total['pending'] == total['formations']


def test_entirely_empty_study_still_reports_all_arms_and_years_without_fake_metrics():
    f, days, ids = prepared_fixture(with_setups=False)
    report, events, paired, setups = study(f, days, ids, start='2024-01-02', end=str(days[-1].date()))
    assert events.empty and paired.empty and not setups['events']
    assert len([r for r in report['summary'] if r['year'] == 'all']) == 24
    assert len(report['summary']) == 96 and len(report['market_context']) == 72
    assert all(r['events'] == 0 and r['mean_net_return'] is None and r['win_rate'] is None for r in report['summary'])
    assert all(r['formations'] == 0 and r['mean_wait_net'] is None for r in report['paired_summary'])
    json.dumps(report, allow_nan=False)


def test_runner_rejects_end_before_prepared_cutoff_and_unobserved_start():
    f, days, ids = prepared_fixture(with_setups=False)
    with pytest.raises(ValueError, match='exactly the observed study end'):
        study(f, days, ids, start='2024-01-02', end=str(days[-2].date()))
    with pytest.raises(ValueError, match='exactly the observed study end'):
        study(f, days, ids, start='2024-01-06', end=str(days[-1].date()))
