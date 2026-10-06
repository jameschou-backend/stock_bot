import numpy as np
import pandas as pd
import pytest

from skills.smc_outcomes import compare_fvg
from skills.strategy_scanner.outcomes import COSTS, _net


def fixture(n=100):
    days = pd.bdate_range('2024-01-02', periods=n)
    ids = ['0050', '2330', '2454']
    f = {name: pd.DataFrame(100., index=days, columns=ids)
         for name in ('open', 'close', 'c', 'h', 'l')}
    f['h'] += 2
    f['l'] -= 2
    f['volume'] = pd.DataFrame(1_000_000., index=days, columns=ids)
    for key in ('valid', 'eligible'):
        f[key] = pd.DataFrame(True, index=days, columns=ids)
    return f, days, ids


def setup(days, *, stock='2330', anchor=10, state='retested', retest=15, identifier='fvg-a'):
    return dict(kind='fvg', stock_id=stock, setup_id=identifier,
                setup_date=str(days[anchor].date()), status=state,
                retest_date=str(days[retest].date()) if state == 'retested' else None)


def price(f, days, i, value, stock='2330', *, opening=None, high=None, low=None):
    op = value if opening is None else opening
    for key in ('close', 'c'):
        f[key].at[days[i], stock] = value
    f['open'].at[days[i], stock] = op
    f['h'].at[days[i], stock] = max(op, value)+2 if high is None else high
    f['l'].at[days[i], stock] = min(op, value)-2 if low is None else low


def run(f, days, ids, setups, horizons=(20,)):
    return compare_fvg(f, days, ids, setups, days[0], days[-1], horizons=horizons)


def test_arms_share_formation_exit_not_retest_plus_horizon_and_charge_both_sides():
    f, d, ids = fixture()
    price(f, d, 11, 105, opening=100)
    price(f, d, 16, 100, opening=80)
    price(f, d, 30, 130)
    price(f, d, 35, 900)  # Would dramatically inflate a retest+horizon comparison.
    price(f, d, 11, 100, '0050', opening=90)
    price(f, d, 16, 95, '0050', opening=95)
    price(f, d, 30, 110, '0050')
    summary, rows = run(f, d, ids, [setup(d)])
    r = rows.iloc[0]
    assert r.outcome_status == 'evaluated'
    assert r.direct_entry_date == str(d[11].date())
    assert r.wait_entry_date == str(d[16].date())
    assert r.exit_date == str(d[30].date())
    assert r.wait_delay_sessions == 5
    assert r.direct_gross == pytest.approx(.3)
    assert r.wait_gross == pytest.approx(130/80-1)
    assert r.direct_net == pytest.approx(_net(100, 130, COSTS['stock_sell_tax']))
    assert r.wait_net == pytest.approx(_net(80, 130, COSTS['stock_sell_tax']))
    assert r.common_benchmark_net == pytest.approx(_net(90, 110, COSTS['benchmark_sell_tax']))
    assert r.wait_exposed_benchmark_net == pytest.approx(_net(95, 110, COSTS['benchmark_sell_tax']))
    assert r.delta == pytest.approx(r.wait_net-r.direct_net)
    assert summary[0]['mean_paired_delta'] == pytest.approx(r.delta)


def test_never_retested_rally_stays_in_cash_cohort_not_dropped_or_cost_charged():
    f, d, ids = fixture()
    price(f, d, 30, 160, '2454')
    setups = [setup(d), setup(d, stock='2454', state='expired', identifier='fvg-b')]
    summary, rows = run(f, d, ids, setups)
    s = summary[0]
    cash = rows.iloc[1]
    assert cash.direct_rally and cash.missed_rally and not cash.wait_entered
    assert cash.wait_net == 0 and cash.wait_gross == 0
    assert pd.isna(cash.wait_exposed_benchmark_net)
    assert pd.isna(cash.wait_mae) and pd.isna(cash.wait_mfe)
    assert s['formations'] == 2 and s['evaluated'] == 2 and s['cash_nonentries'] == 1
    assert s['take_rate'] == .5
    assert s['mean_wait_net'] == pytest.approx(rows.iloc[0].wait_net / 2)
    assert s['executed_only_mean_wait_net'] == rows.iloc[0].wait_net
    assert s['missed_rallies'] == 1 and s['missed_share_of_direct_rallies'] == 1
    assert s['wait_win_rate_including_cash'] == 0
    assert s['wait_cash_is_win'] is False
    assert s['paired_delta_cluster_bootstrap'] is None  # One formation-date cluster.


def test_invalidated_nonentry_is_cash_but_does_not_cancel_direct_arm():
    f, d, ids = fixture()
    price(f, d, 30, 70)
    _, rows = run(f, d, ids, [setup(d, state='invalidated')])
    r = rows.iloc[0]
    assert r.outcome_status == 'evaluated'
    assert r.direct_net < -.3 and r.wait_net == 0 and r.delta > .3
    assert not r.direct_rally and not r.missed_rally


@pytest.mark.parametrize('key,stock,value,expected', [
    ('valid', '2330', False, 'stock_path_missing'),
    ('eligible', '2330', False, 'stock_path_missing'),
    ('volume', '2330', 0, 'stock_path_missing'),
    ('c', '2330', np.nan, 'stock_path_missing'),
    ('h', '2330', np.nan, 'stock_path_missing'),
    ('valid', '0050', False, 'benchmark_path_missing'),
    ('eligible', '0050', False, 'benchmark_path_missing'),
    ('volume', '0050', 0, 'benchmark_path_missing'),
])
def test_full_common_holding_path_required_even_before_delayed_entry(key, stock, value, expected):
    f, d, ids = fixture()
    f[key].at[d[13], stock] = value  # Before wait entry at16; endpoint-only check would miss it.
    summary, rows = run(f, d, ids, [setup(d)])
    r = rows.iloc[0]
    assert r.outcome_status == expected
    assert pd.isna(r.direct_net) and pd.isna(r.wait_net) and pd.isna(r.delta)
    assert summary[0]['missing'] == 1 and summary[0]['evaluated'] == 0


def test_unknown_setup_is_not_zero_cash_and_maturity_is_not_shortened():
    f, d, ids = fixture()
    setups = [setup(d, state='data_missing'),
              setup(d, state='pending', anchor=95, identifier='pending-young'),
              setup(d, state='pending', identifier='pending-unresolved')]
    summary, rows = run(f, d, ids, setups)
    assert rows.outcome_status.tolist() == ['setup_data_missing', 'immature', 'setup_pending']
    assert rows.wait_net.isna().all() and rows.direct_net.isna().all()
    assert summary[0]['missing'] == 1 and summary[0]['immature'] == 1
    assert summary[0]['pending'] == 1 and summary[0]['evaluated'] == 0
    assert rows.iloc[1].exit_date is None


def test_excursions_use_each_actual_held_window_not_account_drawdown():
    f, d, ids = fixture()
    price(f, d, 13, 100, low=70, high=160)
    price(f, d, 20, 100, low=90, high=120)
    summary, rows = run(f, d, ids, [setup(d)])
    r = rows.iloc[0]
    assert r.direct_mae == pytest.approx(-.3)
    assert r.direct_mfe == pytest.approx(.6)
    assert r.wait_mae == pytest.approx(-.1)
    assert r.wait_mfe == pytest.approx(.2)
    assert summary[0]['mean_direct_mae'] == pytest.approx(-.3)
    assert summary[0]['executed_only_mean_wait_mae'] == pytest.approx(-.1)
    assert 'not_account_drawdown' in summary[0]['excursion_definition']


def test_no_retest_same_day_or_after_ten_sessions_and_duplicate_cohorts_rejected():
    f, d, ids = fixture()
    for retest in (10, 21):
        with pytest.raises(ValueError, match='next-10-session|formation/evidence cutoff'):
            run(f, d, ids, [setup(d, retest=retest)])
    with pytest.raises(ValueError, match='unique'):
        run(f, d, ids, [setup(d), setup(d)])
    wrong = setup(d, state='expired')
    wrong['retest_date'] = str(d[15].date())
    with pytest.raises(ValueError, match='Only retested'):
        run(f, d, ids, [wrong])


def test_tenth_session_retest_enters_next_session_and_two_horizons_keep_formation_year():
    f, d, ids = fixture()
    summary, rows = run(f, d, ids, [setup(d, retest=20)], horizons=(20, 60))
    assert rows.wait_entry_date.tolist() == [str(d[21].date())] * 2
    assert rows.exit_date.tolist() == [str(d[30].date()), str(d[70].date())]
    assert [(s['horizon'], s['year']) for s in summary] == [(20, 'all'), (20, '2024'), (60, 'all'), (60, '2024')]


def test_adjusted_open_uses_same_day_scale_and_signal_day_price_never_fills():
    f, d, ids = fixture()
    f['open'].at[d[11], '2330'] = 50
    f['close'].at[d[11], '2330'] = 50
    f['c'].at[d[11], '2330'] = 100
    f['open'].at[d[10], '2330'] = 1
    _, rows = run(f, d, ids, [setup(d)])
    assert rows.iloc[0].direct_gross == 0
    assert rows.iloc[0].direct_net == pytest.approx(_net(100, 100, COSTS['stock_sell_tax']))


def test_end_bounds_all_price_evidence_not_only_formations():
    f, d, ids = fixture()
    spec = setup(d, state='expired')
    summary, rows = compare_fvg(f, d, ids, [spec], d[0], d[25], horizons=(20,))
    assert rows.iloc[0].outcome_status == 'immature'
    assert summary[0]['evaluated'] == 0
    for key in ('open', 'close', 'c', 'h', 'l'):
        f[key].loc[d[26]:] = 999999
    again, other = compare_fvg(f, d, ids, [spec], d[0], d[25], horizons=(20,))
    assert summary == again
    pd.testing.assert_frame_equal(rows, other)


def test_empty_cohort_and_non_fvg_preserve_schema_and_zero_counts():
    f, d, ids = fixture()
    for setups in ([], [dict(kind='orderblock')]):
        summary, rows = run(f, d, ids, setups)
        assert rows.empty and 'direct_net' in rows
        assert summary[0]['formations'] == 0
        assert summary[0]['mean_paired_delta'] is None


def test_clustered_interval_reproducible_and_cash_included():
    f, d, ids = fixture()
    setups = [setup(d, anchor=10, state='expired', identifier='a'),
              setup(d, anchor=12, state='expired', identifier='b'),
              setup(d, anchor=12, state='expired', identifier='c', stock='2454')]
    price(f, d, 30, 160)
    first, rows = run(f, d, ids, setups)
    second, _ = run(f, d, ids, setups)
    ci = first[0]['paired_delta_cluster_bootstrap']
    assert ci == second[0]['paired_delta_cluster_bootstrap']
    assert ci['clusters'] == 2 and ci['samples'] == 500 and ci['seed'] == 20261006
    assert ci['low'] <= rows.delta.mean() <= ci['high']
    assert first[0]['cash_nonentries'] == 3


def test_comparison_requires_explicit_unique_fixed_horizons_and_matching_source_axes():
    f, d, ids = fixture()
    for horizons in ((), (20, 20), (True,), (5,)):
        with pytest.raises(ValueError, match='20/60'):
            run(f, d, ids, [], horizons=horizons)
    f['volume'] = f['volume'].iloc[:-1]
    with pytest.raises(ValueError, match='axes'):
        run(f, d, ids, [])


def test_future_terminal_setup_state_is_not_imported_into_earlier_cutoff():
    f, d, ids = fixture()
    spec = setup(d, state='expired')
    spec['expiry_date'] = str(d[20].date())
    with pytest.raises(ValueError, match='evidence cutoff'):
        compare_fvg(f, d, ids, [spec], d[0], d[18], horizons=(20,))


@pytest.mark.parametrize('stock', ['2330', '0050'])
def test_equal_adjusted_ohlc_roundoff_preserves_path_but_material_violation_is_missing(stock):
    f, d, ids = fixture()
    # Valid raw open==low can round a few ulps below its adjusted low.
    f['l'].at[d[13], stock] = np.nextafter(100., np.inf)
    _, rows = run(f, d, ids, [setup(d)])
    assert rows.iloc[0].outcome_status == 'evaluated'
    f['l'].at[d[13], stock] = 100.00001
    _, rows = run(f, d, ids, [setup(d)])
    assert rows.iloc[0].outcome_status == ('stock_path_missing' if stock == '2330' else 'benchmark_path_missing')


def test_adjustment_factor_uses_consistent_parenthesization_for_equal_open_low():
    f, d, ids = fixture()
    raw, adjusted, raw_open = 33.05, 30.728803, 32.9
    i = 13
    f['open'].at[d[i], '2330'] = raw_open
    f['close'].at[d[i], '2330'] = raw
    f['c'].at[d[i], '2330'] = adjusted
    f['h'].at[d[i], '2330'] = 33.55 * (adjusted / raw)
    f['l'].at[d[i], '2330'] = raw_open * (adjusted / raw)
    _, rows = run(f, d, ids, [setup(d)])
    assert rows.iloc[0].outcome_status == 'evaluated'
