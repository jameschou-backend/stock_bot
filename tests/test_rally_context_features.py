"""Evidence timing and forward-path boundaries for rally context research."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.study_rally_context import fixed_outcomes
from skills.rally_context_features import (
    attach_context, build_signal_features, conjunction, nullable_condition,
)
from skills.strategy_scanner.outcomes import COSTS, _net


def test_nullable_numeric_evidence_keeps_all_nonfinite_values_unknown():
    actual = nullable_condition([1., 0., -1., np.nan, np.inf, -np.inf], lambda x: x > 0)
    assert actual[:3].tolist() == [True, False, False]
    assert actual[3:].isna().all()


def test_false_component_does_not_hide_unknown_component_in_conjunction():
    result = conjunction([True, False, False, True], [True, True, pd.NA, pd.NA])
    assert result[:2].tolist() == [True, False]
    assert result[2:].isna().all()


def context_inputs():
    days = pd.bdate_range('2024-01-02', periods=15)
    date = str(days[10].date())
    events = pd.DataFrame([
        dict(stock_id='2330', signal_date=date, not_extended=True, close_strong=True,
             contraction=True, moderate_volume=False),
        dict(stock_id='2317', signal_date=date, not_extended=False, close_strong=pd.NA,
             contraction=pd.NA, moderate_volume=False),
    ])
    flows = [dict(stock_id='2330', signal_date=date, lag=lag,
                  flow_end=str(days[10-lag].date()), price_start=str(days[5-lag].date()),
                  flow_ratio5=value, flow_issue=None)
             for lag, value in ((1, .1), (3, -.2))]
    peers = [dict(stock_id='2330', signal_date=date, group_cutoff_date='2023-12-29',
                  feature_available_at=date+' after completed close',
                  historical_industry_claimed=False, feature_issue=None,
                  peer_above_ma20_fraction=.5, peer_share_multiple=1.2)]
    return events, flows, peers, days


def test_external_features_join_exact_stock_date_and_keep_lags_separate():
    events, flows, peers, days = context_inputs()
    # A different date's observed feature must not be carried forward onto 2317.
    flows.append(dict(flows[0], stock_id='2317', signal_date=str(days[9].date())))
    actual = attach_context(events, flows, peers, days)
    assert actual.loc[0, 'flow_positive_lag1']
    assert not actual.loc[0, 'flow_positive_lag3']
    assert actual.loc[0, 'peer_breadth'] and actual.loc[0, 'peer_turnover']
    assert actual.loc[0, 'peer_and_flow']
    assert actual.loc[0, 'clean_price']
    assert not actual.loc[0, 'quiet_breakout']
    assert actual.loc[1, 'flow_issue_lag1'] == 'coordinate_not_covered'
    for col in ('flow_positive_lag1', 'peer_breadth', 'peer_and_flow', 'clean_price', 'quiet_breakout'):
        assert pd.isna(actual.loc[1, col])


def test_issue_rows_remain_unknown_even_with_positive_stale_numbers():
    events, flows, peers, days = context_inputs()
    flows[0]['flow_issue'] = 'flow_category_or_volume_missing_or_conflicting'
    peers[0]['feature_issue'] = 'insufficient_common_peer_coverage'
    actual = attach_context(events, flows, peers, days)
    for col in ('flow_positive_lag1', 'peer_breadth', 'peer_turnover', 'peer_and_flow'):
        assert pd.isna(actual.loc[0, col])


@pytest.mark.parametrize('source', ['flow', 'peer'])
def test_duplicate_external_coordinate_is_rejected(source):
    events, flows, peers, days = context_inputs()
    target = flows if source == 'flow' else peers
    target.append(deepcopy(target[0]))
    with pytest.raises(ValueError, match='Duplicate'):
        attach_context(events, flows, peers, days)


@pytest.mark.parametrize('lag', [1, 3])
def test_flow_cutoff_must_equal_declared_market_session_lag(lag):
    events, flows, peers, days = context_inputs()
    row = next(row for row in flows if row['lag'] == lag)
    row['flow_end'] = events.iloc[0].signal_date
    with pytest.raises(ValueError, match='cutoff'):
        attach_context(events, flows, peers, days)


def test_known_flow_requires_an_explicit_cutoff():
    events, flows, peers, days = context_inputs()
    flows[0]['flow_end'] = None
    with pytest.raises(ValueError, match='cutoff|Known flow'):
        attach_context(events, flows, peers, days)


@pytest.mark.parametrize('incorrect_start', [None, '2024-01-15'])
def test_known_flow_price_window_must_match_same_five_sessions(incorrect_start):
    events, flows, peers, days = context_inputs()
    flows[0]['price_start'] = incorrect_start
    with pytest.raises(ValueError, match='window|start|cutoff'):
        attach_context(events, flows, peers, days)


def test_peer_membership_cannot_reach_signal_day():
    events, flows, peers, days = context_inputs()
    peers[0]['group_cutoff_date'] = events.iloc[0].signal_date
    with pytest.raises(ValueError, match='membership'):
        attach_context(events, flows, peers, days)


def test_peer_features_cannot_be_available_after_signal_close():
    events, flows, peers, days = context_inputs()
    peers[0]['feature_available_at'] = str(days[11].date())+' after completed close'
    with pytest.raises(ValueError, match='available|availability|cutoff|Peer feature'):
        attach_context(events, flows, peers, days)


def test_price_peers_cannot_be_mislabeled_historical_industry():
    events, flows, peers, days = context_inputs()
    peers[0]['historical_industry_claimed'] = True
    with pytest.raises(ValueError, match='industry'):
        attach_context(events, flows, peers, days)


@pytest.mark.parametrize('source', ['flow', 'peer'])
def test_known_external_evidence_must_be_finite(source):
    events, flows, peers, days = context_inputs()
    if source == 'flow':
        flows[0]['flow_ratio5'] = np.inf
    else:
        peers[0]['peer_share_multiple'] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        attach_context(events, flows, peers, days)


def synthetic_scanner_inputs():
    days = pd.bdate_range('2022-01-03', periods=480)
    rows = []
    for sid in ('0050', '2330', '2317'):
        for i, day in enumerate(days):
            close = 100.+np.sin(i/4.)
            volume = 7_000_000.
            if sid == '2330' and i in (425, 430, 431, 450):
                close, volume = 120.+i/100., 14_000_000.
            rows.append(dict(date=day, stock_id=sid, open=close-.3, high=close+.5,
                low=close-.6, close=close, adjusted_close=close, volume=volume,
                amount=close*volume, quality=True, eligible=True))
    signals = [dict(signal_date=str(days[i].date()), members=['2330'], priority=.2)
               for i in (425, 430, 431, 450)]
    p = dict(signal_date=str(days[430].date()), stock_id='2330', status='up',
             source_date_end=str(days[429].date()),
             prior_dates=[str(day.date()) for day in days[410:430]],
             poc_before=100., poc_after=105., available=True)
    return pd.DataFrame(rows), days, signals, [p]


def test_context_features_survive_future_perturbation_and_true_prefix_truncation():
    bars, days, signals, profiles = synthetic_scanner_inputs()
    cutoff = days[440]
    def build(source, calendar, end):
        return build_signal_features(source, calendar, start=str(days[420].date()),
            end=str(end.date()), original_signals=signals, poc=profiles,
            provenance={'original_candidates_complete': True})[0]
    full = build(bars, days, days[-1])
    expected = full[full.signal_date <= str(cutoff.date())].reset_index(drop=True)
    assert set(expected.cohort) == {'original_red', 'legacy_course_breakout'}
    original = expected[expected.cohort.eq('original_red')]
    assert str(days[430].date()) in set(original.signal_date)
    assert str(days[431].date()) not in set(original.signal_date)
    assert original.loc[original.signal_date.eq(str(days[430].date())), 'poc_up'].item()
    assert expected.market_breadth.isna().all()  # fewer than 500 eligible symbols
    truncated = build(bars[bars.date <= cutoff], days[:441], cutoff)
    pd.testing.assert_frame_equal(expected, truncated)
    changed = bars.copy()
    future = changed.date > cutoff
    for field in ('open', 'high', 'low', 'close', 'adjusted_close'):
        changed.loc[future, field] *= 9
    changed.loc[future, 'volume'] *= 17
    changed.loc[future, 'amount'] *= 153
    changed.loc[future, 'eligible'] = False
    mutated = build(changed, days, days[-1])
    mutated = mutated[mutated.signal_date <= str(cutoff.date())].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, mutated)


def outcome_inputs():
    days = pd.bdate_range('2024-01-02', periods=85)
    ids = ['0050', '2330']
    c = pd.DataFrame({'0050': np.arange(85)+100., '2330': np.arange(85)+200.}, index=days)
    f = dict(c=c, close=c/2, open=c/2, h=c+1., l=c-1.,
             valid=pd.DataFrame(True, index=days, columns=ids),
             eligible=pd.DataFrame(True, index=days, columns=ids),
             volume=pd.DataFrame(1000., index=days, columns=ids))
    f['open'].iloc[6, 1] = 100.  # next open adjusted by factor 2 gives 200
    f['h'].iloc[5, 1] = 1_000_000.  # signal-day extreme excluded
    f['h'].iloc[10, 1] = 400.
    f['l'].iloc[11, 1] = 90.
    f['h'].iloc[26, 1] = 900.  # outside 20-day window but within 60-day window
    f['l'].iloc[26, 1] = 50.
    events = pd.DataFrame([dict(event_id='x', stock_id='2330', signal_index=5,
                               column_index=1, signal_date=str(days[5].date()))])
    return events, f, days, ids


def test_outcome_uses_next_adjusted_open_exact_terminal_day_and_within_window_extrema():
    events, f, days, ids = outcome_inputs()
    actual = fixed_outcomes(events, f, days, ids).set_index('horizon')
    row = actual.loc[20]
    assert row.entry_date == str(days[6].date())
    assert row.exit_date == str(days[25].date())
    assert row.gross_return == pytest.approx(225./200.-1)
    assert row.net_return == pytest.approx(_net(200., 225., COSTS['stock_sell_tax']))
    assert row.benchmark_net_return == pytest.approx(_net(106., 125., COSTS['benchmark_sell_tax']))
    assert row.mfe == pytest.approx(400./200.-1)
    assert row.mae == pytest.approx(90./200.-1)
    assert actual.loc[60, 'exit_date'] == str(days[65].date())
    assert actual.loc[60, 'gross_return'] == pytest.approx(265./200.-1)
    assert actual.loc[60, 'mfe'] == pytest.approx(900./200.-1)
    assert actual.loc[60, 'mae'] == pytest.approx(50./200.-1)


@pytest.mark.parametrize('field,value', [('valid', False), ('eligible', False), ('volume', 0.)])
def test_missing_holding_path_never_becomes_shorter_successful_outcome(field, value):
    events, f, days, ids = outcome_inputs()
    f[field].iloc[12, 1] = value
    actual = fixed_outcomes(events, f, days, ids)
    assert actual.mature.all() and not actual.complete.any()
    assert actual[['gross_return', 'net_return', 'mfe', 'mae']].isna().all().all()
    assert actual.benchmark_net_return.notna().all()


def test_immature_outcome_keeps_known_entry_but_no_exit_or_return():
    events, f, days, ids = outcome_inputs()
    events.loc[0, 'signal_index'] = 70
    events.loc[0, 'signal_date'] = str(days[70].date())
    actual = fixed_outcomes(events, f, days, ids)
    assert actual.entry_date.eq(str(days[71].date())).all()
    assert actual.exit_date.isna().all()
    assert not actual.mature.any() and not actual.complete.any()
    assert actual[['gross_return', 'net_return', 'mfe', 'mae', 'benchmark_net_return']].isna().all().all()


def test_benchmark_unknown_does_not_erase_complete_stock_outcome():
    events, f, days, ids = outcome_inputs()
    f['valid'].iloc[12, 0] = False
    actual = fixed_outcomes(events, f, days, ids)
    assert actual.complete.all() and actual.net_return.notna().all()
    assert actual.benchmark_net_return.isna().all()
