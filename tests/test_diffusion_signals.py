"""Synthetic economic sequences exercise ordering, missingness and causality."""
import json

import numpy as np
import pandas as pd
import pytest

from skills.diffusion_signals import build_diffusion


def market(*, confirms=True, event_date='2022-01-05', periods=235):
    days = pd.bdate_range('2021-06-01', periods=periods)
    ids = ['0050', '1101', '1102', '1103', '1104']
    steps = np.arange(len(days))
    benchmark = 100 * np.exp(np.cumsum(.0003 + .0001 * np.cos(steps / 4)))
    peer = 100 * np.exp(np.cumsum(.0004 + .004 * np.sin(steps / 3)))
    close = pd.DataFrame(np.column_stack([benchmark, peer, peer, peer, peer]), index=days, columns=ids)
    i = days.get_loc(event_date)
    # An ordinary correlated group first cools off together. Only 1101 breaks
    # out on event day; 1102/1103 respond two sessions later, or never respond.
    anchor = float(close.iloc[i - 11]['1101'])
    close.iloc[i - 10:, 1:] = np.array([anchor * .998 ** k
                                       for k in range(1, len(days) - i + 11)])[:, None]
    leader_price = max(float(close.iloc[:i]['1101'].max()) * 1.08, anchor * 1.10)
    close.loc[days[i]:, '1101'] = leader_price
    if confirms:
        close.loc[days[i + 2]:, '1102'] = anchor * 1.07
        close.loc[days[i + 2]:, '1103'] = anchor * 1.06
    volume = pd.DataFrame(2_000_000., index=days, columns=ids)
    volume.loc[days[i], '1101'] = 4_000_000.
    if confirms:
        volume.loc[days[i + 2]:, ['1102', '1103']] = 20_000_000.
    turnover = close * volume
    companies = pd.DataFrame({'stock_id': ids[1:], 'listed_date': pd.Timestamp('2000-01-01')})
    return close, close.copy(), volume, turnover, companies


def run(inputs, **kwargs):
    return build_diffusion(*inputs, start=kwargs.pop('start', '2022-01-03'),
                           signal_end=kwargs.pop('signal_end', '2022-01-31'), **kwargs)


def event_on(result, date='2022-01-05'):
    return next(event for event in result['events'] if event['leader_date'] == date)


def test_ordered_diffusion_and_four_arms_use_original_priority():
    result = run(market())
    event = event_on(result)
    assert event['leader_id'] == '1101'
    assert event['leader_peer_breadth'] == 0
    assert event['status'] == 'confirmed'
    assert event['confirmation_date'] == '2022-01-07'
    assert event['follower_id'] == '1102'
    assert event['confirmation_peer_breadth'] == pytest.approx(2 / 3)
    assert event['confirmation_new_responders'] == ['1102', '1103']
    share = event['confirmation_turnover_share']
    assert share['last5_mean'] > share['prior20_mean']
    assert result['entries']['leader_now'][0]['entry_date'] == '2022-01-06'
    for arm in ('leader_after', 'follower_after', 'basket_after'):
        assert result['entries'][arm][0]['signal_date'] == '2022-01-07'
        assert result['entries'][arm][0]['entry_date'] == '2022-01-10'
        assert result['entries'][arm][0]['priority'] == event['priority']
    assert result['entries']['basket_after'][0]['members'] == ['1101', '1102', '1103', '1104']
    assert len(result['events']) == 1  # At most one event per monthly group.
    json.dumps(result, allow_nan=False)


def test_unconfirmed_leader_is_kept_in_immediate_arm():
    result = run(market(confirms=False))
    event = event_on(result)
    assert event['status'] == 'no_confirmation'
    assert len(event['confirmation_checks']) == 10
    assert event['follower_id'] is None
    assert len(result['entries']['leader_now']) == 1
    assert all(not result['entries'][arm] for arm in ('leader_after', 'follower_after', 'basket_after'))


def test_leader_own_rally_and_turnover_cannot_confirm_peers():
    inputs = market(confirms=False)
    close, other, volume, turnover, _ = inputs
    close.loc['2022-01-06':, '1101'] *= 1.03
    other.loc[:] = close
    volume.loc['2022-01-06':, '1101'] *= 10
    turnover.loc[:] = close * volume
    event = event_on(run(inputs))
    assert event['status'] == 'no_confirmation'
    assert all(check['peer_breadth'] == 0 for check in event['confirmation_checks'])


def test_simultaneous_group_strength_is_not_an_ordered_leader():
    inputs = market()
    close, other, volume, turnover, _ = inputs
    for sid in ['1102', '1103']:
        close.loc['2022-01-05':, sid] = close.loc['2022-01-07', sid]
    other.loc[:] = close
    turnover.loc[:] = close * volume
    result = run(inputs)
    assert not result['events']
    assert result['stats']['counts']['leader_rejected_already_broad'] >= 1


def test_prefix_and_future_mutation_cannot_change_group_or_earlier_decisions():
    inputs = market()
    full = run(inputs)
    # Mid-month prefix must already have its group and immediate decision.
    prefix_inputs = tuple(frame.loc[:'2022-01-06'].copy() if isinstance(frame.index, pd.DatetimeIndex)
                          else frame.copy() for frame in inputs)
    prefix = run(prefix_inputs)
    assert prefix['groups'] == full['groups']
    assert prefix['entries']['leader_now'] == full['entries']['leader_now']
    event = event_on(prefix)
    assert event['status'] == 'window_incomplete'
    assert event['leader_peer_breadth'] == event_on(full)['leader_peer_breadth']
    assert event['confirmation_checks'] == event_on(full)['confirmation_checks'][:1]
    # A dramatic later anomaly cannot rewrite the already-frozen January group
    # or the decisions made before that future day.
    mutated = tuple(frame.copy() for frame in inputs)
    mutated[0].loc['2022-01-12':, '1101'] *= 3
    changed = run(mutated)
    assert full['groups'] == changed['groups']
    assert full['events'] == changed['events']
    assert full['entries'] == changed['entries']


def test_pending_group_crosses_month_and_can_confirm_after_signal_cutoff():
    inputs = market(event_date='2022-01-31')
    result = run(inputs, signal_end='2022-01-31')
    event = event_on(result, '2022-01-31')
    assert event['group_month'] == '2022-01'
    assert event['confirmation_date'] == '2022-02-02'
    assert event['status'] == 'confirmed'
    assert result['entries']['follower_after'][0]['entry_date'] == '2022-02-03'


def test_missing_peer_price_is_unknown_not_a_zero_return():
    inputs = market()
    for frame in inputs[:2]:
        frame.loc['2022-01-06':'2022-01-19', '1104'] = np.nan
    result = run(inputs)
    event = event_on(result)
    assert event['status'] == 'data_insufficient'
    assert event['insufficient_data_sessions'] == 10
    assert all(check['peer_breadth'] is None for check in event['confirmation_checks'])
    assert not result['entries']['follower_after']


def test_missing_market_company_keeps_denominator_coverage_unknown():
    inputs = list(market())
    inputs[4] = pd.concat([inputs[4], pd.DataFrame({'stock_id': ['9999'],
                                                  'listed_date': [pd.Timestamp('2000-01-01')]})],
                         ignore_index=True)
    result = run(tuple(inputs))
    event = event_on(result)
    assert event['status'] == 'data_insufficient'
    assert event['leader_turnover_share']['coverage_min'] == .8
    assert event['leader_turnover_share']['last5_mean'] is None
    assert not result['entries']['follower_after']


def test_future_listing_does_not_reduce_past_market_coverage():
    inputs = list(market())
    inputs[4] = pd.concat([inputs[4], pd.DataFrame({'stock_id': ['9999'],
                                                  'listed_date': [pd.Timestamp('2022-03-01')]})],
                         ignore_index=True)
    event = event_on(run(tuple(inputs)))
    assert event['status'] == 'confirmed'
    assert event['leader_turnover_share']['coverage_min'] == 1.


def test_dual_price_anomaly_exclusion_uses_only_past_window():
    inputs = market()
    inputs[1].loc['2022-01-12':, '1104'] *= 1.01
    result = run(inputs, signal_end='2022-02-28')
    january, february = result['groups']
    assert any('1104' in group['members'] for group in january['clusters'])
    assert '1104' in february['exclusions']['price_anomaly_in_past126']
    assert event_on(result)['status'] == 'confirmed'


def test_listing_defense_and_liquidity_units_are_real_eligibility_rules():
    inputs = list(market())
    inputs[4].loc[inputs[4].stock_id == '1104', 'listed_date'] = pd.Timestamp('2021-12-01')
    result = run(tuple(inputs))
    assert '1104' in result['groups'][0]['exclusions']['listing_window']
    assert not result['events']
    low_liquidity = list(market())
    low_liquidity[2] /= 1000  # Mistaken thousand-share data cannot clear 50m TWD.
    low_liquidity[3] /= 1000
    result = run(tuple(low_liquidity))
    assert result['groups'][0]['eligible_count'] == 0
    assert not result['events']


def test_missing_benchmark_uses_common_days_without_price_filling():
    inputs = market()
    for frame in inputs[:2]:
        frame.loc['2021-11-01', '0050'] = np.nan
    result = run(inputs)
    assert result['groups'][0]['common_observations'] == 124
    assert event_on(result)['status'] == 'confirmed'
    # A missing stock observation on an otherwise common benchmark day blocks
    # the stock; it is not filled to imitate complete history.
    for frame in inputs[:2]:
        frame.loc['2021-11-10', '1104'] = np.nan
    result = run(inputs)
    assert '1104' in result['groups'][0]['exclusions']['incomplete_common_returns']
    assert not result['events']


@pytest.mark.parametrize('invalid_stock', ['1101', '1102'])
def test_confirmation_requires_current_leader_and_best_new_follower_quality(invalid_stock):
    inputs = market()
    # At confirmation time the second price source disagrees by 100bp. This
    # future discrepancy cannot delete the leader, or select the second-best
    # follower as a replacement for the invalid first-ranked candidate.
    inputs[1].loc['2022-01-07':, invalid_stock] *= 1.01
    result = run(inputs)
    event = event_on(result)
    assert event['status'] == 'data_insufficient'
    first_potential_confirmation = event['confirmation_checks'][1]
    assert first_potential_confirmation['follower_id'] == '1102'
    assert first_potential_confirmation['reason'] == 'leader_or_follower_quality_window_invalid'
    assert result['entries']['leader_now']
    assert not result['entries']['follower_after']


def test_peer_price_breadth_requires_separate_turnover_share_evidence():
    inputs = market()
    close, _, volume, turnover, _ = inputs
    # Hold every peer's estimated trading money flat, even while two prices
    # respond. No peer flow expansion means the confirmation must fail.
    turnover.loc[:, ['1102', '1103', '1104']] = 200_000_000.
    for sid in ['1102', '1103', '1104']:
        volume.loc[:, sid] = turnover[sid] / close[sid]
    event = event_on(run(inputs))
    assert event['status'] == 'no_confirmation'
    assert any(check['peer_breadth'] >= .6 for check in event['confirmation_checks'])
    assert any(check['reason'] == 'peer_turnover_share_not_rising'
               for check in event['confirmation_checks'])


def test_equal_leader_scores_tie_by_ticker_without_reselection():
    inputs = market(confirms=False)
    close, other, volume, turnover, _ = inputs
    close.loc['2022-01-05':, '1102'] = close.loc['2022-01-05':, '1101']
    other.loc[:] = close
    volume.loc['2022-01-05', '1102'] = volume.loc['2022-01-05', '1101']
    turnover.loc[:] = close * volume
    event = event_on(run(inputs))
    assert event['leader_id'] == '1101'
    assert event['leader_responders'] == ['1102']
    assert event['leader_peer_breadth'] == pytest.approx(1 / 3)


def test_top300_liquidity_tie_is_deterministic_and_0050_not_a_candidate():
    original = market()
    days = original[0].index
    symbols = [str(1000 + i) for i in range(304)]
    price = pd.DataFrame(np.repeat(original[0][['1104']].to_numpy(), 304, axis=1),
                         index=days, columns=symbols)
    price.insert(0, '0050', original[0]['0050'])
    volume = price * 0 + 2_000_000.
    amount = price * volume
    companies = pd.DataFrame({'stock_id': symbols, 'listed_date': pd.Timestamp('2000-01-01')})
    result = run((price, price.copy(), volume, amount, companies))
    audit = result['groups'][0]
    assert audit['eligible_count'] == 304
    assert audit['selected_ids'] == symbols[:300]
    assert '0050' not in audit['selected_ids']
    assert not audit['clusters']  # One giant group fails the fixed 4..20 size.
    assert len(audit['discarded_clusters'][0]['members']) == 300


@pytest.mark.parametrize('problem', ['ticker', 'volume_unit', 'turnover_unit', 'negative', 'alignment'])
def test_invalid_ticker_unit_and_alignment_are_explicit_errors(problem):
    inputs = list(market())
    if problem == 'ticker':
        inputs[0] = inputs[0].rename(columns={'1101': '1101.TW'})
    elif problem == 'volume_unit':
        inputs[2].attrs['unit'] = 'thousand_shares'
    elif problem == 'turnover_unit':
        inputs[3].attrs['unit'] = 'thousand_TWD'
    elif problem == 'negative':
        inputs[2].iloc[0, 1] = -1
    else:
        inputs[1] = inputs[1].iloc[1:]
    with pytest.raises(ValueError):
        run(tuple(inputs))
