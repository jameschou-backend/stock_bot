"""Synthetic checks only: no provider access or portfolio returns."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import prepare_million_signals as module


def splice_inputs():
    days = pd.bdate_range('2026-05-20', '2026-06-29')
    old_days = days[(days <= '2026-06-23') & ~days.isin(pd.to_datetime(module.FRESH_EXTRA_DATES))]
    ids = ['0050', '1101', '1102']
    raw = pd.DataFrame(100., index=days, columns=ids)
    old = pd.DataFrame(40., index=old_days, columns=ids)
    fresh = pd.DataFrame(80., index=days, columns=ids)
    fresh.loc['2026-06-24':] = 84.
    return old, fresh, raw


def test_explicit_anchor_rebase_tail_and_declared_calendar_holes():
    old, fresh, raw = splice_inputs()
    stitched, audit = module.stitch_snapshot(old, fresh, raw)
    assert stitched.loc['2026-05-20', '1101'] == 80
    assert stitched.loc['2026-05-21', '1101'] == 80
    assert stitched.loc['2026-06-24', '1101'] == 84
    assert audit['assets'][1]['anchor_date'] == '2026-06-23'
    assert audit['assets'][1]['scale'] == 2
    assert audit['inserted_market_dates'] == ['2026-05-21', '2026-05-22']
    assert audit['historical_revalidation'] is False
    assert audit['no_anchor_assets'] == []


def test_unanchored_asset_keeps_old_and_rejects_all_inserted_prices():
    old, fresh, raw = splice_inputs()
    fresh.loc['2026-06-01':'2026-06-23', '1101'] = np.nan
    stitched, audit = module.stitch_snapshot(old, fresh, raw)
    assert stitched.loc['2026-05-20', '1101'] == 40
    assert pd.isna(stitched.loc['2026-05-21', '1101'])
    assert stitched.loc['2026-06-24':, '1101'].isna().all()
    assert audit['no_anchor_assets'] == ['1101']


def test_original_missing_stock_quote_never_filled_and_raw_missing_blocks_tail():
    old, fresh, raw = splice_inputs()
    old.loc['2026-06-10', '1101'] = np.nan
    raw.loc['2026-06-24', '1101'] = np.nan
    raw.loc['2026-06-25', '1101'] = 0.
    stitched, _ = module.stitch_snapshot(old, fresh, raw)
    assert stitched.loc[['2026-06-10', '2026-06-24', '2026-06-25'], '1101'].isna().all()


def test_last_common_valid_anchor_and_unchanged_source_frames():
    old, fresh, raw = splice_inputs()
    raw.loc['2026-06-23', '1101'] = np.nan
    fresh.loc['2026-06-22', '1101'] = 60.
    before = [frame.copy(deep=True) for frame in (old, fresh, raw)]
    stitched, audit = module.stitch_snapshot(old, fresh, raw)
    assert audit['assets'][1]['anchor_date'] == '2026-06-22'
    assert audit['assets'][1]['scale'] == 1.5
    assert stitched.loc['2026-05-20', '1101'] == 60
    for got, expected in zip((old, fresh, raw), before):
        pd.testing.assert_frame_equal(got, expected)


def test_future_tail_mutations_do_not_change_old_rebased_prefix():
    old, fresh, raw = splice_inputs()
    before, _ = module.stitch_snapshot(old, fresh, raw)
    fresh.loc['2026-06-25':] *= 50
    after, _ = module.stitch_snapshot(old, fresh, raw)
    pd.testing.assert_frame_equal(before.loc[:'2026-06-24'], after.loc[:'2026-06-24'])


def test_unapproved_new_historical_market_day_is_not_silently_filled():
    old, fresh, raw = splice_inputs()
    with pytest.raises(ValueError, match='Unapproved old-calendar holes'):
        module.stitch_snapshot(old.drop(pd.Timestamp('2026-06-02')), fresh, raw)


def test_fresh_alignment_must_be_exact():
    old, fresh, raw = splice_inputs()
    with pytest.raises(ValueError, match='exactly aligned'):
        module.stitch_snapshot(old, fresh[fresh.columns[::-1]], raw)


def test_fetch_validation_rejects_wrong_dates_and_duplicates():
    frame = pd.DataFrame({'stock_id': ['1101'], 'date': ['2026-06-01'], 'close': [100.]})
    with pytest.raises(ValueError, match='another date'):
        module._validate_fresh_rows(frame, '2026-06-02')
    with pytest.raises(ValueError, match='Duplicate'):
        module._validate_fresh_rows(pd.concat([frame, frame]), '2026-06-01')


def test_fetch_checkpoint_resume_has_no_network_calls(tmp_path, monkeypatch):
    days = ['2026-06-01', '2026-06-02']
    monkeypatch.setattr(module, 'fetch_dates', lambda: days)
    calls = []
    def fetcher(dataset, start, end, **kwargs):
        calls.append((dataset, start, end, kwargs))
        return pd.DataFrame({'stock_id': ['1101'], 'date': [str(start)], 'close': [100.]})
    first = module.fetch_adjusted(out_dir=tmp_path, workers=2, fetcher=fetcher)
    second = module.fetch_adjusted(out_dir=tmp_path, workers=2, fetcher=fetcher)
    assert first == second
    assert len(calls) == 2
    assert all(call[3]['requests_per_hour'] == 5400 and call[3]['max_retries'] == 0 for call in calls)
    assert first['calls_reserved'] == 2


def test_fetch_error_stops_new_batches_and_keeps_success(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'fetch_dates', lambda: ['2026-06-01', '2026-06-02', '2026-06-03'])
    calls = []
    def fetcher(dataset, start, end, **kwargs):
        calls.append(str(start))
        if str(start) == '2026-06-02':
            raise RuntimeError('quota paused')
        return pd.DataFrame({'stock_id': ['1101'], 'date': [str(start)], 'close': [100.]})
    with pytest.raises(RuntimeError, match='quota paused'):
        module.fetch_adjusted(out_dir=tmp_path, workers=2, fetcher=fetcher)
    assert set(calls) == {'2026-06-01', '2026-06-02'}
    assert (tmp_path / '2026-06-01.json').exists()
    assert not (tmp_path / 'manifest.json').exists()


def test_fixed_fetch_plan_within_budget_and_contains_both_missing_days():
    assert len(module.fetch_dates()) <= 100
    assert set(module.FRESH_EXTRA_DATES).issubset(module.fetch_dates())
    assert module.fetch_dates()[-1] == '2026-09-09'


def signal_inputs():
    days = pd.bdate_range('2021-06-01', '2022-01-31')
    ids = ['0050', '1101', '1102', '1103', '1104']
    steps = np.arange(len(days))
    benchmark = 100 * np.exp(np.cumsum(.0003 + .0001 * np.cos(steps / 4)))
    peer = 100 * np.exp(np.cumsum(.0004 + .004 * np.sin(steps / 3)))
    close = pd.DataFrame(np.column_stack([benchmark, peer, peer, peer, peer]), index=days, columns=ids)
    i = days.get_loc('2022-01-05')
    anchor = float(close.iloc[i - 11]['1101'])
    close.iloc[i - 10:, 1:] = np.array([anchor * .998 ** k for k in range(1, len(days) - i + 11)])[:, None]
    close.loc[days[i]:, '1101'] = max(float(close.iloc[:i]['1101'].max()) * 1.08, anchor * 1.10)
    volume = pd.DataFrame(2_000_000., index=days, columns=ids)
    volume.loc[days[i], '1101'] = 4_000_000.
    companies = pd.DataFrame({'stock_id': ids[1:], 'listed_date': pd.Timestamp('2000-01-01')})
    return close, close.copy(), close.copy(), volume, companies


def run_signals(inputs):
    return module.build_signals(*inputs, start='2022-01-03', signal_end='2022-01-05')


def test_reuses_original_event_group_priority_and_next_day_with_past_liquidity():
    result = run_signals(signal_inputs())
    assert len(result['entries']) == 1
    row = result['entries'][0]
    old = result['diffusion']['entries']['leader_now'][0]
    assert row['event_id'] == old['event_id']
    assert row['members'] == ['1101']
    assert row['priority'] == old['priority']
    assert row['signal_date'] == '2022-01-05'
    assert row['entry_date'] == '2022-01-06'
    assert row['group_members'] == ['1101', '1102', '1103', '1104']
    assert row['trend_state'] == 'ON'
    assert row['liquidity_before_entry'] == row['liquidity_at_signal']
    assert row['liquidity_before_entry']['adv20_shares'] == 2_100_000.
    json.dumps(result, allow_nan=False)


def test_execution_day_volume_and_future_prices_cannot_change_orders():
    inputs = signal_inputs()
    original = run_signals(inputs)['entries']
    inputs[3].loc['2022-01-06':] *= 200
    for frame in inputs[:3]:
        frame.loc['2022-01-06':] *= 10
    changed = run_signals(inputs)['entries']
    assert changed == original


def test_missing_signal_benchmark_is_not_treated_as_on():
    inputs = signal_inputs()
    # Test the gate independently of the technical screen: a missing quote
    # must not inherit yesterday's bullish state.
    rows = run_signals(inputs)['diffusion']['entries']['leader_now']
    inputs[0].loc['2022-01-05', '0050'] = np.nan
    trend = module.build_trend(inputs[0]['0050'])
    accepted, rejected = module.gate_events(rows, trend)
    assert accepted == []
    assert rejected[0]['reason'] == 'trend_unknown'


def test_short_or_missing_liquidity_window_is_explicit():
    _, _, raw, volume, _ = signal_inputs()
    assert module._liquidity(volume, raw, volume.index[2], '1101')['adv20_shares'] is None
    volume.loc['2022-01-04', '1101'] = np.nan
    value = module._liquidity(volume, raw, '2022-01-05', '1101')
    assert value['observations'] == 19
    assert value['complete_20_sessions'] is False
    assert value['mean_turnover20_twd'] is None


def test_benchmark_coverage_cannot_be_hidden_by_ordinary_stock_coverage():
    days = pd.bdate_range('2025-06-09', '2025-06-20')
    close = pd.DataFrame({'0050': 100., '1101': 50.}, index=days)
    volume = close * 100
    known = pd.to_datetime(['2025-06-11', '2025-06-12', '2025-06-13', '2025-06-16', '2025-06-17'])
    close.loc[known, '0050'] = np.nan
    check = module.validate_benchmark_quotes(close, volume)
    assert check['unexpected_missing_dates'] == []
    close.loc['2025-06-18', '0050'] = np.nan
    with pytest.raises(ValueError, match='Unresolved 0050'):
        module.validate_benchmark_quotes(close, volume)


def test_benchmark_zero_volume_is_not_a_valid_quote():
    days = pd.bdate_range('2022-01-03', periods=3)
    close = pd.DataFrame({'0050': 100.}, index=days)
    volume = close.copy()
    volume.iloc[1, 0] = 0
    with pytest.raises(ValueError, match='2022-01-04'):
        module.validate_benchmark_quotes(close, volume)


def test_official_reference_split_factor_does_not_double_apply():
    days = pd.to_datetime(['2025-06-10', '2025-06-18', '2025-06-19'])
    raw = pd.DataFrame({'0050': [200., 50., 51.], '1101': [100., 98., 99.]}, index=days)
    actions = pd.DataFrame({'stock_id': ['1101'], 'event_date': ['2025-06-18'], 'ratio': [.98]})
    result = module.official_adjusted(raw, actions)
    assert result['0050'].tolist() == [50., 50., 51.]
    assert result['1101'].tolist() == [98., 98., 99.]
    actions.loc[0, 'stock_id'] = '0050'
    with pytest.raises(ValueError, match='double adjustment'):
        module.official_adjusted(raw, actions)


def test_sealed_adjusted_checkpoint_mutation_is_rejected_without_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'fetch_dates', lambda: ['2026-06-01'])
    def fetcher(dataset, start, end, **kwargs):
        return pd.DataFrame({'stock_id': ['1101'], 'date': [str(start)], 'close': [100.]})
    module.fetch_adjusted(out_dir=tmp_path, workers=1, fetcher=fetcher)
    path = tmp_path / '2026-06-01.json'
    value = json.loads(path.read_text())
    value['data'][0]['close'] = 200.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Frozen adjusted checkpoint changed'):
        module.fetch_adjusted(out_dir=tmp_path, workers=1, fetcher=lambda *a, **k: pytest.fail('must not fetch'))


def test_missing_adjusted_market_day_is_not_silently_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'fetch_dates', lambda: ['2026-06-01'])
    module.fetch_adjusted(out_dir=tmp_path, workers=1, fetcher=lambda *a, **k: pd.DataFrame())
    raw = pd.DataFrame({'0050': [100.]}, index=pd.to_datetime(['2026-06-01']))
    with pytest.raises(ValueError, match='missing on open sessions'):
        module.load_fresh(raw, out_dir=tmp_path)


def test_historical_comparison_matches_economic_group_not_number():
    old = {'events': [{'event_id': 'old', 'leader_date': '2022-01-05', 'leader_id': '1101',
                       'members': ['1101', '1102'], 'priority': .1}],
           'groups': [{'month': '2022-01', 'clusters': [{'group_id': 'a', 'members': ['1101', '1102']}]}]}
    new = json.loads(json.dumps(old))
    new['events'][0]['event_id'] = 'new-group-number'
    new['groups'][0]['clusters'][0]['group_id'] = 'b'
    comparison = module.historical_comparison(new, old)
    assert comparison['common_leaders_by_date_stock_and_group'] == 1
    assert comparison['changed_group_months'] == []
    assert comparison['old_only_event_ids'] == []
