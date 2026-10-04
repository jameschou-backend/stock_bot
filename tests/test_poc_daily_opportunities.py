from types import SimpleNamespace
import json
import threading

import pandas as pd
import pytest

from skills import poc_daily_opportunities as mod
from skills.volume_profile import build_volume_profile


def calendar():
    return pd.bdate_range('2026-01-01', periods=21).strftime('%Y-%m-%d').tolist()


def row():
    return mod.base_row(dict(signal_id='candidate', stock_id='2330', signal_date=calendar()[-1], entry_date=None), calendar())


def complete_days():
    return {'2330-'+d: dict(status='usable', prices=pd.DataFrame([
        dict(timestamp=pd.Timestamp(d)+pd.Timedelta('09:05:00'), price=100.+i, shares=1000)]),
        raw_rows=1, shares=1000, audit=dict(ordinary_volume_matched=True, ordinary_amount_matched=True))
        for i, d in enumerate(calendar()[:-1])}


def test_terminal_entry_and_account_decisions_are_irrelevant():
    event = dict(signal_id='candidate', stock_id='2330', signal_date=calendar()[-1], entry_date=None)
    original = mod.base_row(event, calendar())
    event.update(entry_date='1900-01-01', slots=0, cash=0, rank=99999, account_status='full')
    assert mod.base_row(event, calendar()) == original
    assert len(original['prior_dates']) == 20
    assert original['source_date_end'] < original['signal_date']


def test_opportunity_has_no_account_input():
    result = mod.assess(row(), complete_days())
    assert result['status'] == 'up'
    assert result['poc_after'] > result['poc_before']
    assert result['ordinary_daily_matched'] is True
    assert result['source_regular_tick_rows'] == 20


def test_bad_day_takes_precedence_over_missing_data_for_all_windows():
    days = {'2330-'+calendar()[4]: dict(status='unknown', audit={'status': 'official_ohlc_conflict'})}
    result = mod.assess(row(), days)
    assert result['status'] == 'unknown'
    assert result['reason'] == 'ordinary_tape_conflict'
    assert 'poc_before' not in result
    assert mod.assess(row(), {})['status'] == 'pending_data'


def test_equal_poc_is_known_not_up():
    days = complete_days()
    for d in days.values(): d['prices']['price'] = 100.
    result = mod.assess(row(), days)
    assert result['status'] == 'down'
    assert result['poc_before'] == result['poc_after'] == 100.


def test_future_ticks_cannot_become_opportunity():
    days = complete_days()
    days['2330-'+calendar()[0]]['prices']['timestamp'] = pd.Timestamp(calendar()[-1])
    with pytest.raises(ValueError, match='at or after signal_date'):
        mod.assess(row(), days)


def test_exact_daily_aggregation_preserves_duplicates_bins_and_halves(monkeypatch):
    original, aggregated = [], []
    for i, day in enumerate(calendar()[:-1]):
        ticks = pd.DataFrame(dict(time=pd.to_timedelta(['09:00:00', '09:00:00', '09:01:00', '13:30:00', '14:30:00']),
                                 price=[100., 100., 100.+i, 105.+i, 999.], shares=[1000, 2000, 3000, 1000, 5000]))
        monkeypatch.setattr(mod, 'normalize_ticks', lambda *a, frame=ticks: frame.copy())
        grouped, count, shares = mod.exact_price_sums(None, '2330', day, 'TWSE')
        assert count == 4 and shares == 7000
        assert grouped.shares.sum() == 7000
        original.append(ticks.iloc[:4].assign(timestamp=pd.Timestamp(day)+ticks.time.iloc[:4]))
        aggregated.append(grouped)
    kwargs = dict(signal_date=calendar()[-1], session_dates=calendar()[:-1], source_kind='authentic_regular_board_trade_ticks')
    full = build_volume_profile(pd.concat(original), **kwargs)
    small = build_volume_profile(pd.concat(aggregated), **kwargs)
    for key in ('full', 'first_half', 'second_half', 'poc_up', 'bin_edges', 'poc_shift'):
        assert full[key] == small[key]
    assert full['input_tick_count'] > small['input_tick_count']


@pytest.mark.parametrize('value', [.1, float('inf'), 2**53])
def test_aggregation_rejects_inexact_shares(monkeypatch, value):
    ticks = pd.DataFrame(dict(time=[pd.Timedelta('09:00:00')], price=[100.], shares=[value]))
    monkeypatch.setattr(mod, 'normalize_ticks', lambda *a: ticks)
    with pytest.raises(ValueError): mod.exact_price_sums(None, '2330', calendar()[0], 'TWSE')


def test_fill_does_not_fetch_window_containing_known_bad_day():
    obj = mod.DailyProfiles.__new__(mod.DailyProfiles)
    obj.calendar = calendar(); obj.signals = [dict(signal_id='candidate', stock_id='2330', signal_date=calendar()[-1])]
    obj.rows = {'candidate': dict(row(), status='pending_data')}; obj.structural = {}
    obj.days_cache = {'2330-'+calendar()[5]: dict(status='unknown', audit={'status': 'bad'})}
    obj.unavailable = {}
    obj.calls_this_run = 0
    obj.day = lambda *args, **kwargs: pytest.fail('Known invalid window must not fetch')
    obj.fill(100)
    assert obj.rows['candidate']['status'] == 'unknown'
    assert obj.calls_this_run == 0


def test_zero_budget_never_fetches():
    obj = mod.DailyProfiles.__new__(mod.DailyProfiles)
    obj.calendar = calendar(); obj.signals = [dict(signal_id='candidate', stock_id='2330', signal_date=calendar()[-1])]
    obj.rows = {'candidate': dict(row(), status='pending_data')}; obj.structural = {}
    obj.days_cache = {}; obj.calls_this_run = 0
    obj.unavailable = {}
    obj.day = lambda *args, **kwargs: pytest.fail('Zero budget must not fetch')
    obj.fill(0)
    assert obj.rows['candidate']['status'] == 'pending_data'


def test_permanent_download_failure_does_not_waste_other_nineteen_requests():
    obj = mod.DailyProfiles.__new__(mod.DailyProfiles)
    obj.calendar = calendar(); obj.signals = [dict(signal_id='candidate', stock_id='2330', signal_date=calendar()[-1])]
    obj.rows = {'candidate': dict(row(), status='pending_data')}; obj.structural = {}
    obj.days_cache = {}; obj.calls_this_run = 0
    obj.unavailable = {'2330-'+calendar()[0]: 'provider_error'}
    obj.day = lambda *a, **k: pytest.fail('Incomplete unretryable window should not consume requests')
    obj.fill(100)
    assert obj.rows['candidate']['status'] == 'pending_data'


def acquisition(tmp_path):
    obj = mod.DailyProfiles.__new__(mod.DailyProfiles)
    obj.directory = tmp_path; obj.allowed = {('2330', calendar()[0])}
    obj._config = SimpleNamespace(finmind_token='unit-test-only', finmind_requests_per_hour=6000)
    obj._lock = threading.Lock(); obj.attempt_count = obj.calls_this_run = 0
    obj.unavailable = {}
    obj._mark = lambda *a: None
    obj.attempt_paths = None
    return obj


def test_quota_resume_preserves_attempts_and_never_retries_provider_errors(tmp_path, monkeypatch):
    from app import finmind, rate_limiter
    obj = acquisition(tmp_path)
    stats = SimpleNamespace(remaining_requests=1, retry_after_seconds=0)
    monkeypatch.setattr(rate_limiter, 'get_rate_limiter', lambda *a: SimpleNamespace(get_stats=lambda: stats))
    calls = []
    def quota(*a, **k):
        calls.append(k)
        raise finmind.FinMindQuotaError(60)
    monkeypatch.setattr(finmind, 'fetch_dataset', quota)
    assert obj._fetch('2330', calendar()[0])['status'] == 'quota_paused'
    first = sorted((tmp_path/'attempts').glob('*.json'))[0].read_bytes()
    stats.remaining_requests = 0; stats.retry_after_seconds = 60
    assert obj._fetch('2330', calendar()[0]) is None
    assert len(calls) == 1
    stats.remaining_requests = 1; stats.retry_after_seconds = 0
    def failed(*a, **k): raise finmind.FinMindError('no retry')
    monkeypatch.setattr(finmind, 'fetch_dataset', failed)
    assert obj._fetch('2330', calendar()[0])['status'] == 'provider_error'
    assert obj._fetch('2330', calendar()[0]) is None
    assert obj.attempt_count == 2
    assert sorted((tmp_path/'attempts').glob('*.json'))[0].read_bytes() == first
    assert calls[0]['max_retries'] == 0 and calls[0]['requests_per_hour'] == 6000


def test_saved_receipt_cannot_disagree_with_acquisition_evidence(tmp_path):
    obj = acquisition(tmp_path); obj.raw_index = {}
    query = dict(dataset='TaiwanStockPriceTick', data_id='2330', start_date=calendar()[0])
    item = dict(query=query, status='provider_error')
    key = '2330-'+calendar()[0]
    mod.write(tmp_path/'attempts'/(key+'-0001.json'), item)
    pointer = tmp_path/'receipts'/(key+'.json'); mod.write(pointer, item)
    assert obj._stored('2330', calendar()[0]) == item
    mod.write(pointer, dict(item, status='empty'))
    with pytest.raises(ValueError, match='immutable acquisition attempt'):
        obj._stored('2330', calendar()[0])


def test_orphan_attempt_and_out_of_scope_request_are_not_retried(tmp_path, monkeypatch):
    from app import rate_limiter, finmind
    obj = acquisition(tmp_path)
    monkeypatch.setattr(rate_limiter, 'get_rate_limiter', lambda *a: SimpleNamespace(
        get_stats=lambda: SimpleNamespace(remaining_requests=10, retry_after_seconds=0)))
    monkeypatch.setattr(finmind, 'fetch_dataset', lambda *a, **k: pytest.fail('Never retry unknown crash outcome'))
    mod.write(tmp_path/'attempts'/('2330-'+calendar()[0]+'-0001.json'), dict(status='started',
        query=dict(dataset='TaiwanStockPriceTick', data_id='2330', start_date=calendar()[0])))
    assert obj._fetch('2330', calendar()[0]) is None
    with pytest.raises(ValueError, match='outside frozen'): obj._fetch('9999', calendar()[0])


def test_missing_receipt_retains_orphan_attempt_evidence(tmp_path):
    obj = acquisition(tmp_path); obj.raw_index = {}; marked = []
    obj._mark = lambda p, *a: marked.append(p)
    item = dict(status='started', query=dict(dataset='TaiwanStockPriceTick', data_id='2330', start_date=calendar()[0]))
    attempt = tmp_path/'attempts'/('2330-'+calendar()[0]+'-0001.json')
    mod.write(attempt, item)
    assert obj._stored('2330', calendar()[0]) == item
    assert attempt in marked
