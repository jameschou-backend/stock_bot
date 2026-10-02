from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import json

import pytest

from skills.market_input_validation import MarketEvidenceError
from skills.ordinary_volume_evidence import (
    OrdinaryVolumeEvidence, benchmark_split_halt, execution_capacity_report, verify_halts,
)


def fixture():
    days = [(date(2025, 1, 1)+timedelta(days=i)).isoformat() for i in range(22)]
    rows = {('TPEX', d, '5314'): dict(market='TPEX', date=d, stock_id='5314',
            volume_scope='ordinary_session', volume=1_050_000,
            open=10, high=11, low=9, close=10) for d in days}
    market_days = {('TPEX', d) for d in days}
    return days, rows, market_days


def halt(tmp_path, *, start='2025-01-03', end='2025-01-07'):
    path = tmp_path/'notice.html'
    path.write_text('reviewed official halt schedule')
    row = dict(stock_id='5314', market='TPEx', kind='trading_suspension',
               start=start, end=end, announcement_date='2024-12-01',
               source_path='notice.html', source_sha256=sha256(path.read_bytes()).hexdigest())
    refs = {}
    return verify_halts([row], tmp_path, refs)[0], refs


def test_exact_capacity_is_same_scope_and_ignores_future():
    days, rows, observed = fixture()
    rows[('TPEX', days[-1], '5314')]['volume'] = 999_999_999
    result = OrdinaryVolumeEvidence(rows, observed).capacity('TPEx', days[-2], '5314', days, cumulative_qty=10_000)
    assert result['verified_capacity'] == 10_000
    assert result['prior_ordinary_average'] == 1_050_000
    assert result['fill_check'] == 'capacity_matched'
    assert result['prior_required_dates'] == days[:20]


def test_total_volume_is_only_upper_bound_not_ordinary():
    days, rows, observed = fixture()
    rows[('TPEX', days[-2], '5314')]['volume_scope'] = 'all_daily_sessions'
    evidence = OrdinaryVolumeEvidence(rows, observed)
    row = evidence.observation('TPEX', days[-2], '5314')
    assert row['ordinary_volume'] is None
    assert row['total_volume'] == 1_050_000
    result = evidence.capacity('TPEX', days[-2], '5314', days, cumulative_qty=11_000)
    assert result['verified_capacity'] is None
    assert result['lower_capacity'] == 0
    assert result['upper_capacity'] == 10_000
    assert result['fill_check'] == 'capacity_conflict'


def test_unclassified_scope_does_not_even_certify_total_upper_bound():
    days, rows, observed = fixture()
    rows[('TPEX', days[0], '5314')]['volume_scope'] = 'unclassified_daily'
    result = OrdinaryVolumeEvidence(rows, observed).capacity('TPEX', days[-2], '5314', days, cumulative_qty=9_000)
    assert result['upper_capacity'] is None
    assert result['verified_capacity'] is None
    assert result['lower_capacity'] == 9_000
    assert result['fill_check'] == 'bounded_capacity_sufficient'
    assert len(result['gaps']) == 1


def test_missing_stock_in_full_table_is_not_zero():
    days, rows, observed = fixture()
    del rows[('TPEX', days[0], '5314')]
    result = OrdinaryVolumeEvidence(rows, observed).observation('TPEX', days[0], '5314')
    assert result['status'] == 'official_stock_row_missing'
    assert result['ordinary_volume'] is None


def test_halt_zero_activity_needs_full_day_table_and_end_is_exclusive(tmp_path):
    days, rows, observed = fixture()
    verified, refs = halt(tmp_path)
    for d in days[2:6]:
        del rows[('TPEX', d, '5314')]
    evidence = OrdinaryVolumeEvidence(rows, observed, halts=[verified])
    result = evidence.observation('TPEX', days[2], '5314')
    assert result['ordinary_volume'] == result['total_volume'] == 0
    assert result['price_observation'] is None
    assert result['executable'] is False
    assert result['status'] == 'verified_full_session_halt'
    assert evidence.observation('TPEX', days[6], '5314')['ordinary_volume'] == 1_050_000
    assert evidence.capacity('TPEX', days[20], '5314', days)['verified_capacity'] == 8_000
    assert refs['notice.html'] == verified['source_sha256']
    observed.remove(('TPEX', days[2]))
    assert OrdinaryVolumeEvidence(rows, observed, halts=[verified]).observation('TPEX', days[2], '5314')['ordinary_volume'] is None


@pytest.mark.parametrize('field,value', [('volume', 1), ('close', 10), ('open', 10)])
def test_halt_conflicting_official_positive_quote_refused(tmp_path, field, value):
    days, rows, observed = fixture()
    verified, _ = halt(tmp_path)
    row = rows[('TPEX', days[2], '5314')]
    row.update(volume=0, open=None, high=None, low=None, close=None)
    row[field] = value
    with pytest.raises(MarketEvidenceError, match='conflicts'):
        OrdinaryVolumeEvidence(rows, observed, halts=[verified]).observation('TPEX', days[2], '5314')


def test_halt_source_and_advance_notice_required(tmp_path):
    verified, _ = halt(tmp_path)
    row = deepcopy(verified)
    row['known_by'] = row['start']
    with pytest.raises(MarketEvidenceError, match='known before'):
        verify_halts([row], tmp_path, {})
    (tmp_path/'notice.html').write_text('changed')
    with pytest.raises(MarketEvidenceError, match='hash mismatch'):
        verify_halts([verified], tmp_path, {})


def test_halt_rejects_overlaps(tmp_path):
    verified, _ = halt(tmp_path)
    with pytest.raises(MarketEvidenceError, match='Overlapping'):
        verify_halts([verified, verified], tmp_path, {})


def test_halt_cannot_backdate_known_by_before_publication(tmp_path):
    verified, _ = halt(tmp_path)
    verified['known_by'] = '2024-11-30'
    with pytest.raises(MarketEvidenceError, match='known before'):
        verify_halts([verified], tmp_path, {})


def test_exact_matrix_preserves_gaps_and_inputs():
    days, rows, observed = fixture()
    before = deepcopy(rows)
    del rows[('TPEX', days[0], '5314')]
    rows[('TPEX', days[1], '5314')]['volume_scope'] = 'all_daily_sessions'
    evidence = OrdinaryVolumeEvidence(rows, observed)
    frame = evidence.matrix('TPEX', days, ['5314'])
    assert frame.iloc[:2].isna().all().all()
    assert frame.iloc[2, 0] == 1_050_000
    assert before[('TPEX', days[2], '5314')] == rows[('TPEX', days[2], '5314')]


def test_same_day_cumulative_capacity_and_gap_deduplication():
    days, rows, observed = fixture()
    del rows[('TPEX', days[0], '5314')]
    trades = [dict(channel='board', date=days[20], stock_id='5314', qty=9_000,
                   sequence=1, capacity_qty=10_000),
              dict(channel='board', date=days[20], stock_id='5314', qty=2_000,
                   sequence=2, capacity_qty=10_000),
              dict(channel='odd', date=days[20], stock_id='5314', qty=20)]
    result = execution_capacity_report(trades, days, OrdinaryVolumeEvidence(rows, observed), lambda t: 'TPEX')
    assert result['required_stock_days'] == 21
    assert len(result['missing_stock_days']) == 1
    assert result['rows'][1]['cumulative_qty'] == 11_000
    assert result['rows'][0]['fill_check'] == 'bounded_capacity_sufficient'
    assert result['all_capacity_verified'] is False


@pytest.mark.parametrize('field,value', [('volume', -1), ('volume', 1.5), ('volume', True), ('stock_id', '9999')])
def test_invalid_quantity_or_wrong_row_identity_fails(field, value):
    days, rows, observed = fixture()
    rows[('TPEX', days[0], '5314')][field] = value
    with pytest.raises(MarketEvidenceError):
        OrdinaryVolumeEvidence(rows, observed).observation('TPEX', days[0], '5314')


def test_short_or_duplicate_calendar_cannot_certify_capacity():
    days, rows, observed = fixture()
    evidence = OrdinaryVolumeEvidence(rows, observed)
    assert evidence.capacity('TPEX', days[1], '5314', days)['status'] == 'calendar_warmup_incomplete'
    with pytest.raises(MarketEvidenceError, match='calendar'):
        evidence.capacity('TPEX', days[-1], '5314', days+[days[-1]])
    with pytest.raises(MarketEvidenceError, match='Matrix axes'):
        evidence.matrix('TPEX', days, ['5314', '5314'])


def test_benchmark_adapter_uses_advance_schedule_and_does_not_make_prices(tmp_path):
    pdf = tmp_path/'schedule.pdf'
    pdf.write_bytes(b'reviewed official notice')
    metadata = dict(stock_id='0050', suspension_known_before_start_verified=True,
        published_at='2025-06-17T11:35:21+08:00',
        verified_terms=dict(last_trading_date_before_split='2025-06-10',
            suspension_start='2025-06-11', suspension_end='2025-06-17', new_units_listing_date='2025-06-18'),
        schedule_source=dict(url='https://www.twse.com.tw/notice.pdf', published_on='2025-05-14',
            conservative_known_by='2025-05-15', local_path='schedule.pdf', sha256=sha256(pdf.read_bytes()).hexdigest()))
    path = tmp_path/'reviewed.json'
    path.write_text(json.dumps(metadata))
    expected = sha256(path.read_bytes()).hexdigest()
    refs = {}
    with pytest.raises(MarketEvidenceError, match='must be pinned'):
        benchmark_split_halt(path, tmp_path, refs)
    halt_row = benchmark_split_halt(path, tmp_path, refs, expected_metadata_sha256=expected)
    assert halt_row['known_by'] == '2025-05-15'
    assert halt_row['end'] == '2025-06-18'
    assert set(refs) == {'reviewed.json', 'schedule.pdf'}
    evidence = OrdinaryVolumeEvidence({}, {('TWSE', '2025-06-11')}, halts=[halt_row])
    assert evidence.observation('TWSE', '2025-06-11', '0050')['price_observation'] is None
    metadata['schedule_source']['conservative_known_by'] = '2025-06-17'
    path.write_text(json.dumps(metadata))
    with pytest.raises(MarketEvidenceError, match='hash mismatch'):
        benchmark_split_halt(path, tmp_path, refs)
    with pytest.raises(MarketEvidenceError, match='schedule'):
        benchmark_split_halt(path, tmp_path, {},
                             expected_metadata_sha256=sha256(path.read_bytes()).hexdigest())
