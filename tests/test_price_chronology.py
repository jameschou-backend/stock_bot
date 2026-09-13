import pandas as pd
import pytest
from scripts.audit_price_chronology import normalize_source, compare_snapshot, FIELDS


def test_wrong_date_and_duplicate_source_are_rejected():
    frame = pd.DataFrame([dict(date='2026-06-09', stock_id='2456', open=10, high=11, low=9, close=10, volume=20)])
    with pytest.raises(ValueError, match='wrong source date'):
        normalize_source(frame, '2026-06-11')
    with pytest.raises(ValueError, match='duplicate'):
        normalize_source(pd.concat([frame,frame]), '2026-06-09')


def test_absence_is_separate_from_identical_old_quote():
    current = pd.DataFrame([[10,11,9,10,100],[20,21,19,20,200],[30,31,29,30,300]],
                           index=['1101','2456','1589'], columns=FIELDS)
    source = current.loc[['1101']]
    historical = current.copy(); historical.loc['1589','volume'] = 400
    result = compare_snapshot(current, source, historical)
    assert result['absent_rows'] == ['1589','2456']
    assert result['exact_older_ohlcv_matches'] == ['2456']
    assert result['different_rows'] == []


def test_changed_volume_is_not_rounded_away():
    current = pd.DataFrame([[10,11,9,10,10000000]], index=['1101'], columns=FIELDS)
    source = current.copy(); source.loc['1101','volume'] += 1
    assert compare_snapshot(current, source, current)['different_rows'] == ['1101']
