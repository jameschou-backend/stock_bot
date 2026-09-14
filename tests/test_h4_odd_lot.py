import pytest
from scripts.audit_h4_odd_lot import parse_h4


def record(version='legacy190',remark=' ',match='Y',day='20220104'):
    new=version=='20260401_201'
    raw=list(' '*(201 if new else 190))
    raw[:6]=list('8261  ');raw[6:18]=list('091003123456')
    raw[18]=remark;raw[20]=match
    end=29 if new else 28;ds=191 if new else 180
    raw[22:end]=list('0012300' if new else '012300')
    raw[end:end+8]=list('00000694');raw[ds:ds+8]=list(day)
    return ''.join(raw).encode()


def test_actual_trial_and_nonmatch_are_distinct():
    rows=parse_h4(b'\r\n'.join([record(),record(remark='T'),record(match=' ')]),'legacy190')
    assert [r['actual_match'] for r in rows]==[True,False,False]
    assert rows[1]['trial']
    assert rows[0]['raw_volume']==694  # Raw units, deliberately not called shares.
    assert rows[0]['time']=='2022-01-04T09:10:03.123456'


def test_layout_is_explicit_and_does_not_silently_shift_columns():
    with pytest.raises(ValueError,match='201 bytes'):parse_h4(record(),'20260401_201')
    with pytest.raises(ValueError,match='predates'):parse_h4(record('20260401_201'),'20260401_201')
    with pytest.raises(ValueError,match='boundary'):parse_h4(record(day='20260402'),'legacy190')
    row=parse_h4(record('20260401_201',day='20260402'),'20260401_201')[0]
    assert row['raw_price']==12300


def test_truncated_unknown_flags_and_invalid_clock_fail():
    with pytest.raises(ValueError):parse_h4(record()[:-1],'legacy190')
    with pytest.raises(ValueError):parse_h4(record(match='X'),'legacy190')
    with pytest.raises(ValueError):parse_h4(record().replace(b'091003',b'251003'),'legacy190')


def test_concatenated_fixed_width_records():
    assert len(parse_h4(record()+record(),'legacy190'))==2
