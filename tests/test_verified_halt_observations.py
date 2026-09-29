import hashlib
import pandas as pd
import pytest
from skills.verified_halt_observations import add_verified_halt_observations


def case(tmp_path):
    p = tmp_path/'official.html'; p.write_text('verified full-session announcement fixture')
    halt = dict(stock_id='1234', market='TPEx', kind='trading_suspension', start='2025-03-20',
                end='2025-03-24', announcement_date='2025-03-12', source_path=p.name,
                source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    days = pd.bdate_range('2025-03-19', '2025-03-25')
    q = pd.DataFrame([dict(stock_id='1234', date=days[0], open=10., high=10.,
                           low=10., close=10., volume=100.)])
    return q, days, halt


def test_only_verified_halt_is_zero_and_unknown_gap_stays_missing(tmp_path):
    q, days, h = case(tmp_path)
    out, evidence = add_verified_halt_observations(q, days, [h], tmp_path)
    assert len(out) == 3 and len(evidence) == 2
    assert out.iloc[1:][['open','high','low','close','volume']].eq(0).all().all()
    assert not out.date.eq(pd.Timestamp(h['end'])).any()
    again, ev = add_verified_halt_observations(out, days, [h], tmp_path)
    pd.testing.assert_frame_equal(out, again); assert not ev


@pytest.mark.parametrize('failure', ['late', 'hash', 'conflict', 'market'])
def test_missing_proof_and_conflicting_trade_never_fill(tmp_path, failure):
    q, days, h = case(tmp_path)
    if failure == 'late': h['announcement_date'] = h['start']
    if failure == 'hash': h['source_sha256'] = 'invalid'
    if failure == 'conflict': q.loc[0, 'date'] = pd.Timestamp(h['start'])
    if failure == 'market': h.pop('market')
    with pytest.raises(ValueError): add_verified_halt_observations(q, days, [h], tmp_path)
