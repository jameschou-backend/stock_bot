import pandas as pd
import pytest

from skills.leader_chip_replay import features, choose, LeaderChipReplay, audit_order
from skills.residual_slot_replay import ResidualSlotReplay
from test_residual_slot_replay import six_stocks


def sources():
    days = pd.bdate_range('2021-01-01', '2022-04-01')
    entries = [dict(event_id='a', members=['1101'], signal_date='2022-03-14')]
    flows = pd.DataFrame(dict(date=days, stock_id='1101', foreign=10., trust=5.))
    weeks = pd.date_range('2021-01-01', '2022-04-01', freq='W-FRI')
    weekly = pd.DataFrame([dict(date=d, stock_id='1101', valid=True, total_units=1e6,
        large_pct=.1+i*.003, small_pct=.8-i*.003, large_units=1e5+i*3000)
        for i, d in enumerate(weeks)])
    return entries, days, flows, weekly


def test_signal_day_and_week_availability_are_not_leaked():
    e, days, f, w = sources()
    row = features(e, days, f, w, 'main')['a']
    assert row['passed'] is True
    assert row['foreign5'] == 50 and row['trust5'] == 25
    assert row['flow_end'] == '2022-03-11'
    assert row['observed_date'] == '2022-03-04' and row['available_date'] == '2022-03-12'
    f.loc[f.date >= '2022-03-14', ['foreign', 'trust']] = -1e9
    w.loc[w.date >= '2022-03-11', 'large_pct'] = 1e9
    assert features(e, days, f, w, 'main')['a'] == row
    delayed = features(e, days, f, w, 'delayed')['a']
    assert delayed['flow_end'] == '2022-03-09'
    assert delayed['observed_date'] == '2022-02-25'


def test_missing_session_and_bad_latest_week_stay_unknown():
    e, days, f, w = sources()
    f = f[f.date != '2022-03-10']
    row = features(e, days, f, w, 'main')['a']
    assert row['known'] is False and row['passed'] is None and row['foreign5'] is None
    w.loc[w.date == '2022-03-04', 'valid'] = False
    row = features(e, days, sources()[2], w, 'main')['a']
    assert row['observed_date'] == '2022-03-04' and row['chip_known'] is False
    row = features(e, days, sources()[2], w[w.date < '2022-02-18'], 'main')['a']
    assert row['chip_known'] is False and row['chip_reason'] == 'stale_or_unavailable'


def test_rank_preserves_all_candidates_and_coverage_is_distinct():
    events = [dict(event_id=s) for s in 'abcde']
    scores = {s:dict(known=v is not None, passed=v) for s, v in zip('abcde', [False,True,None,True,False])}
    expected = {'rank':'bdace', 'filter':'bd', 'coverage':'abde', 'baseline':'abcde'}
    for arm, order in expected.items():
        selected = [e['event_id'] for e in choose(events, scores, arm)]
        assert selected == list(order)
        journal = [dict(original=list('abcde'), selected=selected)]
        assert audit_order(journal, scores, arm)['passed']
        journal[0]['selected'].reverse()
        with pytest.raises(ValueError, match='ordering'):
            audit_order(journal, scores, arm)


@pytest.mark.parametrize('arm', ['baseline', 'rank', 'filter', 'coverage'])
@pytest.mark.parametrize('mask', [0, 7])
def test_neutral_chip_layer_reproduces_full_account(arm, mask):
    _, args, kwargs = six_stocks()
    kwargs['factor_mask'] = mask
    scores = {e['event_id']:dict(stock_id=e['members'][0], signal_date=e['signal_date'],
        flow_end=None, available_date=None, known=True, passed=True) for e in args[3]}
    expected = ResidualSlotReplay(*args, **kwargs, residual_policy='release').run()
    replay = LeaderChipReplay(*args, **kwargs, residual_policy='release', chip_arm=arm, chip_scores=scores)
    assert replay.run() == expected
    audit_order(replay.chip_orders, scores, arm)


def test_changing_priority_changes_actual_cash_limited_fills():
    _, args, kwargs = six_stocks(next_offset=0)
    scores = {e['event_id']:dict(stock_id=e['members'][0], signal_date=e['signal_date'],
        flow_end=None, available_date=None, known=True, passed=e['members'][0]=='1106') for e in args[3]}
    control = LeaderChipReplay(*args, **kwargs, residual_policy='release', chip_arm='baseline', chip_scores=scores).run()
    replay = LeaderChipReplay(*args, **kwargs, residual_policy='release', chip_arm='rank', chip_scores=scores)
    result = replay.run()
    assert not any(t['stock_id']=='1106' and t['side']=='buy' for t in control['trades'])
    assert any(t['stock_id']=='1106' and t['side']=='buy' for t in result['trades'])
    assert len([t for t in result['trades'] if t['side']=='buy']) == 5
    audit_order(replay.chip_orders, scores, 'rank')


def test_identity_or_future_score_is_rejected():
    _, args, kwargs = six_stocks()
    scores = {e['event_id']:dict(stock_id=e['members'][0], signal_date=e['signal_date'],
        flow_end=e['signal_date'], available_date=None, known=True, passed=True) for e in args[3]}
    with pytest.raises(ValueError, match='availability'):
        LeaderChipReplay(*args, **kwargs, residual_policy='release', chip_arm='rank', chip_scores=scores)


@pytest.mark.parametrize('arm,expected_count', [('rank',5),('filter',1),('coverage',5)])
def test_unknown_candidates_are_not_treated_as_negative_or_passed(arm, expected_count):
    _, args, kwargs = six_stocks(next_offset=0)
    scores = {e['event_id']:dict(stock_id=e['members'][0], signal_date=e['signal_date'],
        flow_end=None, available_date=None, known=e['members'][0]!='1101',
        passed=None if e['members'][0]=='1101' else e['members'][0]=='1106') for e in args[3]}
    replay=LeaderChipReplay(*args, **kwargs, residual_policy='release', chip_arm=arm, chip_scores=scores)
    account=replay.run()
    assert sum(t['side']=='buy' for t in account['trades']) == expected_count
    assert any(t['stock_id']=='1106' and t['side']=='buy' for t in account['trades'])
    if arm in ('filter','coverage'):
        assert not any(t['stock_id']=='1101' for t in account['trades'])
    audit_order(replay.chip_orders,scores,arm)
