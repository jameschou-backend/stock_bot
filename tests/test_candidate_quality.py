from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from skills.candidate_quality import candidate_features, generate_candidates, failed_base, CandidateQuality


def frames():
    days = pd.bdate_range('2018-01-01', periods=220)
    c = pd.DataFrame({'0050': np.linspace(100, 160, len(days)),
                      '1234': np.linspace(100, 200, len(days))}, index=days)
    volume = c*0+2_000_000
    values = {'close-official': c, 'close-quality': c.copy(), 'raw-close': c.copy(),
              'raw-volume': volume, 'eligibility': c.notna()}
    companies = pd.DataFrame([dict(stock_id='1234', listed_date=pd.Timestamp('2000-01-01'))])
    return values, companies


def event(days, i=190):
    return dict(event_id='first', signal_date=str(days[i].date()), entry_date=str(days[i+1].date()),
                members=['1234'], priority=.1, liquidity_before_entry={'as_of': str(days[i].date())})


def test_queue_is_five_sessions_with_fresh_dates_and_no_input_mutation():
    values, companies = frames()
    days = values['close-official'].index
    e = event(days)
    before = deepcopy(e)
    rows = generate_candidates(values, companies, [e], str(days[-2].date()))['queue']
    assert len(rows) == 5
    assert e == before
    assert rows[0] == e
    for offset, row in enumerate(rows):
        assert row['signal_date'] == str(days[190+offset].date())
        assert row['entry_date'] == str(days[191+offset].date())
        if offset:
            assert row['origin_event_id'] == 'first'
            assert 'liquidity_before_entry' not in row


def test_future_data_cannot_change_prior_queue_or_filters():
    values, companies = frames()
    days = values['close-official'].index
    e = event(days)
    cutoff = str(days[192].date())
    full = generate_candidates(values, companies, [e], str(days[-2].date()))
    truncated = generate_candidates({k: v.iloc[:194].copy() for k, v in values.items()}, companies, [e], cutoff)
    for arm in full:
        assert [r for r in full[arm] if r['signal_date'] <= cutoff] == truncated[arm]


def test_failed_revalidation_does_not_renew_stale_signal():
    values, companies = frames()
    days = values['close-official'].index
    values['eligibility'].loc[days[191:195], '1234'] = False
    result = generate_candidates(values, companies, [event(days)], str(days[-2].date()))
    assert len(result['queue']) == 1


def test_contraction_excludes_breakout_day():
    values, companies = frames()
    initial = candidate_features(values, companies)
    for name in ('close-official', 'close-quality', 'raw-close'):
        values[name] = values[name].copy()
        values[name].iloc[190, 1] *= 3
    changed = candidate_features(values, companies)
    assert changed['contraction'].iloc[190, 1] == initial['contraction'].iloc[190, 1]
    assert not changed['not_extended'].iloc[190, 1]


def test_queue_rejects_non_next_session_original():
    values, companies = frames()
    days = values['close-official'].index
    e = event(days); e['entry_date'] = str(days[192].date())
    with pytest.raises(ValueError, match='one session'):
        generate_candidates(values, companies, [e], str(days[-2].date()))


@pytest.mark.parametrize('age,ret,relative,below,expected', [
    (20, .02, -.01, True, True), (19, .02, -.01, True, False),
    (20, .02, -.01, False, False), (20, .03, -.01, True, False),
    (20, .02, 0, True, False), (20, float('nan'), -.01, True, False)])
def test_failed_base_requires_stagnation_and_structural_failure(age, ret, relative, below, expected):
    assert failed_base(age, ret, relative, below) == expected


def test_original_exit_is_never_overridden_by_failed_base():
    class Base:
        def corporate_day(self, day):
            self.exit_states['e']['trigger_reason'] = 'loss12'
            return 7
    class Replay(CandidateQuality, Base):
        pass
    r = object.__new__(Replay)
    day = pd.Timestamp('2024-02-02')
    r.candidate_arm = 'failed_base'; r.positions = {day: 1}; r.days = pd.to_datetime(['2024-02-01', day])
    r.holdings = {'1234': {'event_id': 'e', 'qty': 1000}}
    r.exit_states = {'e': {'trigger_reason': None}}
    assert r.corporate_day(day) == 7
    assert r.exit_states['e']['trigger_reason'] == 'loss12'
