from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skills.poc_weakness import diagnose, indicators
from app.poc_weakness_ui import validate


def fixture_data():
    days = pd.bdate_range('2024-01-01', periods=45)
    prices = [110.] * len(days)
    prices[24:34] = [100, 120, 118, 116, 119, 121, 115, 112, 111, 110]
    close = pd.DataFrame({'1111': prices}, index=days)
    quotes = pd.DataFrame(dict(date=days, stock_id='1111', close=prices,
        open=np.array(prices)-1, high=np.array(prices)+2, low=np.array(prices)-2, volume=1000))
    quotes.loc[[26, 27, 30, 31], 'open'] = quotes.loc[[26, 27, 30, 31], 'close']+1
    date = lambda i:str(days[i].date())
    account = dict(cohorts=[dict(event_id='A', stock_id='1111', name='Test', entry_date=date(24))],
        orders=[dict(event_id='A', side='sell', date=date(32), signal_date=date(31), reason='three_black')],
        # The first fill is delayed. Early-warning comparison must not use this date.
        trades=[dict(event_id='A', side='sell', date=date(33))],
        daily=[dict(date=date(i), nav=100000) for i in range(24,45)],
        holdings=[dict(event_id='A', date=date(i), qty=100, market_value=prices[i]*100,
                       stale=False, mark_date=date(i)) for i in range(24,33)])
    episode = dict(rank=1, peak_date=date(29), trough_date=date(32), drawdown=-.1, recovery_date=None,
        cohorts=[dict(event_id='A', group='existing_winner', peak_weight=.12, gross_pnl=-100)])
    return account, close, quotes, [episode]


def test_first_warning_is_not_reset_at_peak_and_lead_uses_exit_decision_not_fill():
    account, close, quotes, episodes = fixture_data()
    r = diagnose(account, close, quotes, episodes)
    w = r['cohorts'][0]['warnings']['two_black']
    assert w['date'] == str(close.index[27].date())
    assert w['next_session'] == str(close.index[28].date())
    assert w['lead_sessions'] == 4
    assert w['later_new_holding_high'] is True
    assert w['subsequent_max_return'] == pytest.approx(121/116-1)
    assert w['signal_count'] == 1  # The second pair is on the original decision date.
    assert r['episodes'][0]['rows'][0]['warnings']['two_black']['location'] == 'before_peak'


def test_future_changes_do_not_change_prior_signal_or_first_warning_information():
    a, c, q, e = fixture_data()
    before = indicators(c, q)[0]
    altered = c.copy(); altered.iloc[28:] *= 2
    after = indicators(altered, q)[0]
    for key in before:
        pd.testing.assert_frame_equal(before[key].iloc[:28], after[key].iloc[:28])
    one = diagnose(a, c, q, e)['cohorts'][0]['warnings']['two_black']
    two = diagnose(a, altered, q, e)['cohorts'][0]['warnings']['two_black']
    for key in ('date', 'next_session', 'lead_sessions', 'close_return_from_entry',
                'known_peak_drawdown', 'adjusted_close', 'ma20', 'weight'):
        assert one[key] == two[key]
    assert one['subsequent_max_return'] != two['subsequent_max_return']


def test_no_warning_without_current_profit_or_before_two_post_entry_closes():
    a, c, q, e = fixture_data()
    a['cohorts'][0]['entry_date'] = str(c.index[27].date())
    r = diagnose(a, c, q, e)
    assert 'two_black' not in r['cohorts'][0]['warnings']
    a, c, q, e = fixture_data(); c.iloc[24] = 130
    assert 'two_black' not in diagnose(a, c, q, e)['cohorts'][0]['warnings']


def test_ma20_does_not_skip_missing_market_session_and_raw_red_is_not_black():
    a, c, q, e = fixture_data()
    c.iloc[20] = np.nan
    s, ma = indicators(c, q)
    assert pd.isna(ma.iloc[27, 0])
    assert not s['two_black_below_ma20'].iloc[27, 0]
    q.loc[27, 'open'] = q.loc[27, 'close']-1
    assert not indicators(c, q)[0]['two_black'].iloc[27, 0]


def test_reject_duplicate_candles_stale_holding_or_non_t_plus_one_original_order():
    a, c, q, e = fixture_data()
    with pytest.raises(ValueError, match='unique'):
        diagnose(a, c, pd.concat([q, q.iloc[:1]]), e)
    a['holdings'][3]['stale'] = True
    with pytest.raises(ValueError, match='Stale'):
        diagnose(a, c, q, e)
    a['holdings'][3]['stale'] = False
    a['orders'][0]['signal_date'] = str(c.index[30].date())
    with pytest.raises(ValueError, match='one market session'):
        diagnose(a, c, q, e)


def test_pending_cohort_outcomes_are_censored_and_not_added_to_resolved_counts():
    a, c, q, e = fixture_data(); a['orders'] = []; a['trades'] = []
    r = diagnose(a, c, q, e)
    w = r['cohorts'][0]['warnings']['two_black']
    assert w['lead_sessions'] is None and w['outcome_censored'] is True
    assert r['rule_summaries']['two_black']['resolved'] == 0
    assert r['rule_summaries']['two_black']['censored'] == 1


def test_sealed_publication_complete_and_unqualified_with_counterexamples():
    root = Path(__file__).resolve().parents[1]
    p = root/'artifacts/forward_simulation/poc_weakness_20261019.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == p.with_suffix('.sha256').read_text().strip()
    r = validate(json.loads(p.read_text()))
    assert r['original_black_audit'] == dict(decisions_checked=1696, exits=57)
    s = r['rule_summaries']['two_black_below_ma20']
    assert (s['triggered'], s['original_decision_lower'], s['subsequent_new_high']) == (22, 19, 4)
    last = r['episodes'][-1]
    assert not any(c['exit_during_decline'] for c in last['rows'])
    assert not any(c['warnings'].get('two_black_below_ma20', {}).get('location') == 'during_decline'
                   for c in last['rows'])
    for mutate in (lambda x:x.update(backtests_run=1), lambda x:x.update(live_qualified=True),
                   lambda x:x['rule_summaries']['two_black'].update(triggered=0),
                   lambda x:x['cohorts'][0].update(warnings={'two_black':dict(date=x['end'])})):
        bad = deepcopy(r); mutate(bad)
        with pytest.raises((ValueError, KeyError)):
            validate(bad)
