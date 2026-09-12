from copy import deepcopy
from scripts.audit_execution_timing import audit_timings


def sample():
    entries = [dict(event_id='one', signal_date='2026-09-10', entry_date='2026-09-11',
                    group_cutoff_date='2026-08-31', members=['1560'],
                    liquidity_at_signal={'as_of': '2026-09-10'}, liquidity_before_entry={'as_of': '2026-09-10'})]
    common = dict(date='2026-09-11', event_id='one', signal_date='2026-09-10')
    case = {'account': {'trades': [dict(common, sequence=1, side='sell', stock_id='0050', reason='fund_stock'),
                                   dict(common, sequence=2, side='buy', stock_id='1560', reason='leader_entry')]},
            'exit_decisions': [dict(event_id='old', signal_date='2026-09-10', date='2026-09-11')]}
    return case, entries, ['2026-09-10', '2026-09-11', '2026-09-14']


def test_date_pass_does_not_certify_same_day_funding_or_execution():
    result = audit_timings(*sample())
    assert result['recorded_date_checks_pass']
    assert result['fills_without_execution_timestamp'] == 2
    assert result['same_day_etf_funding_events'] == 1
    assert not result['live_qualified']


def test_future_liquidity_same_day_entry_and_exit_information_are_rejected():
    case, entries, days = deepcopy(sample())
    entries[0]['entry_date'] = '2026-09-10'
    entries[0]['liquidity_before_entry']['as_of'] = '2026-09-11'
    case['exit_decisions'][0]['signal_date'] = '2026-09-11'
    result = audit_timings(case, entries, days)
    assert not result['recorded_date_checks_pass']
    assert {v['check'] for v in result['violations']} == {
        'entry_next_session', 'liquidity_before_entry', 'fill_matches_frozen_entry', 'exit_reads_previous_session'}
