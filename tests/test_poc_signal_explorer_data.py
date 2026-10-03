from copy import deepcopy
import json

import pandas as pd
import pytest

from scripts.export_poc_signal_explorer import (account_view, build_decisions,
    build_payload, build_signal_rows, profile_details, verify_refs)
from scripts.export_signal_explorer import digest, json_for_script, unpack_price


def fixture_data():
    days = pd.to_datetime(['2025-12-31', '2026-01-02', '2026-01-05', '2026-01-06'])
    adjusted = pd.DataFrame({'1111': [50., 51., 52., 53.], '2222': [60., 61., 62., 63.],
                             '3333': [70., 71., 72., 73.]}, index=days)
    quotes = pd.DataFrame([dict(date=d, stock_id=sid, open=2 * value + (1 if sid == '2222' else -1),
        high=2 * value + 2, low=2 * value - 2, close=2 * value, volume=1e6)
        for sid in adjusted for d, value in adjusted[sid].items()])
    eligible = adjusted.notna()
    companies = pd.DataFrame([dict(stock_id=sid, name='公司'+sid, market='TWSE') for sid in adjusted])
    entries = [dict(event_id='e'+sid, signal_date='2026-01-02', entry_date='2026-01-05',
        members=[sid], priority=.2, leader_evidence=dict(leader_return20=.3,
            benchmark_return20=.1, leader_volume_ratio=2.)) for sid in ['2222', '1111']]
    return entries, adjusted, quotes, eligible, companies


def signals():
    entries, a, q, _, names = fixture_data()
    return build_signal_rows(entries, a, q, names, end='2026-01-06')


def known_profile(sid='1111'):
    dates = [str(d.date()) for d in pd.bdate_range(end='2025-12-31', periods=20)]
    return dict(event_id='e'+sid, stock_id=sid, signal_date='2026-01-02',
        available=True, poc_up=True, prior_dates=dates, ordinary_daily_matched=False,
        profile=dict(session_dates=dates, poc_up=True, first_half={'poc_price': 80},
                     second_half={'poc_price': 90}, full={'poc_price': 90, 'val': 85, 'vah': 95}))


def case(red=False, original=False):
    candidates = ['e1111'] if red else ['e1111', 'e2222']
    context = dict(date='2026-01-05', original_event_ids=candidates, selected_event_ids=candidates,
        certificate=dict(decisions=[dict(event_id=eid, original_rank=i+1,
            profile_status='not_needed_resource_exhausted',
            selection_status='not_needed_resource_exhausted') for i, eid in enumerate(candidates)]))
    if original:
        context.pop('certificate')
    d = dict(family_rules=dict(red_gate=red, volume_exit_mode='none', anchor='original' if original else 'poc_base'),
        entry_gate_decisions=[dict(event_id=s['signal_id'], signal_date=s['signal_date'],
            entry_date=s['entry_date'], status=s['candle'], passed=s['candle']=='red') for s in signals()] if red else [],
        profile_queries=[], summary={'annual': [{'year': '2026', 'total_return': .03}], 'start': '2024-01-02'},
        account=dict(selection_decisions=[context], orders=[], trades=[], daily=[], holdings=[]))
    for day in ['2025-12-31', '2026-01-02', '2026-01-05', '2026-01-06']:
        d['account']['daily'].append(dict(date=day, nav=2e6, opening_nav=2e6, cash=1e6, holdings=1))
        d['account']['holdings'].append(dict(date=day, stock_id='3333', event_id='old2025',
                                            qty=1000, name='公司3333', price=140, market_value=140000))
    return d


def test_original_daily_population_and_ties_precede_red_filter():
    s = signals()
    assert [(r['signal_id'], r['daily_rank'], r['candidate_count']) for r in s] == [
        ('e1111', 1, 2), ('e2222', 2, 2)]
    decisions = build_decisions(s, case(red=True), [known_profile('2222')])
    assert set(decisions) == {'e1111', 'e2222'}
    assert decisions['e2222']['red_gate'] is False
    assert decisions['e2222']['poc_status'] == 'not_evaluated'
    assert decisions['e2222']['selection_status'] == 'red_gate_rejected'
    assert decisions['e1111']['poc_reason'] == 'not_needed_resource_exhausted'


def test_other_arm_profile_cannot_leak_into_unqueried_candidate():
    d = build_decisions(signals(), case(), [known_profile()])
    assert d['e1111']['poc_status'] == 'not_evaluated'
    assert 'poc_before' not in d['e1111']
    assert d['e1111']['simulated_buy_qty'] == 0
    assert d['e1111']['selected_for_planning'] is True  # planning is NOT a fill


def test_queried_profile_identity_and_pre_signal_window_are_required():
    c = case()
    c['profile_queries'] = [dict(event_id='e1111', stock_id='1111', signal_date='2026-01-02')]
    c['account']['selection_decisions'][0]['certificate']['decisions'][0]['profile_status'] = 'known_true'
    with pytest.raises(ValueError, match='Queried profile is missing'):
        build_decisions(signals(), c, [])
    d = build_decisions(signals(), c, [known_profile()])['e1111']
    assert d['poc_status'] == 'up' and d['poc_before'] == 80 and d['poc_price_basis'] == 'raw'
    p = known_profile()
    p['prior_dates'][-1] = '2026-01-02'
    with pytest.raises(ValueError, match='strictly pre-signal'):
        profile_details(p, signals()[0])
    p = known_profile('2222')
    with pytest.raises(ValueError, match='identity mismatch'):
        profile_details(p, signals()[0])


def test_unknown_fallback_is_not_poc_pass_or_a_buy():
    c = case()
    c['profile_queries'] = [dict(event_id='e1111', stock_id='1111', signal_date='2026-01-02')]
    ctx = c['account']['selection_decisions'][0]
    ctx.update(fallback_to_original=True, fallback_reason='ordinary_tape_conflict')
    ctx['certificate']['decisions'][0]['profile_status'] = 'unknown'
    p = dict(event_id='e1111', stock_id='1111', signal_date='2026-01-02',
             available=False, poc_up=None, reason='ordinary_tape_conflict')
    d = build_decisions(signals(), c, [p])
    assert d['e1111']['poc_status'] == 'unknown'
    assert 'poc_before' not in d['e1111']
    assert d['e2222']['poc_status'] == 'not_evaluated'
    assert d['e2222']['poc_reason'] == 'whole_day_quality_fallback_before_query'
    assert all(r['selection_status'] == 'fallback_original' and r['simulated_buy_qty'] == 0 for r in d.values())


def test_future_prices_and_outcomes_cannot_change_signal_features():
    entries, a, q, _, names = fixture_data()
    before = build_signal_rows(entries, a, q, names, end='2026-01-06')
    changed = a.copy()
    changed.loc['2026-01-05':] *= 100
    q2 = q.copy()
    q2.loc[q2.date >= '2026-01-05', ['open', 'high', 'low', 'close']] *= 100
    entries[0]['net_return'] = 999
    assert before == build_signal_rows(entries, changed, q2, names, end='2026-01-06')
    assert not any('net_return' in r or 'exit_date' in r for r in before)


def test_red_gate_only_uses_signal_candle_not_entry_candle():
    entries, a, q, _, names = fixture_data()
    q.loc[(q.date == '2026-01-05') & (q.stock_id == '2222'), 'open'] = 1
    s = build_signal_rows(entries, a, q, names, end='2026-01-06')
    assert s[1]['candle'] == 'black'
    assert build_decisions(s, case(red=True), [])['e2222']['red_gate'] is False


def test_prior_inventory_and_cumulative_account_survive_2026_slice():
    entries, a, q, eligible, names = fixture_data()
    cases = {'poc_red': case(red=True), 'poc_base': case(), 'original': case(original=True)}
    p = build_payload(entries, cases, {k: [] for k in cases}, a, a.copy(), q, eligible, names, end='2026-01-06')
    assert p['default_strategy'] == 'poc_red'
    assert [d['signal_count'] for d in p['days']] == [2, 0, 0]
    assert p['days'][1]['signal_status'] == 'complete'
    assert p['days'][2]['signal_status'] == 'not_generated_no_next_session'
    assert p['metadata']['zero_signal_days'] == 1
    v = p['strategies']['poc_red']
    assert v['opening_inventory_date'] == '2025-12-31'
    assert v['opening_inventory'][0]['event_id'] == 'old2025'
    assert v['account_days']['2026-01-02']['nav'] == 2e6
    assert v['account_days']['2026-01-02']['holdings'][0]['stock_id'] == '3333'
    assert p['stocks']['3333']['signal_ids'] == []
    assert all(r[0] <= '2026-01-06' for stock in p['stocks'].values() for r in stock['prices'])


def test_trade_markers_share_adjusted_candle_basis_and_keep_raw_reference():
    _, a, q, _, _ = fixture_data()
    c = case()
    c['account']['trades'] = [dict(date='2026-01-05', stock_id='1111', side='buy',
        event_id='e1111', signal_date='2026-01-02', qty=2, reference_price=103., channel='odd')]
    view = account_view(c, '2026-01-02', '2026-01-06', a,
                        q.pivot(index='date', columns='stock_id', values='close'))
    trade = view['trades'][0]
    assert trade['reference_price'] == 103.
    assert trade['adjusted_marker_price'] == 51.5
    assert trade['execution_kind'] == 'research_simulated_fill'
    assert trade['channel'] == 'odd'


def test_inconsistent_red_or_missing_context_fails_instead_of_guessing():
    c = case(red=True)
    c['entry_gate_decisions'][0]['passed'] = False
    with pytest.raises(ValueError, match='Red gate disagrees'):
        build_decisions(signals(), c, [])
    c = case()
    c['account']['selection_decisions'] = []
    with pytest.raises(ValueError, match='absent from account'):
        build_decisions(signals(), c, [])


def test_hash_verification_and_inert_json_protect_provenance(tmp_path):
    p = tmp_path / 'source.json'
    p.write_text('{}')
    refs = {'source.json': digest(p)}
    verify_refs(refs, tmp_path)
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed sealed source'):
        verify_refs(refs, tmp_path)
    payload = {'name': '</script><script>alert(1)</script>\u2028&'}
    encoded = json_for_script(payload)
    assert '</script>' not in encoded and '\\u2028' in encoded
    assert json.loads(encoded) == payload
