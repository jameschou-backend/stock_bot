from copy import deepcopy
import json

import pandas as pd
import pytest

from skills import all_entry_accounts as candidates
from scripts import research_all_entry_accounts as research
from scripts import research_strategy_account_comparison as frozen


def small_candidate_bundle(monkeypatch):
    days = ['2024-01-02', '2024-01-03', '2024-01-04']
    by_arm = {x['id']: [candidates.event(x['id'], '2330', days[0], days[1], 1e9)]
              for x in candidates.registry()}
    manifest = dict(ranking='T_amount20_desc_stock_id_asc', counts={k: dict(executable=1) for k in by_arm})
    monkeypatch.setattr(research, 'load_candidates', lambda directory: (by_arm, days, manifest, {}))
    return by_arm, days


def test_exact_entry_registry_excludes_rank_and_filter_modules():
    rows = candidates.registry()
    assert len(rows) == len({r['id'] for r in rows}) == 31
    assert set(candidates.CONTEXT) <= {r['id'] for r in rows}
    assert all(r['kind'] == 'entry' for r in rows)
    assert not {'risk_momentum', 'poc_red_priority', 'entry_not_extended'} & {r['id'] for r in rows}


def test_private_adapter_preserves_frozen_runner_and_common_financial_classes(monkeypatch):
    by_arm, _ = small_candidate_bundle(monkeypatch)
    before = (frozen.ARMS, frozen.RUN_ROOT, frozen.RSI_ARMS, frozen.select_arm_entries)
    runner, _ = research.configured_runner(candidates.BASE/'test', ['momentum', 'poc_up_red', 'benchmark'])
    assert before == (frozen.ARMS, frozen.RUN_ROOT, frozen.RSI_ARMS, frozen.select_arm_entries)
    assert len(runner.ARMS) == 32 and len(runner.RSI_ARMS) == 31
    assert set(runner.ARM_RULES.values()) == {(False, 'none', .7, .3, False)}
    assert runner.DrawdownControl is frozen.DrawdownControl
    assert runner.MidpointExitReplay is frozen.MidpointExitReplay
    assert runner.HistoricalOddEra is frozen.HistoricalOddEra
    assert runner.SELECTION_ARMS['poc_up_red'] == 'poc_filter'
    assert all(v == 'original' for k,v in runner.SELECTION_ARMS.items() if k != 'poc_up_red')
    for arm in ['momentum', 'poc_up_red']:
        selected, red, first = runner.select_arm_entries(arm, [], None, candidates.START, candidates.END)
        assert selected == by_arm[arm] and not red and not first


def test_actual_financial_clock_rejects_same_day_signals(monkeypatch):
    by_arm, days = small_candidate_bundle(monkeypatch)
    by_arm['momentum'][0]['entry_date'] = days[0]
    with pytest.raises(ValueError, match=r'T\+1'):
        research.configured_runner(candidates.BASE/'test', ['momentum'])


def test_unknown_rule_or_duplicate_run_is_rejected_before_replay(monkeypatch):
    small_candidate_bundle(monkeypatch)
    for arms in [[], ['not_a_strategy'], ['momentum', 'momentum']]:
        with pytest.raises(ValueError, match='registered rules'):
            research.configured_runner(candidates.BASE/'test', arms)


def test_sealed_evidence_is_idempotent_but_cannot_change(tmp_path):
    path = tmp_path/'evidence.json'
    first = candidates.write(path, {'x':1})
    assert candidates.write(path, {'x':1}) == first
    with pytest.raises(ValueError, match='replace sealed'):
        candidates.write(path, {'x':2})
    assert json.loads(path.read_text()) == {'x':1}


def test_cross_family_admission_is_rejected_by_independent_identity_audit(monkeypatch):
    by_arm, days = small_candidate_bundle(monkeypatch)
    runner, _ = research.configured_runner(candidates.BASE/'test', ['momentum','original_red'])
    entries = by_arm['momentum'] + by_arm['original_red']
    account = dict(cohorts=[dict(event_id=e['event_id'], stock_id='2330',
                               signal_date=days[0], entry_date=days[1]) for e in entries])
    with pytest.raises(ValueError, match='more than one'):
        runner.audit_candidate_identity(account, entries)


def test_budget_is_persistent_and_offline_never_dispatches(tmp_path, monkeypatch):
    from skills import all_entry_account_data as data
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    monkeypatch.setattr(data, 'BASE', tmp_path/'run')
    prereg = tmp_path/'spec.md'; prereg.write_text('fixed experiment')
    monkeypatch.setattr(data, 'PREREG', prereg)
    monkeypatch.setattr(data, 'LIMITS', dict(financial=1, board=1, profiles=1))
    query = dict(dataset='TaiwanStockPriceTick', data_id='2330', start_date='2024-01-03')
    offline = data.AccountBudget()
    with pytest.raises(data.ComparisonDataRequired, match='offline'):
        offline.reserve('board', query)
    first = data.AccountBudget(online=True)
    first.reserve('board', query)
    second = data.AccountBudget(online=True)
    assert second.snapshot()['attempted_calls'] == 1
    with pytest.raises(data.ComparisonDataRequired, match='no automatic retry'):
        second.reserve('board', query)
    with pytest.raises(data.ComparisonDataRequired, match='exhausted'):
        second.reserve('board', dict(query, data_id='2317'))
    assert second.snapshot()['attempted_calls'] == 1


def test_new_odd_authorization_binds_its_own_preregistration(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from skills import all_entry_account_data as data
    root=tmp_path;prereg=root/'new.md';prereg.write_text('new registered scope')
    obj=object.__new__(data.OddData)
    obj.root=root;obj.refs={};obj.clock=lambda:1000
    obj.expected_auth=dict(prereg_path='new.md',prereg_sha256=data.digest(prereg))
    obj.auth=root/'authorization.json'
    obj.auth.write_text(json.dumps(dict(obj.expected_auth,created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat())))
    assert obj._authorization(fresh=True)['path']=='authorization.json'
    assert obj.refs['new.md']==data.digest(prereg)
    obj.clock=lambda:1000+86401
    with pytest.raises(data.ComparisonDataRequired,match='expired'):obj._authorization(fresh=True)


def test_completed_replay_cannot_hide_unmodeled_certificate_inventory():
    from scripts.summarize_all_entry_accounts import guard_certificate_inventory, UnverifiedCertificateInventory
    guard_certificate_inventory(dict(account=dict(corporate_actions=[])))
    for sid,day in [('5876','2026-07-28'),('6446','2026-07-09')]:
        with pytest.raises(UnverifiedCertificateInventory,match='新股權利證書'):
            guard_certificate_inventory(dict(account=dict(corporate_actions=[
                dict(stock_id=sid,date=day,kind='stock_dividend')])) )
