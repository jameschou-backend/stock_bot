from copy import deepcopy
import hashlib
import json

import pytest

from app.all_entry_account_ui import validate, case_detail, IDENTIFIER
from app.research_terminal_archive import overview
from skills.all_entry_accounts import registry


def publication():
    cases = {r['id']: dict(id=r['id'], name=r['name'], completed=False, metrics=None,
        rank=None, reason='missing evidence') for r in registry()}
    cases['benchmark'] = dict(id='benchmark', name='0050', completed=False, metrics=None, rank=None)
    cases['momentum'].update(completed=True, completed_sessions=666, rank=1,
        metrics=dict(total_return=.2), summary=dict(total_return=.2), annual=[],cohorts={})
    return dict(schema='all_entry_account_comparison_publication_v1',cases=cases,strategy_count=31,
        completed_strategies=1,ranking=['momentum'],best_completed_strategy='momentum',
        best_is_global=False,all_completed=False,live_qualified=False,actual_fill_verified=False,
        unseen_validation=False,start='2024-01-02',end='2026-10-02',initial_cash=1_000_000,position_count=3)


def test_partial_results_cannot_claim_a_global_winner_or_expose_partial_returns():
    original=publication()
    assert validate(original)['best_is_global'] is False
    for mutate in [lambda r:r.update(best_is_global=True),
                   lambda r:r.update(all_completed=True),
                   lambda r:r.update(ranking=['momentum','momentum']),
                   lambda r:r['cases']['original_red'].update(metrics=dict(total_return=10)),
                   lambda r:r['cases']['momentum'].update(completed_sessions=665),
                   lambda r:r.update(live_qualified=True)]:
        report=deepcopy(original);mutate(report)
        with pytest.raises(ValueError):validate(report)


def test_actual_detail_hash_qualification_and_summary_must_all_match(tmp_path):
    report=publication();row=report['cases']['momentum'];path=tmp_path/'account.json'
    account=dict(completed=True,summary=row['summary'],live_qualified=False,actual_fill_verified=False,
        unseen_validation=False,account=dict(daily=[dict(date='2024-01-02',nav=1_000_000)],trades=[]))
    path.write_text(json.dumps(account));row.update(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=case_detail(report,'momentum',tmp_path)
    assert result['daily'][0]['nav']==1_000_000 and result['live_qualified'] is False
    with pytest.raises(ValueError,match='尚未完成'):case_detail(report,'original_red',tmp_path)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='指紋'):case_detail(report,'momentum',tmp_path)
    row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    account['actual_fill_verified']=True;path.write_text(json.dumps(account))
    row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='不一致'):case_detail(report,'momentum',tmp_path)


def test_archive_exposes_comparison_only_after_publication_checks(tmp_path):
    report=publication();path=tmp_path/'artifacts/forward_simulation'/(IDENTIFIER+'.json')
    path.parent.mkdir(parents=True);path.write_text(json.dumps(report))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    row=next(r for r in overview(tmp_path)['items'] if r['id']==IDENTIFIER)
    assert row['status']=='verified_publication' and row['comparison']['completed_strategies']==1
    path.write_text(path.read_text()+' ')
    row=next(r for r in overview(tmp_path)['items'] if r['id']==IDENTIFIER)
    assert row['status']=='unavailable' and 'comparison' not in row
