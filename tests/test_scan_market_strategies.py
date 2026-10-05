import hashlib
import json
from pathlib import Path

import pytest

from scripts.scan_market_strategies import entry_events, load_poc


def test_entry_export_keeps_all_stocks_and_next_session_timing():
    result=dict(status='matched',first_signal=True,reasons=['rule'],metrics={},regime_fit=True)
    payload=dict(strategies=[dict(id='x',version='1')],evaluated_strategy_ids=['x'],
        days=[dict(date='2026-10-02',market_regime='trend_up',stocks=[
            dict(stock_id=str(2300+i),regime='trend_up',results={'x':result}) for i in range(8)])])
    events=entry_events(payload,'x')
    assert len(events)==8
    assert {e['earliest_execution'] for e in events}=={'next_market_session'}
    assert all(e['account_independent'] and not e['live_qualified'] for e in events)
    assert len({e['event_id'] for e in events})==8
    payload['days'][0]['stocks'][0]['results']={'x':dict(result,first_signal=None)}
    assert len(entry_events(payload,'x',first_only=True))==7
    with pytest.raises(ValueError):entry_events(payload,'not_selected')


def seal(tmp_path,rows):
    p=tmp_path/'profiles.json';p.write_text(json.dumps(rows))
    r=tmp_path/'report.json';r.write_text(json.dumps(dict(schema='poc_daily_profiles_v1',
        all_signals_materialized=True,end='2026-10-02',profiles=dict(path=p.name,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()))))
    r.with_suffix('.sha256').write_text(hashlib.sha256(r.read_bytes()).hexdigest())
    return r,p


def test_poc_loader_only_accepts_hash_bound_account_independent_evidence(tmp_path):
    r,p=seal(tmp_path,[dict(account_independent=True)])
    rows,info=load_poc(r,root=tmp_path)
    assert rows and info['scope']=='original_candidate_stockdays_only_not_every_stockday'
    assert info['ancestor_raw_files_reverified'] is False
    with pytest.raises(ValueError,match='input bundle'):
        load_poc(r,root=tmp_path,bundle=tmp_path,manifest_hash='0'*64)
    p.write_text('[]')
    with pytest.raises(ValueError,match='profile hash'):load_poc(r,root=tmp_path)
    r,p=seal(tmp_path,[dict(account_independent=False)])
    with pytest.raises(ValueError,match='independent'):load_poc(r,root=tmp_path)
    assert load_poc(None)[1]['status']=='not_supplied'
