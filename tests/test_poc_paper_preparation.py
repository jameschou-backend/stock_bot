from copy import deepcopy
from datetime import datetime
import json

import pandas as pd
import pytest

from app import poc_paper_preparation as p, forward_journal as j
from skills.all_entry_accounts import event


AT = datetime.fromisoformat('2026-10-15T18:20:00+08:00')
SESSIONS = ['2026-10-14','2026-10-15','2026-10-16']


@pytest.fixture
def protocol():
    return dict(strategy_id='poc_up_red',rules=dict(ranking='signal_amount20_desc_stock_id_asc'))


@pytest.fixture
def snapshot():
    return dict(schema='poc_original_observation_inputs_v1',signal_date='2026-10-15',first_only=False,
        ranking='signal_amount20_desc_stock_id_asc',rank_unknown_count=0,matched_count=2,
        candidates=[event('poc_up_red','2330','2026-10-15',None,1000),
                    event('poc_up_red','2317','2026-10-15',None,900)])


def observe(path,protocol,snapshot,at=AT,sessions=SESSIONS):
    with j.connection(path) as con:
        return p.observe(con,protocol,{},snapshot,dict(data_ready=True,price_date=snapshot['signal_date']),sessions,lambda:at)


def test_future_observation_is_idempotent_and_never_initializes_money_or_orders(tmp_path,protocol,snapshot):
    path=tmp_path/'observations.db'
    assert p.observation_summary(path)['initialized'] is False
    assert not path.exists()
    a=observe(path,protocol,snapshot)
    assert a==observe(path,protocol,snapshot)
    assert a['kind']=='poc_signal_observation'
    assert a['body']['next_market_session']=='2026-10-16'
    assert a['body']['orders_created']==a['body']['fills_created']==0
    summary=p.observation_summary(path)
    assert summary['prospective_signal_days']==1
    assert summary['orders']==summary['fills']==0
    changed=deepcopy(snapshot);changed['candidates'][0]['priority']+=1
    with pytest.raises(ValueError,match='Frozen event'):
        observe(path,protocol,changed)
    with pytest.raises(ValueError,match='Different POC'):
        observe(path,dict(protocol,other_rule=True),snapshot)


@pytest.mark.parametrize('date,clock,sessions,reason',[
    ('2026-10-14',AT,SESSIONS,'signal_not_observed_on_its_own_day'),
    ('2026-10-15',datetime.fromisoformat('2026-10-15T17:59:00+08:00'),SESSIONS,'daily_publication_window_not_finished'),
    ('2026-10-15',AT,SESSIONS[:-1],'next_market_session_not_future_or_unknown'),
])
def test_cannot_backfill_or_guess_next_session(tmp_path,protocol,snapshot,date,clock,sessions,reason):
    snapshot['signal_date']=date
    for c in snapshot['candidates']:c['signal_date']=date
    row=observe(tmp_path/'obs.db',protocol,snapshot,clock,sessions)
    assert row['kind']=='poc_blocked_observation'
    assert reason in row['body']['reasons']
    assert not row['body']['eligible_observation']


@pytest.mark.parametrize('change',['first_only','dropped','wrong_rank','wrong_strategy','reordered'])
def test_rejects_substitution_or_incomplete_candidates(tmp_path,protocol,snapshot,change):
    if change=='first_only':snapshot['first_only']=True
    if change=='dropped':snapshot['candidates'].pop()
    if change=='wrong_rank':snapshot['ranking']='display_order'
    if change=='wrong_strategy':snapshot['candidates'][0]['strategy_id']='poc_priority'
    if change=='reordered':snapshot['candidates'].reverse()
    with pytest.raises(ValueError):observe(tmp_path/'obs.db',protocol,snapshot)


def test_time_cannot_go_backwards_and_hash_chain_cannot_be_edited(tmp_path,protocol,snapshot):
    path=tmp_path/'obs.db';observe(path,protocol,snapshot)
    with pytest.raises(ValueError,match='precede'):
        observe(path,protocol,snapshot,datetime.fromisoformat('2026-10-15T18:19:00+08:00'))
    with j.connection(path) as con:con.execute("UPDATE events SET body='{}' WHERE seq=2")
    with pytest.raises(ValueError,match='integrity'):p.observation_summary(path)


def test_snapshot_keeps_continuing_signals_and_uses_quality_masked_original_ranking(monkeypatch):
    from skills.strategy_scanner import data
    dates=pd.bdate_range('2026-09-17',periods=20)
    day=str(dates[-1].date())
    rows=[]
    for sid,amount in [('2330',2000),('2317',2000),('3008',3000),('0050',1000)]:
        for date in dates:
            rows.append(dict(date=date,stock_id=sid,open=10,high=11,low=10,close=11,
                adjusted_close=11,volume=100,amount=amount,quality=not(sid=='3008' and date==dates[0]),eligible=True))
    monkeypatch.setattr(data,'load_bundle',lambda *args:dict(bars=pd.DataFrame(rows),calendar=dates))
    class Terminal:
        bundle='unused';daily_descriptor={};source_hashes={}
        _days={day:dict(stocks=[dict(stock_id=sid,name=sid,results={'poc_up_red':dict(
            status='matched',first_signal=False,metrics={})}) for sid in ('2330','2317','3008')])}
        def _load(self):pass
        def _unchanged(self):pass
    result=p.scanner_snapshot(Terminal())
    assert [c['members'][0] for c in result['candidates']]==['2317','2330']
    assert all(c['first_signal'] is False for c in result['candidates'])
    assert result['rank_unknown_count']==1 and result['unranked']==['3008']
    assert result['historical_backtests_replayed']==0


def test_frozen_protocol_rejects_changed_source(monkeypatch,tmp_path):
    path=tmp_path/p.PROTOCOL.relative_to(p.ROOT);path.parent.mkdir(parents=True)
    payload=dict(strategy_id='poc_up_red',source_code_sha256={'file':'old'})
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(p,'protocol_payload',lambda root:payload)
    assert p.load_protocol(tmp_path)[0]==payload
    payload['source_code_sha256']['file']='new'
    with pytest.raises(ValueError,match='changed'):p.load_protocol(tmp_path)


@pytest.mark.parametrize('change',['paper','fill','drop_benchmark','hide_gap','capital'])
def test_published_preparation_cannot_be_promoted_or_hide_missing_evidence(change):
    report=json.loads((p.ROOT/'artifacts/forward_simulation'/f'{p.IDENTIFIER}.json').read_text())
    assert p.validate(report) is report
    if change=='paper':report['paper_enabled']=True
    if change=='capital':report['capital_initialized']=True
    if change=='fill':report['observation']['body']['fills_created']=1
    if change=='hide_gap':report['observation']['body']['reasons']=[]
    if change=='drop_benchmark':del report['completed_controls']['benchmark:original:baseline']
    with pytest.raises(ValueError):p.validate(report)
