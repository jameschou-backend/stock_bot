"""Broker account evidence gates, immutable reuse, and quota guards; offline only."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.prepare_poc_latest_inputs import digest
import skills.poc_broker_data as broker
import skills.poc_latest_data as latest
import skills.volume_profile_data as profiles
import scripts.prepare_volume_profile as preparation


def event(eid, sid='2330', signal='2026-09-01', entry='2026-09-02'):
    return dict(event_id=eid, members=[sid], signal_date=signal,
                entry_date=entry, priority=.5, leader_evidence={'volume_ratio':2.})


def evidence(e, *, known=True, persistent=True, concentrated=True):
    return dict(stock_id=e['members'][0], signal_date=e['signal_date'], known5=known,
                persistent5=persistent if known else None,
                concentrated=concentrated if known else None,
                source_event_id='different-old-cohort-id',
                unknown_reason=None if known else 'missing_branch_market_day')


@pytest.mark.parametrize('arm,indices', [
    ('poc_red',[0,1,2,3]), ('poc_persist_guard',[0,1,3]),
    ('poc_combined_guard',[0,3]), ('poc_known5_control',[0,1,2]),
    ('poc_known5_filter',[0,1]), ('poc_known5_combined',[0]),
])
def test_six_preregistered_arms_preserve_order_and_distinguish_unknown(arm,indices):
    rows=[event(str(i),sid) for i,sid in enumerate(('2330','2317','2308','2382'))]
    ev={r['event_id']:evidence(r,persistent=i!=2,concentrated=i!=1,known=i!=3)
        for i,r in enumerate(rows)}
    saved_rows,saved_ev=deepcopy(rows),deepcopy(ev)
    selected,log=broker.apply_broker_gate(arm,rows,ev)
    assert selected==[rows[i] for i in indices]
    assert [d['event_id'] for d in log]==[r['event_id'] for r in rows]
    assert rows==saved_rows and ev==saved_ev
    assert log[-1]['known5'] is False and log[-1]['persistent5'] is None
    assert log[-1]['combined'] is None
    assert log[-1]['unknown_reason']=='missing_branch_market_day'
    if arm.endswith('_guard'):
        assert log[-1]['kept'] is True and log[-1]['reason']=='unknown_kept_by_guard_policy'
    elif arm.startswith('poc_known5_'):
        assert log[-1]['kept'] is False and log[-1]['reason']=='outside_matched_coverage'
    selected[0]['leader_evidence']['volume_ratio']=-99
    assert rows==saved_rows


def test_matched_control_and_filters_use_identical_known_evidence_denominator():
    rows=[event(str(i)) for i in range(4)]
    ev={r['event_id']:evidence(r,known=i<3,persistent=i%2==0,concentrated=i==0)
        for i,r in enumerate(rows)}
    logs=[broker.apply_broker_gate(arm,rows,ev)[1]
          for arm in ('poc_known5_control','poc_known5_filter','poc_known5_combined')]
    assert all([d['known5'] for d in log]==[True,True,True,False] for log in logs)
    assert all(log[-1]['reason']=='outside_matched_coverage' for log in logs)


@pytest.mark.parametrize('change', ['missing','stock','date','known_string','persistent_missing','concentrated_number'])
def test_gate_fails_closed_on_wrong_or_implicit_evidence(change):
    e=event('current');ev={'current':evidence(e)}
    if change=='missing':ev={}
    elif change=='stock':ev['current']['stock_id']='2317'
    elif change=='date':ev['current']['signal_date']='2026-09-02'
    elif change=='known_string':ev['current']['known5']='false'
    elif change=='persistent_missing':del ev['current']['persistent5']
    else:ev['current']['concentrated']=1
    with pytest.raises(ValueError):
        broker.apply_broker_gate('poc_known5_filter',[e],ev)


@pytest.mark.parametrize('change',['duplicate','same_day','earlier_entry','multiple_members'])
def test_gate_rejects_invalid_candidate_identity(change):
    e=event('current');rows=[e]
    if change=='duplicate':rows.append(deepcopy(e))
    elif change=='same_day':e['entry_date']=e['signal_date']
    elif change=='earlier_entry':e['entry_date']='2026-08-31'
    else:e['members'].append('2317')
    with pytest.raises(ValueError):
        broker.apply_broker_gate('poc_red',rows,{'current':evidence(e)})


def test_unknown_is_never_promoted_by_future_outcome_or_stale_condition_values():
    e=event('future-winner');item=evidence(e,known=False)
    item.update(persistent5=True,concentrated=True,outcome={'return':10000,'win':True})
    selected,log=broker.apply_broker_gate('poc_known5_filter',[e],{e['event_id']:item})
    assert selected==[] and log[0]['persistent5'] is None and log[0]['combined'] is None
    for arm in broker.ARMS:
        first=broker.apply_broker_gate(arm,[e],{e['event_id']:item})
        item['outcome']={'return':-1,'win':False}
        assert broker.apply_broker_gate(arm,[e],{e['event_id']:item})==first


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def provider_fixture(tmp_path,monkeypatch):
    """Keep actual evidence loader/binding; stub unrelated 60k-file market startup."""
    monkeypatch.setattr(broker,'ROOT',tmp_path)
    monkeypatch.setattr(broker,'BASE',tmp_path/'.cache/new-account')
    monkeypatch.setattr(broker,'BUNDLE',tmp_path/'bundle')
    monkeypatch.setattr(broker,'DIAGNOSTIC',tmp_path/'diagnostic')
    monkeypatch.setattr(broker,'LATEST',tmp_path/'latest/report.json')
    code=tmp_path/'skills/poc_broker_data.py';code.parent.mkdir();code.write_text('# bound adapter fixture')
    monkeypatch.setattr(broker,'__file__',str(code))
    spec=tmp_path/'spec.md';spec.write_text('fixed six arms')
    monkeypatch.setattr(broker,'PREREG',spec)
    def initialize(self,*args,**kwargs):
        assert kwargs['online'] is False
        self.refs={};self.profiles=SimpleNamespace(refs={});self.profile_index={'poc_red':{}}
        self.frozen_odds=SimpleNamespace(refs={})
    monkeypatch.setattr(broker.LatestAccountData,'__init__',initialize)
    monkeypatch.setattr(broker,'BrokerProfiles',lambda **kw:SimpleNamespace(refs={},online=kw['online']))
    monkeypatch.setattr(broker,'LatestExecutionData',lambda *a,**kw:SimpleNamespace(refs={}))
    entries=[event('new-'+str(i),str(2300+i)) for i in range(5)]
    pending=event('terminal','2492',signal='2026-10-02',entry=None)
    from scripts import research_poc_latest_account
    monkeypatch.setattr(research_poc_latest_account,'load_candidate_bundle',lambda *a:(deepcopy(entries),[deepcopy(pending)]))
    rows=[]
    for i,e in enumerate(entries[:4]):
        branch_known=i!=2;p_known=i!=3
        rows.append(dict(event_id='old-'+str(i),stock_id=e['members'][0],signal_date=e['signal_date'],
            branch=dict(known=branch_known,concentrated_directional=i==0,
                        reason=None if branch_known else 'raw_market_imbalance'),
            persistence5=dict(known=p_known,passed=i==0,
                              reason=None if p_known else 'missing_branch_market_day'),
            outcome={'return':9999,'future_winner':True},name='unused',first_for_stock=True))
    paths=dict(rows=tmp_path/'diagnostic/rows.json',report=tmp_path/'diagnostic/report.json',
               publication=tmp_path/'artifacts/forward_simulation/broker_branch_diagnostics_20261004.json')
    source=tmp_path/'raw-source.json';source.write_text('{}')
    def seal(new_rows=rows):
        write_json(paths['rows'],new_rows)
        write_json(paths['report'],dict(source_sha256={'raw-source.json':digest(source)}))
        write_json(paths['publication'],dict(evidence={str(p.relative_to(tmp_path)):digest(p)
                                    for p in (paths['rows'],paths['report'])}))
        paths['publication'].with_suffix('.sha256').write_text(digest(paths['publication']))
    seal()
    write_json(broker.LATEST,dict(source_sha256={}))
    broker.LATEST.with_suffix('.sha256').write_text(digest(broker.LATEST))
    return SimpleNamespace(root=tmp_path,rows=rows,entries=entries,pending=pending,
                           source=source,paths=paths,seal=seal)


def test_provider_maps_stock_and_signal_not_old_event_id_and_only_causal_fields(provider_fixture):
    f=provider_fixture;p=broker.BranchAccountData(f.root)
    assert list(p.broker_evidence)==[e['event_id'] for e in f.entries]+[f.pending['event_id']]
    rows=[p.broker_evidence[e['event_id']] for e in f.entries]
    assert [r['known5'] for r in rows]==[True,True,False,False,False]
    assert [r['source_event_id'] for r in rows[:4]]==['old-0','old-1','old-2','old-3']
    assert rows[2]['unknown_reason']=='raw_market_imbalance'
    assert rows[3]['unknown_reason']=='missing_branch_market_day'
    assert rows[4]['unknown_reason']=='no_cached_branch_snapshot'
    assert p.broker_evidence[f.pending['event_id']]['known5'] is False
    assert all(not set(r)&{'outcome','future_winner','return','name','first_for_stock'} for r in rows)
    original=deepcopy(p.broker_evidence)
    for r in f.rows:r['outcome']={'return':-9999,'future_winner':False}
    f.seal()
    assert broker.BranchAccountData(f.root).broker_evidence==original


def test_provider_refuses_duplicate_diagnostic_identity(provider_fixture):
    f=provider_fixture;f.rows.append(deepcopy(f.rows[0]));f.seal()
    with pytest.raises(ValueError,match='Duplicate diagnostic'):
        broker.BranchAccountData(f.root)


@pytest.mark.parametrize('component',['branch','persistence5'])
def test_provider_refuses_nonboolean_source_known_flags(provider_fixture,component):
    f=provider_fixture;f.rows[0][component]['known']='false';f.seal()
    with pytest.raises(ValueError):
        broker.BranchAccountData(f.root)


@pytest.mark.parametrize('which',['rows','report','raw'])
def test_provider_refuses_changed_published_or_underlying_data(provider_fixture,which):
    f=provider_fixture;path=f.source if which=='raw' else f.paths[which]
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='source hash changed'):
        broker.BranchAccountData(f.root)


def test_bind_cannot_replace_previously_bound_source(provider_fixture):
    f=provider_fixture;p=broker.BranchAccountData(f.root)
    f.source.write_text('changed')
    with pytest.raises(ValueError,match='Conflicting source hashes'):
        p._bind(f.source)


@pytest.fixture
def raw_profile(tmp_path,monkeypatch):
    # Exercise the actual inherited cache path with a small initialized state.
    monkeypatch.chdir(tmp_path)
    for module in (broker,latest,profiles,preparation):
        monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(profiles,'PILOT',tmp_path/'no-pilot')
    p=broker.BrokerProfiles.__new__(broker.BrokerProfiles)
    p.directory=tmp_path/'.cache/new-profiles';p.directory.mkdir(parents=True)
    p.refs={};p.online=False;p.maximum=2400;p.receipt_hashes={};p.reuse_index={}
    p._config=None;p._lock=Lock();p._stop=Event()
    return p,tmp_path


def receipt(root,*,stock='2330',day='2026-09-01',folder='latest',status='received'):
    query=dict(dataset='TaiwanStockPriceTick',data_id=stock,start_date=day)
    folders={'latest':'.cache/poc-latest-20261003/profiles-v1/receipts',
             'account':'.cache/volume-profile-account-20261003/profiles-v1/receipts',
             'daily':'.cache/poc-daily-opportunities-20261004/data-v1/attempts'}
    suffix='-20261004T000000.json' if folder=='daily' else '.json'
    path=root/folders[folder]/(stock+'-'+day+suffix)
    raw=root/'raw'/f'{stock}-{day}-{folder}.parquet';raw.parent.mkdir(exist_ok=True)
    pd.DataFrame([dict(date=day,stock_id=stock,price=100,volume=1)]).to_parquet(raw,index=False)
    item=dict(query=query,status=status,raw_path=str(raw.relative_to(root)),raw_sha256=digest(raw))
    write_json(path,item)
    return path,raw,item


@pytest.mark.parametrize('folder',['latest','account','daily'])
def test_profiles_reuse_authentic_receipt_without_a_new_attempt(raw_profile,folder):
    p,root=raw_profile;source,raw,item=receipt(root,folder=folder)
    result=p._raw(('2330','2026-09-01'))
    assert result['query']==item['query'] and result['raw_sha256']==digest(raw)
    assert result['reused_receipt']==str(source.relative_to(root))
    assert str(source.relative_to(root)) in p.refs and str(raw.relative_to(root)) in p.refs
    assert not list(p.directory.rglob('attempts/*.json'))
    assert p._raw(('2330','2026-09-01'))==result


@pytest.mark.parametrize('change',['query','raw_bytes'])
def test_bad_high_priority_reuse_source_never_silently_falls_back(raw_profile,change):
    p,root=raw_profile;source,raw,item=receipt(root,folder='latest')
    receipt(root,folder='account')
    if change=='query':
        item['query']['data_id']='2317';write_json(source,item)
    else:raw.write_bytes(b'corrupted parquet')
    with pytest.raises(ValueError):p._raw(('2330','2026-09-01'))
    assert not (p.directory/'receipts/2330-2026-09-01.json').exists()
    assert not list(p.directory.rglob('attempts/*.json'))


def test_existing_failure_receipt_is_preserved_despite_available_valid_reuse(raw_profile):
    p,root=raw_profile;receipt(root)
    dest=p.directory/'receipts/2330-2026-09-01.json'
    failure=dict(query=dict(dataset='TaiwanStockPriceTick',data_id='2330',start_date='2026-09-01'),
                 status='provider_error',error_type='FinMindError')
    write_json(dest,failure);before=dest.read_bytes()
    assert p._raw(('2330','2026-09-01'))==failure
    assert dest.read_bytes()==before and not list(p.directory.rglob('attempts/*.json'))


def resumed_profile(previous):
    value=broker.BrokerProfiles.__new__(broker.BrokerProfiles)
    value.__dict__=dict(previous.__dict__)
    value.refs={}
    return value


def test_reused_receipt_lineage_is_verified_and_bound_after_process_restart(raw_profile):
    p,root=raw_profile;source,raw,item=receipt(root)
    first=p._raw(('2330','2026-09-01'))
    assert first['reused_receipt_sha256']==digest(source)
    resumed=resumed_profile(p)
    assert resumed._raw(('2330','2026-09-01'))==first
    assert resumed.refs[str(source.relative_to(root))]==digest(source)
    assert resumed.refs[str(raw.relative_to(root))]==item['raw_sha256']


@pytest.mark.parametrize('change',['source_bytes','source_query','copied_raw'])
def test_resume_rejects_changed_source_receipt_query_or_raw_copy(raw_profile,change):
    p,root=raw_profile;source,raw,item=receipt(root)
    p._raw(('2330','2026-09-01'))
    dest=p.directory/'receipts/2330-2026-09-01.json'
    copied=json.loads(dest.read_text())
    if change=='source_bytes':
        source.write_text(source.read_text()+' ')
    elif change=='source_query':
        original=json.loads(source.read_text());original['query']['data_id']='2317'
        write_json(source,original)
        # Both documents remain independently hashed: identity must also be checked.
        copied['reused_receipt_sha256']=digest(source);write_json(dest,copied)
    else:
        alternative=root/'raw/alternative.parquet'
        pd.DataFrame([dict(date='2026-09-01',stock_id='2330',price=999,volume=1)]).to_parquet(alternative,index=False)
        copied.update(raw_path=str(alternative.relative_to(root)),raw_sha256=digest(alternative))
        write_json(dest,copied)
    before=dest.read_bytes()
    with pytest.raises(ValueError):resumed_profile(p)._raw(('2330','2026-09-01'))
    assert dest.read_bytes()==before and not list(p.directory.rglob('attempts/*.json'))


@pytest.mark.parametrize('remaining,retry',[(3,0),(6000,60)])
def test_shared_quota_pause_writes_no_attempt_or_receipt(raw_profile,monkeypatch,remaining,retry):
    p,root=raw_profile;p.online=True
    p._config=SimpleNamespace(finmind_requests_per_hour=6000)
    from app import rate_limiter
    seen=[]
    def limiter(maximum):
        seen.append(maximum)
        return SimpleNamespace(get_stats=lambda:SimpleNamespace(remaining_requests=remaining,retry_after_seconds=retry))
    monkeypatch.setattr(rate_limiter,'get_rate_limiter',limiter)
    assert p._raw(('2330','2026-09-01'))['status']=='request_budget_or_quota_paused'
    # The limiter applies its 10% buffer: 6000 requested becomes 5400 effective.
    assert seen==[6000] and not list(p.directory.rglob('*.json'))


def test_cached_profiles_reuse_even_when_quota_is_exhausted(raw_profile,monkeypatch):
    p,root=raw_profile;p.online=True;receipt(root)
    from app import rate_limiter
    def no_quota_call(*args):raise AssertionError('A cache hit needs no quota check')
    monkeypatch.setattr(rate_limiter,'get_rate_limiter',no_quota_call)
    assert p._raw(('2330','2026-09-01'))['status']=='received'


@pytest.mark.parametrize('configured,expected',[(6000,6000),(4000,4000)])
def test_fresh_tick_request_and_precheck_use_same_single_buffer_rate(raw_profile,monkeypatch,configured,expected):
    p,root=raw_profile;p.online=True
    p._config=SimpleNamespace(finmind_token='test-token-never-persist',finmind_requests_per_hour=configured)
    from app import config,finmind,rate_limiter
    monkeypatch.setattr(config,'load_config',lambda:p._config)
    quota_calls=[];fetches=[]
    def limiter(limit):
        quota_calls.append(limit)
        return SimpleNamespace(get_stats=lambda:SimpleNamespace(remaining_requests=20,retry_after_seconds=0))
    def fetch(dataset,start,**kwargs):
        fetches.append((dataset,start,kwargs))
        return pd.DataFrame([dict(date='2026-09-01',stock_id='2330',price=100,volume=1)])
    monkeypatch.setattr(rate_limiter,'get_rate_limiter',limiter)
    monkeypatch.setattr(finmind,'fetch_dataset',fetch)
    result=p._raw(('2330','2026-09-01'))
    assert result['status']=='received' and len(fetches)==1
    dataset,start,kwargs=fetches[0]
    assert dataset=='TaiwanStockPriceTick' and str(start)=='2026-09-01'
    assert kwargs['data_id']=='2330' and kwargs['requests_per_hour']==expected
    assert kwargs['max_retries']==0 and kwargs['token']=='test-token-never-persist'
    assert quota_calls and set(quota_calls)=={expected}
    assert len(list((p.directory/'attempts').glob('*.json')))==1
    assert p._raw(('2330','2026-09-01'))==result and len(fetches)==1
    assert all('test-token-never-persist' not in path.read_text() for path in p.directory.rglob('*.json'))


def test_parent_pilot_reuse_is_normalized_before_publication_and_resume(raw_profile,monkeypatch):
    p,root=raw_profile;p.online=True
    _,raw,item=receipt(root,folder='daily')
    # Make this visible only through the inherited pilot-index path.
    for path in (root/'.cache/poc-daily-opportunities-20261004/data-v1/attempts').glob('*.json'):
        path.unlink()
    source=profiles.PILOT/'tapes-v1/receipts/2330-2026-09-01.json'
    write_json(source,item)
    p.receipt_hashes[str(source.relative_to(root))]=digest(source)
    from app import finmind,rate_limiter
    def no_online(*args,**kwargs):raise AssertionError('Parent pilot reuse must precede quota and network')
    monkeypatch.setattr(rate_limiter,'get_rate_limiter',no_online)
    monkeypatch.setattr(finmind,'fetch_dataset',no_online)
    first=p._raw(('2330','2026-09-01'))
    dest=p.directory/'receipts/2330-2026-09-01.json'
    assert first['raw_sha256']==digest(raw)
    assert first['reused_receipt']==str(source.relative_to(root))
    assert first['reused_receipt_sha256']==digest(source)
    assert p.refs[str(dest.relative_to(root))]==digest(dest)
    assert json.loads(dest.read_text())==first
    resumed=resumed_profile(p)
    assert resumed._raw(('2330','2026-09-01'))==first
    assert resumed.refs[str(source.relative_to(root))]==digest(source)
    assert not list((p.directory/'attempts').glob('*.json'))


def test_local_persistent_tick_budget_blocks_a_different_request_after_restart(raw_profile,monkeypatch):
    p,root=raw_profile;p.online=True;p.maximum=1
    p._config=SimpleNamespace(finmind_token='unit-test',finmind_requests_per_hour=6000)
    from app import config,finmind,rate_limiter
    monkeypatch.setattr(config,'load_config',lambda:p._config)
    monkeypatch.setattr(rate_limiter,'get_rate_limiter',lambda limit:SimpleNamespace(
        get_stats=lambda:SimpleNamespace(remaining_requests=20,retry_after_seconds=0)))
    calls=[]
    def fetch(dataset,start,**kwargs):
        calls.append(kwargs['data_id'])
        return pd.DataFrame([dict(date=str(start),stock_id=kwargs['data_id'],price=100,volume=1)])
    monkeypatch.setattr(finmind,'fetch_dataset',fetch)
    assert p._raw(('2330','2026-09-01'))['status']=='received'
    resumed=resumed_profile(p)
    assert resumed._raw(('2317','2026-09-01'))['status']=='request_budget_or_quota_paused'
    assert calls==['2330']
    assert len(list((p.directory/'attempts').glob('*.json')))==1
    assert not (p.directory/'receipts/2317-2026-09-01.json').exists()


@pytest.mark.parametrize('field,new',[('members',['2317']),('signal_date','2026-09-02')])
def test_profile_reuse_verifies_frozen_stock_and_signal_identity(field,new):
    p=broker.BranchAccountData.__new__(broker.BranchAccountData)
    e=event('historical');record=dict(event_id=e['event_id'],stock_id='2330',
        signal_date=e['signal_date'],available=True,poc_up=True)
    p.broker_evidence={'historical':evidence(e)}
    e[field]=new
    p.profile_index={'poc_red':{'historical':record}};p.used={arm:{} for arm in broker.ARMS}
    with pytest.raises(ValueError):p.profile('poc_red',e)


def test_profile_baseline_never_computes_unobserved_historical_candidate():
    p=broker.BranchAccountData.__new__(broker.BranchAccountData)
    p.profile_index={'poc_red':{}};p.used={arm:{} for arm in broker.ARMS}
    p.broker_evidence={'never-queried':evidence(event('never-queried'))}
    with pytest.raises(ValueError,match='Original baseline profile query changed'):
        p.profile('poc_red',event('never-queried'))


def test_changed_arm_may_compute_new_historical_path_without_mutating_frozen_profile():
    p=broker.BranchAccountData.__new__(broker.BranchAccountData)
    a,b=event('sealed'),event('new-path','2317')
    original=dict(event_id='sealed',stock_id='2330',signal_date=a['signal_date'],
                  available=True,poc_up=True,profile={'poc':100})
    p.profile_index={'poc_red':{'sealed':original}}
    p.broker_evidence={e['event_id']:evidence(e) for e in (a,b)}
    p.used={arm:{} for arm in broker.ARMS};seen=[]
    def compute(e):
        seen.append(deepcopy(e));return dict(event_id=e['event_id'],available=False,reason='not_requested')
    p.profiles=compute
    reused=p.profile('poc_known5_filter',a);reused['profile']['poc']=-1
    assert original['profile']['poc']==100 and seen==[]
    assert p.profile('poc_known5_filter',b)['reason']=='not_requested' and seen==[b]


def cached_execution(provider_fixture,*,meta_change=None):
    f=provider_fixture;p=broker.BranchAccountData(f.root)
    frame=pd.DataFrame([dict(stock_id='2330',date='2026-10-02',reference_price=100,limit_up=110,limit_down=90)])
    path=f.root/'.cache/poc-latest-20261003/execution-v1/2330-TaiwanStockPriceLimit.parquet'
    path.parent.mkdir(parents=True);frame.to_parquet(path,index=False)
    meta=dict(stock_id='2330',dataset='TaiwanStockPriceLimit',start='2018-01-01',end='2026-10-02',sha256=digest(path))
    if meta_change:meta.update(meta_change)
    write_json(path.with_suffix('.json'),meta)
    p.latest_refs={str(t.relative_to(f.root)):digest(t) for t in (path,path.with_suffix('.json'))}
    def never_fetch(*args):raise AssertionError('Bound execution cache needs no provider call')
    p.execution.finmind=never_fetch
    return p,path,frame


def test_bound_execution_cache_is_checked_and_returned_as_private_copy(provider_fixture):
    p,path,frame=cached_execution(provider_fixture)
    first=p.finmind('2330','TaiwanStockPriceLimit')
    assert first.equals(frame)
    first.loc[0,'limit_up']=999
    assert p.finmind('2330','TaiwanStockPriceLimit').equals(frame)
    assert str(path.relative_to(provider_fixture.root)) in p.refs


def test_bound_execution_cache_hash_error_never_falls_back(provider_fixture):
    p,path,_=cached_execution(provider_fixture);path.write_bytes(b'bad bytes')
    with pytest.raises(ValueError,match='source hash changed'):
        p.finmind('2330','TaiwanStockPriceLimit')


@pytest.mark.parametrize('change',[{'stock_id':'2317'},{'end':'2026-10-03'},{'dataset':'TaiwanStockDividend'}])
def test_bound_execution_receipt_identity_must_match_request(provider_fixture,change):
    p,_,_=cached_execution(provider_fixture,meta_change=change)
    with pytest.raises(ValueError,match='execution identity changed'):
        p.finmind('2330','TaiwanStockPriceLimit')


def test_shared_execution_quota_pauses_before_adapter_attempt(provider_fixture,monkeypatch):
    p=broker.BranchAccountData(provider_fixture.root,online=True)
    from app import config,rate_limiter
    monkeypatch.setattr(config,'load_config',lambda:SimpleNamespace(finmind_requests_per_hour=6000))
    def limiter(maximum):
        assert maximum==6000
        return SimpleNamespace(get_stats=lambda:SimpleNamespace(remaining_requests=2,retry_after_seconds=0))
    monkeypatch.setattr(rate_limiter,'get_rate_limiter',limiter)
    def no_attempt(*args):raise AssertionError('Quota pause must precede data adapter reservation')
    p.execution.finmind=no_attempt
    with pytest.raises(broker.ReplayDataUnavailable,match='shared quota paused'):
        p.finmind('2330','TaiwanStockPriceLimit')
    assert not list(p.execution.directory.rglob('attempts/*.json'))


def test_financial_fetch_uses_injected_single_buffer_rate_and_preserves_request_policy(provider_fixture,monkeypatch):
    from datetime import date
    from app import config,finmind
    p=broker.BranchAccountData(provider_fixture.root,online=True)
    assert p.execution.fetcher is broker.fetch_broker_execution
    calls=[];expected=object()
    def fetch(*args,**kwargs):
        calls.append((args,kwargs));return expected
    monkeypatch.setattr(config,'load_config',lambda:SimpleNamespace(finmind_requests_per_hour=6000))
    monkeypatch.setattr(finmind,'fetch_dataset',fetch)
    args=('TaiwanStockDividend',date(2018,1,1),date(2026,10,2))
    kwargs=dict(data_id='2330',token='unit-test-only',requests_per_hour=5400,max_retries=0,timeout=40)
    assert p.execution.fetcher(*args,**kwargs) is expected
    assert calls==[(args,dict(kwargs,requests_per_hour=6000))]
    assert kwargs['requests_per_hour']==5400


@pytest.mark.parametrize('status',['received','failed'])
def test_existing_execution_receipt_bypasses_quota_but_preserves_failure(provider_fixture,monkeypatch,status):
    p=broker.BranchAccountData(provider_fixture.root,online=True)
    path=p.execution.directory/'receipts/2330-TaiwanStockPriceLimit.json'
    write_json(path,dict(status=status))
    from app import config,rate_limiter
    def no_quota(*args):raise AssertionError('Existing receipt must be validated without reserving quota')
    monkeypatch.setattr(config,'load_config',no_quota)
    monkeypatch.setattr(rate_limiter,'get_rate_limiter',no_quota)
    calls=[]
    frame=pd.DataFrame([dict(stock_id='2330',date='2026-10-02',reference_price=100,limit_up=110,limit_down=90)])
    def cached(sid,dataset):
        calls.append((sid,dataset))
        if status=='failed':raise broker.ReplayDataUnavailable('Previous latest execution request failed; no automatic retry')
        return frame.copy()
    p.execution.finmind=cached
    if status=='received':assert p.finmind('2330','TaiwanStockPriceLimit').equals(frame)
    else:
        with pytest.raises(broker.ReplayDataUnavailable,match='Previous latest execution request failed'):
            p.finmind('2330','TaiwanStockPriceLimit')
    assert calls==[('2330','TaiwanStockPriceLimit')]
    assert not list(p.execution.directory.rglob('attempts/*.json'))


def odd_provider_stub():
    value=broker.BranchAccountData.__new__(broker.BranchAccountData)
    value.online=True
    value.refs={'original.json':'a'*64}
    value.supplemental_odds=SimpleNamespace(refs={'supplement.json':'b'*64})
    value.broker_odds=SimpleNamespace(refs={'acquisition.json':'c'*64})
    return value


def test_original_odd_success_preserves_authoritative_row_without_any_fallback(monkeypatch):
    value=odd_provider_stub();expected={'odd_shares':1200,'odd_high':102.,'odd_low':99.}
    def original(self,*args,**kwargs):return deepcopy(expected)
    def prohibited(*args,**kwargs):raise AssertionError('An authoritative original day cannot be replaced')
    monkeypatch.setattr(broker.LatestAccountData,'get_odd',original)
    value.supplemental_odds.get=prohibited;value.broker_odds.get=prohibited
    assert value.get_odd('2024-02-01','2330','TWSE')==expected
    assert value.refs=={'original.json':'a'*64}


@pytest.mark.parametrize('original_missing',['Frozen odd source absent: twse 2024-02-01',
                                           'Latest official odd day missing: twse 2026-09-10'])
def test_only_missing_market_day_uses_isolated_odd_acquisition_and_binds_both_sources(monkeypatch,original_missing):
    from skills.poc_broker_odd import OddMarketDayMissing
    value=odd_provider_stub();seen=[];engine=object()
    def original(self,*args,**kwargs):raise broker.ReplayDataUnavailable(original_missing)
    def supplement(*args):
        seen.append(('supplement',args))
        raise OddMarketDayMissing('Supplementary official odd day missing: twse 2024-02-01')
    def acquire(*args):
        seen.append(('acquire',args));return {'odd_shares':1500,'odd_high':101.}
    monkeypatch.setattr(broker.LatestAccountData,'get_odd',original)
    value.supplemental_odds.get=supplement;value.broker_odds.get=acquire
    assert value.get_odd('2024-02-01','2330','TWSE',engine)=={'odd_shares':1500,'odd_high':101.}
    assert seen==[('supplement',('2024-02-01','2330','TWSE')),
                  ('acquire',('2024-02-01','2330','TWSE',engine))]
    assert value.refs=={'original.json':'a'*64,'supplement.json':'b'*64,'acquisition.json':'c'*64}


@pytest.mark.parametrize('stage,reason',[
    ('original','Frozen odd stock absent: 2330 2024-02-01'),
    ('original','Conflicting frozen odd rows: 2330 2024-02-01'),
    ('supplement','Supplementary odd stock absent: 2330 2024-02-01'),
    ('supplement','Conflicting supplementary odd rows: 2330 2024-02-01'),
    ('supplement','Supplementary odd receipt chain is invalid'),
])
def test_odd_stock_gaps_conflicts_and_bad_receipts_never_trigger_acquisition(monkeypatch,stage,reason):
    value=odd_provider_stub();seen=[]
    def original(self,*args,**kwargs):
        raise broker.ReplayDataUnavailable(reason if stage=='original' else 'Frozen odd source absent: twse 2024-02-01')
    def supplement(*args):
        seen.append('supplement');raise broker.ReplayDataUnavailable(reason)
    def prohibited(*args,**kwargs):raise AssertionError('Invalid evidence must not be replaced with a new request')
    monkeypatch.setattr(broker.LatestAccountData,'get_odd',original)
    value.supplemental_odds.get=supplement;value.broker_odds.get=prohibited
    with pytest.raises(broker.ReplayDataUnavailable,match=reason):
        value.get_odd('2024-02-01','2330','twse')
    assert seen==([] if stage=='original' else ['supplement'])
    if stage=='supplement':assert value.refs['supplement.json']=='b'*64


def test_failed_odd_acquisition_preserves_failure_and_binds_attempt_evidence(monkeypatch):
    from skills.poc_broker_odd import OddMarketDayMissing
    value=odd_provider_stub();calls=[]
    def original(self,*args,**kwargs):raise broker.ReplayDataUnavailable('Frozen odd source absent: twse 2024-02-01')
    def supplement(*args):raise OddMarketDayMissing('Supplementary official odd day missing: twse 2024-02-01')
    def failure(*args):
        calls.append(args)
        value.broker_odds.refs['failed-attempt.json']='d'*64
        raise broker.ReplayDataUnavailable('Official odd provider failed; retry prohibited')
    monkeypatch.setattr(broker.LatestAccountData,'get_odd',original)
    value.supplemental_odds.get=supplement;value.broker_odds.get=failure
    with pytest.raises(broker.ReplayDataUnavailable,match='retry prohibited'):
        value.get_odd('2024-02-01','2330','twse')
    assert calls==[('2024-02-01','2330','twse',None)]
    assert value.refs['supplement.json']=='b'*64 and value.refs['failed-attempt.json']=='d'*64
