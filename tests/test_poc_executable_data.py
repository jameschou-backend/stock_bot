"""No-network contract tests for executable POC data and durable provenance."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skills import poc_executable_data as data


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def raw(sid='2330', day='2024-01-02'):
    return pd.DataFrame(dict(date=[day]*3, stock_id=[sid]*3,
        deal_price=[100.,101.,100.5], volume=[100,200,100],
        Time=['09:00:00','10:00:00','13:30:00'], TickType=[1,1,2]))


def receipt(root, *, sid='2330', day='2024-01-02', status='received', frame=None):
    folder=root/'.cache/poc-broker-account-20261004/profiles-v1'
    path=folder/'receipts'/(sid+'-'+day+'.json')
    value=dict(query=dict(dataset='TaiwanStockPriceTick',data_id=sid,start_date=day),status=status)
    if status=='received':
        source=folder/'raw'/(sid+'-'+day+'.parquet');source.parent.mkdir(parents=True,exist_ok=True)
        (raw(sid,day) if frame is None else frame).to_parquet(source,index=False)
        value.update(raw_path=str(source.relative_to(root)),raw_sha256=data.sha(source))
    write(path,value)
    return path,value


def store(root, **kwargs):
    return data.TickReceipts(root,root/'.cache/new/board-v1',maximum=2,**kwargs)


def config():
    return SimpleNamespace(finmind_requests_per_hour=6000,finmind_token='fixture-not-a-secret')


def limiter(remaining=5400, delay=0, seen=None):
    def factory(requested):
        if seen is not None:seen.append(requested)
        return SimpleNamespace(get_stats=lambda:SimpleNamespace(remaining_requests=remaining,retry_after_seconds=delay))
    return factory


def test_reuse_validates_raw_and_receipt_chain_on_restart_without_quota(tmp_path):
    source,item=receipt(tmp_path)
    noquota=lambda _:pytest.fail('Cached evidence must not check or reserve quota')
    first=store(tmp_path,online=True,limiter=noquota)
    result=first.get('2330','2024-01-02')
    assert result['reused_receipt_sha256']==data.sha(source)
    assert result['raw_sha256']==item['raw_sha256']
    assert first.calls==0
    again=store(tmp_path,online=True,limiter=noquota)
    assert again.get('2330','2024-01-02')==result
    assert str(source.relative_to(tmp_path)) in again.refs
    assert item['raw_path'] in again.refs
    source.write_text(source.read_text()+' ')
    with pytest.raises(ValueError,match='hash changed'):
        store(tmp_path).get('2330','2024-01-02')


@pytest.mark.parametrize('broken',['query','raw','copied_raw','cycle'])
def test_reuse_rejects_query_raw_copy_and_recursive_provenance_change(tmp_path,broken):
    source,item=receipt(tmp_path)
    if broken=='query':
        item['query']['data_id']='2317';write(source,item)
    elif broken=='raw':
        (tmp_path/item['raw_path']).write_bytes(b'invalid parquet')
    elif broken=='cycle':
        item['reused_receipt']=str(source.relative_to(tmp_path));write(source,item)
    else:
        s=store(tmp_path);s.get('2330','2024-01-02')
        target=s.directory/'receipts/2330-2024-01-02.json';v=json.loads(target.read_text())
        v['status']='empty';write(target,v)
    with pytest.raises(ValueError):store(tmp_path).get('2330','2024-01-02')


def test_prior_failure_no_fetch_or_overwrite(tmp_path):
    p,_=receipt(tmp_path,status='provider_error');before=data.sha(p)
    s=store(tmp_path,online=True,fetcher=lambda *a,**k:pytest.fail('Must not retry an old error'))
    assert s.get('2330','2024-01-02')['status']=='prior_cached_attempt_unavailable'
    assert data.sha(p)==before and s.calls==0


def test_frozen_reuse_index_binds_all_inspected_versions_on_resume(tmp_path,monkeypatch):
    import scripts.prepare_volume_profile as prepare
    monkeypatch.setattr(prepare,'ROOT',tmp_path)
    query=dict(dataset='TaiwanStockPriceTick',data_id='2330',start_date='2024-01-02')
    versions=[]
    for i in range(2):
        path=tmp_path/f'old/{i}.parquet';path.parent.mkdir(exist_ok=True)
        raw().to_parquet(path,index=False);meta=path.with_suffix('.json')
        write(meta,dict(query=query,raw_sha256=data.sha(path)))
        versions.append(dict(path=str(path),raw_sha256=data.sha(path),query=query,
            metadata_path=str(meta),metadata_sha256=data.sha(meta)))
    entries={'2330-2024-01-02':dict(content_conflict=False,sources=versions)}
    index=tmp_path/'old/index.json';write(index,entries)
    s=store(tmp_path,reuse_index=entries,reuse_index_ref=(index,data.sha(index)))
    assert s.get('2330','2024-01-02')['status']=='cached'
    resumed=store(tmp_path)
    assert resumed.get('2330','2024-01-02')['status']=='cached'
    assert {str(Path(x[k]).relative_to(tmp_path)) for x in versions for k in ('path','metadata_path')} <= resumed.refs.keys()
    assert 'old/index.json' in resumed.refs
    Path(versions[1]['metadata_path']).write_text('{}')
    with pytest.raises(ValueError,match='hash changed'):
        store(tmp_path).get('2330','2024-01-02')


def test_orphan_and_known_versions_conflict_cannot_hide_behind_another_cache(tmp_path):
    _,item=receipt(tmp_path)
    s=store(tmp_path,online=True)
    p=s.directory/'attempts/2330-2024-01-02.json';write(p,dict(query=item['query']))
    assert s.get('2330','2024-01-02')['status']=='orphaned_started_attempt'
    other=data.TickReceipts(tmp_path,tmp_path/'.cache/other',maximum=2,online=True,
        reuse_index={'2330-2024-01-02':{'content_conflict':True}})
    assert other.get('2330','2024-01-02')['status']=='cached_versions_conflict'


@pytest.mark.parametrize('remaining,delay',[(0,0),(3,0),(100,60)])
def test_shared_quota_pause_leaves_no_reservation(tmp_path,remaining,delay):
    seen=[]
    s=store(tmp_path,online=True,config=config(),limiter=limiter(remaining,delay,seen),
        fetcher=lambda *a,**k:pytest.fail('No HTTP when quota is paused'))
    assert s.get('2330','2024-01-02')['status']=='request_budget_or_quota_paused'
    assert seen==[6000] and s.calls==0
    assert not list(s.directory.glob('attempts/*.json'))


def test_new_fetch_uses_shared_sponsor_limit_no_retry_and_persistent_budget(tmp_path):
    calls=[]
    def fetch(*args,**kwargs):
        calls.append((args,kwargs));return raw(kwargs['data_id'],str(args[1]))
    s=store(tmp_path,online=True,config=config(),limiter=limiter(),fetcher=fetch)
    for sid in ('2330','2317'):
        assert s.get(sid,'2024-01-02')['status']=='received'
    args,kw=calls[0]
    assert args==('TaiwanStockPriceTick',date(2024,1,2))
    assert kw['requests_per_hour']==6000 and kw['max_retries']==0 and kw['timeout']==40
    restarted=store(tmp_path,online=True,config=config(),limiter=limiter(),fetcher=fetch)
    assert restarted.get('2454','2024-01-02')['status']=='request_budget_or_quota_paused'
    assert restarted.get('2330','2024-01-02')['status']=='received'
    assert len(calls)==2 and len(list(s.directory.glob('attempts/*.json')))==2


def test_empty_tape_is_durable_unavailable_not_zero_fill(tmp_path):
    s=store(tmp_path,online=True,config=config(),limiter=limiter(),fetcher=lambda *a,**k:raw().iloc[:0])
    assert s.get('2330','2024-01-02')['status']=='empty'
    s.fetcher=lambda *a,**k:pytest.fail('Empty response may not be refetched')
    assert s.get('2330','2024-01-02')['status']=='empty' and s.calls==1


def test_failure_receipt_resumes_without_retry(tmp_path):
    from app.finmind import FinMindError
    def fail(*a,**k):raise FinMindError('fixture failure')
    s=store(tmp_path,online=True,config=config(),limiter=limiter(),fetcher=fail)
    assert s.get('2330','2024-01-02')['status']=='provider_error'
    restarted=store(tmp_path,online=True,fetcher=lambda *a,**k:pytest.fail('No automatic retry'))
    assert restarted.get('2330','2024-01-02')['status']=='provider_error'
    assert len(list(s.directory.glob('attempts/*.json')))==1


def board(tmp_path, *, exact=None, official_changes=None):
    receipt(tmp_path)
    official=dict(market='TWSE',open=100.,high=101.,low=100.,close=100.5,volume=410000.,
                  source_id='fixture',volume_scope='all_daily_sessions')
    official.update(official_changes or {})
    profiles=SimpleNamespace(refs={},_source=lambda *a:(official,exact))
    obj=data.BoardTicks.__new__(data.BoardTicks)
    obj.profiles=profiles;obj.refs={};obj.queries=[];obj.audits={};obj.loaded={};obj.store=store(tmp_path)
    return obj


def test_board_normalizes_keeps_times_and_does_not_promote_aggregate_bound(tmp_path):
    obj=board(tmp_path);frame,h=obj.get('2330','2024-01-02','twse')
    assert list(frame.shares)==[100000,200000,100000]
    assert list(frame.time)==list(pd.to_timedelta(['09:00:00','10:00:00','13:30:00']))
    quote=dict(open=100.,high=101.,low=100.,close=100.5,volume=410000.)
    result=obj.audit_day('2330','2024-01-02','TWSE',frame,quote)
    assert result['raw_sha256']==h and result['ordinary_volume_matched'] is False
    assert result['tick_sequence_complete'] is False and result['own_order_fill_proven'] is False
    frame.loc[0,'shares']=1
    with pytest.raises(ValueError,match='differs from bound'):obj.audit_day('2330','2024-01-02','TWSE',frame,quote)
    assert obj.get('2330','2024-01-02','TWSE')[0].shares.iloc[0]==100000


@pytest.mark.parametrize('change',[{'high':105.},{'volume':399999.},{'market':'TPEX'}])
def test_board_rejects_official_conflicts_and_wrong_market(tmp_path,change):
    with pytest.raises(data.ReplayDataUnavailable):board(tmp_path,official_changes=change).get('2330','2024-01-02','TWSE')


def test_board_missing_and_explicit_ordinary_conflict_stop_instead_of_zero_order(tmp_path):
    obj=board(tmp_path)
    exact=dict(stock_id='2330',date='2024-01-02',market='TWSE',shares=399000,amount_cents=4025000000,
        open_cents=10000,high_cents=10100,low_cents=10000,close_cents=10050)
    obj.profiles._source=lambda *a:(dict(market='TWSE',open=100.,high=101.,low=100.,close=100.5,
        volume=410000.,source_id='fixture',volume_scope='all_daily_sessions'),exact)
    with pytest.raises(data.ReplayDataUnavailable,match='quality conflict'):obj.get('2330','2024-01-02','TWSE')
    assert obj.refs and obj.audits['2330-2024-01-02']['status']=='official_ordinary_aggregate_conflict'
    with pytest.raises(data.ReplayDataUnavailable,match='not_requested'):obj.get('2317','2024-01-02','TWSE')


def test_profile_preserves_original_known_or_unknown_and_new_query_only_uses_old_algorithm():
    p=data.ExecutableAccountData.__new__(data.ExecutableAccountData)
    event=dict(event_id='frozen',members=['2330'],signal_date='2024-01-02',entry_date='2024-01-03')
    original=dict(event_id='frozen',stock_id='2330',signal_date='2024-01-02',available=False,poc_up=None,reason='quality')
    p.original_profiles={'frozen':original};p.used={data.ARM:{}};p.refs={}
    class Dynamic:
        refs={}
        def __call__(self,e):return dict(event_id=e['event_id'],available=True,poc_up=True)
    p.profiles=Dynamic()
    assert p.profile(data.ARM,event)==original
    result=p.profile(data.ARM,event);result['reason']='mutated'
    assert original['reason']=='quality' and p.used[data.ARM]['frozen']['reason']=='quality'
    wrong=deepcopy(event);wrong['members']=['2317']
    with pytest.raises(ValueError,match='stock/date mismatch'):p.profile(data.ARM,wrong)
    new=deepcopy(event);new['event_id']='new'
    assert p.profile(data.ARM,new)['poc_up'] is True
    new['entry_date']=new['signal_date']
    with pytest.raises(ValueError):p.profile(data.ARM,new)


def test_financial_gateway_applies_reserve_only_once(monkeypatch):
    from app import config as conf, finmind
    seen=[]
    monkeypatch.setattr(conf,'load_config',config)
    monkeypatch.setattr(finmind,'fetch_dataset',lambda *a,**k:seen.append((a,k)))
    data.fetch_execution('TaiwanStockDividend',date(2018,1,1),requests_per_hour=5400,max_retries=0)
    assert seen[0][1]['requests_per_hour']==6000 and seen[0][1]['max_retries']==0


def test_after_hours_is_only_odd_route_and_source_refs_survive_failure(monkeypatch):
    from skills import poc_executable_odd
    p=data.ExecutableAccountData.__new__(data.ExecutableAccountData);p.refs={}
    p.after_hours=SimpleNamespace(refs={'fake-raw':'hash'})
    def fail(*args):raise data.ReplayDataUnavailable('missing auction')
    p.after_hours.get=fail
    with pytest.raises(data.ReplayDataUnavailable):p.get_odd('2024-01-02','2330','twse')
    assert p.refs=={'fake-raw':'hash'}


def financial_donor(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'ROOT',tmp_path)
    p=data.ExecutableAccountData.__new__(data.ExecutableAccountData)
    p.refs={};p.execution=SimpleNamespace(_old=lambda *a:(None,{}),refs={})
    folder=tmp_path/'.cache/poc-broker-account-20261004/execution-v1'
    key='2330-TaiwanStockPriceLimit';start='2018-01-01'
    query=dict(stock_id='2330',dataset='TaiwanStockPriceLimit',start=start,end=data.END)
    frame=pd.DataFrame(dict(stock_id=['2330'],date=['2024-01-02'],limit_down=[90.],limit_up=[110.],reference_price=[100.]))
    dest=folder/(key+'.parquet');raw_path=folder/'raw'/(key+'.parquet')
    raw_path.parent.mkdir(parents=True);frame.to_parquet(raw_path,index=False);frame.to_parquet(dest,index=False)
    attempt=folder/'attempts'/(key+'.json');write(attempt,dict(query=query,status='started'))
    receipt_path=folder/'receipts'/(key+'.json')
    write(receipt_path,dict(query=query,status='received',attempt_sha256=data.sha(attempt),raw_sha256=data.sha(raw_path),raw_path=str(raw_path.relative_to(tmp_path))))
    write(dest.with_suffix('.json'),dict(stock_id='2330',dataset='TaiwanStockPriceLimit',start=start,end=data.END,
        historical_rights_frozen_through=None,old_source_sha256={},new_query_start=start,preparation_rule='append_only',sha256=data.sha(dest)))
    return p,frame,dest,receipt_path


def test_financial_reuse_reconstructs_from_query_bound_raw_and_binds_ancestors(tmp_path,monkeypatch):
    p,frame,dest,receipt_path=financial_donor(tmp_path,monkeypatch)
    assert p._broker_financial('2330','TaiwanStockPriceLimit').equals(frame)
    assert len(p.refs)==5
    assert all(data.sha(tmp_path/k)==v for k,v in p.refs.items())
    frame.loc[0,'limit_up']=111;frame.to_parquet(dest,index=False)
    meta=json.loads(dest.with_suffix('.json').read_text());meta['sha256']=data.sha(dest);write(dest.with_suffix('.json'),meta)
    fresh=data.ExecutableAccountData.__new__(data.ExecutableAccountData);fresh.refs={};fresh.execution=p.execution
    with pytest.raises(ValueError,match='reconstruction differs'):fresh._broker_financial('2330','TaiwanStockPriceLimit')


def test_financial_incomplete_prior_attempt_blocks_unrequested_retry(tmp_path,monkeypatch):
    p,_,dest,_=financial_donor(tmp_path,monkeypatch);dest.unlink()
    with pytest.raises(data.ReplayDataUnavailable,match='no automatic retry'):
        p._broker_financial('2330','TaiwanStockPriceLimit')


def test_network_counts_include_only_current_odd_attempts_and_financial_dispatches():
    p=data.ExecutableAccountData.__new__(data.ExecutableAccountData)
    p.board_ticks=SimpleNamespace(store=SimpleNamespace(calls=3))
    p.profiles=SimpleNamespace(raw_store=SimpleNamespace(calls=4))
    p._financial_calls=2;p._odd_initial_attempts=5
    p.after_hours=SimpleNamespace(snapshot=lambda:dict(attempted_calls=7))
    assert p.finmind_calls==9 and p.network_calls==11
