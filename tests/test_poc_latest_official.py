from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from scripts import prepare_poc_latest_official as m
from skills.official_daily_acquisition import create_plan,request_item,digest,encoded,AcquisitionBlocked


def body(market='TWSE',day='2026-09-10'):
    if market=='TWSE':return dict(stat='OK',date=day.replace('-',''),type='ALLBUT0999',tables=[dict(
        fields=['證券代號','證券名稱','開盤價','最高價','最低價','收盤價','成交股數'],
        notes=['本統計資訊含一般、零股、盤後定價、鉅額交易'],data=[['2330','台積電','100','110','90','105','100,050']])])
    return dict(stat='ok',date=day.replace('-',''),tables=[dict(fields=['代號','名稱','開盤','最高','最低','收盤','成交股數'],
        title='上櫃股票每日收盤行情(不含定價)',date='115/'+day[5:].replace('-','/'),
        category='所有證券(不含權證、牛熊證)',totalCount=1,data=[['2330','測試','100','110','90','105','100,000']])])


class Fake:
    def __init__(self,responses):self.now=datetime(2026,10,3,tzinfo=timezone.utc).timestamp();self.calls=[];self.responses=responses
    def clock(self):return self.now
    def sleep(self,delta):self.now+=delta
    def get(self,url,**kw):return self.request('GET',url,**kw)
    def request(self,method,url,**kw):
        self.calls.append((self.now,method,url,kw));assert kw['allow_redirects'] is False
        r=self.responses.pop(0)
        if isinstance(r,Exception):raise r
        return r


def response(payload=None,status=200):return SimpleNamespace(status_code=status,content=json.dumps(payload or body()).encode(),history=[])


def setup(tmp_path,responses):
    cache=tmp_path/'cache';items=[request_item('TWSE',d) for d in ('2026-09-10','2026-09-11')]
    create_plan(tmp_path,cache/'daily',items,{})
    m.sealed(cache/'actions-plan.json',dict(schema='poc_latest_actions_plan_v1',entries=m.action_items()))
    runtime=Fake(responses);auth=cache/'authorization.json'
    m.sealed(auth,dict(schema='official_daily_authorization_v1',scope='poc_latest_official_extension',user_request='go',
        task_scope=m.SCOPE,security_bypass_authorized=False,created_at=m.stamp(runtime.clock),
        plan_sha256=digest(cache/'daily/plan.json'),actions_plan_sha256=digest(cache/'actions-plan.json')))
    c=m.CurrentRequestAcquisition(tmp_path,cache/'daily',authorization_path=auth,session=runtime,
                                  clock=runtime.clock,sleep=runtime.sleep)
    hold=tmp_path/'.cache/official-origin-holds/www.twse.com.tw.json';hold.parent.mkdir(parents=True)
    hold.write_text(json.dumps(dict(status='blocked',observed_at='2026-10-02T00:00:00Z',evidence_sha256={})))
    return c,runtime,items,hold


def test_current_go_probe_preserves_hold_and_reuses_success(tmp_path):
    c,r,items,hold=setup(tmp_path,[response(),response(body(day='2026-09-11'))]);h=digest(hold)
    first=c.probe(items[0],allow_probe=True);assert first['accepted']
    assert c.probe(items[0],allow_probe=True)==first
    assert c.fetch(items[1])['accepted'];assert len(r.calls)==2 and r.calls[1][0]-r.calls[0][0]>=3.1
    assert digest(hold)==h and first['volume_scope']=='all_daily_sessions'


@pytest.mark.parametrize('mutation',['old_quote','scope','expired','plan'])
def test_authorization_exact_current_scope(tmp_path,mutation):
    c,r,items,_=setup(tmp_path,[]);a=m.read(c.authorization)
    if mutation=='old_quote':a['user_request']='但仍缺 3,745張官方市場日表 你可以補？'
    if mutation=='scope':a['task_scope']['end']='2026-10-05'
    if mutation=='expired':a['created_at']='2026-10-01T00:00:00+00:00'
    if mutation=='plan':a['plan_sha256']='0'*64
    c.authorization.write_text(json.dumps(a))
    with pytest.raises(AcquisitionBlocked):c.probe(items[0],allow_probe=True)
    assert not r.calls


def test_security_denial_never_retries_or_continues(tmp_path):
    c,r,items,hold=setup(tmp_path,[response(status=428)]);h=digest(hold)
    assert not c.probe(items[0],allow_probe=True)['accepted']
    assert not c.fetch(items[0])['accepted']
    with pytest.raises(AcquisitionBlocked):c.fetch(items[1])
    assert len(r.calls)==1 and digest(hold)==h


def test_calendar_respects_nontrading_weekdays():
    x=m.dates_from_calendar(pd.DataFrame({'date':['2026-09-09','2026-09-10','2026-09-24','2026-09-29','2026-10-02']}))
    assert x==['2026-09-10','2026-09-24','2026-09-29','2026-10-02']


def action_fixture(tmp_path):
    kind='twse_capital_reduction';p=tmp_path/'old-action.json'
    fields=['恢復買賣日期','股票代號','名稱','停止買賣前收盤價格','恢復買賣參考價','漲停','跌停','基準','除權','原因']
    payload=dict(stat='OK',fields=fields,data=[['115/09/14','2330','台積電','100','200','220','180','200','200','彌補虧損']],strDate='20260910',endDate='20261002')
    p.write_text(json.dumps(payload));inventory={'old_actions':{kind:{'path':'old-action.json'}}}
    item=next(i for i in m.action_items() if i['kind']==kind)
    return inventory,item,payload


def test_action_schema_and_date_cannot_be_silently_skipped(tmp_path):
    inv,item,p=action_fixture(tmp_path)
    assert len(m.parse_actions(tmp_path,inv,item,p))==1
    p['data'][0][0]='115/09/09'
    with pytest.raises(Exception):m.parse_actions(tmp_path,inv,item,p)
    p['data']=[];p['fields']=[]
    with pytest.raises(AcquisitionBlocked):m.parse_actions(tmp_path,inv,item,p)


def test_explicit_empty_is_different_from_missing_schema(tmp_path):
    inv,item,p=action_fixture(tmp_path)
    assert m.parse_actions(tmp_path,inv,item,{'stat':'沒有符合條件的資料!'}).empty
    with pytest.raises(AcquisitionBlocked):m.parse_actions(tmp_path,inv,item,{'stat':'OK'})


def test_action_probe_same_shared_lock_and_no_retries(tmp_path):
    inv,item,p=action_fixture(tmp_path);c,r,_,hold=setup(tmp_path,[response(p)]);h=digest(hold)
    a=m.action_probe(c,inv,item);assert a['accepted'] and a['events']==1
    assert m.action_probe(c,inv,item)==a and len(r.calls)==1 and digest(hold)==h


def test_action_denial_blocks_other_endpoint(tmp_path):
    inv,item,p=action_fixture(tmp_path);c,r,_,hold=setup(tmp_path,[response(status=403)])
    assert not m.action_probe(c,inv,item)['accepted']
    other=next(i for i in m.action_items() if i['kind']=='twse_ex_rights')
    with pytest.raises(AcquisitionBlocked):m.action_probe(c,inv,other)
    assert len(r.calls)==1


def test_action_proof_expires_during_throttle_no_dispatch(tmp_path):
    inv,item,p=action_fixture(tmp_path);c,r,_,_=setup(tmp_path,[])
    r.now+=86399
    sp,_,_=c._origin_paths(item);sp.parent.mkdir(parents=True,exist_ok=True)
    sp.write_text(json.dumps({'last_start_epoch':r.now}))
    with pytest.raises(AcquisitionBlocked):m.action_probe(c,inv,item)
    assert not r.calls and not (c.cache.parent/'actions'/item['kind']/'attempt.json').exists()


def test_missing_market_and_event_sources_are_not_complete(tmp_path,monkeypatch):
    c,r,items,_=setup(tmp_path,[response()]);c.probe(items[0],allow_probe=True)
    inv=dict(days=['2026-09-10','2026-09-11'],required_market_days=4,reusable=[],source_sha256={})
    monkeypatch.setattr(m,'initialize',lambda root,cache:inv)
    report=m.normalize(tmp_path,c.cache.parent)
    assert report['accepted_market_days']==1 and len(report['missing_market_days'])==3
    assert report['daily_tables_extension_complete'] is False
    assert report['corporate_events_extension_complete'] is False and report['event_count']==0
    df=pd.read_parquet(tmp_path/report['normalized_path'])
    assert df['volume_scope'].tolist()==['all_daily_sessions']
    assert set(df['date'])=={'2026-09-10'} and len(df)==1
    assert all(digest(tmp_path/p)==h for p,h in report['source_sha256'].items())
    assert all(digest(tmp_path/p)==h for p,h in report['output_sha256'].items())


def test_changed_daily_raw_rejected_during_normalization(tmp_path,monkeypatch):
    c,r,items,_=setup(tmp_path,[response()]);receipt=c.probe(items[0],allow_probe=True)
    (tmp_path/receipt['raw_path']).write_text('{}')
    monkeypatch.setattr(m,'initialize',lambda root,cache:dict(days=['2026-09-10'],required_market_days=2,reusable=[],source_sha256={}))
    with pytest.raises(AcquisitionBlocked):m.normalize(tmp_path,c.cache.parent)


def test_action_attached_response_denial_is_preserved(tmp_path):
    import requests
    inv,item,p=action_fixture(tmp_path)
    exc=requests.ConnectionError('attached',response=response(status=403))
    c,r,_,_=setup(tmp_path,[exc]);result=m.action_probe(c,inv,item)
    assert not result['accepted'] and result['http_status']==403 and result['security_denied'] is True
    assert result['exception_response_present'] is True and result['status']=='origin_stopped'
    assert m.action_probe(c,inv,item)==result and len(r.calls)==1
