from copy import deepcopy
from datetime import timedelta
from decimal import Decimal as D
import pytest
from app import forward_simulation as s,forward_journal as j,forward_portfolio as p,forward_comparison as c,forward_automation as a,forward_market_quotes as m
from tests.test_forward_portfolio import setup,buy,clock


def initialized(tmp_path):
    original=tmp_path/'original';benchmark=tmp_path/'benchmark';signals=tmp_path/'signals'
    with j.connection(original) as con:setup(con);buy(con,n=999)
    c.seed_benchmark(original,benchmark,clock())
    with j.connection(signals) as con:j.initialize(con,clock())
    root=tmp_path/'sim'
    s.initialize(root,original,benchmark,clock(),signals)
    return root,original,benchmark


def order():return dict(stock_id='2492',market='tse',side='buy',channel='odd',qty=999,filled=0,limit_price='100',recorded_at='2026-09-11T20:00:00+08:00',session='2026-09-14')

def quote(seconds=0,volume=1000):
    at=clock('2026-09-14',10)()+timedelta(seconds=seconds)
    return dict(stock_id='2492',market='tse',channel='odd',quantity_unit='shares',quote_at=at.isoformat(),retrieved_at=at.isoformat(),asks=[dict(price='99',shares=5000)],bids=[dict(price='98.9',shares=5000)],volume_shares=volume)


def observation(seconds=0,volume=1000):
    body=dict(status='ok',quote=quote(seconds,volume));return dict(body=body,hash=j.digest(body))


def test_sealed_initialization_preserves_sources_and_blocks_late_or_changed_rules(tmp_path,monkeypatch):
    root,source,bench=initialized(tmp_path)
    before=(s.read(source),s.read(bench))
    assert s.initialize(root,source,bench,clock())==root
    assert (s.read(source),s.read(bench))==before
    assert (root/'signals.sqlite3').exists()
    with pytest.raises(ValueError,match='開盤'):s.initialize(tmp_path/'late',source,bench,clock('2026-09-14',10))
    monkeypatch.setitem(s.RULES,'adverse_slippage','.01')
    with j.connection(root/'strategy.sqlite3') as con:
        with pytest.raises(ValueError,match='已變更'):s.verify(con)


def test_two_observations_capacity_slippage_partial_costs():
    now=clock('2026-09-14',10)()+timedelta(seconds=20)
    result=s.proposal(order(),quote(),quote(20,3000),now)
    assert result['qty']==200 and result['price']=='99.1' and result['fee']=='28' and result['tax']=='0'
    assert result['capacity_shares']==200
    # A volume increase is not enough without visible shares.
    thin=quote(20,3000);thin['asks'][0]['shares']=90
    assert s.proposal(order(),quote(),thin,now)['qty']==9
    assert s.proposal(dict(order(),channel='board'),dict(quote(),channel='board'),dict(quote(20,3000),channel='board'),now)['qty']==0


@pytest.mark.parametrize('case',['first','stale','future','gap','no_volume','reverse_volume','wrong_day','slippage','delay','stale_previous'])
def test_no_invented_fills(case):
    o=order();prev=quote();cur=quote(20,3000);now=clock('2026-09-14',10)()+timedelta(seconds=20)
    if case=='first':prev=None
    if case=='stale':now+=timedelta(seconds=30)
    if case=='future':now-=timedelta(seconds=30)
    if case=='gap':cur=quote(200,3000);now+=timedelta(seconds=180)
    if case=='no_volume':cur['volume_shares']=1000
    if case=='reverse_volume':cur['volume_shares']=999
    if case=='wrong_day':o['session']='2026-09-15'
    if case=='slippage':cur['asks'][0]['price']='100'
    if case=='delay':o['recorded_at']=now.isoformat()
    if case=='stale_previous':prev['retrieved_at']=(clock('2026-09-14',10)()+timedelta(seconds=16)).isoformat()
    assert s.proposal(o,prev,cur,now)['qty']==0


def test_observation_idempotence_and_original_isolation(tmp_path):
    root,original,bench=initialized(tmp_path);path=root/'strategy.sqlite3';before=s.read(original)
    first=s.match(path,observation(),clock('2026-09-14',10));assert not first['fills']
    now=lambda:clock('2026-09-14',10)()+timedelta(seconds=20)
    second=s.match(path,observation(20,3000),now);assert len(second['fills'])==1
    assert s.match(path,observation(20,3000),now)==second
    state=p.summary(path,now)
    assert state['fill_count']==1 and state['holdings'][0]['qty']==200
    assert state['cash']=='980152.0' and s.read(original)==before
    fills=[r for r in state['rows'] if r['kind']=='fill']
    assert fills[0]['body']['evidence']['classification']==s.RULES['classification']


def test_regular_and_odd_provider_units_are_distinct():
    row=dict(c='0050',ex='tse',ch='0050.tw',d='20260914',tlong=str(int(clock('2026-09-14',10)().timestamp()*1000)),a='99_',f='4_',b='98_',g='5_',v='10')
    raw=dict(rtcode='0000',msgArray=[row]);at=clock('2026-09-14',10)().isoformat()
    board=m.parse(raw,'tse','0050','board',at);odd=m.parse(raw,'tse','0050','odd',at)
    assert board['asks'][0]['shares']==4000 and board['volume_shares']==10000
    assert odd['asks'][0]['shares']==4 and odd['volume_shares']==10
    row['v']='-'
    with pytest.raises(ValueError):m.parse(raw,'tse','0050','board',at)


def test_closed_market_and_outside_window_do_not_call_network(tmp_path,monkeypatch):
    root,original,bench=initialized(tmp_path)
    monkeypatch.setattr(a,'calendar_day',lambda day:False)
    def never(*args,**kw):raise AssertionError('network should not run')
    assert a.run(root,clock=clock('2026-09-12',10),fetcher=never,sleeper=never)['status']=='closed_market'
    monkeypatch.setattr(a,'calendar_day',lambda day:True)
    assert a.run(root,clock=clock('2026-09-14',7),fetcher=never,sleeper=never)['status']=='outside_window'
    assert len(s.read(original))==4


def test_stage_retry_and_repeated_success_do_not_redo_work(tmp_path):
    calls=[]
    def ok():calls.append(1);return {'done':True}
    assert a._stage(tmp_path,'daily',ok,clock())['status']=='ok'
    assert a._stage(tmp_path,'daily',ok,clock())['status']=='already_done' and len(calls)==1
    def fail():calls.append(2);raise ValueError('waiting evidence')
    assert a._stage(tmp_path,'blocked',fail,clock())['status']=='blocked'
    assert a._stage(tmp_path,'blocked',fail,clock())['status']=='cooldown' and calls==[1,2]


def test_missing_initialization_does_not_create_broken_root(tmp_path):
    root=tmp_path/'missing'
    with pytest.raises(ValueError,match='初始化'):a.run(root)
    assert not root.exists()


def test_daily_company_review_cannot_be_invented_or_reused_next_day(tmp_path,monkeypatch):
    root,_,_=initialized(tmp_path);path=root/'strategy.sqlite3'
    s.match(path,observation(),clock('2026-09-14',10))
    s.match(path,observation(20,3000),lambda:clock('2026-09-14',10)()+timedelta(seconds=20))
    monkeypatch.setattr(a,'approval_signature',lambda *args:'valid-source')
    monkeypatch.setattr(a.corporate,'inspect',lambda *args,**kwargs:dict(blocked=False))
    calls=[]
    monkeypatch.setattr(a.corporate,'capture_close',lambda *args,**kw:calls.append(kw) or {'ok':True})
    with pytest.raises(ValueError,match='待人工'):a.close(path,clock=clock('2026-09-14'))
    a.approve(path,'tester','已核對當日全部公告及權益測試資料',clock=clock('2026-09-14'))
    assert a.close(path,clock=clock('2026-09-14'))=={'ok':True} and calls[0]['actions_reviewed']
    with pytest.raises(ValueError,match='待人工'):a.close(path,clock=clock('2026-09-15'))
    monkeypatch.setattr(a,'approval_signature',lambda *args:'changed-source')
    with pytest.raises(ValueError,match='待人工'):a.close(path,clock=clock('2026-09-14'))


def test_intraday_uses_shared_quotes_but_independent_counterfactual_capacity(tmp_path,monkeypatch):
    root,_,_=initialized(tmp_path)
    monkeypatch.setattr(a,'calendar_day',lambda day:True)
    monkeypatch.setattr(a.corporate,'refresh',lambda *args,**kw:{})
    monkeypatch.setattr(a.corporate,'inspect',lambda *args,**kw:dict(blocked=False))
    monkeypatch.setattr(a,'markets',lambda ids:{sid:'tse' for sid in ids})
    current=[clock('2026-09-14',10)()];calls=[]
    def fetch(market,sid,channel):
        calls.append((market,sid,channel))
        secs=int((current[0]-clock('2026-09-14',10)()).total_seconds())
        obs=observation(secs,1000+secs*100);q=obs['body']['quote'];q.update(stock_id=sid,channel=channel)
        # Enough board liquidity to exercise all channels.
        if channel=='board':
            q['volume_shares']*=1000
            for x in q['asks']+q['bids']:x['shares']*=1000
        obs['hash']=j.digest(obs['body']);return obs
    def sleep(seconds):current[0]+=timedelta(seconds=seconds)
    result=a.run(root,clock=lambda:current[0],sleeper=sleep,fetcher=fetch)
    assert result['status']=='ok'
    assert len(calls)==6  # one 2492 odd, one 0050 board, one 0050 odd, twice
    assert p.summary(root/'strategy.sqlite3',lambda:current[0])['fill_count']==1
    assert p.summary(root/'benchmark.sqlite3',lambda:current[0])['fill_count']==2
    assert (root/'latest-report.json').exists()


def test_close_workflow_keeps_other_stages_and_no_duplicate_pipeline(tmp_path,monkeypatch):
    root,_,_=initialized(tmp_path);calls=[]
    monkeypatch.setattr(a,'calendar_day',lambda day:True)
    monkeypatch.setattr(a,'command',lambda *args,**kw:calls.append('pipeline'))
    monkeypatch.setattr(a,'_signals',lambda root:dict(hash='fixture-signal'))
    monkeypatch.setattr(a.corporate,'refresh',lambda *args,**kw:{})
    def blocked(*args,**kw):raise ValueError('待人工核對')
    monkeypatch.setattr(a,'close',blocked)
    result=a.run(root,clock=clock('2026-09-14'))  # 20:00 is outside the bounded settlement window
    assert result['status']=='outside_window'
    result=a.run(root,clock=clock('2026-09-14',19))
    assert result['status']=='needs_attention'
    assert calls==['pipeline']
    assert all(o['closed'] for o in p.state(s.read(root/'strategy.sqlite3'))['orders'].values())
    a.run(root,clock=clock('2026-09-14',19));assert calls==['pipeline']


def test_status_before_init_does_not_create_files(tmp_path):
    root=tmp_path/'missing'
    with pytest.raises(ValueError):a.export(root)
    assert not root.exists()
