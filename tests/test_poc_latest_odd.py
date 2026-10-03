from datetime import datetime, timezone
from types import SimpleNamespace
import json

import pytest

from skills import poc_latest_odd as mod
from skills.replay_market_feeds import FIELDS


def payload(market, day):
    table=dict(date=day.replace('-',''),fields=list(FIELDS[market.lower()].values()),
        data=[['2330','10000','10','11','9','9.9','10.1','100','100']])
    if market=='TWSE':
        table.update(stat='OK',title=f'115年{day[5:7]}月{day[8:]}日 盤中零股交易行情單',total=1)
        return table
    table.update(title='盤中零股每日收盤行情',totalCount=1)
    return dict(date=day.replace('-',''),stat='OK',tables=[table])


def response(body,status=200):
    return SimpleNamespace(status_code=status,history=[],
        content=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False).encode())


class Session:
    def __init__(self, rows):self.rows=list(rows);self.calls=[]
    def get(self,url,**kwargs):
        self.calls.append((url,kwargs));return self.rows.pop(0)


def engine(day,quantity=1):
    return SimpleNamespace(day_plans={'x':dict(date=day,stock_id='2330',planned_qty=quantity,odd_qty=quantity)})


@pytest.fixture
def context(tmp_path):
    prereg=tmp_path/mod.PREREG;prereg.parent.mkdir();prereg.write_text('fixed plan')
    now=[datetime(2026,10,3,tzinfo=timezone.utc).timestamp()]
    def sleep(seconds):now[0]+=seconds
    for host in ('www.twse.com.tw','www.tpex.org.tw'):
        mod._write(tmp_path/'.cache/official-origin-holds'/(host+'.json'),
            dict(status='blocked',observed_at='2026-10-02T00:00:00Z',evidence_sha256={}))
    return tmp_path,now,sleep


def client(context,session,online=True):
    root,now,sleep=context
    return mod.LatestOddData(root,online=online,session=session,clock=lambda:now[0],sleep=sleep)


def test_exact_endpoint_followup_and_offline_closure(context):
    root,now,_=context
    s=Session([response(payload('TWSE','2026-09-10')),response(payload('TWSE','2026-09-11'))])
    c=client(context,s);hold=root/'.cache/official-origin-holds/www.twse.com.tw.json';before=hold.read_bytes()
    assert c.get('2026-09-10','2330','twse',engine('2026-09-10'))['odd_shares']==10000
    c.get('2026-09-11','2330','twse',engine('2026-09-11'))
    assert hold.read_bytes()==before and now[0]>=datetime(2026,10,3,tzinfo=timezone.utc).timestamp()+3.1
    assert all(k['allow_redirects'] is False for _,k in s.calls)
    now[0]+=90000
    c=client(context,Session([]),False)
    c.get('2026-09-11','2330','twse')
    assert any('/raw/TWSE-2026-09-10.json' in r for r in c.refs)
    assert any('/source-snapshots/' in r for r in c.refs)
    assert any('/attempts/TWSE-2026-09-10.json' in r for r in c.refs)


@pytest.mark.parametrize('status,body',[(403,b'denied'),(429,b'rate'),(302,b'redirect'),(200,b'captcha')])
def test_denial_preserves_stop_and_never_retries(context,status,body):
    s=Session([response(body,status)]);c=client(context,s)
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-10','2330','twse',engine('2026-09-10'))
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-10','2330','twse',engine('2026-09-10'))
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-11','2330','twse',engine('2026-09-11'))
    assert len(s.calls)==1
    state=mod.read(context[0]/'.cache/official-daily-origin-dispatch/www.twse.com.tw.json')
    assert 'stopped' in state and 'in_flight' not in state


def test_bad_payload_is_missing_data_never_zero_volume(context):
    s=Session([response(dict(stat='OK',date='20260909'))]);c=client(context,s)
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-10','2330','twse',engine('2026-09-10'))
    state=mod.read(context[0]/'.cache/official-daily-origin-dispatch/www.twse.com.tw.json')
    assert 'in_flight' not in state


@pytest.mark.parametrize('day,sid,qty',[('2026-09-09','2330',1),('2026-10-05','2330',1),('2026-09-10','00631L',1),('2026-09-10','2330',0)])
def test_scope_and_precommitted_demand_required(context,day,sid,qty):
    s=Session([]);c=client(context,s)
    with pytest.raises(ValueError):c.get(day,sid,'twse',engine(day,qty))
    assert not s.calls


@pytest.mark.parametrize('ancestor',['raw','attempts','demands','source-snapshots'])
def test_changed_probe_ancestor_rejected_offline(context,ancestor):
    s=Session([response(payload('TWSE','2026-09-10')),response(payload('TWSE','2026-09-11'))]);c=client(context,s)
    for day in ('2026-09-10','2026-09-11'):c.get(day,'2330','twse',engine(day))
    path=sorted((c.cache/ancestor).glob('*'))[0];path.write_bytes(path.read_bytes()+b' ')
    offline=client(context,Session([]),False)
    with pytest.raises(ValueError):offline.get('2026-09-11','2330','twse')


def test_offline_missing_does_not_dispatch(context):
    s=Session([]);c=client(context,s,False)
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-10','2330','twse',engine('2026-09-10'))
    assert not s.calls
