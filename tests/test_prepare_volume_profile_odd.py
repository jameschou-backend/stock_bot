from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_volume_profile_odd as mod
from skills.replay_market_feeds import FIELDS

REAL=Path(mod.__file__).resolve().parents[1]


def payload(market,day,sid):
    table=dict(date=day.replace('-',''),fields=list(FIELDS[market.lower()].values()),
        data=[[sid,'10000','10','11','9','9.9','10.1','100','100']])
    if market=='TWSE':
        table.update(stat='OK',title=f'{int(day[:4])-1911}年{day[5:7]}月{day[8:]}日 盤中零股交易行情單',total=1)
        return table
    table.update(title='盤中零股每日收盤行情',totalCount=1)
    return dict(date=day.replace('-',''),stat='OK',tables=[table])


class Session:
    def __init__(self,responses):self.responses=list(responses);self.calls=[]
    def get(self,url,**kwargs):
        self.calls.append((url,kwargs))
        return self.responses.pop(0)


def response(body,status=200):
    return SimpleNamespace(status_code=status,content=json.dumps(body,ensure_ascii=False).encode() if not isinstance(body,bytes) else body,history=[])


@pytest.fixture
def context(tmp_path,monkeypatch):
    prereg=tmp_path/mod.PREREG;prereg.parent.mkdir();prereg.write_bytes((REAL/mod.PREREG).read_bytes())
    helper=tmp_path/'scripts/prepare_volume_profile_odd.py';helper.parent.mkdir();helper.write_bytes(Path(mod.__file__).read_bytes())
    monkeypatch.setattr(mod,'__file__',str(helper))
    now=[datetime(2026,10,3,tzinfo=timezone.utc).timestamp()]
    def sleep(seconds):now[0]+=seconds
    for host in ['www.twse.com.tw','www.tpex.org.tw']:
        mod._write(tmp_path/'.cache/official-origin-holds'/(host+'.json'),dict(status='blocked',observed_at='2026-09-25T00:00:00+00:00',evidence_sha256={}))
    def evidence(market,day,sid):
        p=tmp_path/'demands'/(market+'-'+day+'.json')
        mod._write(p,dict(completed=False,summary=None,reason='Offline official odd cache missing (holds retained): odd:'+market.lower()+':'+day,
            partial_journal=dict(day_plans=[dict(date=day,stock_id=sid,planned_qty=1001,odd_qty=1)])))
        return p
    return tmp_path,now,sleep,evidence


def test_probe_and_followup_bound_endpoint_spacing_and_unchanged_hold(context):
    root,now,sleep,evidence=context
    session=Session([response(payload('TWSE','2025-02-13','6558')),response(payload('TWSE','2025-02-14','6558'))])
    client=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    hold=root/'.cache/official-origin-holds/www.twse.com.tw.json';before=hold.read_bytes()
    first=client.fetch('TWSE','2025-02-13','6558',evidence('TWSE','2025-02-13','6558'),probe=True)
    second=client.fetch('TWSE','2025-02-14','6558',evidence('TWSE','2025-02-14','6558'))
    assert first['accepted'] and second['accepted'] and hold.read_bytes()==before
    assert mod._epoch(second['started_at'])-mod._epoch(first['started_at'])>=3.1
    assert all(c[0]==mod.URLS['twse'] and c[1]['allow_redirects'] is False for c in session.calls)
    assert (root/mod.DEST/'odd-twse-2025-02-13.json').exists()
    assert client.fetch('TWSE','2025-02-13','6558',evidence('TWSE','2025-02-13','6558'))['resumed_without_retry']
    assert len(session.calls)==2


def test_each_market_needs_its_own_probe(context):
    root,now,sleep,evidence=context
    session=Session([response(payload('TWSE','2025-02-13','6558')),response(payload('TPEX','2025-06-23','5475'))])
    c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    c.fetch('TWSE','2025-02-13','6558',evidence('TWSE','2025-02-13','6558'),probe=True)
    with pytest.raises(mod.AcquisitionBlocked):c.fetch('TPEX','2025-06-23','5475',evidence('TPEX','2025-06-23','5475'))
    result=c.fetch('TPEX','2025-06-23','5475',evidence('TPEX','2025-06-23','5475'),probe=True)
    assert result['accepted'] and session.calls[-1][0]==mod.URLS['tpex']
    assert session.calls[-1][1]['params']['date']=='2025/06/23'


@pytest.mark.parametrize('status,body',[(403,b'denied'),(302,b'moved'),(200,b'FOR SECURITY REASONS'),(429,b'quota')])
def test_new_security_stops_and_failed_probe_never_retries(context,status,body):
    root,now,sleep,evidence=context
    session=Session([response(body,status)]);c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    p=evidence('TWSE','2025-02-13','6558');r=c.fetch('TWSE','2025-02-13','6558',p,probe=True)
    assert not r['accepted'] and r['security_denied']
    assert c.fetch('TWSE','2025-02-13','6558',p,probe=True)['resumed_without_retry']
    with pytest.raises(mod.AcquisitionBlocked):c.fetch('TWSE','2025-02-14','6558',evidence('TWSE','2025-02-14','6558'))
    assert len(session.calls)==1 and not (root/mod.DEST/'odd-twse-2025-02-13.json').exists()


def test_schema_failure_gets_receipt_and_does_not_leave_inflight(context):
    root,now,sleep,evidence=context
    session=Session([response(dict(date='20250212',stat='OK'))]);c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    r=c.fetch('TWSE','2025-02-13','6558',evidence('TWSE','2025-02-13','6558'),probe=True)
    assert r['status']=='schema_error_no_retry' and not r['accepted']
    assert 'in_flight' not in mod.read(root/'.cache/official-daily-origin-dispatch/www.twse.com.tw.json')


def test_no_arbitrary_date_or_more_than_fifty_attempts(context):
    root,now,sleep,evidence=context
    session=Session([]);c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    with pytest.raises(mod.AcquisitionBlocked):c.fetch('TWSE','2025-02-14','6558',evidence('TWSE','2025-02-14','6558'),probe=True)
    for i in range(50):mod._write(root/mod.BASE/'attempts'/f'{i}.json',dict(started=True))
    with pytest.raises(mod.AcquisitionBlocked,match='50-attempt'):c.fetch('TWSE','2025-02-13','6558',evidence('TWSE','2025-02-13','6558'),probe=True)
    assert not session.calls


def test_utc_z_and_explicit_offset_have_identical_epochs():
    assert mod._epoch('2026-10-02T07:34:28Z')==mod._epoch('2026-10-02T07:34:28+00:00')
    assert mod._epoch('2026-10-02T07:34:28.125Z')==mod._epoch('2026-10-02T15:34:28.125+08:00')
    with pytest.raises(mod.AcquisitionBlocked,match='timezone'):
        mod._epoch('2026-10-02T07:34:28')


def test_probe_reads_z_hold_without_changing_its_bytes(context):
    root,now,sleep,evidence=context
    hold=root/'.cache/official-origin-holds/www.tpex.org.tw.json'
    mod._write(hold,dict(status='blocked',observed_at='2026-10-02T07:34:28Z',evidence_sha256={}))
    before=hold.read_bytes()
    session=Session([response(payload('TPEX','2025-06-23','5475'))])
    c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    result=c.fetch('TPEX','2025-06-23','5475',evidence('TPEX','2025-06-23','5475'),probe=True)
    assert result['accepted'] and hold.read_bytes()==before and len(session.calls)==1


def completed_followup(context):
    root,now,sleep,evidence=context
    session=Session([response(payload('TWSE','2025-02-13','6558')),
                     response(payload('TWSE','2025-02-14','6558'))])
    c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    first=c.fetch('TWSE','2025-02-13','6558',evidence('TWSE','2025-02-13','6558'),probe=True)
    # An already accepted receipt must retain the exact old helper, not assume
    # the currently installed source is what dispatched that earlier request.
    helper=Path(mod.__file__);helper.write_bytes(helper.read_bytes()+b'\n# later reviewed closure version\n')
    second=c.fetch('TWSE','2025-02-14','6558',evidence('TWSE','2025-02-14','6558'))
    assert first['helper_sha256']!=second['helper_sha256']
    return first,second,root/mod.DEST/'odd-twse-2025-02-14.json'


def test_offline_cached_wrapper_binds_both_helper_versions_and_probe_ancestors(context):
    root,now,sleep,_=context;first,second,p=completed_followup(context)
    now[0]+=90000  # Old valid data may be read; this must not authorize new I/O.
    session=Session([]);c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    verified=c.verify_cached(p);refs=verified['source_sha256']
    for day in ('2025-02-13','2025-02-14'):
        for sub,extension in [('attempts','json'),('receipts','json'),('receipts','sha256'),('raw','bin')]:
            assert f'{mod.BASE}/{sub}/TWSE-{day}.{extension}' in refs
        assert f'demands/TWSE-{day}.json' in refs
    for receipt in (first,second):
        name=f"{mod.BASE}/source-snapshots/{receipt['helper_sha256']}.py"
        assert refs[name]==receipt['helper_sha256']
    assert f'{mod.BASE}/authorization.json' in refs and mod.PREREG in refs
    assert '.cache/official-origin-holds/www.twse.com.tw.json' in refs
    assert verified['rows']['6558']['odd_shares']==10000 and not session.calls


@pytest.mark.parametrize('ancestor',['raw','demand','attempt','helper','authorization','receipt_hash'])
def test_offline_cache_rejects_changed_probe_ancestor_without_network(context,ancestor):
    root,now,sleep,_=context;first,_,p=completed_followup(context)
    targets=dict(raw=first['raw_path'],demand=first['requirement']['path'],
        attempt=f'{mod.BASE}/attempts/TWSE-2025-02-13.json',
        helper=f"{mod.BASE}/source-snapshots/{first['helper_sha256']}.py",
        authorization=f'{mod.BASE}/authorization.json',
        receipt_hash=f'{mod.BASE}/receipts/TWSE-2025-02-13.sha256')
    target=root/targets[ancestor]
    if ancestor=='attempt':
        value=mod.read(target);value['started_at']='2026-10-03T00:00:01+00:00';mod._write(target,value)
    elif ancestor=='receipt_hash':target.write_text('0'*64+'\n')
    else:target.write_bytes(target.read_bytes()+b' ')
    session=Session([]);c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    with pytest.raises(mod.AcquisitionBlocked):c.verify_cached(p)
    assert not session.calls


def test_missing_positive_odd_requirement_stops_before_io(context):
    root,now,sleep,evidence=context;p=evidence('TWSE','2025-02-13','6558')
    value=mod.read(p);value['partial_journal']['day_plans'][0]['odd_qty']=0;mod._write(p,value)
    session=Session([]);c=mod.OddCompletion(root,session=session,clock=lambda:now[0],sleep=sleep)
    with pytest.raises(mod.AcquisitionBlocked,match='positive odd-share'):
        c.fetch('TWSE','2025-02-13','6558',p,probe=True)
    assert not session.calls and not (root/mod.BASE/'authorization.json').exists()
