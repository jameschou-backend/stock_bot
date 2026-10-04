from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import requests

from skills import poc_broker_odd_acquisition as mod
from test_prepare_volume_profile_odd import payload, response


class Session:
    def __init__(self, values):
        self.values, self.calls = list(values), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def engine(day, sid='1216', qty=1001, odd=1):
    return SimpleNamespace(day_plans={'x':dict(date=day, stock_id=sid, planned_qty=qty,
                                              odd_qty=odd, board_qty=qty-odd)})


@pytest.fixture
def context(tmp_path):
    p = tmp_path / mod.PREREG; p.parent.mkdir(); p.write_text('registered broker account rules')
    now = [datetime(2026, 10, 4, 12, tzinfo=timezone.utc).timestamp()]
    def sleep(seconds):
        now[0] += seconds
    for host in ('www.twse.com.tw', 'www.tpex.org.tw'):
        mod._write(tmp_path / '.cache/official-origin-holds' / (host+'.json'),
                   dict(status='blocked', observed_at='2026-10-02T00:00:00Z', evidence_sha256={}))
    return tmp_path, now, sleep


def client(context, session, online=True):
    root, now, sleep = context
    return mod.BrokerOddData(root, online=online, session=session, clock=lambda:now[0], sleep=sleep)


def test_both_real_missing_dates_use_own_exact_first_probe(context):
    root, _, _ = context
    s = Session([response(payload('TWSE','2026-09-23','1216')),
                 response(payload('TPEX','2025-09-03','4931'))])
    before = {p:p.read_bytes() for p in (root/'.cache/official-origin-holds').glob('*.json')}
    c = client(context, s)
    for market, day, sid in [('TWSE','2026-09-23','1216'), ('TPEX','2025-09-03','4931')]:
        row = c.get(day, sid, market, engine(day,sid))
        assert row['odd_shares'] == 10000 and row['market'] == market.lower()
        r = mod.read(c.cache/'receipts'/(market+'-'+day+'.json'))
        assert r['request_kind'] == 'single_normal_probe' and r['recovery_proof'] is None
        assert r['response_present'] and r['global_hold_unchanged']
    assert all(p.read_bytes() == value for p,value in before.items())
    assert s.calls[0] == (mod.URLS['twse'],dict(params={'date':'20260923','response':'json'},timeout=(10,30),allow_redirects=False))
    assert s.calls[1][1]['params']['date'] == '2025/09/03'
    assert c.snapshot()['accepted_count'] == 2


def test_followup_spacing_complete_closure_and_expired_offline_replay(context):
    root, now, _ = context
    s = Session([response(payload('TWSE',day,'1216')) for day in ['2026-09-23','2026-09-24']])
    c = client(context,s)
    for day in ['2026-09-23','2026-09-24']:
        c.get(day,'1216','twse',engine(day))
    a,b = [mod.read(c.cache/'receipts'/('TWSE-'+day+'.json')) for day in ['2026-09-23','2026-09-24']]
    assert mod._epoch(b['started_at'])-mod._epoch(a['started_at']) >= 3.1
    assert b['recovery_proof']['path'].endswith('TWSE-2026-09-23.json')
    now[0] += 90000
    offline = client(context, Session([]),False)
    assert offline.get('2026-09-24','1216','twse')['odd_shares'] == 10000
    for kind, extension in [('raw','bin'),('attempts','json'),('receipts','json'),('receipts','sha256')]:
        assert mod.BASE+'/'+kind+'/TWSE-2026-09-23.'+extension in offline.refs
    assert any('/demands/' in p for p in offline.refs)
    assert any('/source-snapshots/' in p for p in offline.refs)
    assert mod.PREREG in offline.refs and mod.BASE+'/authorization.json' in offline.refs


@pytest.mark.parametrize('status,body',[(401,b'denied'),(403,b'denied'),(428,b'challenge'),
                                     (429,b'rate'),(302,b'redirect'),(200,b'captcha')])
def test_security_stop_persistent_no_retries(context,status,body):
    s = Session([response(body,status)]); c = client(context,s)
    for day in ['2026-09-23','2026-09-23','2026-09-24']:
        with pytest.raises(mod.ReplayDataUnavailable):
            c.get(day,'1216','twse',engine(day))
    assert len(s.calls) == 1
    state = mod.read(context[0]/'.cache/official-daily-origin-dispatch/www.twse.com.tw.json')
    assert state['stopped'] and 'in_flight' not in state
    r = mod.read(c.cache/'receipts/TWSE-2026-09-23.json')
    assert not r['accepted'] and r['security_denied'] and r['http_status'] == status


@pytest.mark.parametrize('attached_status',[None,200,403,429])
def test_transport_failure_cannot_become_success_or_new_probe(context,attached_status):
    attached = None if attached_status is None else response(b'access denied' if attached_status != 200 else b'{}',attached_status)
    s = Session([requests.ConnectionError('provider details must not be copied',response=attached)])
    c = client(context,s)
    for day in ['2026-09-23','2026-09-23','2026-09-24']:
        with pytest.raises(mod.ReplayDataUnavailable):
            c.get(day,'1216','twse',engine(day))
    r = mod.read(c.cache/'receipts/TWSE-2026-09-23.json')
    assert len(s.calls) == 1 and r['http_status'] == attached_status
    assert r['response_present'] == (attached_status is not None)
    assert not r['accepted'] and 'provider details' not in json.dumps(r)
    if attached_status in (403,429):
        assert r['security_denied']


def test_absent_demand_stock_is_failed_schema_not_zero(context):
    s = Session([response(payload('TWSE','2026-09-23','2330'))]); c = client(context,s)
    with pytest.raises(mod.ReplayDataUnavailable):
        c.get('2026-09-23','1216','twse',engine('2026-09-23'))
    r = mod.read(c.cache/'receipts/TWSE-2026-09-23.json')
    assert not r['accepted'] and r['status'] == 'schema_error_no_retry'


def test_accepted_full_market_table_missing_other_stock_stays_missing(context):
    s = Session([response(payload('TWSE','2026-09-23','1216'))]);c = client(context,s)
    c.get('2026-09-23','1216','twse',engine('2026-09-23'))
    with pytest.raises(mod.ReplayDataUnavailable,match='table lacks stock'):
        c.get('2026-09-23','2330','twse')
    assert len(s.calls) == 1


def test_offline_constructor_does_not_create_authorization_or_attempt(context):
    root,_,_ = context;s = Session([]);c = client(context,s,False)
    with pytest.raises(mod.ReplayDataUnavailable,match='day missing'):
        c.get('2026-09-23','1216','twse')
    assert not c.cache.exists() and not s.calls


@pytest.mark.parametrize('bad',['outside_date','wrong_market','nonstock','zero','negative','boolean','bad_split'])
def test_scope_and_frozen_positive_plan_required(context,bad):
    day,sid,market = '2026-09-23','1216','twse';e=engine(day)
    if bad == 'outside_date':day='2026-10-05'
    elif bad == 'wrong_market':market='otc'
    elif bad == 'nonstock':sid='00631L'
    elif bad == 'zero':e.day_plans['x']['odd_qty']=0
    elif bad == 'negative':e.day_plans['x']['planned_qty']=-1
    elif bad == 'boolean':e.day_plans['x']['odd_qty']=True
    else:e.day_plans['x']['board_qty']=2000
    s=Session([]);c=client(context,s)
    with pytest.raises(ValueError):c.get(day,sid,market,e)
    assert not s.calls and not (c.cache/'attempts').exists()


def test_auth_expiry_during_origin_wait_forbids_dispatch(context):
    root,now,_=context;s=Session([]);c=client(context,s)
    now[0]+=86399
    mod._write(root/'.cache/official-daily-origin-dispatch/www.twse.com.tw.json',dict(last_start_epoch=now[0],probes={}))
    with pytest.raises(mod.ReplayDataUnavailable,match='authorization expired'):
        c.get('2026-09-23','1216','twse',engine('2026-09-23'))
    assert not s.calls and not (c.cache/'attempts').exists()


def test_changed_global_hold_during_wait_forbids_dispatch(context):
    root,now,_=context;s=Session([]);c=client(context,s)
    mod._write(root/'.cache/official-daily-origin-dispatch/www.twse.com.tw.json',dict(last_start_epoch=now[0],probes={}))
    def changed_sleep(seconds):
        now[0]+=seconds
        mod._write(root/'.cache/official-origin-holds/www.twse.com.tw.json',dict(status='blocked',observed_at=c._now()))
    c.sleep=changed_sleep
    with pytest.raises(mod.ReplayDataUnavailable,match='hold changed during wait'):
        c.get('2026-09-23','1216','twse',engine('2026-09-23'))
    assert not s.calls and not (c.cache/'attempts').exists()


@pytest.mark.parametrize('state_type',['in_flight','newer_stop'])
def test_origin_interrupt_or_newer_security_stop_forbids_probe(context,state_type):
    root,now,_=context;s=Session([]);c=client(context,s)
    state=dict(probes={})
    if state_type=='in_flight':state['in_flight']=dict(started_at='2026-10-01T00:00:00Z')
    else:
        p=root/'.cache/stop-evidence.json';mod._write(p,dict(blocked=True))
        state['stopped']=dict(observed_at=c._now(),receipt_path=str(p.relative_to(root)),receipt_sha256=mod.digest(p))
    mod._write(root/'.cache/official-daily-origin-dispatch/www.twse.com.tw.json',state)
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-23','1216','twse',engine('2026-09-23'))
    assert not s.calls


def test_persistent_budget_and_interrupted_identity_never_dispatch(context):
    s=Session([]);c=client(context,s)
    mod._write(c.cache/'attempts/TWSE-2026-09-23.json',dict(started=True))
    with pytest.raises(mod.ReplayDataUnavailable,match='unfinished'):
        c.get('2026-09-23','1216','twse',engine('2026-09-23'))
    for i in range(39):mod._write(c.cache/'attempts'/f'{i}.json',dict(started=True))
    with pytest.raises(mod.ReplayDataUnavailable,match='40-attempt'):
        c.get('2026-09-24','1216','twse',engine('2026-09-24'))
    assert not s.calls


@pytest.mark.parametrize('ancestor',['raw','attempts','demands','source-snapshots','authorization','probe_receipt'])
def test_changed_accepted_ancestor_rejected_offline(context,ancestor):
    s=Session([response(payload('TWSE',d,'1216')) for d in ['2026-09-23','2026-09-24']]);c=client(context,s)
    for day in ['2026-09-23','2026-09-24']:c.get(day,'1216','twse',engine(day))
    if ancestor=='authorization':p=c.auth
    elif ancestor=='probe_receipt':p=c.cache/'receipts/TWSE-2026-09-23.json'
    else:p=sorted((c.cache/ancestor).glob('*'))[0]
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):
        offline=client(context,Session([]),False);offline.get('2026-09-24','1216','twse')


def test_no_proxy_or_identity_change_by_default(context):
    root,now,sleep=context
    c=mod.BrokerOddData(root,clock=lambda:now[0],sleep=sleep)
    assert c.session.trust_env is False
    assert 'User-Agent' in c.session.headers
