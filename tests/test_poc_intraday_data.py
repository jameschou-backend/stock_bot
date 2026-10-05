from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import requests

from skills import poc_intraday_data as mod
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
    return mod.IntradayOddAcquisition(root, online=online, session=session, clock=lambda:now[0], sleep=sleep)


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
    for i in range(399):mod._write(c.cache/'attempts'/f'{i}.json',dict(started=True))
    with pytest.raises(mod.ReplayDataUnavailable,match='400-attempt'):
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
    c=mod.IntradayOddAcquisition(root,clock=lambda:now[0],sleep=sleep)
    assert c.session.trust_env is False
    assert 'User-Agent' in c.session.headers


def parsed(day='2026-09-23', sid='1216', market='TWSE'):
    request = mod.request_item(day, market)
    return mod.parse_odd(mod.wrapper(dict(request, http_status=200, retrieved_at='2026-10-05T00:00:00Z'),
        payload(market,day,sid)), market.lower(),day)[sid]


@pytest.fixture
def composed(tmp_path, monkeypatch):
    from copy import deepcopy
    item = parsed()
    class Supplement:
        refs = {}; available_market_days = ('odd:twse:2026-09-23',); selected_sources = {}
        error = None
        def get(self, day, sid, market):
            if self.error: raise self.error
            return deepcopy(item)
    class Acquisition:
        cache = tmp_path/mod.BASE
        refs = {}; calls_seen = 0
        def get(self, day, sid, market, engine=None):
            self.calls_seen += 1
            return deepcopy(item)
        def snapshot(self): return dict(attempted_calls=0)
    supplement, acquisition = Supplement(), Acquisition()
    monkeypatch.setattr(mod,'SupplementaryOddSources', lambda *a,**k:supplement)
    value = mod.IntradayOddData(tmp_path,source_refs={},acquisition=acquisition)
    return value, supplement, acquisition, item


def test_frozen_daily_data_tagged_explicitly_not_ticks_or_afterhours(composed):
    value, _, acquisition, item = composed
    result = value.get('2026-09-23','1216','twse')
    assert result['odd_shares']==10000 and result['source_date']=='2026-09-23'
    assert result['volume_scope']==mod.VOLUME_SCOPE and result['after_hours'] is False
    assert result['intraday_tick_verified'] is False and result['actual_fill_verified'] is False
    assert result['evidence_status']=='official_intraday_daily_table'
    assert 'daily_participation_ceiling' not in result and item['daily_participation_ceiling']==.05
    assert acquisition.calls_seen==0 and value.snapshot()['participation']==.01


def test_only_missing_whole_day_can_reach_new_acquisition(composed):
    value, supplement, acquisition, _ = composed
    supplement.error = mod.OddMarketDayMissing('Absent whole day')
    value.get('2026-09-23','1216','twse',engine('2026-09-23'))
    assert acquisition.calls_seen==1


@pytest.mark.parametrize('message', ['Supplementary odd stock absent: 1216', 'Conflicting supplementary odd rows: 1216'])
def test_stock_absence_or_conflict_cannot_fallback_to_new_source(composed,message):
    value, supplement, acquisition, _ = composed
    supplement.error = mod.ReplayDataUnavailable(message)
    with pytest.raises(mod.ReplayDataUnavailable):value.get('2026-09-23','1216','twse')
    assert acquisition.calls_seen==0


@pytest.mark.parametrize('message',['Supplementary odd source changed: test','Supplementary odd receipt chain is invalid: test'])
def test_integrity_errors_cannot_be_treated_as_normal_order_gap(composed,message):
    value,supplement,acquisition,_=composed
    supplement.error=mod.ReplayDataUnavailable(message)
    with pytest.raises(ValueError,match='integrity failure'):value.get('2026-09-23','1216','twse')
    assert not acquisition.calls_seen


def test_existing_second_source_must_agree_and_may_not_hide_missing_stock(composed,tmp_path):
    value,_,acquisition,item=composed
    path=tmp_path/'legacy/receipts/TWSE-2026-09-23.json';path.parent.mkdir(parents=True);path.write_text('{}')
    other=dict(item,odd_shares=10001)
    legacy=SimpleNamespace(cache=path.parent.parent,refs={},cached=lambda *a:{'1216':other})
    value.legacy['latest']=legacy
    with pytest.raises(mod.ReplayDataUnavailable,match='Conflicting official'):value.get('2026-09-23','1216','twse')
    legacy.cached=lambda *a:{}
    with pytest.raises(mod.ReplayDataUnavailable,match='lacks stock'):value.get('2026-09-23','1216','twse')
    legacy.cached=lambda *a:{'1216':item}
    assert value.get('2026-09-23','1216','twse')['odd_shares']==10000
    assert value.queries[-1]['sources']==['frozen_inventory','latest'] and acquisition.calls_seen==0


def test_accepted_same_experiment_receipt_is_also_crosschecked(composed):
    value,_,acquisition,item=composed
    p=acquisition.cache/'receipts/TWSE-2026-09-23.json';p.parent.mkdir(parents=True);p.write_text('{}')
    acquisition.get=lambda *a,**k:dict(item,odd_high=12.)
    with pytest.raises(mod.ReplayDataUnavailable,match='Conflicting official'):value.get('2026-09-23','1216','twse')


@pytest.mark.parametrize('change',[{'after_hours':True},{'volume_scope':'ordinary_session'},
    {'source_date':'2026-09-24'},{'market':'tpex'},{'odd_shares':True},{'odd_shares':-1},
    {'odd_high':float('nan')},{'odd_low':12.}])
def test_scope_units_and_prices_not_silently_relabelled(change):
    row=dict(parsed(),**change)
    with pytest.raises(ValueError):mod.intraday_row(row,'2026-09-23','1216','TWSE')


def test_explicit_official_zero_preserved_without_invented_price():
    row=dict(parsed(),odd_shares=0,odd_high=None,odd_low=None,odd_last=None)
    assert mod.intraday_row(row,'2026-09-23','1216','twse')['odd_high'] is None
    row['odd_last']=10.
    with pytest.raises(ValueError):mod.intraday_row(row,'2026-09-23','1216','twse')


def test_account_swaps_only_odd_provider_and_counts_new_official_calls(monkeypatch,tmp_path):
    # Exercise inherited facade without expensive actual research initialization.
    obj=object.__new__(mod.IntradayAccountData);obj.refs={};obj.after_hours=None
    obj.profiles=SimpleNamespace(raw_store=SimpleNamespace(calls=3),refs={})
    obj.board_ticks=SimpleNamespace(store=SimpleNamespace(calls=2),refs={})
    obj.execution=SimpleNamespace(refs={});obj._financial_calls=1
    seen=[]
    class Intraday:
        refs={};calls=4
        def get(self,*args):seen.append(args);return {'scope':'intraday'}
        def _merge(self):pass
    obj.intraday_odds=Intraday()
    e=engine('2026-09-23')
    assert obj.get_odd('2026-09-23','1216','twse',e)=={'scope':'intraday'}
    assert seen==[('2026-09-23','1216','twse',e)] and obj.network_calls==10
    assert obj.after_hours is None
