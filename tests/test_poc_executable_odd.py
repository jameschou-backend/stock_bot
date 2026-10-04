from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import requests

from skills import poc_executable_odd as mod
from test_prepare_volume_profile_odd import response


def payload(market, day, sid):
    tw = market.upper() == 'TWSE'
    table = dict(fields=mod.FIELDS['twse' if tw else 'tpex'],
        data=[[sid,'Test','10,000','10','1,000,000','100','99','100','101','100']],
        title=f'{int(day[:4])-1911}年{day[5:7]}月{day[8:]}日 盤後零股交易行情單' if tw else '盤後零股每日收盤行情')
    if tw:
        return dict(table, date=day.replace('-',''), stat='OK', type='ALL', total=1, hints='單位：元、股')
    return dict(date=day.replace('-',''), stat='ok', template='/template/afterTrading/odd',
                tables=[dict(table, date=f'{int(day[:4])-1911}/{day[5:7]}/{day[8:]}', totalCount=1)])


def source(market='TWSE', day='2026-09-23', sid='1216'):
    r = dict(mod.request_item(day,market), http_status=200, retrieved_at='2026-10-04T00:00:00+00:00')
    return mod.wrapper(r,payload(market,day,sid))


def with_summary(data):
    row=['\u3000','合計','0','0','0','--','--','0','--','0']
    for index in (2,3,4,7,9):
        row[index]=str(sum(int(str(r[index]).replace(',','')) for r in data['data']))
    data['data'].append(row);data['total']=len(data['data'])
    return data


def failed_old_schema(context,monkeypatch,more_responses=None):
    p=with_summary(payload('TWSE','2026-09-23','1216'))
    s=Session([response(p),*(more_responses or [])]);c=client(context,s)
    def old_parser(*args,**kwargs):
        raise mod.ReplayDataUnavailable('Old parser rejected a legitimate total row')
    old_helper=context[0]/'former-helper.py'
    old_helper.write_bytes(Path(mod.__file__).read_bytes()+b'\n# Previous schema version test snapshot.\n')
    with monkeypatch.context() as patch:
        patch.setattr(mod,'parse_after_hours',old_parser)
        patch.setattr(mod,'__file__',str(old_helper))
        with pytest.raises(mod.ReplayDataUnavailable):
            c.get('2026-09-23','1216','TWSE',engine('2026-09-23'))
    return c,s


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
                                              odd_qty=odd, board_qty=qty-odd, side='buy', signal_date='2024-01-01',
                                              odd_limit=101, odd_order_time='13:40:00', odd_expires_at='14:30:00')})


@pytest.fixture
def context(tmp_path):
    p = tmp_path / mod.PREREG; p.parent.mkdir(); p.write_text('registered executable auction account rules')
    now = [datetime(2026, 10, 4, 12, tzinfo=timezone.utc).timestamp()]
    def sleep(seconds):
        now[0] += seconds
    for host in ('www.twse.com.tw', 'www.tpex.org.tw'):
        mod._write(tmp_path / '.cache/official-origin-holds' / (host+'.json'),
                   dict(status='blocked', observed_at='2026-10-02T00:00:00Z', evidence_sha256={}))
    return tmp_path, now, sleep


def client(context, session, online=True):
    root, now, sleep = context
    return mod.ExecutableOddData(root, online=online, session=session, clock=lambda:now[0], sleep=sleep)


def test_both_markets_need_own_exact_after_hours_first_probe(context):
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
    assert s.calls[0] == (mod.URLS['twse'],dict(params={'date':'20260923','response':'json','type':'ALL'},timeout=(10,30),allow_redirects=False))
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
    c=mod.ExecutableOddData(root,clock=lambda:now[0],sleep=sleep)
    assert c.session.trust_env is False
    assert 'User-Agent' in c.session.headers


@pytest.mark.parametrize('market',['TWSE','TPEX'])
@pytest.mark.parametrize('mutation',['date','inner_date','title','scope','units','count','width','duplicate','missing_qty','negative_qty','fraction_qty','no_price','nonfinite_price','intraday'])
def test_wrong_or_incomplete_auction_never_becomes_fill(market,mutation):
    record=source(market);p=record['payload'];t=p if market=='TWSE' else p['tables'][0]
    if mutation=='date':p['date']='20260924'
    elif mutation=='inner_date':
        if market=='TWSE':record['day']='2026-09-24'
        else:t['date']='115/09/24'
    elif mutation=='title':t['title']='盤中零股每日收盤行情'
    elif mutation=='scope':record['params']['type']='SomeStock'
    elif mutation=='units':t['fields']=[x.replace('成交股數','成交仟股') for x in t['fields']]
    elif mutation=='count':t['total' if market=='TWSE' else 'totalCount']=99
    elif mutation=='width':t['data'][0].pop()
    elif mutation=='duplicate':
        t['data'].append(list(t['data'][0]));t['total' if market=='TWSE' else 'totalCount']=2
    elif mutation=='missing_qty':t['data'][0][2]='--'
    elif mutation=='negative_qty':t['data'][0][2]='-1'
    elif mutation=='fraction_qty':t['data'][0][2]='1.5'
    elif mutation=='no_price':t['data'][0][5]='0.00'
    elif mutation=='nonfinite_price':t['data'][0][5]='NaN'
    else:record['url']=record['url'].replace('TWT53U','MI_INDEX').replace('afterTrading/odd','intradayOdd')
    with pytest.raises(mod.ReplayDataUnavailable):mod.parse_after_hours(record,market,'2026-09-23')


@pytest.mark.parametrize('market,zero',[('TWSE','--'),('TPEX','0.00')])
def test_explicit_zero_auction_is_known_not_missing(market,zero):
    record=source(market);t=record['payload'] if market=='TWSE' else record['payload']['tables'][0]
    t['data'][0][2:6]=['0','0','0',zero]
    rows=mod.parse_after_hours(record,market,'2026-09-23')
    assert '2330' not in rows and rows['1216']['auction_price'] is None
    out=mod.match_after_hours(rows['1216'],'buy',101,500)
    assert out['filled_qty']==0 and out['failure']=='no_auction_trades'
    assert out['evidence_status']=='complete_auction_table' and out['reference_price'] is None


def test_strict_through_participation_and_cumulative_no_double_spend():
    row=mod.parse_after_hours(source(),'TWSE','2026-09-23')['1216']
    buy=mod.match_after_hours(row,'buy',101,150)
    assert buy['filled_qty']==100 and buy['reference_price']==100 and buy['last_fill_time']=='14:30:00'
    assert buy['failure']=='partial_auction_capacity' and buy['actual_fill_verified'] is False
    assert mod.match_after_hours(row,'sell',99,50,used_shares=80)['filled_qty']==20
    assert mod.match_after_hours(row,'buy',101,1,used_shares=100)['filled_qty']==0
    with pytest.raises(ValueError):mod.match_after_hours(row,'sell',99,50,used_shares=101)
    for side,limit in [('buy',100),('sell',100),('buy',99),('sell',101)]:
        result=mod.match_after_hours(row,side,limit,50)
        assert result['filled_qty']==0 and result['reference_price'] is None
        assert result['failure']==('same_price_queue_uncredited' if limit==100 else 'limit_not_through_auction')


@pytest.mark.parametrize('shares,expected',[(99,0),(100,1),(299,2),(300,3)])
def test_capacity_decimal_floors_shares_not_board_lots(shares,expected):
    row=mod.parse_after_hours(source(),'TWSE','2026-09-23')['1216'];row['odd_shares']=shares
    assert mod.match_after_hours(row,'buy',101,999)['filled_qty']==expected


@pytest.mark.parametrize('kwargs',[{'quantity':0},{'quantity':1000},{'quantity':True},
    {'participation':.02},{'participation':float('nan')},{'limit_price':0},
    {'limit_price':float('inf')},{'used_shares':-1},{'side':'hold'}])
def test_invalid_order_rejected(kwargs):
    row=mod.parse_after_hours(source(),'TWSE','2026-09-23')['1216']
    args=dict(side='buy',limit_price=101,quantity=10);args.update(kwargs)
    with pytest.raises(ValueError):mod.match_after_hours(row,**args)


@pytest.mark.parametrize('field,value',[('volume_scope','intraday_odd'),('auction_time','13:30:00'),
    ('odd_shares',None),('auction_price',None),('odd_high',999),('volume_unit','lots')])
def test_unclassified_or_missing_source_is_blocker_not_zero(field,value):
    row=mod.parse_after_hours(source(),'TWSE','2026-09-23')['1216'];row[field]=value
    with pytest.raises(mod.ReplayDataUnavailable):mod.match_after_hours(row,'buy',101,100)


@pytest.mark.parametrize('field,value',[('signal_date','2026-09-23'),('signal_date','2026-09-24'),
    ('odd_order_time','09:00:00'),('odd_expires_at','13:30:00'),('odd_limit',None),
    ('odd_limit',101.2),('side','hold')])
def test_precommitted_timing_side_and_odd_limit_required(context,field,value):
    e=engine('2026-09-23');e.day_plans['x'][field]=value;s=Session([]);c=client(context,s)
    with pytest.raises(ValueError):c.get('2026-09-23','1216','TWSE',e)
    assert not s.calls and not (c.cache/'attempts').exists()


def test_receipt_binds_auction_demand_and_recomputed_outcome(context):
    s=Session([response(payload('TWSE','2026-09-23','1216'))]);c=client(context,s)
    e=engine('2026-09-23',qty=150,odd=150)
    c.get('2026-09-23','1216','twse',e)
    p=c.cache/'receipts/TWSE-2026-09-23.json';r=mod.read(p)
    assert r['demand_outcomes'][0]['filled_qty']==100
    assert r['demand_outcomes'][0]['limit_price']==101 and r['actual_fill_verified'] is False
    r['demand_outcomes'][0]['filled_qty']=150;mod._write(p,r)
    p.with_suffix('.sha256').write_text(mod.digest(p)+'\n')
    with pytest.raises(ValueError,match='outcome'):
        client(context,Session([]),False).get('2026-09-23','1216','twse')


def test_cached_other_stock_needs_its_own_frozen_plan(context):
    p=payload('TWSE','2026-09-23','1216');p['data'].append(['2330',*p['data'][0][1:]]);p['total']=2
    s=Session([response(p)]);c=client(context,s)
    c.get('2026-09-23','1216','twse',engine('2026-09-23'))
    with pytest.raises(ValueError,match='precommitted plan'):c.get('2026-09-23','2330','twse')
    assert c.get('2026-09-23','2330','twse',engine('2026-09-23','2330'))['stock_id']=='2330'
    assert len(s.calls)==1


@pytest.mark.parametrize('body',[[],None,{'stat':'ok','date':'20260923','tables':[None]}])
def test_malformed_body_preserves_failed_attempt_without_retry(context,body):
    s=Session([response(body)]);c=client(context,s)
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-23','1216','TPEX',engine('2026-09-23'))
    r=mod.read(c.cache/'receipts/TPEX-2026-09-23.json')
    assert r['status']=='schema_error_no_retry' and not r['accepted'] and r['response_present']
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-23','1216','TPEX',engine('2026-09-23'))
    assert len(s.calls)==1


def test_twse_unique_total_checks_all_source_rows_before_stock_filter():
    r=source();p=r['payload']
    p['data'].append(['00631L',*p['data'][0][1:]])
    with_summary(p)
    rows=mod.parse_after_hours(r,'TWSE','2026-09-23')
    assert set(rows)=={'1216'} and rows['1216']['odd_shares']==10000


@pytest.mark.parametrize('bad',['blank_detail','two_totals','wrong_name','not_last','price','count',
    'shares','trades','amount','bid_qty','ask_qty','none_code'])
def test_total_row_is_not_a_blank_identity_bypass(bad):
    r=source();p=with_summary(r['payload']);last=p['data'][-1]
    if bad=='blank_detail':p['data'][0][0]=' '
    elif bad=='two_totals':p['data'].append(list(last));p['total']=3
    elif bad=='wrong_name':last[1]='一般證券'
    elif bad=='not_last':p['data'].reverse()
    elif bad=='price':last[5]='100'
    elif bad=='count':p['total']=1
    elif bad=='none_code':p['data'][0][0]=None
    else:last[{'shares':2,'trades':3,'amount':4,'bid_qty':7,'ask_qty':9}[bad]]='123'
    with pytest.raises(mod.ReplayDataUnavailable):mod.parse_after_hours(r,'TWSE','2026-09-23')


def test_declared_foreign_currency_amount_exclusion_is_exact():
    r=source();p=r['payload'];p['data'].append(['01234K',*p['data'][0][1:]])
    with_summary(p);p['data'][-1][4]='1000000'
    with pytest.raises(mod.ReplayDataUnavailable):mod.parse_after_hours(r,'TWSE','2026-09-23')
    p['notes']=['ETF證券代號第六碼為K、M、S、C者，表示該ETF以外幣交易。','不加計外幣交易證券交易金額。']
    assert set(mod.parse_after_hours(r,'TWSE','2026-09-23'))=={'1216'}


def test_append_only_schema_reparse_binds_original_bytes_and_continues_without_reprobe(context,monkeypatch):
    c,s=failed_old_schema(context,monkeypatch,[response(payload('TWSE','2026-09-24','1216'))])
    kept=[p for folder in ['raw','receipts','attempts','demands'] for p in (c.cache/folder).glob('*')]
    kept += list((context[0]/'.cache/official-origin-holds').glob('*.json'))
    original={p:p.read_bytes() for p in kept}
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-24','1216','TWSE',engine('2026-09-24'))
    r=c.reparse_schema('2026-09-23','TWSE',reason='Recognize and sum the official final total row')
    assert len(s.calls)==1 and all(p.read_bytes()==value for p,value in original.items())
    assert r['request_kind']=='offline_schema_reparse' and r['network_requests']==0
    assert r['helper_sha256']!=mod.read(c.cache/'receipts/TWSE-2026-09-23.json')['helper_sha256']
    assert 'skills/poc_executable_odd.py' not in r['source_sha256']
    assert r['demand_outcomes'][0]['filled_qty']==1 and r['original_failure_preserved']
    assert c.get('2026-09-23','1216','TWSE')['odd_shares']==10000
    again=c.reparse_schema('2026-09-23','TWSE',reason='Repeat local verification')
    assert again==r and len(s.calls)==1
    proof=mod.read(c.cache/'proof-TWSE.json')
    assert proof['path'].endswith('schema-supplements/TWSE-2026-09-23.json')
    assert proof['retrieved_at']==r['retrieved_at']
    c.get('2026-09-24','1216','TWSE',engine('2026-09-24'))
    later=mod.read(c.cache/'receipts/TWSE-2026-09-24.json')
    assert len(s.calls)==2 and later['request_kind']=='necessary_preplanned_day'
    assert later['recovery_proof']==proof
    assert mod._epoch(later['started_at'])-mod._epoch(r['started_at'])>=3.1
    snap=c.snapshot()
    assert (snap['attempted_calls'],snap['accepted_count'],snap['failed_count'],
            snap['offline_schema_reparsed_count'],snap['effective_accepted_count'],snap['unresolved_failure_count'])==(2,1,1,1,2,0)
    offline=client(context,Session([]),False)
    assert offline.get('2026-09-24','1216','twse')['odd_shares']==10000
    for key in ['raw_path','demand_path','base_receipt_path']:
        assert r[key] in offline.refs
    assert mod.BASE+'/schema-supplements/TWSE-2026-09-23.json' in offline.refs


def test_offline_reparse_does_not_renew_original_probe_age(context,monkeypatch):
    c,s=failed_old_schema(context,monkeypatch);original=mod.read(c.cache/'receipts/TWSE-2026-09-23.json')
    context[1][0]+=90000
    r=c.reparse_schema('2026-09-23','TWSE',reason='Local schema fix after original request expired')
    assert mod._epoch(r['created_at'])-mod._epoch(original['retrieved_at'])==90000
    assert mod.read(c.cache/'proof-TWSE.json')['retrieved_at']==original['retrieved_at']
    with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-24','1216','TWSE',engine('2026-09-24'))
    assert len(s.calls)==1


@pytest.mark.parametrize('field,value',[('status','request_failed_no_retry'),('http_status',403),
    ('http_status',429),('security_denied',True),('response_present',False),
    ('redirect_statuses',[302]),('automatic_redirects_disabled',False),('global_hold_unchanged',False)])
def test_reparse_cannot_reclassify_transport_or_security_failure(context,monkeypatch,field,value):
    c,s=failed_old_schema(context,monkeypatch);p=c.cache/'receipts/TWSE-2026-09-23.json'
    r=mod.read(p);r[field]=value;mod._write(p,r);p.with_suffix('.sha256').write_text(mod.digest(p)+'\n')
    fresh=client(context,Session([]),False)
    with pytest.raises((ValueError,mod.ReplayDataUnavailable)):
        fresh.reparse_schema('2026-09-23','TWSE',reason='Must reject')
    assert not (c.cache/'schema-supplements').exists() and len(s.calls)==1


def test_no_newer_origin_stop_may_be_cleared_by_offline_schema_reparse(context,monkeypatch):
    c,s=failed_old_schema(context,monkeypatch)
    c.reparse_schema('2026-09-23','TWSE',reason='Official table total')
    root,now,_=context;now[0]+=1
    p=root/'.cache/new-stop.json';mod._write(p,dict(status='blocked'))
    statepath=root/'.cache/official-daily-origin-dispatch/www.twse.com.tw.json';state=mod.read(statepath)
    state['stopped']=dict(observed_at=c._now(),receipt_path=str(p.relative_to(root)),receipt_sha256=mod.digest(p))
    mod._write(statepath,state)
    with pytest.raises(mod.ReplayDataUnavailable,match='Newer origin stop'):
        c.get('2026-09-24','1216','TWSE',engine('2026-09-24'))
    assert len(s.calls)==1 and mod.read(statepath)==state


@pytest.mark.parametrize('ancestor',['raw','attempt','demand','original_helper','new_helper','base_receipt','supplement'])
def test_offline_reparse_complete_ancestry_is_hash_bound(context,monkeypatch,ancestor):
    c,_=failed_old_schema(context,monkeypatch)
    r=c.reparse_schema('2026-09-23','TWSE',reason='Official table total')
    original=mod.read(c.cache/'receipts/TWSE-2026-09-23.json')
    paths=dict(raw=c.root/r['raw_path'],attempt=c.cache/'attempts/TWSE-2026-09-23.json',
        demand=c.root/r['demand_path'],original_helper=c.cache/'source-snapshots'/(original['helper_sha256']+'.py'),
        new_helper=c.cache/'source-snapshots'/(r['helper_sha256']+'.py'),base_receipt=c.root/r['base_receipt_path'],
        supplement=c.cache/'schema-supplements/TWSE-2026-09-23.json')
    target=paths[ancestor];target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError):client(context,Session([]),False).get('2026-09-23','1216','TWSE')


def test_reparse_does_not_turn_an_accepted_receipt_into_a_new_proof(context):
    c=client(context,Session([response(payload('TWSE','2026-09-23','1216'))]))
    c.get('2026-09-23','1216','TWSE',engine('2026-09-23'))
    with pytest.raises(ValueError,match='original schema failure'):
        c.reparse_schema('2026-09-23','TWSE',reason='Unnecessary reparse')
    assert not (c.cache/'schema-supplements').exists()


def test_failed_followup_schema_can_be_reparsed_without_replacing_original_probe(context,monkeypatch):
    s=Session([response(payload('TWSE',day,'1216')) for day in ['2026-09-23','2026-09-24','2026-09-25']])
    c=client(context,s);c.get('2026-09-23','1216','TWSE',engine('2026-09-23'))
    proof=c.cache/'proof-TWSE.json';before=proof.read_bytes();parser=mod.parse_after_hours
    def formerly_bad(record,market,day):
        if day=='2026-09-24':raise mod.ReplayDataUnavailable('Earlier schema')
        return parser(record,market,day)
    with monkeypatch.context() as patch:
        patch.setattr(mod,'parse_after_hours',formerly_bad)
        with pytest.raises(mod.ReplayDataUnavailable):c.get('2026-09-24','1216','TWSE',engine('2026-09-24'))
    r=c.reparse_schema('2026-09-24','TWSE',reason='Independent local schema validation')
    assert r['original_request_kind']=='necessary_preplanned_day' and proof.read_bytes()==before
    c.get('2026-09-25','1216','TWSE',engine('2026-09-25'))
    assert len(s.calls)==3 and c.snapshot()['unresolved_failure_count']==0


def test_reparse_supplement_claims_and_closure_cannot_be_stripped(context,monkeypatch):
    c,_=failed_old_schema(context,monkeypatch);c.reparse_schema('2026-09-23','TWSE',reason='Official total')
    p=c.cache/'schema-supplements/TWSE-2026-09-23.json';r=mod.read(p);r['source_sha256']={}
    mod._write(p,r);p.with_suffix('.sha256').write_text(mod.digest(p)+'\n')
    with pytest.raises(ValueError,match='closure'):
        client(context,Session([]),False).get('2026-09-23','1216','TWSE')
