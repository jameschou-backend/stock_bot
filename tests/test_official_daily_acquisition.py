from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import requests

from skills.official_daily_acquisition import (AcquisitionBlocked, OfficialDailyAcquisition,
    USER_REQUEST, create_plan, digest, encoded, request_item)


def payload(market='TWSE', day='2026-09-09'):
    if market == 'TWSE':
        table = dict(fields=['證券代號','證券名稱','開盤價','最高價','最低價','收盤價','成交股數'],
            notes=['本統計資訊含一般、零股、盤後定價、鉅額交易'],
            data=[['2330','台積電','100','110','90','105','100,050']])
        return dict(stat='OK', date=day.replace('-',''), type='ALLBUT0999', tables=[table])
    y,m,d = map(int, day.split('-'))
    table = dict(fields=['代號','名稱','開盤','最高','最低','收盤','成交股數'],
        title='上櫃股票每日收盤行情(不含定價)', date=f'{y-1911}/{m:02d}/{d:02d}',
        category='所有證券(不含權證、牛熊證)', totalCount=1,
        data=[['2330','測試','100','110','90','105','100,000']])
    return dict(stat='ok', date=day.replace('-',''), tables=[table])


class Runtime:
    def __init__(self, root, responses):
        self.root, self.responses, self.calls = root, list(responses), []
        self.now = datetime(2026,10,2,tzinfo=timezone.utc).timestamp()

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def get(self, url, **kwargs):
        assert kwargs['allow_redirects'] is False
        self.calls.append((url, self.now, kwargs))
        attempts = list(self.root.glob('cache*/attempts/*.json'))
        assert any(json.loads(p.read_text())['url'] == url for p in attempts)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def response(market='TWSE', day='2026-09-09', *, status=200, body=None, history=None):
    return SimpleNamespace(status_code=status, content=body if body is not None else encoded(payload(market,day)).encode(),
                           history=history or [])


def setup(tmp_path, responses, *, days=('2026-09-09','2026-09-10'), market='TWSE', cache='cache'):
    items = [request_item(market, d) for d in days]
    folder = tmp_path/cache
    create_plan(tmp_path, folder, items, {})
    runtime = Runtime(tmp_path,responses)
    client = OfficialDailyAcquisition(tmp_path,folder,session=runtime,clock=runtime.clock,sleep=runtime.sleep)
    return client, runtime, items


def auth(client, runtime):
    path = client.cache/'authorization.json'
    path.write_text(encoded(dict(schema='official_daily_authorization_v1', user_request=USER_REQUEST,
        scope='missing_official_daily_tables', security_bypass_authorized=False,
        created_at=datetime.fromtimestamp(runtime.now,timezone.utc).isoformat(), plan_sha256=client.plan_hash)))
    client.authorization = path
    return path


def hold(tmp_path, runtime, *, delta=-100, market='TWSE'):
    host = 'www.twse.com.tw' if market == 'TWSE' else 'www.tpex.org.tw'
    path = tmp_path/'.cache/official-origin-holds'/(host+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(encoded(dict(status='blocked', observed_at=datetime.fromtimestamp(runtime.now+delta,timezone.utc).isoformat(),
        evidence_sha256={}, no_automatic_recovery=True)))
    return path


def test_success_requires_complete_schema_resumes_and_exports(tmp_path):
    c,r,items = setup(tmp_path,[response(),response(day='2026-09-10')])
    first = c.fetch(items[0])
    assert first['accepted'] and first['volume_scope'] == 'all_daily_sessions'
    assert c.fetch(items[0]) == first and len(r.calls) == 1
    c.fetch(items[1])
    assert r.calls[1][1]-r.calls[0][1] >= 3.1
    manifest = c.export_manifest(tmp_path/'supplement.json')
    assert manifest['schema'] == 'official_market_supplement_v1'
    assert manifest['accepted_market_days'] == 2 and manifest['live_qualified'] is False
    assert (tmp_path/'supplement.sha256').read_text().strip() == digest(tmp_path/'supplement.json')
    with pytest.raises(FileExistsError): c.export_manifest(tmp_path/'supplement.json')


@pytest.mark.parametrize('mutation', ['date','scope','count','duplicate','ohlc'])
def test_semantic_failures_are_not_accepted_or_retried(tmp_path,mutation):
    p = payload('TPEX')
    table = p['tables'][0]
    if mutation == 'date': p['date'] = '20260908'
    elif mutation == 'scope': table['category'] = '半導體'
    elif mutation == 'count': table['totalCount'] = 2
    elif mutation == 'duplicate': table['data'] *= 2; table['totalCount'] = 2
    else: table['data'][0][4] = '200'
    c,r,items = setup(tmp_path,[response('TPEX',body=encoded(p).encode())],market='TPEX')
    value = c.fetch(items[0])
    assert not value['accepted'] and value['status'] == 'schema_error_no_retry'
    assert c.fetch(items[0])['resumed_without_retry'] and len(r.calls) == 1
    assert c.export_manifest(tmp_path/'supplement.json')['entries'] == []


@pytest.mark.parametrize('status,body', [(401,b'x'),(403,b'x'),(428,b'x'),(429,b'x'),(302,b'x'),
    (200,b'FOR SECURITY REASONS'),(200,b'<html>captcha</html>')])
def test_security_response_stops_other_dates_across_processes(tmp_path,status,body):
    c,r,items = setup(tmp_path,[response(status=status,body=body)])
    assert c.fetch(items[0])['status'] == 'origin_stopped'
    other = OfficialDailyAcquisition(tmp_path,c.cache,session=r,clock=r.clock,sleep=r.sleep)
    with pytest.raises(AcquisitionBlocked,match='hold active'):
        other.fetch(items[1])
    assert len(r.calls) == 1


def test_global_hold_requires_explicit_probe_and_is_never_changed(tmp_path):
    c,r,items = setup(tmp_path,[response(),response(day='2026-09-10')])
    h = hold(tmp_path,r)
    before = h.read_bytes()
    with pytest.raises(AcquisitionBlocked,match='hold active'): c.fetch(items[0])
    with pytest.raises(AcquisitionBlocked,match='disabled'): c.probe(items[0])
    with pytest.raises(AcquisitionBlocked,match='authorization'): c.probe(items[0],allow_probe=True)
    auth(c,r)
    proof = c.probe(items[0],allow_probe=True)
    assert proof['accepted'] and proof['request_kind'] == 'single_normal_probe'
    assert c.fetch(items[1])['accepted']
    assert h.read_bytes() == before and len(r.calls) == 2


def test_probe_failure_cannot_retry_another_date(tmp_path):
    c,r,items = setup(tmp_path,[response(status=503)])
    hold(tmp_path,r); auth(c,r)
    assert not c.probe(items[0],allow_probe=True)['accepted']
    with pytest.raises(AcquisitionBlocked,match='already consumed'):
        c.probe(items[1],allow_probe=True)
    assert len(r.calls) == 1


@pytest.mark.parametrize('mutation', ['stale','new_hold','wrong_endpoint','changed_bytes','wrong_authorization'])
def test_recovery_requires_fresh_exact_endpoint_bound_proof(tmp_path,mutation):
    c,r,items = setup(tmp_path,[response()])
    h = hold(tmp_path,r); a = auth(c,r)
    proof = c.probe(items[0],allow_probe=True)
    if mutation == 'stale': r.now += 86401
    elif mutation == 'new_hold': hold(tmp_path,r,delta=1)
    elif mutation == 'wrong_endpoint':
        row = json.loads((tmp_path/proof['receipt_path']).read_text())
        row['url'] = 'https://www.twse.com.tw/rwd/zh/afterTrading/TWTC7U'
        p = tmp_path/proof['receipt_path']; p.write_text(encoded(row)); p.with_suffix('.sha256').write_text(digest(p))
    elif mutation == 'changed_bytes': (tmp_path/proof['raw_path']).write_text('{}')
    else:
        row = json.loads(a.read_text()); row['security_bypass_authorized'] = True; a.write_text(encoded(row))
    with pytest.raises((AcquisitionBlocked,ValueError)):
        c.fetch(items[1])
    assert len(r.calls) == 1


def test_transport_and_interruption_never_repeat_attempt(tmp_path):
    c,r,items = setup(tmp_path,[requests.Timeout()])
    assert c.fetch(items[0])['status'] == 'transport_error_no_retry'
    assert c.fetch(items[0])['resumed_without_retry']
    c2,r2,i2 = setup(tmp_path,[RuntimeError('process interrupted')],cache='cache2',market='TPEX')
    with pytest.raises(RuntimeError): c2.fetch(i2[0])
    assert c2.fetch(i2[0])['status'] == 'interrupted_no_retry'
    with pytest.raises(AcquisitionBlocked,match='hold active'): c2.fetch(i2[1])
    assert len(r.calls) == len(r2.calls) == 1


def test_plan_source_mutation_and_off_plan_requests_fail(tmp_path):
    source = tmp_path/'input.json'; source.write_text('{}')
    item = request_item('TWSE','2026-09-09')
    create_plan(tmp_path,tmp_path/'cache',[item],{'input.json':digest(source)})
    source.write_text('[]')
    with pytest.raises(AcquisitionBlocked,match='source changed'):
        OfficialDailyAcquisition(tmp_path,tmp_path/'cache')
    c,r,items = setup(tmp_path,[],cache='cache2')
    with pytest.raises(AcquisitionBlocked,match='outside'): c.fetch(request_item('TWSE','2020-01-01'))
    with pytest.raises(AcquisitionBlocked,match='differs'):
        create_plan(tmp_path,c.cache,[items[0]],{})
    assert not r.calls


def test_per_origin_interval_is_shared_across_cache_instances(tmp_path):
    c,r,items = setup(tmp_path,[response(),response(day='2026-09-10')])
    create_plan(tmp_path,tmp_path/'cache2',[items[1]],{})
    other = OfficialDailyAcquisition(tmp_path,tmp_path/'cache2',session=r,clock=r.clock,sleep=r.sleep)
    c.fetch(items[0]); other.fetch(items[1])
    assert r.calls[1][1]-r.calls[0][1] >= 3.1


def test_low_interval_is_rejected(tmp_path):
    c,r,items = setup(tmp_path,[])
    with pytest.raises(AcquisitionBlocked,match='3.1'):
        OfficialDailyAcquisition(tmp_path,c.cache,min_interval=1.5)


def test_changed_success_receipt_or_payload_cannot_be_reused(tmp_path):
    c,r,items = setup(tmp_path,[response()])
    value = c.fetch(items[0])
    p = tmp_path/value['receipt_path']
    row = json.loads(p.read_text()); row['security_denied'] = True
    p.write_text(encoded(row))
    with pytest.raises(AcquisitionBlocked,match='hash changed'): c.fetch(items[0])
    p.with_suffix('.sha256').write_text(digest(p))
    with pytest.raises(AcquisitionBlocked,match='transport differs'): c.fetch(items[0])
    assert len(r.calls) == 1


def test_new_instance_can_use_successful_probe_only_for_same_plan(tmp_path):
    c,r,items = setup(tmp_path,[response(),response(day='2026-09-10')])
    h = hold(tmp_path,r); a = auth(c,r)
    proof = c.probe(items[0],allow_probe=True)
    other = OfficialDailyAcquisition(tmp_path,c.cache,authorization_path=a,
        recovery_proofs={'TWSE':proof['receipt_path']},session=r,clock=r.clock,sleep=r.sleep)
    assert other.fetch(items[1])['accepted'] and h.exists()


def test_hold_evidence_and_plan_mutation_stop_before_transport(tmp_path):
    c,r,items = setup(tmp_path,[])
    h = hold(tmp_path,r)
    evidence = tmp_path/'denied.html'; evidence.write_text('security challenge')
    row = json.loads(h.read_text()); row['evidence_sha256'] = {'denied.html':digest(evidence)}
    h.write_text(encoded(row)); evidence.write_text('changed')
    with pytest.raises(AcquisitionBlocked,match='evidence changed'): c.fetch(items[0])
    c.plan_path.write_text('{}')
    with pytest.raises(AcquisitionBlocked,match='plan changed'): c.fetch(items[0])
    assert not r.calls


def test_authorization_for_previous_odd_endpoint_is_not_accepted(tmp_path):
    c,r,items = setup(tmp_path,[])
    auth(c,r)
    row = json.loads(c.authorization.read_text())
    row['user_request'] = 'user go after explicit connection permission explanation'
    c.authorization.write_text(encoded(row))
    with pytest.raises(AcquisitionBlocked,match='only this user request'):
        c.probe(items[0],allow_probe=True)
    assert not r.calls


def test_independent_markets_dispatch_concurrently_in_same_cache(tmp_path):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    from threading import Event
    items = [request_item(m, '2026-09-09') for m in ('TWSE','TPEX')]
    create_plan(tmp_path,tmp_path/'cache',items,{})
    entered, release = Event(), Event()
    class Session:
        def get(self,url,**kwargs):
            assert kwargs['allow_redirects'] is False
            market = 'TWSE' if 'twse.com.tw' in url else 'TPEX'
            if market == 'TWSE':
                entered.set()
                assert release.wait(5), 'Test release timed out'
            return response(market)
    a = OfficialDailyAcquisition(tmp_path,tmp_path/'cache',session=Session())
    b = OfficialDailyAcquisition(tmp_path,tmp_path/'cache',session=Session())
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(a.fetch,items[0])
        assert entered.wait(2)
        try:
            second = pool.submit(b.fetch,items[1])
            assert second.result(timeout=2)['accepted']
            assert not first.done(), 'TWSE remains blocked on its own response'
            export = pool.submit(b.export_manifest,tmp_path/'concurrent-export.json')
            with pytest.raises(FutureTimeout):
                export.result(timeout=0.05)
        finally:
            release.set()
        assert first.result(timeout=2)['accepted']
        assert export.result(timeout=2)['accepted_market_days'] == 2


def test_exported_acquisition_receipts_are_accepted_by_supplement_reader(tmp_path):
    from skills.official_market_supplement import collect_supplement
    items = [request_item(m,'2026-09-09') for m in ('TWSE','TPEX')]
    create_plan(tmp_path,tmp_path/'cache',items,{})
    runtime = Runtime(tmp_path,[response('TWSE'),response('TPEX')])
    client = OfficialDailyAcquisition(tmp_path,tmp_path/'cache',session=runtime,
        clock=runtime.clock,sleep=runtime.sleep)
    for item in items:
        assert client.fetch(item)['accepted']
    manifest = tmp_path/'supplement.json'
    client.export_manifest(manifest)
    refs = {}
    official, descriptors = collect_supplement(manifest,tmp_path,refs)
    assert len(descriptors) == 2
    assert official['TWSE','2026-09-09','2330']['volume_scope'] == 'all_daily_sessions'
    assert official['TPEX','2026-09-09','2330']['volume_scope'] == 'ordinary_session'
    assert all(digest(tmp_path/name) == value for name,value in refs.items())
