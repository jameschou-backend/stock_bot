from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest
import requests

from skills import official_market_completion as completion
from skills.market_input_validation import MarketEvidenceError
from skills.official_daily_acquisition import (OfficialDailyAcquisition, USER_REQUEST,
    create_plan, digest, encoded, request_item)
from skills.official_market_supplement import validate_entry


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(encoded(value))
    path.with_suffix('.sha256').write_text(digest(path)+'\n')
    return path


def payload(market,day):
    if market == 'TWSE':
        table = dict(fields=['證券代號','證券名稱','開盤價','最高價','最低價','收盤價','成交股數'],
            notes=['本統計資訊含一般、零股、盤後定價、鉅額交易'],data=[['2330','測試','100','110','90','105','100050']])
        return dict(stat='OK',date=day.replace('-',''),type='ALLBUT0999',tables=[table])
    y,m,d = map(int,day.split('-'))
    table = dict(fields=['代號','名稱','開盤','最高','最低','收盤','成交股數'],
        title='上櫃股票每日收盤行情(不含定價)',date=f'{y-1911}/{m:02d}/{d:02d}',
        category='所有證券(不含權證、牛熊證)',totalCount=1,data=[['2330','測試','100','110','90','105','100000']])
    return dict(stat='ok',date=day.replace('-',''),tables=[table])


@pytest.fixture
def evidence(tmp_path,monkeypatch,request):
    # Small real receipt graph exercises the same exact set partition as production.
    counts = dict(REQUIRED_DAYS=8,ORIGINAL_MISSING=7,OFFLINE_RECOVERED=3,PLANNED_DOWNLOADS=4)
    for name,value in counts.items(): monkeypatch.setattr(completion,name,value)
    cached = [('TPEX','2026-09-07')]
    offline_keys = [('TWSE','2026-09-07'),('TWSE','2026-09-08'),('TPEX','2026-09-08')]
    planned = [request_item(m,d) for m in ('TWSE','TPEX') for d in ('2026-09-09','2026-09-10')]

    def legacy(market,day):
        raw = write(tmp_path/'old'/f'{market}-{day}.json',payload(market,day))
        item = request_item(market,day)
        receipt = write(tmp_path/'old'/f'{market}-{day}.source.json',dict(url=item['url'],params=item['params'],
            sha256=digest(raw),http_status=200,retrieved_at='2026-09-29T00:00:00+00:00'))
        return dict(market=market,date=day,raw_path=str(raw.relative_to(tmp_path)),raw_sha256=digest(raw),
            receipt_path=str(receipt.relative_to(tmp_path)),receipt_sha256=digest(receipt))
    cached_rows = [legacy(*key) for key in cached]
    parameter = getattr(request,'param',False)
    recover_count = parameter.get('recover',0) if isinstance(parameter,dict) else 0
    if parameter is True:
        cached_rows.append(legacy('TWSE','2018-01-02'))
    offline_rows = [legacy(*key) for key in offline_keys]
    original = tmp_path/'original.json'
    refs = {r[k]:r[h] for r in cached_rows for k,h in [('raw_path','raw_sha256'),('receipt_path','receipt_sha256')]}
    required = [dict(market=m,date=d,status='cached' if (m,d) in cached else 'source_missing')
                for m,d in cached+offline_keys+[(r['market'],r['date']) for r in planned]]
    original_descriptors = [dict(market=r['market'],date=r['date'],path=r['raw_path'],sha256=r['raw_sha256'],
        receipt=r['receipt_path'],rows=1,
        volume_scope='ordinary_session' if r['market']=='TPEX' else 'all_daily_sessions') for r in cached_rows]
    write(original,dict(schema='market_input_validation_v1',live_qualified=False,source_sha256=refs,
        request_plan=required,requests_lower_bound=7,sources=original_descriptors))
    monkeypatch.setattr(completion,'ORIGINAL_REPORT_SHA256',digest(original))
    offline = write(tmp_path/'offline.json',dict(schema='official_market_supplement_v1',live_qualified=False,
        entries=offline_rows,source_sha256={}))
    cache = tmp_path/'acquisition'
    create_plan(tmp_path,cache,planned,{'original.json':digest(original)})
    now = [datetime(2026,10,2,tzinfo=timezone.utc).timestamp()]
    failed_once = set()
    class Session:
        def get(self,url,params,**kwargs):
            market = 'TWSE' if 'twse.com.tw' in url else 'TPEX'
            day = (params['date'][:4]+'-'+params['date'][4:6]+'-'+params['date'][6:]
                   if market == 'TWSE' else str(int(params['d'][:3])+1911)+params['d'][3:].replace('/','-'))
            if recover_count and day == '2026-09-10' and (market == 'TWSE' or recover_count == 2) and market not in failed_once:
                failed_once.add(market)
                raise requests.ConnectionError()
            return SimpleNamespace(status_code=200,content=encoded(payload(market,day)).encode(),history=[])
    authorization = write(cache/'authorization.json',dict(schema='official_daily_authorization_v1',
        user_request=USER_REQUEST,scope='missing_official_daily_tables',security_bypass_authorized=False,
        created_at=datetime.fromtimestamp(now[0],timezone.utc).isoformat(),plan_sha256=digest(cache/'plan.json')))
    def sleep(seconds): now[0] += seconds
    client = OfficialDailyAcquisition(tmp_path,cache,session=Session(),authorization_path=authorization,
        clock=lambda:now[0],sleep=sleep)
    for item in planned:
        if item['date'] == '2026-09-09': client.probe(item,allow_probe=True)
        else:
            result = client.fetch(item)
            if not result['accepted']:
                legacy_hash = None
                if isinstance(parameter,dict) and parameter.get('legacy'):
                    base = cache/'receipts'/(item['identity']+'.json')
                    legacy = json.loads(base.read_text()); legacy.pop('exception_response_present')
                    write(base,legacy); legacy_hash = digest(base)
                client.recover_transport(item,'Agent reviewed connection failure and acknowledges any unrecorded response presence.',
                                         legacy_failure_sha256=legacy_hash)
    exported = tmp_path/'downloaded.json'
    downloaded = client.export_manifest(exported)
    final = write(tmp_path/'final.json',dict(schema='official_market_supplement_v1',live_qualified=False,
        entries=offline_rows+downloaded['entries'],source_sha256=downloaded['source_sha256']))
    hold = write(tmp_path/completion.HOLD_PATH,dict(status='blocked',no_automatic_recovery=True,evidence_sha256={}))
    monkeypatch.setattr(completion,'ORIGINAL_HOLD_SHA256',digest(hold))
    auditrefs = dict(refs)
    for row in offline_rows+downloaded['entries']:
        auditrefs[row['raw_path']] = row['raw_sha256']; auditrefs[row['receipt_path']] = row['receipt_sha256']
    auditrefs['final.json'] = digest(final)
    audit = write(tmp_path/'audit.json',dict(schema='market_input_validation_v2',live_qualified=False,
        source_sha256=auditrefs,request_plan=[dict(r,status='cached') for r in required],requests_lower_bound=0,
        sources=original_descriptors+[validate_entry(r,tmp_path)[1] for r in offline_rows+downloaded['entries']],
        complete_verified_data=False,checks={'all_historical_market_days_observed':True}))
    args = (tmp_path,original,offline,cache,final)
    return dict(args=args,audit=audit,hold=hold,planned=planned,cache=cache,root=tmp_path,final=final)


def test_full_receipt_partition_and_audit_are_required_for_completion(evidence):
    pending = completion.verify_completion(*evidence['args'])
    assert not pending['complete'] and pending['status'] == 'acquisition_complete_pending_audit'
    result = completion.verify_completion(*evidence['args'],final_audit=evidence['audit'],output=evidence['root']/'completion.json')
    assert result['complete'] and result['final_supplement_count'] == 7 and result['total_required_days'] == 8
    assert result['new_accepted_receipts_by_market'] == {'TWSE':2,'TPEX':2}
    assert result['single_normal_probes'] == 2
    assert result['original_hold_preserved'] and result['network_requests'] == 0
    assert result['live_qualified'] is result['backtest_qualified'] is result['return_recomputed'] is False
    assert result['final_audit']['complete_verified_data'] is False


@pytest.mark.parametrize('evidence',[True],indirect=True)
def test_original_outside_period_table_is_preserved_without_inflating_required_coverage(evidence):
    result = completion.verify_completion(*evidence['args'],final_audit=evidence['audit'])
    assert result['complete'] and result['total_required_days'] == 8
    assert result['original_cached_market_days'] == 1
    assert len(json.loads(evidence['audit'].read_text())['sources']) == 9


@pytest.mark.parametrize('evidence',[{'recover':1},{'recover':2},{'recover':2,'legacy':True}],indirect=True)
def test_completed_transport_recoveries_preserve_failure_history_and_actual_request_count(evidence):
    result = completion.verify_completion(*evidence['args'],final_audit=evidence['audit'])
    recovered = len(list((evidence['cache']/'recoveries').glob('*/receipt.json')))
    assert result['complete'] and recovered in (1,2)
    assert result['original_failed_receipts'] == result['failed_receipts'] == result['recovered_receipts'] == recovered
    assert result['new_http_requests'] == 4+recovered and result['new_accepted_receipts'] == 4
    assert result['unresolved_failures'] == 0
    assert len(result['original_failure_history']) == recovered
    for row in result['original_failure_history']:
        receipt = json.loads((evidence['root']/row['receipt_path']).read_text())
        assert receipt['accepted'] is False and receipt['http_status'] is None and 'raw_path' not in receipt
        assert row['exception_response_presence'] == ('recorded_absent' if 'exception_response_present' in receipt
                                                      else 'not_recorded_in_original_version')


@pytest.mark.parametrize('evidence',[{'recover':1}],indirect=True)
def test_completion_rejects_unknown_recovery_attempt(evidence):
    write(evidence['cache']/'recoveries'/'unknown'/'attempt.json',{})
    with pytest.raises(MarketEvidenceError,match='Unknown or unfinished'):
        completion.verify_completion(*evidence['args'])


@pytest.mark.parametrize('mutation',['missing','duplicate','unknown','raw_changed','receipt_changed','unfinished','failed','hold_changed','plan_changed'])
def test_incomplete_or_changed_acquisition_cannot_be_certified(evidence,mutation):
    root,original,offline,cache,final = evidence['args']
    item = evidence['planned'][0]
    receipt = cache/'receipts'/(item['identity']+'.json')
    if mutation in ('missing','duplicate','unknown'):
        value = json.loads(final.read_text())
        if mutation == 'missing': value['entries'].pop()
        elif mutation == 'duplicate': value['entries'].append(value['entries'][0])
        else: value['entries'][0] = dict(value['entries'][0],date='2020-01-01')
        write(final,value)
    elif mutation == 'raw_changed': (cache/'raw'/(item['identity']+'.bin')).write_text('{}')
    elif mutation == 'receipt_changed': receipt.write_text('{}')
    elif mutation == 'unfinished': receipt.unlink()
    elif mutation == 'failed':
        value = json.loads(receipt.read_text()); value['accepted'] = False
        write(receipt,value)
    elif mutation == 'hold_changed': evidence['hold'].write_text('{}')
    else: (cache/'plan.json').write_text('{}')
    with pytest.raises((MarketEvidenceError,FileNotFoundError,KeyError)):
        completion.verify_completion(*evidence['args'],final_audit=evidence['audit'])


@pytest.mark.parametrize('mutation',['missing_day','unknown_day','unbound_manifest','source_changed','duplicate_source','unknown_source',
    'wrong_scope','wrong_row_count','wrong_receipt_hash','wrong_http_status'])
def test_final_audit_must_bind_exact_complete_sources(evidence,mutation):
    value = json.loads(evidence['audit'].read_text())
    if mutation == 'missing_day': value['requests_lower_bound'] = 1; value['request_plan'][0]['status'] = 'source_missing'
    elif mutation == 'unknown_day': value['request_plan'][0]['date'] = '2020-01-01'
    elif mutation == 'unbound_manifest': value['source_sha256'].pop('final.json')
    elif mutation == 'source_changed': value['sources'][0]['sha256'] = '0'*64
    elif mutation == 'duplicate_source': value['sources'].append(value['sources'][0])
    elif mutation == 'unknown_source': value['sources'].append(dict(value['sources'][0],date='2020-01-01'))
    elif mutation == 'wrong_scope': value['sources'][1]['volume_scope'] = 'ordinary_session'
    elif mutation == 'wrong_row_count': value['sources'][1]['rows'] += 1
    elif mutation == 'wrong_receipt_hash': value['sources'][0]['receipt_sha256'] = '0'*64
    else: value['sources'][1]['http_status'] = None
    write(evidence['audit'],value)
    with pytest.raises(MarketEvidenceError):
        completion.verify_completion(*evidence['args'],final_audit=evidence['audit'])


def test_offline_plan_partition_cannot_overlap(evidence):
    root,original,offline,cache,final = evidence['args']
    value = json.loads(offline.read_text())
    key = evidence['planned'][0]
    value['entries'][0] = dict(value['entries'][0],market=key['market'],date=key['date'])
    write(offline,value)
    with pytest.raises(MarketEvidenceError): completion.verify_completion(*evidence['args'])


def test_unknown_attempt_cannot_be_hidden_by_complete_manifest(evidence):
    write(evidence['cache']/'attempts'/'unknown.json',{})
    with pytest.raises(MarketEvidenceError,match='unknown acquisition attempts'):
        completion.verify_completion(*evidence['args'])


def test_legacy_unknown_http_status_remains_unknown(evidence,monkeypatch):
    from skills import official_market_supplement as supplement
    root,original,offline,cache,final = evidence['args']
    old = json.loads(offline.read_text())
    row = old['entries'][0]
    receipt = root/row['receipt_path']; value = json.loads(receipt.read_text()); value.pop('http_status')
    write(receipt,value)
    row['receipt_sha256'] = digest(receipt)
    monkeypatch.setattr(supplement,'LEGACY_RECEIPT_SHA256',frozenset({digest(receipt)}))
    write(offline,old)
    complete = json.loads(final.read_text()); complete['entries'][0] = row; write(final,complete)
    result = completion.verify_completion(*evidence['args'])
    assert result['supplemental_http_status_evidence']['legacy_status_unknown'] == 1
    assert result['new_accepted_receipts'] == 4
