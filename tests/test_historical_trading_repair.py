from copy import deepcopy
import json

import pandas as pd
import pytest

from skills import historical_trading_repair as repair
from skills.historical_universe_completion import resolve_completion
from skills.historical_selector_replay import eligibility_matrix


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def seal(root, value):
    path = root / 'repair.json'
    write(path, value)
    path.with_suffix('.sha256').write_text(repair.sha(path))
    return path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    sources, closure, approved = {}, {}, {}

    def source(key, value, raw=False):
        p = tmp_path / (key + ('.html' if raw else '.json'))
        p.write_text(value) if raw else write(p, value)
        approved[key] = closure[p.name] = repair.sha(p)
        sources[key] = dict(path=p.name, sha256=repair.sha(p))

    source('managed_period', dict(source_kind='official_pdf_web_extraction',
        source_url=repair.TAIFEX_URL, raw_pdf_bytes_obtained=False,
        pdf_pages_1based=[1], publication_date='2019-11-29',
        extraction='4415 台原藥公司 普通股股票自106年11月21日開始為櫃檯買賣管理股票 '
                   '自108年12月16日起終止該公司之有價證券櫃檯買賣'))
    fields = ['代號', '開盤', '最高', '最低', '收盤', '成交股數']
    source('managed_daily', dict(date='20190703', stat='ok', tables=[dict(title='管理股票',
        totalCount=1, fields=fields, data=[['4415', '6.00', '6.00', '6.00', '6.00', '2,364']])]))
    source('managed_daily_receipt', dict(url=repair.DAILY_URL,
        params={'date':'2019/07/03','response':'json'}, http_status=200,
        raw_sha256=sources['managed_daily']['sha256']))
    source('face_announcement', '<p>7780 發言日期114/12/22 發言時間17:51:30 '
        '舊股票最後交易日:民國115年1月8日 '
        '舊股票停止交易期間:民國115年1月9日至115年1月17日 '
        '新股票上市買賣日:民國115年1月19日</p>', raw=True)
    source('face_announcement_receipt', dict(url=repair.MOPS_URL, http_status=200,
        raw_sha256=sources['face_announcement']['sha256'],
        params=dict(h311='7780', h312='20251222', h313='175130', h315='1')))
    source('face_restore', dict(data=[['115/01/19','7780','7780,20260109,20260119']]))
    source('face_summary', dict(note='legacy source closure is retained'))
    source('daily_scope_excerpt', dict(source_kind='official_page_search_excerpt',
        raw_html_obtained=False, original_tool_output_saved=False,
        excerpt='上櫃股票行情(含等價、零股、盤後、鉅額交易)'))
    dates = ['2019-06-14','2019-06-25','2019-06-27','2019-07-01','2019-07-02',
             '2019-07-03','2019-07-04','2019-07-08','2019-07-09']
    original, gaps = [], []
    for day in dates:
        raw_path, receipt_path = day+'.json', day+'.receipt.json'
        write(tmp_path/raw_path, dict(date=day)); write(tmp_path/receipt_path, dict(http_status=200))
        for name in (raw_path, receipt_path): closure[name] = repair.sha(tmp_path/name)
        descriptor = dict(path=raw_path, sha256=closure[raw_path], receipt=receipt_path,
                          receipt_sha256=closure[receipt_path], volume_scope='ordinary_session')
        quote = dict(stock_id='4415', date=day, open=6., high=6., low=6., close=6.)
        original.append(dict(quote=quote, ordinary_session_shares=2000, source_descriptor=descriptor))
        known = day == '2019-07-03'
        gaps.append(dict(date=day, stock_id='4415', raw_ohlc={k:quote[k] for k in ('open','high','low','close')},
            ordinary_session_shares=2000, ordinary_source=descriptor,
            independent_adjusted_price_verified=False, quality_gate_passed=False,
            ordinary_quote_not_promoted_to_total=True, total_daily_shares=2364 if known else None,
            total_source=sources['managed_daily'] if known else None,
            total_volume_scope='dailyQuotes_reported_all_sessions' if known else 'unknown'))
    source('ordinary_gap_audit', dict(missing_positive_quotes=dict(ordinary_quotes=original)))
    entries = [dict(stock_id='4415', market='TPEx', kind='managed_board', start='2017-11-21',
        end='2019-12-16', interval='start_inclusive_end_exclusive', legal_security_type='ordinary_share',
        publication_date='2019-11-29', source_path=sources['managed_period']['path']),
        dict(stock_id='7780', market='TWSE', kind='trading_suspension', start='2026-01-09',
        end='2026-01-19', interval='start_inclusive_end_exclusive',
        announcement_at='2025-12-22T17:51:30+08:00', announcement_date='2025-12-22', known_by='2025-12-23',
        announced_last_old_trading_date='2026-01-08', announced_suspension_last_date_inclusive='2026-01-17',
        announced_new_trading_date='2026-01-19', source_path=sources['face_announcement']['path'])]
    manifest = dict(schema=repair.SCHEMA, live_qualified=False, performance_recomputed=False,
        publication_time_archive_complete=False, complete_historical_universe=False,
        sources=sources, source_sha256=closure, entries=entries, quote_gaps_4415=gaps)
    base = dict(episodes=[dict(stock_id='4415',market='TPEx',category='股票',start='2018-01-02',end='2019-12-16'),
        dict(stock_id='7780',market='TWSE',category='股票',start='2025-09-09',end=None,snapshot_date='2026-09-14'),
        dict(stock_id='0050',market='TWSE',category='ETF',start='2018-01-02',end=None,snapshot_date='2026-09-14')],
        trading_exclusions=[],coverage_start='2018-01-02',coverage_end='2026-09-09',source_sha256={})
    monkeypatch.setattr(repair, 'REVIEWED_SOURCES', approved)
    return tmp_path, manifest, base


def load(evidence):
    root, manifest, _ = evidence
    return repair.load_trading_repair(seal(root, manifest), root)


def test_exclusion_boundaries_preserve_legal_listing_and_missing_data(evidence):
    before = deepcopy(evidence[2])
    result = repair.apply_trading_repair(evidence[2], load(evidence))
    assert before == evidence[2]
    assert result['episodes'] == before['episodes']
    assert resolve_completion(result,'4415','2019-06-14')['status'] == 'not_general_board'
    assert resolve_completion(result,'4415','2019-12-15')['status'] == 'not_general_board'
    assert resolve_completion(result,'4415','2019-12-16')['status'] != 'not_general_board'
    for day, expected in [('2026-01-08','identified'),('2026-01-09','official_trading_suspension'),
                          ('2026-01-18','official_trading_suspension'),('2026-01-19','identified')]:
        assert resolve_completion(result,'7780',day)['status'] == expected
    gaps = result['trading_repair']['quote_gaps_4415']
    assert sum(r['total_daily_shares'] is None for r in gaps) == 8
    assert not any(r['quality_gate_passed'] for r in gaps)
    assert not any(result[k] for k in ('performance_recomputed','live_qualified','complete_historical_universe'))
    assert 'quotes' not in result


def test_eligibility_masks_management_and_halt_without_dropping_listing(evidence):
    result = repair.apply_trading_repair(evidence[2], load(evidence))
    days = pd.to_datetime(['2019-06-14','2026-01-08','2026-01-09','2026-01-19'])
    mask = eligibility_matrix(result,pd.DataFrame({'stock_id':['4415','7780']}),days)
    assert not mask['4415'].any()
    assert mask['7780'].tolist() == [False,True,False,True]


@pytest.mark.parametrize('mutation', ['date','end','known_by','kind','market','duplicate','target','live',
    'fake_quality','fake_total','fake_ordinary','fake_price','fake_source','remove_closure','escape'])
def test_resealed_semantic_changes_are_rejected(evidence, mutation):
    root, m, _ = evidence
    if mutation=='date':m['entries'][0]['start']='2017-11-20'
    elif mutation=='end':m['entries'][1]['end']='2026-01-20'
    elif mutation=='known_by':m['entries'][1]['known_by']='2025-12-22'
    elif mutation=='kind':m['entries'][0]['kind']='trading_suspension'
    elif mutation=='market':m['entries'][0]['market']='TWSE'
    elif mutation=='duplicate':m['entries'].append(deepcopy(m['entries'][0]))
    elif mutation=='target':m['entries'][0]['stock_id']='2330'
    elif mutation=='live':m['live_qualified']=True
    elif mutation=='fake_quality':m['quote_gaps_4415'][0]['quality_gate_passed']=True
    elif mutation=='fake_total':m['quote_gaps_4415'][0]['total_daily_shares']=2000
    elif mutation=='fake_ordinary':m['quote_gaps_4415'][5]['ordinary_session_shares']=2364
    elif mutation=='fake_price':m['quote_gaps_4415'][0]['raw_ohlc']['high']=7
    elif mutation=='fake_source':m['quote_gaps_4415'][5]['total_source']=None
    elif mutation=='remove_closure':m['source_sha256'].pop(m['quote_gaps_4415'][0]['ordinary_source']['path'])
    elif mutation=='escape':m['source_sha256']['../out.json']='0'*64
    with pytest.raises(ValueError):load(evidence)


def test_source_reseal_requires_fresh_review(evidence):
    root, m, _ = evidence
    source=m['sources']['face_announcement'];p=root/source['path']
    p.write_text(p.read_text().replace('1月9日','1月8日'))
    m['source_sha256'][source['path']]=source['sha256']=repair.sha(p)
    with pytest.raises(ValueError,match='independent review'):load(evidence)


def test_full_closure_and_manifest_seal(evidence):
    root,m,_=evidence;path=seal(root,m);refs={}
    repair.load_trading_repair(path,root,refs)
    assert refs == dict(m['source_sha256'],**{'repair.json':repair.sha(path)})
    with pytest.raises(ValueError,match='closure conflict'):
        repair.load_trading_repair(path,root,{'managed_period.json':'0'*64})
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='manifest hash'):repair.load_trading_repair(path,root)


def test_changed_loaded_payload_unrepaired_ipo_and_overlap_rejected(evidence):
    r=load(evidence);r['entries'][0]['start']='2000-01-01'
    with pytest.raises(ValueError,match='unchanged loaded'):repair.apply_trading_repair(evidence[2],r)
    r=load(evidence);base=deepcopy(evidence[2]);base['episodes'][1]['start']='2026-01-19'
    with pytest.raises(ValueError):repair.apply_trading_repair(base,r)
    base=deepcopy(evidence[2]);base['trading_exclusions']=[dict(stock_id='7780',market='TWSE',
        kind='information_halt',start='2026-01-08',end='2026-01-10')]
    with pytest.raises(ValueError,match='Overlapping'):repair.apply_trading_repair(base,r)
    result=repair.apply_trading_repair(evidence[2],r)
    with pytest.raises(ValueError,match='already applied'):repair.apply_trading_repair(result,r)
