from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from app import backtest_full_pass_ui as ui
from skills.backtest_case_dependencies import configurations, dependencies
from skills.publication_versions import digest, encoded


PROJECT=Path(__file__).resolve().parents[1]


def put(root,name,value):
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(encoded(value))
    return dict(path=name,sha256=digest(path.read_bytes()))


def seal(path,value):
    path.write_bytes(encoded(value))
    path.with_suffix('.sha256').write_text(digest(path.read_bytes()))


@pytest.fixture
def full_pass(tmp_path,monkeypatch):
    root=tmp_path
    for name in ui.CODE:
        target=root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((PROJECT/name).read_bytes())
    base=put(root,'base.json',dict(schema='backtest_data_completion_v1',case_count=20))
    (root/'scripts/prepare_historical_cohort_supplement.py').write_text('# fixture\n')
    (root/'quotes.parquet').write_bytes(b'fixture raw prices')
    old={key:put(root,'previous/'+key+'.json',dict(kind=key))
         for key in ('ordinary','identity','odd_lot','publication')}
    parent=put(root,'artifacts/forward_simulation/backtest_data_followup_20260925.json',
        dict(schema='backtest_data_followup_v1',live_qualified=False,performance_recomputed=False,
             base_data_report=base,reports=old))
    def binding(descriptor):
        return {descriptor['path']:descriptor['sha256']}
    cases={name:dependencies(name,config,dict(name=name,ordinary={'required_sessions':12},
        odd_lot={'required_sessions':int(not config['board_only'])}))
        for name,config in configurations().items()}
    counts=Counter(row['code'] for case in cases.values() for row in case['dependencies'] if row['required'])
    sources=dict(
        ordinary=dict(summary=dict(required_sessions=917,same_scope_aggregate_matched=194,
            aggregate_conflicts=42,independent_daily_source_missing=681),input_sha256=binding(old['ordinary'])),
        identity=dict(remaining_unknown_starts=0,unconfirmed_categories=0,
            unresolved_current_date_discrepancies=0,source_sha256=binding(old['identity'])),
        odd_daily=dict(required_stock_days=13,statuses={'official_daily_positive_trade':13},
            input_sha256=binding(old['odd_lot'])),
        provider_refresh=dict(summary=dict(required=42,refreshed=42,aggregate_matched=0,conflicts=42),
            input_sha256=binding(old['ordinary'])),
        dependencies=dict(summary=dict(case_count=20,required_case_counts=dict(counts)),cases=cases,
            input_sha256=binding(base)),
        cohort_prices=dict(summary=dict(stock_count=49,quote_rows=35664,warmup_rows=11556,
            research_period_rows=24108,quarantined_rows=641),plan={'input_sha256':binding(old['identity'])}))
    for key,value in sources.items():
        value['schema']=ui._SCHEMAS[key]
    descriptors={key:put(root,key+'.json',value) for key,value in sources.items()}
    sources['external']=dict(historical_auction_sessions_acquired=0,missing_historical_auction_sessions=1303,
        issuer={'current_observed_historical_documents_received':1},input_sha256=binding(descriptors['odd_daily']),
        schema=ui._SCHEMAS['external'])
    descriptors['external']=put(root,'external.json',sources['external'])
    index=dict(schema='backtest_full_pass_v1',strict_data_ready=False,live_qualified=False,
        performance_recomputed=False,parent_followup=parent,reports=descriptors,
        code_sha256={name:digest((root/name).read_bytes()) for name in ui.CODE})
    path=root/'index.json'
    seal(path,index)
    seen=[]
    def validators(root):
        def checker(kind):
            def verify(path):
                seen.append(kind)
                return deepcopy(sources[kind])
            return verify
        return {key:checker(key) for key in ui.KINDS}
    monkeypatch.setattr(ui,'_validators',validators)
    return root,path,index,sources,seen


def test_all_reports_revalidate_their_sources_before_display(full_pass):
    root,path,index,sources,seen=full_pass
    actual,reports=ui.load(path,root)
    assert actual==index and reports==sources
    assert set(seen)==ui.KINDS and len(seen)==7


@pytest.mark.parametrize('key',['ordinary','identity','odd_daily','provider_refresh','dependencies','external','cohort_prices'])
def test_same_case_count_does_not_allow_coverage_from_another_scope(full_pass,key):
    root,path,index,sources,_=full_pass
    field='source_sha256' if key=='identity' else 'input_sha256'
    target=sources[key]['plan'] if key=='cohort_prices' else sources[key]
    target[field]={'different-experiment.json':'b'*64}
    index['reports'][key]=put(root,key+'.json',sources[key])
    seal(path,index)
    with pytest.raises(ValueError,match='範圍|本輪零股日表'):
        ui.load(path,root)


def test_changed_report_bytes_cannot_keep_a_previous_descriptor(full_pass):
    root,path,_,_,_=full_pass
    (root/'ordinary.json').write_text('{}')
    with pytest.raises(ValueError,match='已變動'):
        ui.load(path,root)


def test_changed_ui_code_requires_a_new_verified_publication(full_pass):
    root,path,_,_,_=full_pass
    code=root/ui.CODE[0]
    code.write_bytes(code.read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError,match='已變動'):
        ui.load(path,root)


@pytest.mark.parametrize('flag',['live_qualified','strict_data_ready','performance_recomputed'])
def test_full_pass_cannot_promote_strategy_or_claim_new_performance(full_pass,flag):
    root,path,index,_,_=full_pass
    index[flag]=True
    seal(path,index)
    with pytest.raises(ValueError,match='資格標記'):
        ui.load(path,root)


def test_validator_output_must_equal_displayed_report_bytes(full_pass,monkeypatch):
    root,path,_,_,_=full_pass
    original=ui._validators(root)
    original['ordinary']=lambda path:{'different':True}
    monkeypatch.setattr(ui,'_validators',lambda root:original)
    with pytest.raises(ValueError,match='核對期間'):
        ui.load(path,root)


def test_dependency_table_separates_sector_pool_and_board_only_execution(full_pass):
    _,_,_,sources,_=full_pass
    rows=ui.dependency_rows(sources['dependencies'])
    assert len(rows)==8
    for row in rows:
        assert row['公司行動證據']=='需要'
        assert row['營收／新聞版本']=='未使用'
        assert row['完整零股時序']==('需要' if row['交易方式']=='整張＋零股' else '不下零股單')
        assert row['歷史產業成員']==('需要' if row['策略'].startswith('相對強勢') else '未使用')
        assert row['歷史股票全集']==('僅核對0050自身' if row['策略']=='0050 持有基準' else '需要')


def test_inconsistent_aggregate_counts_cannot_be_displayed(full_pass):
    root,path,index,sources,_=full_pass
    sources['ordinary']['summary']['aggregate_conflicts']=0
    index['reports']['ordinary']=put(root,'ordinary.json',sources['ordinary'])
    seal(path,index)
    with pytest.raises(ValueError,match='同一股日集合'):
        ui.load(path,root)


def test_render_retains_conflicts_missing_sources_and_no_live_qualification(full_pass,monkeypatch):
    from streamlit.testing.v1 import AppTest
    root,path,_,_,_=full_pass
    monkeypatch.setattr(ui,'ROOT',root)
    monkeypatch.setattr(ui,'REPORT',path)
    app=AppTest.from_string('from app.backtest_full_pass_ui import render\nrender()').run()
    assert not app.exception and not app.error
    assert len(app.warning)==1 and '尚未補齊' in app.warning[0].value
    assert len(app.dataframe)==2 and len(app.get('download_button'))==7
    table=app.dataframe[0].value.to_string(index=False)
    assert '194 股日一致；42 股日衝突；681 股日缺完整來源' in table
    assert '仍缺 1,303 股日' in table
    assert '13／13 股日查得實際成交' in table
    assert '641 列價格缺漏／矛盾隔離' in table
    assert not app.success


@pytest.mark.parametrize('failure',[ValueError,RuntimeError])
def test_failed_source_verification_hides_counts_and_downloads(full_pass,monkeypatch,failure):
    from streamlit.testing.v1 import AppTest
    root,path,_,_,_=full_pass
    monkeypatch.setattr(ui,'ROOT',root)
    monkeypatch.setattr(ui,'REPORT',path)
    def failed(root):
        raise failure('底層原始來源已變動')
    monkeypatch.setattr(ui,'_validators',failed)
    app=AppTest.from_string('from app.backtest_full_pass_ui import render\nrender()').run()
    assert not app.exception and '底層原始來源已變動' in app.error[0].value
    assert not app.dataframe and not app.get('download_button') and not app.success


def test_publisher_checks_every_source_before_writing_output(full_pass,monkeypatch):
    from scripts import publish_backtest_full_pass as publisher
    root,_,index,_,seen=full_pass
    monkeypatch.setattr(publisher,'ROOT',root)
    output=root/'artifacts/full.json'
    value=publisher.publish(output,{key:root/item['path'] for key,item in index['reports'].items()})
    assert set(seen)==ui.KINDS
    assert ui.load(output,root)[0]==value
    assert value['code_sha256']==index['code_sha256']
    with pytest.raises(ValueError,match='new artifact'):
        publisher.publish(output,{key:root/item['path'] for key,item in index['reports'].items()})


def test_publisher_source_failure_leaves_no_published_artifact(full_pass,monkeypatch):
    from scripts import publish_backtest_full_pass as publisher
    root,_,index,_,_=full_pass
    monkeypatch.setattr(publisher,'ROOT',root)
    def failed(*args):
        raise ValueError('raw source changed')
    monkeypatch.setattr(publisher,'load',failed)
    output=root/'artifacts/full.json'
    with pytest.raises(ValueError,match='raw source changed'):
        publisher.publish(output,{key:root/item['path'] for key,item in index['reports'].items()})
    assert not output.exists() and not output.with_suffix('.sha256').exists()


def test_existing_sidecar_does_not_leave_an_orphan_index(full_pass,monkeypatch):
    from scripts import publish_backtest_full_pass as publisher
    root,_,index,_,_=full_pass
    monkeypatch.setattr(publisher,'ROOT',root)
    output=root/'artifacts/full.json'
    output.with_suffix('.sha256').write_text('prior artifact hash')
    with pytest.raises(ValueError,match='new artifact'):
        publisher.publish(output,{key:root/item['path'] for key,item in index['reports'].items()})
    assert not output.exists()


@pytest.fixture
def cached_full_pass(full_pass,monkeypatch):
    root,path,index,sources,seen=full_pass
    raw=put(root,'raw.json',{'price':10})
    code=root/'scripts/prepare_historical_cohort_supplement.py'
    code.parent.mkdir(parents=True,exist_ok=True)
    code.write_text('# cohort verifier fixture\n')
    quotes=root/'quotes.parquet'
    quotes.write_bytes(b'fixture raw prices')
    nested=put(root,'nested.json',dict(
        input_sha256={raw['path']:raw['sha256']},
        code_sha256={str(code.relative_to(root)):digest(code.read_bytes())},
        description={'path':'not-an-input-and-does-not-exist'}))
    sources['odd_daily']['source_sha256']={nested['path']:nested['sha256']}
    # Formal reports flatten their consumed transitive refs into these maps.
    sources['odd_daily']['input_sha256'].update({raw['path']:raw['sha256']})
    sources['cohort_prices'].update(schema='historical_cohort_local_prices_v1',
        quotes_sha256=digest(quotes.read_bytes()),code_sha256=digest(code.read_bytes()))
    for key in ('odd_daily','cohort_prices'):
        index['reports'][key]=put(root,key+'.json',sources[key])
    sources['external']['input_sha256']={index['reports']['odd_daily']['path']:index['reports']['odd_daily']['sha256']}
    index['reports']['external']=put(root,'external.json',sources['external'])
    seal(path,index)
    parent=root/index['parent_followup']['path']
    base=root/'base.json'
    for file in [parent,base,*[root/d['path'] for d in index['reports'].values()]]:
        file.with_suffix('.sha256').write_text(digest(file.read_bytes()))
    expected={p:digest(p.read_bytes()) for p in (
        root/raw['path'],root/nested['path'],code,quotes,parent,base,
        *[p.with_suffix('.sha256') for p in [parent,base,*[root/d['path'] for d in index['reports'].values()]]])}
    validators=ui._validators(root)
    original=validators['odd_daily']
    def verify(p):
        for source,wanted in expected.items():
            if digest(source.read_bytes())!=wanted:
                raise ValueError('fixture source changed')
        return original(p)
    validators['odd_daily']=verify
    monkeypatch.setattr(ui,'_validators',lambda root:validators)
    with ui._CACHE_LOCK:
        ui._VERIFIED_CACHE.clear()
    return full_pass


def test_ui_cache_reuses_success_without_reopening_json_or_verifying(cached_full_pass,monkeypatch):
    root,path,_,_,seen=cached_full_pass
    first=ui.load_for_ui(path,root)
    assert len(seen)==7
    def no_cold_read(*args,**kwargs):
        raise AssertionError('warm hit must only stat the retained dependency list')
    monkeypatch.setattr(ui,'_source_signatures',no_cold_read)
    second=ui.load_for_ui(path,root)
    assert second==first and second is not first and len(seen)==7
    # CLI/publication must continue to invoke all verifiers on every call.
    ui.load(path,root)
    assert len(seen)==14


@pytest.mark.parametrize('name',[
    'raw.json','nested.json','odd_daily.sha256','base.sha256',
    'artifacts/forward_simulation/backtest_data_followup_20260925.sha256',
    'scripts/prepare_historical_cohort_supplement.py','quotes.parquet',
    'app/backtest_full_pass_ui.py','base.json','identity.json','index.sha256',
])
def test_any_bound_source_change_invalidates_success_even_with_restored_mtime(cached_full_pass,name):
    root,path,_,_,_=cached_full_pass
    ui.load_for_ui(path,root)
    source=root/name
    before=source.stat()
    data=source.read_bytes()
    source.write_bytes(bytes([data[0]^1])+data[1:])
    os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert source.stat().st_size==before.st_size
    assert source.stat().st_mtime_ns==before.st_mtime_ns
    with pytest.raises((ValueError,OSError)):
        ui.load_for_ui(path,root)
    assert (root,path) not in ui._VERIFIED_CACHE


def test_missing_source_cannot_return_previous_success(cached_full_pass):
    root,path,_,_,_=cached_full_pass
    ui.load_for_ui(path,root)
    (root/'raw.json').unlink()
    with pytest.raises(OSError):
        ui.load_for_ui(path,root)
    assert not ui._VERIFIED_CACHE


def test_verification_time_mutation_is_rejected_even_if_validator_returns_success(cached_full_pass,monkeypatch):
    root,path,_,_,_=cached_full_pass
    original=ui.load
    def changing(*args,**kwargs):
        result=original(*args,**kwargs)
        (root/'raw.json').write_text('{"price":20}')
        return result
    monkeypatch.setattr(ui,'load',changing)
    with pytest.raises(ValueError,match='完整驗證期間'):
        ui.load_for_ui(path,root)
    assert not ui._VERIFIED_CACHE


def test_errors_are_not_cached(cached_full_pass,monkeypatch):
    root,path,_,_,seen=cached_full_pass
    original=ui.load
    def failed(*args,**kwargs):
        raise ValueError('temporary verification failure')
    monkeypatch.setattr(ui,'load',failed)
    with pytest.raises(ValueError):
        ui.load_for_ui(path,root)
    assert not ui._VERIFIED_CACHE
    monkeypatch.setattr(ui,'load',original)
    ui.load_for_ui(path,root)
    assert len(seen)==7


def test_returned_values_cannot_mutate_the_success_cache(cached_full_pass):
    root,path,_,_,_=cached_full_pass
    cold=ui.load_for_ui(path,root)
    cold[1]['ordinary']['summary']['aggregate_conflicts']=0
    warm=ui.load_for_ui(path,root)
    assert warm[1]['ordinary']['summary']['aggregate_conflicts']==42
    warm[0]['live_qualified']=True
    assert ui.load_for_ui(path,root)[0]['live_qualified'] is False


def test_concurrent_sessions_share_one_successful_verification(cached_full_pass,monkeypatch):
    root,path,_,_,seen=cached_full_pass
    started,release=Event(),Event()
    original=ui.load
    calls=[]
    def slow(*args,**kwargs):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return original(*args,**kwargs)
    monkeypatch.setattr(ui,'load',slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(ui.load_for_ui,path,root)
        assert started.wait(5)
        second=pool.submit(ui.load_for_ui,path,root)
        release.set()
        assert first.result()==second.result()
    assert len(calls)==1 and len(seen)==7


def test_unknown_report_schema_requires_an_explicit_closure_review(cached_full_pass):
    root,path,index,sources,_=cached_full_pass
    sources['ordinary']['schema']='new-unreviewed-schema'
    index['reports']['ordinary']=put(root,'ordinary.json',sources['ordinary'])
    seal(path,index)
    with pytest.raises(ValueError,match='尚未審核'):
        ui.load_for_ui(path,root)
    assert not ui._VERIFIED_CACHE
