"""Synthetic account research tests; never open historical sources or use an API."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import research_exit_scenarios as driver
from skills.million_replay import Replay
from skills.scenario_exit_replay import ExitSignals


class Feeds:
    def __init__(self, quotes, directory, offline):
        self.quotes = quotes.set_index(['date','stock_id'])
        self.days = sorted(quotes.date.unique())
        self.directory, self.offline = directory, offline

    def get_limits(self, sid):
        return {str(pd.Timestamp(day).date()):{'upper':100_000.,'lower':.001} for day in self.days}

    def get_odd(self, day, sid, market):
        price = float(self.quotes.loc[(pd.Timestamp(day),sid),'close'])
        return dict(odd_shares=100_000,odd_last=price,odd_bid=price-.01,
                    odd_ask=price+.01,bid_qty=10_000,ask_qty=10_000)

    def manifest(self):
        index = self.directory/'index.json'
        result = driver.read(index)
        for name,digest in result['files_sha256'].items():
            if driver.sha(self.directory/name) != digest:
                raise ValueError('Synthetic execution evidence changed')
        return {**result,'manifest_sha256':driver.sha(index),
                'cache_directory':str(self.directory.resolve())}


class Corporate:
    def __init__(self, directory, offline):
        self.directory, self.offline = directory, offline
        self.requests, self.loaded, self.request_observer = 0,set(),None

    def prepare(self, sid):
        if not self.offline and sid not in self.loaded:
            self.requests += 1
            if self.request_observer:
                self.request_observer(self.requests)
        self.loaded.add(sid)

    def on_date(self, sid, day):
        return []

    def manifest(self):
        return {'files_sha256':{'seed.parquet':driver.sha(self.directory/'seed.parquet')},
                'overrides':{},'requests':self.requests}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    child = tmp_path/'inputs'
    feedpath,divpath = child/'execution-feeds',child/'dividends'
    feedpath.mkdir(parents=True); divpath.mkdir()
    (feedpath/'source.json').write_text('{}')
    (divpath/'seed.parquet').write_bytes(b'synthetic dividend evidence')
    seed_index = dict(schema=1,parser_sha256='synthetic',entries={'one':'source'},
        files_sha256={'source.json':driver.sha(feedpath/'source.json')},limitations=['daily only'],
        request_counters={'official_http_requests':10,'finmind_requests':2})
    driver.write(feedpath/'index.json',seed_index)
    driver.write(child/'manifest.json',dict(seed_files_sha256={
        'execution-feeds/source.json':driver.sha(feedpath/'source.json'),
        'dividends/seed.parquet':driver.sha(divpath/'seed.parquet')},
        seed_execution_index=seed_index,seed_dividend_hashes={'seed.parquet':driver.sha(divpath/'seed.parquet')}))
    (tmp_path/'spec.md').write_text('fixed synthetic specification')
    (tmp_path/'code.py').write_text('# synthetic code identity')
    driver.write(tmp_path/'overrides.json',{'overrides':{}})
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'INPUT',child)
    monkeypatch.setattr(driver,'OVERRIDES',tmp_path/'overrides.json')

    days = pd.bdate_range('2021-01-04',periods=210)
    adjusted = pd.DataFrame({'0050':100.+np.arange(len(days))*.1,
                            '1101':np.full(len(days),100.)},index=days)
    adjusted.loc[days[132]:,'1101'] = 85.
    quotes = pd.DataFrame([dict(date=day,stock_id=sid,open=price,high=price+1,
        low=price-1,close=price,volume=2_000_000) for day in days for sid,price in [('0050',100.),('1101',50.)]])
    companies = pd.DataFrame([dict(stock_id='1101',name='Synthetic',market='TWSE')])
    entries = [dict(event_id='first',members=['1101'],priority=.1,
        signal_date=str(days[129].date()),entry_date=str(days[130].date()))]
    data = driver.RunInputs(quotes,companies,days,entries,pd.DataFrame(),ExitSignals(adjusted,days),{},
                            str(days[129].date()),str(days[-1].date()))
    calls = []
    def pair(data, *, offline):
        calls.append(offline)
        return Feeds(data.quotes,feedpath,offline),Corporate(divpath,offline)
    for mode in ('strategy','benchmark'):
        feeds,corp = pair(data,offline=True)
        data.parent[mode] = Replay(quotes,companies,days,entries,feeds,corp,
            start=data.start,end=data.end,benchmark=mode=='benchmark').run()
    calls.clear()
    def context():
        seed = driver.read(child/'manifest.json')
        for name,digest in seed['seed_files_sha256'].items():
            if driver.sha(child/name) != digest:
                raise ValueError('Synthetic seed changed')
        return dict(schema=1,mode_order=list(driver.MODES),source_files_sha256={
            name:driver.sha(tmp_path/name) for name in ('spec.md','overrides.json','inputs/manifest.json')},
            code_sha256={'code.py':driver.sha(tmp_path/'code.py')},runtime_versions={'synthetic':'1'},
            child_cache_directory=str(child))
    monkeypatch.setattr(driver,'verify_sources',context)
    monkeypatch.setattr(driver,'load_inputs',lambda:data)
    monkeypatch.setattr(driver,'provider_pair',pair)
    monkeypatch.setattr(driver,'ReplayMarketFeeds',lambda *args,**kwargs:Feeds(quotes,feedpath,True))
    return dict(root=tmp_path,input=child,output=tmp_path/'output',data=data,pair=pair,calls=calls)


def test_seals_every_case_then_offline_reproduces_and_exposes_light_summary(sandbox):
    output = sandbox['output']
    report = driver.research(output)
    assert list(report['cases']) == list(driver.MODES)
    assert all(report[key] is False for key in ('unseen_validation','live_qualified','auto_promote'))
    assert report['cases']['fixed63']['account'] == sandbox['data'].parent['strategy']
    assert report['benchmark']['account'] == sandbox['data'].parent['benchmark']
    summary = driver.read(output/'summary.json')
    assert 'cases' not in summary and 'account' not in summary
    assert len(summary['comparisons']) == 7
    assert {r['mode'] for r in summary['annual']} == {*driver.MODES,'benchmark'}
    assert summary['case_files']['loss12'] == 'cases/loss12.json'
    assert '配股' in summary['complete_exit_date_definition']
    assert report['performance']['online_preparation_corporate_requests'] == 2
    assert report['performance']['offline_api_requests'] == 0
    manifest = driver.verify_report(output)
    inventory = manifest['verification_files_sha256']
    assert {'code.py','spec.md','inputs/manifest.json','inputs/execution-feeds/index.json',
            'inputs/dividends/seed.parquet','output/summary.json','output/cases/adaptive.json'} <= set(inventory)
    assert 'output/manifest.json' not in inventory
    assert driver.research(output,offline=True) == report
    with pytest.raises(ValueError,match='Sealed exit research exists'):
        driver.research(output)


def test_interrupted_preparation_resumes_completed_cases_and_deduplicates_provider_counts(sandbox,monkeypatch):
    original = driver.run_case
    attempted = []
    failed = False
    def interrupted(mode,data,feeds,corp):
        nonlocal failed
        attempted.append((mode,feeds.offline))
        if mode == 'trail20_12' and not feeds.offline and not failed:
            failed = True
            raise RuntimeError('Synthetic source unavailable')
        return original(mode,data,feeds,corp)
    monkeypatch.setattr(driver,'run_case',interrupted)
    with pytest.raises(RuntimeError,match='source unavailable'):
        driver.research(sandbox['output'])
    assert not (sandbox['output']/'manifest.json').exists()
    assert (sandbox['output']/'cases/loss12.manifest.json').exists()
    attempted.clear()
    report = driver.research(sandbox['output'])
    assert ('loss12',False) not in attempted
    assert ('fixed63',True) in attempted  # Its mandatory final offline verification still runs.
    assert report['performance']['online_preparation_corporate_requests'] == 4
    assert len(driver.read(sandbox['output']/'preparation_sessions.json')['sessions']) == 2


def test_failed_case_is_not_checkpointed_and_source_error_is_not_a_fallback(sandbox,monkeypatch):
    original = driver.run_case
    def fail(mode,*args):
        if mode == 'loss12':
            raise ValueError('Missing official historical date')
        return original(mode,*args)
    monkeypatch.setattr(driver,'run_case',fail)
    with pytest.raises(ValueError,match='Missing official'):
        driver.research(sandbox['output'])
    assert not (sandbox['output']/'cases/loss12.manifest.json').exists()
    assert not (sandbox['output']/'manifest.json').exists()


@pytest.mark.parametrize('name',['spec.md','code.py','overrides.json'])
def test_changed_context_refuses_resume(sandbox,name):
    output = sandbox['output']; output.mkdir()
    driver.write(output/'run.json',{'context':driver.verify_sources()})
    (sandbox['root']/name).write_text('changed')
    with pytest.raises(ValueError,match='new explicit --output'):
        driver.research(output)


@pytest.mark.parametrize('mode',['fixed63','benchmark'])
def test_control_and_benchmark_must_exactly_match_parent(sandbox,mode):
    data = deepcopy(sandbox['data'])
    data.parent['strategy' if mode=='fixed63' else 'benchmark']['daily'][0]['nav'] += 1
    feeds,corp = sandbox['pair'](data,offline=True)
    with pytest.raises(ValueError,match='exactly reproduce'):
        driver.run_case(mode,data,feeds,corp)


def test_offline_decision_difference_prevents_sealing(sandbox,monkeypatch):
    original = driver.run_case
    def mismatch(mode,data,feeds,corp):
        case = original(mode,data,feeds,corp)
        if mode=='loss12' and feeds.offline:
            case['exit_decisions'][0]['phase'] = 'tampered'
        return case
    monkeypatch.setattr(driver,'run_case',mismatch)
    with pytest.raises(ValueError,match='Offline account/decision/state'):
        driver.research(sandbox['output'])
    assert not (sandbox['output']/'manifest.json').exists()


def test_checkpoint_hash_and_append_only_provider_evidence(sandbox):
    context = driver.verify_sources()
    feeds,corp = sandbox['pair'](sandbox['data'],offline=True)
    case = driver.run_case('fixed63',sandbox['data'],feeds,corp)
    driver.save_case(sandbox['output'],case,context,driver.provider_snapshot(feeds,corp))
    index_path = sandbox['input']/'execution-feeds/index.json'
    index = driver.read(index_path)
    index['entries']['new'] = 'new evidence'
    index['request_counters']['official_http_requests'] += 1
    driver.write(index_path,index)
    assert driver.load_case(sandbox['output'],'fixed63',context) == case
    index['entries']['one'] = 'mutated evidence'
    driver.write(index_path,index)
    with pytest.raises(ValueError,match='execution evidence changed'):
        driver.load_case(sandbox['output'],'fixed63',context)
    path = sandbox['output']/'cases/fixed63.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='Completed case changed'):
        driver.load_case(sandbox['output'],'fixed63',context)


@pytest.mark.parametrize('changed',['output','code','seed','new_dividend','index','inventory'])
def test_sealed_verification_detects_changed_evidence(sandbox,changed):
    output = sandbox['output']
    driver.research(output)
    if changed=='output':
        (output/'summary.json').write_text('{}')
    elif changed=='code':
        (sandbox['root']/'code.py').write_text('different')
    elif changed=='seed':
        (sandbox['input']/'dividends/seed.parquet').write_bytes(b'changed')
    elif changed=='new_dividend':
        (sandbox['input']/'dividends/new.parquet').write_bytes(b'new source')
    elif changed=='index':
        index = sandbox['input']/'execution-feeds/index.json'
        index.write_text(index.read_text()+' ')
    else:
        manifest = driver.read(output/'manifest.json')
        del manifest['verification_files_sha256']['code.py']
        driver.write(output/'manifest.json',manifest)
    with pytest.raises(ValueError):
        driver.verify_report(output)


def test_exit_reason_count_is_once_per_position_and_delay_excludes_etf_funding():
    days = pd.bdate_range('2026-01-05',periods=6)
    iso = [str(day.date()) for day in days]
    states = {'one':dict(stock_id='1101',trigger_reason='loss12',signal_date=iso[0],target_date=iso[1])}
    def trade(seq,sid,side,day,qty,**kwargs):
        return dict(sequence=seq,event_id='one',stock_id=sid,side=side,date=day,qty=qty,channel='odd',**kwargs)
    account = dict(receivables=[], corporate_actions=[], cohorts=[dict(event_id='one',exit_date=iso[5])],trades=[
        trade(1,'0050','sell',iso[0],100,odd_bid_qty=100),
        trade(2,'1101','buy',iso[0],200,odd_ask_qty=100,odd_bid_qty=1000),
        trade(3,'1101','sell',iso[3],100,odd_bid_qty=100,odd_ask_qty=0),
        trade(4,'1101','sell',iso[5],100)])
    result = driver.exit_statistics(account,states,days)
    wait = result['exit_waits'][0]
    assert result['reason_counts']=={'loss12':1}
    assert wait['first_fill_date']==iso[3] and wait['complete_exit_date']==iso[5]
    assert wait['signal_to_first_fill_sessions']==3
    assert wait['target_to_first_fill_sessions']==2 and wait['target_to_complete_sessions']==4
    assert result['depth_audit']['exceeds_depth_trade_sequences']==[2]
    assert result['depth_audit']['missing_depth_trade_sequences']==[4]
    account['cohorts'][0]['exit_date'] = None
    assert driver.exit_statistics(account,states,days)['pending_exit_positions']==1
    states['one']['target_date'] = iso[0]
    with pytest.raises(ValueError,match='immediately previous'):
        driver.exit_statistics(account,states,days)


def test_load_reads_full_pool_once_and_uses_explicit_date_column(tmp_path,monkeypatch):
    child = tmp_path/'child'; parent = tmp_path/'parent'
    child.mkdir(); parent.mkdir()
    refs = {}
    for name in ('quotes','companies','calendar','close_official','events','signals'):
        path = tmp_path/(name+'.json')
        path.write_text('{}')
        refs[name] = {'path':path.name}
    entries = [dict(event_id=str(i),members=['1101']) for i in range(458)]
    driver.write(tmp_path/'signals.json',{'entries':entries})
    driver.write(child/'manifest.json',{'references':refs})
    driver.write(parent/'report.json',{})
    monkeypatch.setattr(driver,'ROOT',tmp_path); monkeypatch.setattr(driver,'INPUT',child)
    monkeypatch.setattr(driver,'PARENT',parent)
    days = pd.bdate_range('2022-01-03',periods=125)
    calls = []
    def parquet(path,**kwargs):
        calls.append((Path(path).stem,kwargs))
        if Path(path).stem=='calendar':
            return pd.DataFrame({'date':days.astype(str),'is_open':True})
        if Path(path).stem=='close_official':
            assert kwargs['columns']==['date','0050','1101']
            return pd.DataFrame({'date':days.astype(str),'0050':100.,'1101':50.})
        return pd.DataFrame()
    monkeypatch.setattr(pd,'read_parquet',parquet)
    loaded = driver.load_inputs()
    assert loaded.features.days.equals(days)
    assert loaded.features.adjusted_close.index.equals(days)
    assert [name for name,_ in calls].count('quotes') == 1
    assert next(options for name,options in calls if name=='quotes')['filters']==[('stock_id','in',['0050','1101'])]


def test_unsafe_or_duplicate_json_evidence_is_rejected(tmp_path):
    path = tmp_path/'value.json'
    path.write_text('{"same":1,"same":2}')
    with pytest.raises(ValueError,match='Duplicate'):
        driver.read(path)
    path.write_text('{"same":NaN}')
    with pytest.raises(ValueError,match='Non-finite'):
        driver.read(path)
    with pytest.raises(ValueError,match='Unsafe'):
        driver._safe(tmp_path,'../elsewhere')
    (tmp_path/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='symlinked'):
        driver._safe(tmp_path,'link/value.json')


def test_recorded_corporate_requests_survive_a_post_fetch_failure(monkeypatch):
    corp = driver.TrackedCorporateActions.__new__(driver.TrackedCorporateActions)
    corp.requests = 0
    observed = []
    corp.request_observer = observed.append
    def fetched_then_invalid(self,sid):
        self.requests += 1
        raise ValueError('Synthetic invalid dividend date')
    monkeypatch.setattr(driver.CorporateActions,'prepare',fetched_then_invalid)
    with pytest.raises(ValueError,match='invalid dividend'):
        corp.prepare('1101')
    assert observed == [1]


def test_source_change_during_a_case_stops_before_checkpoint(sandbox,monkeypatch):
    original = driver.run_case
    def changed(mode,*args):
        case = original(mode,*args)
        if mode=='loss12':
            (sandbox['root']/'code.py').write_text('changed during run')
        return case
    monkeypatch.setattr(driver,'run_case',changed)
    with pytest.raises(ValueError,match='changed during execution'):
        driver.research(sandbox['output'])
    assert not (sandbox['output']/'cases/loss12.manifest.json').exists()


def diagnostic_corporate(rows):
    corp = driver.TrackedCorporateActions.__new__(driver.TrackedCorporateActions)
    corp.requests = 0
    corp.loaded = {'2885':deepcopy(rows)}
    return corp


def stock_dividend(**changes):
    return dict(dict(kind='stock_dividend',stock_id='2885',date='2023-08-11',
        shares_per_share=.05,pay_date='2023-10-13',fractional_cash_per_share=0.),**changes)


@pytest.mark.parametrize('field,value',[
    ('shares_per_share',None),('shares_per_share',float('nan')),
    ('shares_per_share',float('inf')),('shares_per_share',-.1),
    ('shares_per_share','0.05'),('shares_per_share',True),
    ('shares_per_share',Decimal('NaN')),('pay_date',None),('pay_date',''),
    ('fractional_cash_per_share',float('nan')),('fractional_cash_per_share',float('inf')),
    ('fractional_cash_per_share',-1),('fractional_cash_per_share','1'),
])
def test_stock_dividend_diagnostic_identifies_stock_day_and_invalid_field(field,value):
    corp = diagnostic_corporate([stock_dividend(**{field:value})])
    with pytest.raises(driver.UnresolvedAction) as error:
        corp.on_date('2885','2023-08-11')
    assert all(text in str(error.value) for text in ('2885','2023-08-11',field))


@pytest.mark.parametrize('rate,fraction',[(.05,0.),(0,0),(Decimal('.05'),Decimal('0')),(.05,None)])
def test_valid_dividends_return_unchanged_copies_including_optional_fraction_term(rate,fraction):
    row = stock_dividend(shares_per_share=rate,fractional_cash_per_share=fraction)
    corp = diagnostic_corporate([row])
    result = corp.on_date('2885','2023-08-11')
    assert result == [row] and result[0] is not corp.loaded['2885'][0]
    assert corp.loaded['2885'] == [row]


def test_unheld_past_stock_dividend_does_not_block_prepare_or_another_date():
    old = stock_dividend(shares_per_share=None,pay_date=None)
    current = dict(kind='cash_dividend',stock_id='2885',date='2023-09-01',cash_per_share=1.)
    corp = diagnostic_corporate([old,current])
    corp.prepare('2885')
    assert corp.on_date('2885','2023-09-01') == [current]
    assert corp.on_date('2885','2023-09-04') == []
    with pytest.raises(driver.UnresolvedAction,match='shares_per_share, pay_date'):
        corp.on_date('2885','2023-08-11')
