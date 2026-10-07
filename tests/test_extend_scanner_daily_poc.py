from types import SimpleNamespace
import json

import pandas as pd
import pytest

from scripts import extend_scanner_daily_poc as mod


def actions(day):
    return dict(start=day, end=day, corporate_events_extension_complete=True,
                corporate_action_coverage=[dict(kind=m+'_'+k, complete=True, start=day, end=day)
                    for m in ('twse', 'tpex') for k in ('ex_rights', 'capital_reduction', 'par_value_change')])


def test_each_signal_date_needs_its_own_complete_action_interval():
    manifest = dict(events_extension_complete=True, event_extension_report='actions.json',
                    source_sha256={'actions.json': 'hash'})
    reports = [actions('2026-10-06'), actions('2026-10-07')]
    assert mod.action_coverage_reason(manifest, '2026-10-06', reports) is None
    assert mod.action_coverage_reason(manifest, '2026-10-07', reports) is None
    assert mod.action_coverage_reason(manifest, '2026-10-07', reports[:1]) == 'corporate_event_interval_incomplete'
    reports[1]['corporate_action_coverage'].pop()
    assert mod.action_coverage_reason(manifest, '2026-10-07', reports) == 'corporate_event_interval_incomplete'
    assert mod.action_coverage_reason({}, '2026-10-07', reports) == 'corporate_event_coverage_incomplete'


def test_prefix_rejects_changed_old_prices_on_arbitrary_extension(tmp_path):
    class Reader:
        def __init__(self, path, end): self.path, self.manifest = path, {'end': end}
        def verify(self, name): return self.path/name
    base = tmp_path/'base'; base.mkdir()
    new = tmp_path/'new'; new.mkdir()
    pd.DataFrame({'date': pd.to_datetime(['2026-10-05']), '2330': [100.]}).to_parquet(base/'close-official.parquet')
    pd.DataFrame({'date': pd.to_datetime(['2026-10-05','2026-10-06','2026-10-07']), '2330': [101.,102.,103.]}).to_parquet(new/'close-official.parquet')
    with pytest.raises(ValueError, match='matrix prefix changed'):
        mod.verify_prefix(Reader(base,'2026-10-05'), Reader(new,'2026-10-07'))
    with pytest.raises(ValueError, match='strictly newer'):
        mod.verify_prefix(Reader(base,'2026-10-07'), Reader(new,'2026-10-07'))


def test_new_signal_poc_excludes_signal_day_even_in_multiday_continuation(monkeypatch):
    import numpy as np
    obj = mod.Continuation.__new__(mod.Continuation)
    obj.days = pd.bdate_range(end='2026-10-07', periods=22)
    obj.calendar = obj.days.strftime('%Y-%m-%d').tolist()
    obj.manifest = dict(events_extension_complete=True, event_extension_report='actions.json', source_sha256={'actions.json':'h'})
    obj.action_reports = [actions('2026-10-06'),actions('2026-10-07')]
    obj.signals = [dict(signal_id='s-'+d, signal_date=d, stock_id='2330') for d in obj.calendar[-2:]]
    obj.events = pd.DataFrame({'stock_id':pd.Series(dtype=str), 'event_date':pd.Series(dtype='datetime64[ns]')})
    obj.official = pd.DataFrame(index=pd.MultiIndex.from_product([['2330'],obj.calendar]))
    obj.rows={};obj.structural={};obj.allowed=set()
    obj._path=lambda sid:SimpleNamespace(close=np.ones(22)*100,raw_close=np.ones(22)*100)
    monkeypatch.setattr(mod,'path_issue',lambda *args:None)
    obj._seed()
    assert len(obj.rows)==2 and len(obj.allowed)==21
    assert ('2330','2026-10-07') not in obj.allowed
    assert ('2330','2026-10-06') in obj.allowed
    assert obj.rows['s-2026-10-06']['window_end']=='2026-10-05'
    assert obj.rows['s-2026-10-07']['window_end']=='2026-10-06'


def test_poc_cache_only_never_attempts_network():
    obj=mod.Continuation.__new__(mod.Continuation)
    with pytest.raises(RuntimeError, match='cache-only'):
        obj._fetch('2330','2026-10-06')


def test_additional_receipts_require_exact_query(tmp_path):
    obj=mod.Continuation.__new__(mod.Continuation)
    obj.tick_directories=[tmp_path];obj._mark=lambda *a:None
    folder=tmp_path/'receipts';folder.mkdir()
    (folder/'2330-2026-10-06.json').write_text(json.dumps({'query':{'dataset':'TaiwanStockPriceTick','data_id':'2330','start_date':'2026-10-05'},'status':'empty'}))
    with pytest.raises(ValueError,match='query changed'):
        obj._stored('2330','2026-10-06')


def test_official_report_missing_one_market_is_rejected(tmp_path, monkeypatch):
    sources = tmp_path/'sources.json'
    sources.write_text(json.dumps({'sources':{'new-twse':{}}}))
    normalized = tmp_path/'official.parquet'
    pd.DataFrame([dict(stock_id='2330',date='2026-10-06',market='TWSE',source_id='new-twse')]).to_parquet(normalized)
    report = dict(daily_tables_extension_complete=True, start='2026-10-06',end='2026-10-06',
                  normalized_path='official.parquet',sources_path='sources.json',
                  output_sha256={'official.parquet':'h','sources.json':'h'},
                  required_market_days=2,accepted_market_days=2,missing_market_days=[])
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    monkeypatch.setattr(mod,'sealed_json',lambda *args:report)
    obj=mod.Continuation.__new__(mod.Continuation)
    obj.refs={};obj.calendar=['2026-10-06','2026-10-07'];obj._mark=lambda *args:None
    with pytest.raises(ValueError,match='market-day coverage'):
        obj._extend_official(tmp_path/'report.json')


def test_known_missing_official_identity_does_not_fetch_tapes(monkeypatch):
    import numpy as np
    obj=mod.Continuation.__new__(mod.Continuation)
    obj.days=pd.bdate_range(end='2026-10-07',periods=21)
    obj.calendar=obj.days.strftime('%Y-%m-%d').tolist()
    obj.manifest=dict(events_extension_complete=True,event_extension_report='actions.json',source_sha256={'actions.json':'h'})
    obj.action_reports=[actions('2026-10-07')]
    obj.signals=[dict(signal_id='s',signal_date='2026-10-07',stock_id='2330')]
    obj.events=pd.DataFrame({'stock_id':pd.Series(dtype=str),'event_date':pd.Series(dtype='datetime64[ns]')})
    obj.official=pd.DataFrame(index=pd.MultiIndex.from_product([['2330'],obj.calendar[1:]]))
    obj.rows={};obj.structural={};obj.allowed=set()
    obj._path=lambda sid:SimpleNamespace(close=np.ones(21)*100,raw_close=np.ones(21)*100)
    monkeypatch.setattr(mod,'path_issue',lambda *args:None)
    obj._seed()
    assert obj.rows['s']['reason']=='official_identity_missing_or_ambiguous'
    assert not obj.allowed


@pytest.mark.parametrize('historical_price,raises', [(100.,False),(101.,True)])
def test_additional_receipt_cannot_mask_different_historical_tape(tmp_path, monkeypatch, historical_price, raises):
    from skills import intraday_limit_replay
    obj=mod.Continuation.__new__(mod.Continuation)
    obj.tick_directories=[tmp_path/'extra'];obj._mark=lambda *args:None
    obj._source=lambda *args:({'market':'TWSE'},None)
    query=dict(dataset='TaiwanStockPriceTick',data_id='2330',start_date='2026-10-06')
    extra=dict(query=query,status='received',raw_path='new.parquet',raw_sha256='newhash')
    historical=dict(query=query,status='received',raw_path='old.parquet',raw_sha256='oldhash')
    receipts=tmp_path/'extra'/'receipts';receipts.mkdir(parents=True)
    (receipts/'2330-2026-10-06.json').write_text(json.dumps(extra))
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    monkeypatch.setattr(mod,'verify_saved',lambda item,expected:item)
    monkeypatch.setattr(mod.DailyProfiles,'_stored',lambda *args:historical)
    monkeypatch.setattr(intraday_limit_replay,'normalize_ticks',lambda frame,*args:frame)
    pd.DataFrame({'price':[100.],'shares':[1000]}).to_parquet(tmp_path/'new.parquet')
    pd.DataFrame({'price':[historical_price],'shares':[1000]}).to_parquet(tmp_path/'old.parquet')
    if raises:
        with pytest.raises(ValueError,match='Conflicting normalized'):
            obj._stored('2330','2026-10-06')
    else:
        assert obj._stored('2330','2026-10-06')==extra
