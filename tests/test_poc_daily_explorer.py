from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path

import pytest

from scripts import export_poc_daily_explorer as mod


def fixture():
    calendar=[];day=date(2025, 12, 1)
    while day<=date(2026, 10, 2):
        if day.weekday()<5:calendar.append(day.isoformat())
        day+=timedelta(days=1)
    rows=[('up','2026-01-02','2026-01-05','red'),
          ('down','2026-01-05','2026-01-06','black'),
          ('unknown','2026-01-06','2026-01-07','red'),
          ('up','2026-10-02',None,'red'),('pending_data','2026-10-02',None,'red')]
    signals=[];profiles=[];stocks={}
    for i,(status,day,entry,candle) in enumerate(rows):
        sid=str(2330+i);eid='signal-'+str(i)
        signals.append(dict(signal_id=eid,stock_id=sid,signal_date=day,entry_date=entry,candle=candle))
        at=calendar.index(day);prior=calendar[at-20:at]
        p=dict(signal_id=eid,stock_id=sid,signal_date=day,status=status,available=status in ('up','down'),
            reason=None if status in ('up','down') else 'ordinary_tape_conflict' if status=='unknown' else 'raw_data_missing',
            prior_dates=prior,window_start=prior[0],window_end=prior[-1],source_date_end=prior[-1],
            account_independent=True,reconstructed=True,computed_at='2026-10-04T00:00:00Z')
        if p['available']:p.update(poc_before=100.,poc_after=110. if status=='up' else 100.)
        profiles.append(p);stocks[sid]={'prices':[[d,100.] for d in calendar]}
    decisions={s['signal_id']:{'poc_status':'not_evaluated','pending_entry':s['entry_date'] is None,
        'entry_date':s['entry_date'],'simulated_buy_qty':0} for s in signals}
    base=dict(schema='poc_signal_explorer_v1',default_strategy='poc_red',signals=signals,stocks=stocks,
        metadata=dict(date_start='2026-01-02',date_end='2026-10-02'),price_columns=['date','close'],
        days=[{'date':d} for d in calendar if d.startswith('2026')],
        strategies={'poc_red':{'decisions':decisions,'trades':[],'summary':{'final_nav':6766236.96}}},
        benchmark={'summary':{'final_nav':3542492.54}})
    return base,profiles


def test_all_candidates_join_independently_including_black_and_terminal():
    base,profiles=fixture();before=deepcopy(base)
    result=mod.build_payload(base,profiles)
    assert base==before and result['strategies']==before['strategies'] and result['benchmark']==before['benchmark']
    assert result['signals']==before['signals'] and result['stocks']==before['stocks']
    assert len(result['opportunities'])==5
    assert result['opportunities']['signal-1']['status']=='down'  # A black candle still gets evaluated.
    terminal=result['opportunities']['signal-3']
    assert terminal['status']=='up' and terminal['source_date_end']<'2026-10-02'
    assert result['signals'][3]['entry_date'] is None
    assert result['strategies']['poc_red']['decisions']['signal-3']['pending_entry']
    assert result['strategies']['poc_red']['decisions']['signal-3']['poc_status']=='not_evaluated'
    assert result['opportunity_metadata']['status_counts']=={'up':2,'down':1,'unknown':1,'pending_data':1}
    assert result['opportunity_metadata']['red_status_counts']=={'up':2,'down':0,'unknown':1,'pending_data':1}
    assert result['opportunity_metadata']['historical_availability_certified'] is False
    assert result['opportunity_metadata']['all_profiles_available'] is False


@pytest.mark.parametrize('change',['missing','extra','duplicate','wrong_stock','wrong_day','future_window',
    'skipped_session','window_label','unknown_price','pending_no_reason','bad_available','bad_direction',
    'nonpositive','boolean_price','infinite_price','naive_timestamp','local_timestamp','account_dependent'])
def test_incomplete_uncausal_or_invented_opportunity_stops(change):
    base,profiles=fixture()
    if change=='missing':profiles.pop()
    elif change=='extra':profiles.append({**profiles[0],'signal_id':'extra'})
    elif change=='duplicate':profiles.append(deepcopy(profiles[0]))
    elif change=='wrong_stock':profiles[0]['stock_id']='9999'
    elif change=='wrong_day':profiles[0]['signal_date']='2026-01-05'
    elif change=='future_window':profiles[0]['prior_dates'][-1]=profiles[0]['signal_date']
    elif change=='skipped_session':profiles[0]['prior_dates'][0]='2025-12-01'
    elif change=='window_label':profiles[0]['source_date_end']='2026-01-02'
    elif change=='unknown_price':profiles[2]['poc_after']=123.
    elif change=='pending_no_reason':profiles[-1]['reason']=None
    elif change=='bad_available':profiles[0]['available']=1
    elif change=='bad_direction':profiles[0]['poc_after']=99.
    elif change=='nonpositive':profiles[0]['poc_before']=0.
    elif change=='boolean_price':profiles[0]['poc_before']=True
    elif change=='infinite_price':profiles[0]['poc_before']=float('inf')
    elif change=='naive_timestamp':profiles[0]['computed_at']='2026-10-04T00:00:00'
    elif change=='local_timestamp':profiles[0]['computed_at']='2026-10-04T00:00:00+08:00'
    elif change=='account_dependent':profiles[0]['account_independent']=False
    with pytest.raises(ValueError):mod.build_payload(base,profiles)


def test_future_candles_and_account_results_do_not_select_or_change_opportunities():
    base,profiles=fixture();expected=mod.build_payload(base,profiles)['opportunities']['signal-0']
    # Change post-signal prices and all account outcomes, retaining the observed calendar.
    for row in base['stocks']['2330']['prices']:
        if row[0]>'2026-01-02':row[1]=999999.
    base['strategies']['poc_red']['summary']['final_nav']=1.
    base['strategies']['poc_red']['decisions']['signal-0']['simulated_buy_qty']=999999
    assert mod.build_payload(base,profiles)['opportunities']['signal-0']==expected


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return mod.digest(path)


def sealed_fixture(root,monkeypatch):
    base,profiles=fixture()
    payload=root/mod.BASE/'payload.json';payload_sha=write_json(payload,base)
    evidence=root/'evidence.json';evidence_sha=write_json(evidence,{'raw':'fixed'})
    receipt=root/mod.BASE/'receipt.json'
    receipt_sha=write_json(receipt,dict(schema='poc_latest_explorer_receipt_v1',
        source_sha256={'evidence.json':evidence_sha},
        output_sha256={str(payload.relative_to(root)):payload_sha}))
    monkeypatch.setattr(mod,'BASE_PAYLOAD_SHA',payload_sha);monkeypatch.setattr(mod,'BASE_RECEIPT_SHA',receipt_sha)
    profile_file=root/'daily/profiles.json';profile_sha=write_json(profile_file,profiles)
    report=dict(schema='poc_daily_profiles_v1',all_signals_materialized=True,year=2026,end='2026-10-02',
        base_payload={'path':str(payload.relative_to(root)),'sha256':payload_sha},
        base_receipt={'path':str(receipt.relative_to(root)),'sha256':receipt_sha},
        profiles={'path':'daily/profiles.json','sha256':profile_sha},source_sha256={})
    path=root/'daily/report.json';write_json(path,report);path.with_suffix('.sha256').write_text(mod.digest(path))
    return path


def test_parent_payload_profile_and_source_closure_are_verified(tmp_path,monkeypatch):
    path=sealed_fixture(tmp_path,monkeypatch)
    base,profiles,report,receipt,refs=mod.load_daily(path,tmp_path)
    assert len(profiles)==5 and len(base['signals'])==5
    assert {'evidence.json','daily/report.json','daily/report.sha256','daily/profiles.json'}.issubset(refs)
    (tmp_path/'evidence.json').write_text('{"raw":"changed"}')
    with pytest.raises(ValueError,match='changed sealed'):mod.load_daily(path,tmp_path)


def test_unbound_profile_mutation_and_wrong_base_are_rejected(tmp_path,monkeypatch):
    path=sealed_fixture(tmp_path,monkeypatch)
    profile=tmp_path/'daily/profiles.json';profile.write_text(profile.read_text()+' ')
    with pytest.raises(ValueError,match='changed sealed'):mod.load_daily(path,tmp_path)
    path=sealed_fixture(tmp_path,monkeypatch)
    report=mod.read(path);report['base_payload']['path']='some-other-payload.json'
    write_json(path,report);path.with_suffix('.sha256').write_text(mod.digest(path))
    with pytest.raises(ValueError,match='another sealed'):mod.load_daily(path,tmp_path)


@pytest.mark.parametrize('key,value',[('year',None),('year',2025),('end',None),('end','2026-09-09')])
def test_report_scope_must_be_explicit_and_exact(tmp_path,monkeypatch,key,value):
    path=sealed_fixture(tmp_path,monkeypatch)
    report=mod.read(path)
    if value is None:report.pop(key)
    else:report[key]=value
    write_json(path,report);path.with_suffix('.sha256').write_text(mod.digest(path))
    with pytest.raises(ValueError,match='must cover 2026'):
        mod.load_daily(path,tmp_path)


def test_atomic_write_keeps_old_file_until_complete_replacement(tmp_path,monkeypatch):
    dest=tmp_path/'public.html';dest.write_text('old complete page')
    actual_replace=mod.os.replace
    seen=[]
    def inspect_then_replace(source,target):
        assert Path(source).parent==dest.parent
        assert dest.read_text()=='old complete page'
        assert Path(source).read_text()=='新頁面'*10000
        seen.append(True)
        actual_replace(source,target)
    monkeypatch.setattr(mod.os,'replace',inspect_then_replace)
    mod.atomic_write(dest,'新頁面'*10000)
    assert seen==[True] and dest.read_text()=='新頁面'*10000
    assert list(tmp_path.iterdir())==[dest]


def test_failed_atomic_publication_leaves_previous_file_and_cleans_temporary(tmp_path,monkeypatch):
    dest=tmp_path/'receipt.json';dest.write_text('old receipt')
    def fail(source,target):raise OSError('simulated publication interruption')
    monkeypatch.setattr(mod.os,'replace',fail)
    with pytest.raises(OSError,match='publication interruption'):
        mod.atomic_write(dest,b'new receipt')
    assert dest.read_text()=='old receipt' and list(tmp_path.iterdir())==[dest]


def test_old_account_html_is_archived_exactly_before_overwrite(tmp_path):
    active=tmp_path/'artifacts/reports/signal_explorer_2026.html'
    active.parent.mkdir(parents=True);active.write_text('<html>sealed account</html>')
    receipt={'output_sha256':{str(active.relative_to(tmp_path)):mod.digest(active)}}
    archived=mod.archive_account_page(active,receipt,tmp_path)
    assert archived==tmp_path/mod.ARCHIVE and archived.read_bytes()==active.read_bytes()
    active.write_text('<html>daily opportunity update</html>')
    assert mod.archive_account_page(active,receipt,tmp_path)==archived
    archived.write_text('changed')
    with pytest.raises(ValueError,match='archive differs'):
        mod.archive_account_page(active,receipt,tmp_path)
