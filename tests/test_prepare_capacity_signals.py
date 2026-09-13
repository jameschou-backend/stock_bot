import json
from pathlib import Path
import pandas as pd
import pytest
from scripts import prepare_capacity_signals as script


def test_quarantine_precedes_signal_generation_and_changed_source_cannot_resume(tmp_path,monkeypatch):
    source=tmp_path/'parent';source.mkdir();base=tmp_path/'baseline';base.mkdir()
    names=('close-official.parquet','close-quality.parquet','raw-close.parquet','raw-volume.parquet')
    day='2026-06-09'
    for name in names:pd.DataFrame({'date':[day],'1435':[10.],'2330':[100.]}).to_parquet(source/name,index=False)
    pd.DataFrame({'stock_id':['1435','2330']}).to_parquet(base/'companies.parquet',index=False)
    (base/'manifest.json').write_text(json.dumps({'files_sha256':{'companies.parquet':script.original.sha(base/'companies.parquet')}}))
    (source/'manifest.json').write_text(json.dumps(dict(sha256={n:script.original.sha(source/n) for n in names},code_sha256={},next_session='2026-09-14')))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'quarantine':[dict(stock_id='1435',date=day)]}))
    monkeypatch.setattr(script,'AUDIT',audit);monkeypatch.setattr(script.original,'BASE',base)
    monkeypatch.setattr(script,'extend',lambda root:source/'signals.json')
    calls=[]
    def generate(close,quality,raw,volume,*a,**k):
        for frame in (close,quality,raw,volume):
            assert pd.isna(frame.at[pd.Timestamp(day),'1435']) and frame.at[pd.Timestamp(day),'2330']==100
        calls.append(1);return dict(entries=[])
    monkeypatch.setattr(script,'build_signals',generate)
    result=script.prepare(tmp_path/'result')
    assert json.loads(result.read_text())['historical_membership_verified'] is False
    assert script.prepare(tmp_path/'result')==result and len(calls)==1
    audit.write_text(json.dumps({'quarantine':[]}))
    with pytest.raises(ValueError,match='來源改變'):script.prepare(tmp_path/'result')
