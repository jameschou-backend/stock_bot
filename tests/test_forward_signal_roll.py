import json
import pandas as pd
import pytest
from scripts import prepare_rolling_forward as r,prepare_forward_signals as legacy


def test_incremental_bridge_trims_future_empty_row_preserves_source_and_verifies_hash(tmp_path,monkeypatch):
    base=tmp_path/'base';base.mkdir();root=tmp_path/'sim';prior=root/'signals'/'2026-09-10';prior.mkdir(parents=True)
    companies=pd.DataFrame([dict(stock_id='2492',market='TWSE')]);companies.to_parquet(base/'companies.parquet')
    (base/'manifest.json').write_text(json.dumps(dict(files_sha256={'companies.parquet':legacy.sha(base/'companies.parquet')})))
    for name in r.NAMES[:-1]:pd.DataFrame({'date':pd.to_datetime(['2026-09-09','2026-09-10','2026-09-11']),'2492':[100,101,None]}).to_parquet(prior/name,index=False)
    metadata=dict(sha256={n:legacy.sha(prior/n) for n in r.NAMES[:-1]},code_sha256={})
    (prior/'manifest.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(legacy,'BASE',base)
    calls=[]
    def prepare():
        frame=pd.read_parquet(legacy.BASE/'raw-close.parquet')
        calls.append(dict(base=legacy.BASE,dates=list(frame.date.astype(str)),out=legacy.OUTPUT))
        return legacy.OUTPUT/'new.json'
    monkeypatch.setattr(legacy,'prepare',prepare)
    output=r.prepare(root)
    assert calls[0]['dates']==['2026-09-09','2026-09-10'] and output==root/'signals'/'new.json'
    assert legacy.BASE==base and len(pd.read_parquet(prior/'raw-close.parquet'))==3
    r.prepare(root)
    (prior/'raw-close.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='已變更'):r.prepare(root)
