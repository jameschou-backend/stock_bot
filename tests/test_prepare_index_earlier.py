from types import SimpleNamespace
import pandas as pd
import pytest
from scripts import prepare_index_earlier as prepare

def test_six_request_ceiling_and_resume_never_refetches(tmp_path,monkeypatch):
    calls=[]
    def fetch(dataset,*args,**kwargs):
        calls.append((dataset,kwargs));return pd.DataFrame([dict(stock_id=kwargs['data_id'],date='2016-01-04',close=100)])
    monkeypatch.setattr(prepare,'OUT',tmp_path)
    monkeypatch.setattr(prepare,'load_config',lambda:SimpleNamespace(finmind_token='dummy-sensitive-token'))
    monkeypatch.setattr(prepare,'fetch_dataset',fetch)
    prepare.run();prepare.run()
    assert len(calls)==6 and all(kw['max_retries']==0 and kw['requests_per_hour']==6000 for _,kw in calls)
    assert all('dummy-sensitive-token' not in p.read_text() for p in tmp_path.glob('*.json'))
    (tmp_path/'0050-TaiwanStockPrice.json').write_text('{}')
    with pytest.raises(ValueError,match='source changed'):prepare.run()
    assert len(calls)==6
