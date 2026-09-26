from types import SimpleNamespace
import json

import pandas as pd
import pytest

from scripts import prepare_index_exposure as prepare


@pytest.fixture
def acquisition(tmp_path,monkeypatch):
    calls=[]
    def fetch(dataset,*args,**kwargs):
        calls.append((dataset,kwargs))
        frame=pd.DataFrame([dict(stock_id='00631L',date='2021-01-04',close=100.)])
        frame.attrs.update(retrieved_at=123,cache_hit=False,source='finmind')
        return frame
    monkeypatch.setattr(prepare,'OUT',tmp_path)
    monkeypatch.setattr(prepare,'load_config',lambda:SimpleNamespace(finmind_token='test-only-secret'))
    monkeypatch.setattr(prepare,'fetch_dataset',fetch)
    return tmp_path,calls


def test_three_bounded_fetches_are_reused_without_refetch_or_secret_persistence(acquisition):
    folder,calls=acquisition
    first=prepare.run();assert prepare.run()==first
    assert len(calls)==3 and all(kw['max_retries']==0 and kw['requests_per_hour']==6000 for _,kw in calls)
    assert all('test-only-secret' not in p.read_text() for p in folder.glob('*.json'))
    assert first['database_writes']==0


@pytest.mark.parametrize('mutate',['source','hash','plan'])
def test_corrupt_source_or_changed_plan_cannot_be_silently_refetched(acquisition,mutate):
    folder,calls=acquisition;prepare.run()
    file=folder/('plan.json' if mutate=='plan' else 'TaiwanStockPrice.'+('sha256' if mutate=='hash' else 'json'))
    file.write_text('{}')
    with pytest.raises(ValueError):prepare.run()
    assert len(calls)==3


def test_mismatched_identity_retains_attempt_but_does_not_publish_source(acquisition,monkeypatch):
    folder,_=acquisition
    monkeypatch.setattr(prepare,'fetch_dataset',lambda *a,**kw:pd.DataFrame([dict(stock_id='2330',date='2021-01-04')]))
    for i in range(3):
        with pytest.raises(ValueError,match='identity'):prepare.run()
    with pytest.raises(ValueError,match='ceiling'):prepare.run()
    assert len(json.loads((folder/'attempts.json').read_text()))==3
    assert not (folder/'manifest.json').exists()
