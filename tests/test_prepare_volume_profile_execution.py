from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import prepare_volume_profile_execution as prep


def test_only_exact_existing_execution_queries_are_permitted(tmp_path, monkeypatch):
    monkeypatch.setattr(prep,'BASE',tmp_path/'prep')
    monkeypatch.setattr(prep,'DEST',tmp_path/'execution')
    with pytest.raises(ValueError,match='Only four-digit'):
        prep.prepare('1231','TaiwanStockPriceTick')
    assert not list((tmp_path/'prep').glob('attempts/*'))


def test_query_hash_copy_and_no_retry(tmp_path, monkeypatch):
    from app import config, finmind
    monkeypatch.setattr(prep,'ROOT',tmp_path)
    monkeypatch.setattr(prep,'BASE',tmp_path/'prep')
    monkeypatch.setattr(prep,'DEST',tmp_path/'execution')
    monkeypatch.setattr(config,'load_config',lambda:SimpleNamespace(finmind_token='test-secret',finmind_requests_per_hour=6000))
    seen=[]
    def fetch(dataset,start,end,**kwargs):
        seen.append((dataset,str(start),str(end),kwargs))
        return pd.DataFrame([dict(stock_id='1231',date='2024-01-01',CashEarningsDistribution=2.)])
    monkeypatch.setattr(finmind,'fetch_dataset',fetch)
    result=prep.prepare('1231','TaiwanStockDividend')
    assert result['status']=='received' and result['rows']==1
    assert seen[0][:3]==('TaiwanStockDividend','2018-01-01','2026-09-09')
    assert seen[0][3]['requests_per_hour']==5400 and seen[0][3]['max_retries']==0
    assert 'test-secret' not in (tmp_path/'prep/attempts/1231-TaiwanStockDividend.json').read_text()
    assert pd.read_parquet(tmp_path/'execution/dividends/1231.parquet').equals(pd.read_parquet(tmp_path/result['raw_path']))
    assert prep.prepare('1231','TaiwanStockDividend')['status']=='already_present'
    assert len(seen)==1


def test_persistent_failed_attempt_cannot_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(prep,'BASE',tmp_path/'prep')
    monkeypatch.setattr(prep,'DEST',tmp_path/'execution')
    attempt=tmp_path/'prep/attempts/1231-TaiwanStockDividend.json'
    prep.write(attempt,dict(status='failed',error_type='FinMindError'))
    with pytest.raises(ValueError,match='automatic retry forbidden'):
        prep.prepare('1231','TaiwanStockDividend')
    with pytest.raises(ValueError,match='automatic retry forbidden'):
        prep.prepare('1231','TaiwanStockDividend',resume_permission_failure=True)
