from datetime import date
from types import SimpleNamespace
import pandas as pd
import pytest

from skills.price_coverage import incomplete_dates
from skills import daily_pick
from app.finmind import FinMindError


def test_missing_market_is_gap_even_when_latest_date_exists():
    counts=pd.DataFrame([
        (date(2026,9,7),'TWSE',1087),(date(2026,9,7),'TPEX',875),
        (date(2026,9,8),'TPEX',866),
    ],columns=['trading_date','market','rows_count'])
    assert incomplete_dates(counts)==[date(2026,9,8)]
    counts.loc[len(counts)]=[date(2026,9,8),'TWSE',1080]
    assert incomplete_dates(counts)==[]


def test_market_filter_reads_actual_index_not_price_level_average(monkeypatch):
    monkeypatch.setattr('app.config.load_config', lambda: SimpleNamespace(finmind_token='',finmind_requests_per_hour=6000))
    frame=pd.DataFrame({'date':['2026-09-07','2026-09-08'],'price':[100.,101.], 'stock_id':['TAIEX','TAIEX']})
    monkeypatch.setattr('app.finmind.fetch_dataset',lambda *a,**k:frame)
    out=daily_pick._load_market_price_df(None,date(2026,9,8),60)
    assert out.avg_close.tolist()==[100,101]
    with pytest.raises(FinMindError, match='尚未更新'):
        daily_pick._load_market_price_df(None,date(2026,9,9),60)
