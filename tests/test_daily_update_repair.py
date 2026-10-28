from datetime import date
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, RawPrice, Stock, TradingCalendar
from app.twse_client import TWSEClient, TWSEError
from skills.ingest_prices import _resolve_start_date


@pytest.mark.parametrize('twse_day,tpex_day', [(None, '1151027'), ('1151022', '1151027')])
def test_latest_quotes_reject_partial_or_different_market_dates(monkeypatch, twse_day, tpex_day):
    client = TWSEClient()
    payloads = iter([
        [] if twse_day is None else [dict(Code='2330', Date=twse_day, ClosingPrice='100')],
        [dict(SecuritiesCompanyCode='6488', Date=tpex_day, Close='100')],
    ])
    monkeypatch.setattr(client, '_get_json', lambda *a, **k: next(payloads))
    with pytest.raises(TWSEError, match='market|市場'):
        client.fetch_prices_latest()


def test_incremental_start_repairs_a_day_missing_from_both_markets():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        for sid, market in [('2330', 'TWSE'), ('6488', 'TPEX')]:
            s.add(Stock(stock_id=sid, market=market, security_type='stock', is_listed=True))
            for day in (20, 22):
                s.add(RawPrice(stock_id=sid, trading_date=date(2026, 10, day), close=100))
        s.add_all(TradingCalendar(trading_date=date(2026, 10, d), is_open=True) for d in (20, 21, 22))
        s.commit()
        assert _resolve_start_date(s, date(2026, 10, 1)) == date(2026, 10, 21)
    engine.dispose()


def test_daily_cli_source_is_explicit_and_overrides_loaded_legacy_setting(monkeypatch):
    from scripts import run_daily
    monkeypatch.setenv('INGEST_PRICES_SOURCE', 'twse')
    run_daily.configure_market_source('finmind')
    import os
    assert all(os.environ[k] == 'finmind' for k in run_daily.MARKET_SOURCE_KEYS)
    with pytest.raises(ValueError):
        run_daily.configure_market_source('misspelled')


@pytest.mark.parametrize('dataset', ['TaiwanStockMarginPurchaseShortSale', 'TaiwanStockInstitutionalInvestorsBuySell'])
def test_documented_daily_chip_endpoints_use_one_market_request(monkeypatch, dataset):
    import pandas as pd
    from app import finmind
    calls = []
    def fetch(*a, **kw):
        calls.append(kw.get('data_id'))
        return pd.DataFrame([{'stock_id': '2330'}])
    monkeypatch.setattr(finmind, 'fetch_dataset', fetch)
    finmind.fetch_dataset_by_stocks(dataset, date(2026, 10, 27), date(2026, 10, 27), ['2330', '6488'])
    assert calls == [None]


def test_daily_ingest_end_cannot_fetch_unpublished_today(monkeypatch):
    from app.ingest_dates import resolve_ingest_end_date
    monkeypatch.setenv('DAILY_DATA_END_DATE', '2026-10-27')
    assert resolve_ingest_end_date(date(2026, 10, 28)) == date(2026, 10, 27)
    monkeypatch.setenv('DAILY_DATA_END_DATE', '2026-10-29')
    with pytest.raises(ValueError):
        resolve_ingest_end_date(date(2026, 10, 28))


def test_institutional_normalizer_excludes_warrants_without_allowlist():
    import pandas as pd
    from skills.ingest_institutional import _normalize_institutional
    raw = pd.DataFrame([dict(stock_id=sid, date='2026-10-27', name='Foreign_Investor', buy=100, sell=50)
                        for sid in ['2330','00400A','123456']])
    assert _normalize_institutional(raw).stock_id.tolist() == ['2330']
