from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.finmind import FinMindError, FinMindQuotaError
from app.models import Base, TradingCalendar, SecuritiesLendingDay, SecuritiesLendingDaily
from skills import ingest_securities_lending as lending


def raw(day='2026-10-27'):
    return pd.DataFrame([
        dict(date=day, stock_id='2330', transaction_type='議借', volume=100, fee_rate=1.0),
        dict(date=day, stock_id='2330', transaction_type='競價', volume=300, fee_rate=2.0),
        dict(date=day, stock_id='00403A', transaction_type='競價', volume=50, fee_rate=1.0),
    ])


def test_transactions_are_flow_with_volume_weighted_fee_not_balance():
    rows = lending._aggregate_lending(raw())
    assert list(rows.stock_id) == ['2330']
    row = rows.iloc[0]
    assert row.transaction_volume == 400
    assert row.transaction_count == 2
    assert row.fee_rate == pytest.approx(1.75)
    assert 'lending_balance' not in rows


@pytest.mark.parametrize('column,value', [('volume', -1), ('volume', None), ('volume', 1.5), ('fee_rate', None), ('transaction_type', '還券')])
def test_invalid_source_semantics_fail_closed(column, value):
    df = raw(); df.loc[0, column] = value
    with pytest.raises(FinMindError):
        lending._aggregate_lending(df)


def test_duplicate_identical_trades_are_not_silently_deduplicated():
    df = pd.concat([raw().iloc[:1]] * 2)
    row = lending._aggregate_lending(df).iloc[0]
    assert row.transaction_volume == 200
    assert row.transaction_count == 2


def test_completed_days_survive_failure_and_rerun_skips_them(monkeypatch):
    engine = create_engine('sqlite:///:memory:'); Base.metadata.create_all(engine)
    cfg = SimpleNamespace(tz='Asia/Taipei', finmind_token=None, finmind_requests_per_hour=5400,
                          finmind_retry_max=0, finmind_retry_backoff=0)
    days = [date(2026, 10, 22), date(2026, 10, 27)]
    calls = []
    def fetch(**kw):
        assert kw.get('data_id') is None
        day = kw['start_date']; calls.append(day)
        if day == days[1]:
            raise FinMindError('temporary source error')
        return raw(day.isoformat())
    monkeypatch.setattr(lending, 'fetch_dataset', fetch)
    with Session(engine) as s:
        s.add_all(TradingCalendar(trading_date=d, is_open=True) for d in days); s.commit()
        with pytest.raises(FinMindError):
            lending.run(cfg, s, start_date=days[0], end_date=days[1])
        assert s.get(SecuritiesLendingDay, days[0]).source_rows == 3
        assert s.get(SecuritiesLendingDay, days[1]) is None
        monkeypatch.setattr(lending, 'fetch_dataset', lambda **kw: (calls.append(kw['start_date']) or raw(kw['start_date'].isoformat())))
        result = lending.run(cfg, s, start_date=days[0], end_date=days[1])
        assert result['requests'] == 1
        assert calls == [days[0], days[1], days[1]]
        assert len(s.execute(select(SecuritiesLendingDaily)).scalars().all()) == 2
        assert lending.run(cfg, s, start_date=days[0], end_date=days[1])['requests'] == 0
    engine.dispose()


def test_empty_published_market_day_is_not_a_successful_checkpoint(monkeypatch):
    engine = create_engine('sqlite:///:memory:'); Base.metadata.create_all(engine)
    cfg = SimpleNamespace(tz='Asia/Taipei', finmind_token=None, finmind_requests_per_hour=5400,
                          finmind_retry_max=0, finmind_retry_backoff=0)
    day = date(2026, 10, 27)
    monkeypatch.setattr(lending, 'fetch_dataset', lambda **kw: pd.DataFrame())
    with Session(engine) as s:
        s.add(TradingCalendar(trading_date=day, is_open=True)); s.commit()
        with pytest.raises(FinMindError): lending.run(cfg, s, start_date=day, end_date=day)
        assert s.get(SecuritiesLendingDay, day) is None
    engine.dispose()
