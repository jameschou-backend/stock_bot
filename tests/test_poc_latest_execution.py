from datetime import date
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.poc_latest_execution import (
    CACHE_ORDER, DIRECTORY, LatestExecutionData, economic_components, merge_dividends,
)
from skills.replay_market_feeds import ReplayDataUnavailable
from scripts.prepare_poc_latest_inputs import digest


def dividend(*, cash_ex='2026-08-01', cash=1., stock_ex='2026-09-15', stock=2.,
             payment='2026-08-20', announced='2026-05-01'):
    return dict(date=announced, stock_id='2330', AnnouncementDate=announced,
        CashExDividendTradingDate=cash_ex, CashDividendPaymentDate=payment,
        CashEarningsDistribution=cash, CashStatutorySurplus=0.,
        StockExDividendTradingDate=stock_ex, StockEarningsDistribution=stock,
        StockStatutorySurplus=0., TotalNumberOfCashCapitalIncrease=0.)


def limits(day, sid='2330'):
    return pd.DataFrame([dict(date=day, stock_id=sid, reference_price=100., limit_up=110., limit_down=90.)])


def sealed(tmp_path, frame, dataset):
    path = tmp_path / CACHE_ORDER[0] / ('2330-' + dataset + '.parquet')
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    meta = path.with_suffix('.json')
    meta.write_text(json.dumps(dict(stock_id='2330', dataset=dataset, start='2018-01-01',
                                    end='2026-09-09', sha256=digest(path))))
    return {str(p.relative_to(tmp_path)): digest(p) for p in (path, meta)}


def config():
    return SimpleNamespace(finmind_token='unit-test-only', finmind_requests_per_hour=6000)


def test_mixed_dividend_row_keeps_old_cash_and_new_shares_by_ex_date():
    old = pd.DataFrame([dividend(stock=1.)])
    fresh = pd.DataFrame([dividend(cash=99., stock=2., payment='2026-08-30')])
    merged = merge_dividends(old, fresh)
    assert economic_components(merged) == economic_components(old)
    cash = merged[merged.CashExDividendTradingDate.eq('2026-08-01')]
    stock = merged[merged.StockExDividendTradingDate.eq('2026-09-15')]
    assert len(cash) == len(stock) == 1
    assert cash.iloc[0].CashEarningsDistribution == 1.
    assert cash.iloc[0].CashDividendPaymentDate == '2026-08-20'
    assert cash.iloc[0].StockEarningsDistribution == 0.
    assert stock.iloc[0].StockEarningsDistribution == 2.
    assert stock.iloc[0].CashEarningsDistribution == 0.


def test_early_announced_new_cash_is_retained_while_old_stock_is_frozen():
    old = pd.DataFrame([dividend(cash_ex='2026-09-20', cash=4., stock_ex='2026-08-01', stock=1.)])
    fresh = pd.DataFrame([dividend(cash_ex='2026-09-20', cash=5., stock_ex='2026-08-01', stock=99.,
                                  announced='2026-05-01', payment='2026-10-01')])
    merged = merge_dividends(old, fresh)
    assert economic_components(merged) == economic_components(old)
    assert merged[merged.CashExDividendTradingDate.eq('2026-09-20')].iloc[0].CashEarningsDistribution == 5.
    assert merged[merged.StockExDividendTradingDate.eq('2026-08-01')].iloc[0].StockEarningsDistribution == 1.


def test_conflicting_new_dividend_revisions_are_not_added_together():
    old = pd.DataFrame([dividend(stock_ex='', stock=0.)])
    fresh = pd.DataFrame([dividend(cash_ex='2026-09-20', cash=2.),
                          dividend(cash_ex='2026-09-20', cash=3.)])
    with pytest.raises(ValueError, match='Conflicting dividend revisions'):
        merge_dividends(old, fresh)


def test_limits_fetch_only_added_period_and_offline_reuses_receipt(tmp_path):
    old = limits('2026-09-09')
    refs = sealed(tmp_path, old, 'TaiwanStockPriceLimit')
    calls = []
    def fetch(dataset, start, end, **kwargs):
        calls.append((dataset, start, end, kwargs))
        return limits('2026-09-10')
    provider = LatestExecutionData(tmp_path, True, source_refs=refs, fetcher=fetch, config=config())
    result = provider.finmind('2330', 'TaiwanStockPriceLimit')
    assert result.iloc[:1].reset_index(drop=True).equals(old)
    assert calls[0][1:3] == (date(2026, 9, 10), date(2026, 10, 2))
    assert calls[0][3]['requests_per_hour'] == 5400 and calls[0][3]['max_retries'] == 0
    offline = LatestExecutionData(tmp_path, False, source_refs=refs)
    assert offline.finmind('2330', 'TaiwanStockPriceLimit').equals(result)
    assert len(calls) == 1


def test_dividend_fetch_spans_full_announcements_and_copies_only_new_directory(tmp_path):
    old = pd.DataFrame([dividend(stock=1.)])
    refs = sealed(tmp_path, old, 'TaiwanStockDividend')
    calls = []
    def fetch(dataset, start, end, **kwargs):
        calls.append((start, end))
        return pd.DataFrame([dividend(stock=2., cash=99.)])
    provider = LatestExecutionData(tmp_path, True, source_refs=refs, fetcher=fetch, config=config())
    result = provider.finmind('2330', 'TaiwanStockDividend')
    assert calls == [(date(2018, 1, 1), date(2026, 10, 2))]
    assert result.equals(pd.read_parquet(provider.dividend_directory / '2330.parquet'))
    assert economic_components(result) == economic_components(old)
    for name, value in refs.items():
        assert digest(tmp_path / name) == value


def test_new_stock_requires_full_limit_history(tmp_path):
    calls = []
    def fetch(dataset, start, end, **kwargs):
        calls.append(start)
        return limits('2026-09-10')
    provider = LatestExecutionData(tmp_path, True, source_refs={}, fetcher=fetch, config=config())
    provider.finmind('2330', 'TaiwanStockPriceLimit')
    assert calls == [date(2018, 1, 1)]


def test_offline_missing_data_creates_no_request_attempt(tmp_path):
    provider = LatestExecutionData(tmp_path, False, source_refs={})
    with pytest.raises(ReplayDataUnavailable, match='cache missing'):
        provider.finmind('2330', 'TaiwanStockPriceLimit')
    assert not list((tmp_path / DIRECTORY / 'attempts').glob('*.json'))


def test_failure_is_redacted_reserved_and_never_retried(tmp_path):
    calls = []
    def fetch(*args, **kwargs):
        calls.append(True)
        raise RuntimeError('SECRET_SHOULD_NOT_BE_PERSISTED')
    provider = LatestExecutionData(tmp_path, True, source_refs={}, fetcher=fetch, config=config())
    for _ in range(2):
        with pytest.raises(ReplayDataUnavailable):
            provider.finmind('2330', 'TaiwanStockPriceLimit')
    assert len(calls) == 1
    receipts = list((tmp_path / DIRECTORY / 'receipts').glob('*.json'))
    assert 'SECRET_SHOULD_NOT_BE_PERSISTED' not in receipts[0].read_text()
    assert len(list((tmp_path / DIRECTORY / 'attempts').glob('*.json'))) == 1


def test_persistent_budget_cannot_be_reset_by_new_instance(tmp_path):
    def fetch(*args, **kwargs):
        return limits('2026-09-10', sid=kwargs['data_id'])
    one = LatestExecutionData(tmp_path, True, source_refs={}, maximum_requests=1, fetcher=fetch, config=config())
    one.finmind('2330', 'TaiwanStockPriceLimit')
    two = LatestExecutionData(tmp_path, True, source_refs={}, maximum_requests=1, fetcher=fetch, config=config())
    with pytest.raises(ValueError, match='budget exhausted'):
        two.finmind('2317', 'TaiwanStockPriceLimit')


def test_wrong_stock_response_and_changed_historical_source_fail_closed(tmp_path):
    refs = sealed(tmp_path, limits('2026-09-09'), 'TaiwanStockPriceLimit')
    provider = LatestExecutionData(tmp_path, True, source_refs=refs,
        fetcher=lambda *a, **kw: limits('2026-09-10', sid='2317'), config=config())
    with pytest.raises(ReplayDataUnavailable, match='ValueError'):
        provider.finmind('2330', 'TaiwanStockPriceLimit')
    (tmp_path / next(iter(refs))).write_bytes(b'changed')
    other = LatestExecutionData(tmp_path, False, source_refs=refs)
    with pytest.raises(ValueError, match='Sealed source changed'):
        other.finmind('2330', 'TaiwanStockPriceLimit')
