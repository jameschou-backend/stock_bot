from __future__ import annotations

from datetime import date

from skills import ingest_institutional, ingest_margin_short, ingest_prices


class _Cfg:
    tz = "Asia/Taipei"
    train_lookback_years = 5
    finmind_token = ""
    finmind_requests_per_hour = 6000
    finmind_retry_max = 1
    finmind_retry_backoff = 0.1
    chunk_days = 30
    margin_bulk_chunk_days = 30


def _install_common_monkeypatch(monkeypatch, module):
    monkeypatch.setattr(module, "start_job", lambda *_a, **_k: "job-1")
    monkeypatch.setattr(module, "finish_job", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_resolve_start_date", lambda *_a, **_k: date(2026, 2, 12))
    monkeypatch.setattr(
        module,
        "probe_dataset_has_data",
        lambda **_k: {"has_data": False, "probe_stock_id": None, "rows": 0},
    )


def test_ingest_prices_skip_when_probe_has_no_data(monkeypatch):
    _install_common_monkeypatch(monkeypatch, ingest_prices)
    monkeypatch.setattr(
        ingest_prices,
        "fetch_stock_list",
        lambda **_k: (_ for _ in ()).throw(AssertionError("fetch_stock_list should not be called")),
    )
    out = ingest_prices.run(_Cfg(), db_session=None)
    assert out["rows"] == 0


def test_ingest_institutional_skip_when_probe_has_no_data(monkeypatch):
    _install_common_monkeypatch(monkeypatch, ingest_institutional)
    monkeypatch.setattr(
        ingest_institutional,
        "fetch_stock_list",
        lambda **_k: (_ for _ in ()).throw(AssertionError("fetch_stock_list should not be called")),
    )
    out = ingest_institutional.run(_Cfg(), db_session=None)
    assert out["rows"] == 0


def test_ingest_margin_skip_when_probe_has_no_data(monkeypatch):
    _install_common_monkeypatch(monkeypatch, ingest_margin_short)
    monkeypatch.setattr(
        ingest_margin_short,
        "_get_stock_list",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("_get_stock_list should not be called")),
    )
    out = ingest_margin_short.run(_Cfg(), db_session=None)
    assert out["rows"] == 0
