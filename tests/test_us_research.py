"""Research invariants: chronological features, cash/fill timing and data receipts."""
import io
import json
import gzip
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from us_market.research_backtest import blackhorse_labels, simulate
from us_market.research_data import Acquisition, bars, protocol, universe
from us_market.research_signals import ARMS, Panel, features
from us_market.store import Store, encoded, sha


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "us", min_free_bytes=0)


def panel(rows=320):
    days = pd.Index(pd.bdate_range("2020-01-01", periods=rows).strftime("%Y-%m-%d"), name="date")
    rng = np.random.default_rng(17)
    values = np.cumprod(1 + rng.normal(.002, .015, (rows, 3)), axis=0) * 100
    close = pd.DataFrame(values, index=days, columns=["AAA", "BBB", "SPY"])
    fields = {"open": close * .999, "close": close, "high": close * 1.01, "low": close * .99,
              "volume": close * 0 + 1_000_000, "turnover": close * 1_000_000}
    return Panel(days, list(close.columns), fields, ["AAA", "BBB"], {})


def manual_features(data, entries):
    empty = data.fields["close"].fillna(0) * 0
    signals = empty.astype(bool)
    for i, name in entries:
        signals.loc[data.dates[i], name] = True
    return {"signals": {arm: signals.copy() for arm in ARMS}, "rs": empty + .1,
            "adv": empty + 1e7, "exit_low": empty, "eligible": signals.copy()}


def test_features_are_prefix_invariant_and_obey_calendar_warmup():
    data = panel()
    original = features(data)
    changed = panel()
    for key in changed.fields:
        changed.fields[key].iloc[280:] *= 100
    alternate = features(changed)
    for arm in ARMS:
        pd.testing.assert_frame_equal(original["signals"][arm].iloc[:280], alternate["signals"][arm].iloc[:280])
        assert not original["signals"][arm].iloc[:199].any().any()
        assert not original["signals"][arm].SPY.any()
    data.fields["close"].iloc[250, 0] = np.nan
    after_gap = features(data)
    assert not after_gap["eligible"].AAA.iloc[250:].any()


def test_common_trend_does_not_use_future_prices():
    data = panel()
    actual = features(data)
    for i in (250, 300):
        short = Panel(data.dates[:i], data.symbols, {k: v.iloc[:i] for k, v in data.fields.items()}, data.universe, {})
        earlier = features(short)
        for arm in ARMS:
            pd.testing.assert_frame_equal(actual["signals"][arm].iloc[:i], earlier["signals"][arm])


def test_entry_is_next_open_exit_is_after_close_and_costs_reconcile():
    data = panel(6)
    for key in ("open", "high", "low", "close"):
        data.fields[key].iloc[:] = 100
    data.fields["close"].iloc[1, 0] = 85  # Entry day close triggers stop.
    data.fields["open"].iloc[2, 0] = 80
    feat = manual_features(data, [(0, "AAA")])
    result = simulate(data, feat, "rs_breakout", data.dates[0], data.dates[-1], 15)
    assert [f["date"] for f in result["fills"]] == [data.dates[1], data.dates[2]]
    assert [f["side"] for f in result["fills"]] == ["buy", "sell"]
    assert result["trades"][0]["exit_signal_date"] == data.dates[1]
    assert result["equity"][-1]["equity"] == pytest.approx(100000 - 10015 + 7988)
    assert result["metrics"]["fees_usd"] == 27


def test_missing_next_open_is_skipped_and_never_postponed():
    data = panel(6)
    data.fields["open"].iloc[1, 0] = np.nan
    result = simulate(data, manual_features(data, [(0, "AAA")]), "obv_breakout", data.dates[0], data.dates[-1], 15)
    assert result["fills"] == []
    assert result["metrics"]["skipped_missing_open"] == 1


def test_pending_exit_waits_for_actual_open_and_marks_missing_risk():
    data = panel(5)
    for key in ("open", "close"):
        data.fields[key].iloc[:] = 100
    data.fields["close"].iloc[1, 0] = 85
    data.fields["open"].iloc[2:4, 0] = np.nan
    data.fields["close"].iloc[2:4, 0] = np.nan
    result = simulate(data, manual_features(data, [(0, "AAA")]), "obv_breakout", data.dates[0], data.dates[-1], 15)
    assert result["trades"][0]["exit_date"] == data.dates[4]
    assert result["metrics"]["stale_position_days"] == 2
    data.fields["open"].iloc[4, 0] = np.nan
    data.fields["close"].iloc[4, 0] = np.nan
    stale = simulate(data, manual_features(data, [(0, "AAA")]), "obv_breakout", data.dates[0], data.dates[-1], 15)
    assert stale["metrics"]["final_pending_exits"] == 1
    assert stale["metrics"]["final_stale_value"] == 8500
    assert stale["metrics"]["final_return_stale_marked_zero"] == pytest.approx(-.10015)


def test_no_same_day_sale_proceeds_fund_entries():
    data = panel(5)
    for key in ("open", "close"):
        data.fields[key].iloc[:] = 100
    data.fields["close"].iloc[1, 0] = 85
    feat = manual_features(data, [(0, "AAA"), (1, "BBB")])
    spec = {**protocol(), "slots": 1}
    result = simulate(data, feat, "rs_breakout", data.dates[0], data.dates[-1], 15, spec)
    assert result["metrics"]["buy_fills"] == 1
    assert [f["ticker"] for f in result["fills"]] == ["AAA", "AAA"]


def test_period_start_never_uses_a_signal_from_prior_period():
    data = panel(5)
    result = simulate(data, manual_features(data, [(0, "AAA")]), "rs_breakout", data.dates[1], data.dates[-1], 15)
    assert not result["fills"]


def test_blackhorse_denominators_separate_missing_and_immature():
    data = panel(70)
    for key in ("open", "close"):
        data.fields[key].iloc[:] = 100
    data.fields["close"].iloc[60, 0] = 140
    data.fields["close"].iloc[40, 1] = np.nan
    feat = manual_features(data, [(0, "AAA"), (0, "BBB"), (65, "AAA")])
    actual = blackhorse_labels(data, feat, data.dates[0], data.dates[-1])["rs_breakout"]
    assert actual == {"candidates": 3, "evaluated": 1, "hits": 1, "unknown": 1, "immature": 1, "hit_rate": 1.0}


def fake_bar(day="2020-01-02", **overrides):
    return {"t": day + "T05:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000, **overrides}


def test_multi_symbol_pagination_and_verified_offline_replay(store, monkeypatch):
    runner = Acquisition(store)
    pages = [{"bars": {"AAA": [fake_bar()]}, "next_page_token": "two"},
             {"bars": {"BBB": [fake_bar()]}, "next_page_token": None}]
    urls = []
    def get(url):
        urls.append(url)
        return pages.pop(0)
    monkeypatch.setattr(runner, "get", get)
    spec = {**protocol(), "data_start": "2020-01-02", "data_end": "2020-01-02"}
    receipt = bars(runner, ["AAA", "BBB"], "all", spec, {"2020-01-02"})
    assert receipt["rows"] == 2 and len(urls) == 2
    assert parse_qs(urlsplit(urls[1]).query)["page_token"] == ["two"]
    assert bars(runner, ["AAA", "BBB"], "all", spec, {"2020-01-02"}) == receipt
    (store.root / receipt["parquet"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        bars(runner, ["AAA", "BBB"], "all", spec, {"2020-01-02"})


@pytest.mark.parametrize("data", [[fake_bar(), fake_bar()], [fake_bar("2020-01-04")], [fake_bar(h=99)]])
def test_bad_bars_never_publish_complete_batch(store, monkeypatch, data):
    runner = Acquisition(store)
    monkeypatch.setattr(runner, "get", lambda url: {"bars": {"AAA": data}, "next_page_token": None})
    spec = {**protocol(), "data_start": "2020-01-02", "data_end": "2020-01-02"}
    with pytest.raises(ValueError):
        bars(runner, ["AAA"], "all", spec, {"2020-01-02"})
    assert not list((store.root / "bars").glob("*.json"))


def test_url_boundary_rejects_orders_account_and_wrong_feed():
    with pytest.raises(ValueError):
        Acquisition.validate_url("https://paper-api.alpaca.markets/v2/orders", "alpaca")
    with pytest.raises(ValueError):
        Acquisition.validate_url("https://paper-api.alpaca.markets/v2/account", "alpaca")
    with pytest.raises(ValueError):
        Acquisition.validate_url("https://data.alpaca.markets/v2/stocks/bars?feed=iex", "alpaca")


def test_historical_universe_retains_inactive_today_without_future_filter(store, monkeypatch):
    runner = Acquisition(store)
    row = {"ticker": "OLD", "type": "CS", "locale": "us", "market": "stocks", "primary_exchange": "XNYS"}
    captured = []
    def get(url, provider):
        captured.append(url)
        return {"results": [row, {**row, "ticker": "OLDw"}]}
    monkeypatch.setattr(runner, "get", get)
    names, _ = universe(runner, protocol())
    assert names["stocks"] == [row]
    assert len(names["excluded_security_variants"]) == 1
    assert parse_qs(urlsplit(captured[0]).query)["date"] == ["2018-12-31"]
    assert not names["present_day_survivor_filter"]


def test_offline_comparison_publishes_reproducible_complete_report(store, monkeypatch):
    from us_market import research_cli, research_signals
    data = panel()
    spec = {**protocol(), "periods": {
        "development": [data.dates[0], data.dates[199]],
        "validation": [data.dates[200], data.dates[259]],
        "holdout": [data.dates[260], data.dates[-1]],
    }}
    monkeypatch.setattr(research_cli, "protocol", lambda: spec)
    monkeypatch.setattr(research_signals, "protocol", lambda: spec)
    store.put("protocol.json", encoded(spec))
    names = store.put("inputs/universe.json", encoded({"stocks": [{"ticker": s} for s in data.universe], "excluded_security_variants": []}))
    days = store.put("inputs/calendar.json", encoded([{"date": d} for d in data.dates]))
    receipts = []
    for adjustment in ("all", "split"):
        rows = [{"ticker": s, "date": d, **{k: data.fields[k].loc[d, s] for k in ("open", "high", "low", "close", "volume")}}
                for s in data.symbols for d in data.dates]
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buffer, compression="zstd")
        ref = store.put(f"bars/{adjustment}.parquet", buffer.getvalue())
        receipts.append({"query": {"symbols": data.symbols, "adjustment": adjustment}, "rows": len(rows), "parquet": ref, "complete_pagination": True})
    store.put("inputs/bundle.json", encoded({"complete_acquisition": True, "protocol_sha256": sha(encoded(spec)),
                                            "universe": names, "calendar": days, "bars": receipts}))
    first = research_cli.compare(store)
    size = store.capacity()["used_bytes"]
    assert research_cli.compare(store) == first
    assert store.capacity()["used_bytes"] == size
    result = json.loads((store.root / f"reports/{first['run_id']}/results.json").read_bytes())
    assert set(result["comparisons"]["holdout"]) == set(ARMS) | {"SPY"}
    assert result["live_qualified"] is False
    assert (store.root / f"reports/{first['run_id']}/report.md").is_file()


def test_cash_merger_converts_units_once_and_cash_is_available_next_day():
    data = panel(6)
    for key in ("open", "close"):
        data.fields[key].iloc[:] = 100
    feat = manual_features(data, [(0, "AAA"), (1, "BBB"), (2, "BBB")])
    event = {"adjusted_cash_per_unit": 150, "rate": 75, "unit_factor": 2, "ids": ["cash-event"]}
    spec = {**protocol(), "slots": 1}
    result = simulate(data, feat, "rs_breakout", data.dates[0], data.dates[-1], 15, spec,
                      cash_actions={2: {0: event}, 3: {0: event}})
    assert len(result["settlements"]) == 1
    assert result["settlements"][0]["cash_usd"] == pytest.approx(100000 / 1.0015 * 1.5)
    assert result["trades"][0]["reason"] == "cash_merger"
    assert [f["date"] for f in result["fills"]] == [data.dates[1], data.dates[3]]
    assert all(f["side"] == "buy" for f in result["fills"])
    assert result["metrics"]["final_pending_exits"] == 0


def action_inputs(store, data, events):
    from us_market.research_actions import POLICY
    body = gzip.compress(encoded({"corporate_actions": {"cash_mergers": events}}), mtime=0)
    source = store.put("actions/source.json.gz", body)
    bundle = {"complete_pagination": True, "pages": [{"source": source}]}
    store.put("actions/bundle.json", encoded(bundle))
    records = [{"ticker": "AAA", "date": day, "close": data.fields["close"].loc[day, "AAA"] / 2} for day in data.dates]
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records), buf)
    raw = store.put("actions/raw.parquet", buf.getvalue())
    store.put("actions/cash-inputs.json", encoded({"policy": POLICY, "source_bundle_sha256": sha(encoded(bundle)), "raw_bars": [{"parquet": raw}]}))


def test_cash_action_duplicates_and_missing_payment_dates_are_explicit(store):
    from us_market.research_actions import cash_events
    data = panel(15)
    base = {"acquiree_symbol": "AAA", "acquiree_cusip": "000000001", "rate": 75, "process_date": data.dates[4], "payable_date": data.dates[4]}
    events = [{**base, "id": "first"}, {**base, "id": "second", "payable_date": data.dates[5]},
              {"id": "missing", "acquiree_symbol": "BBB", "process_date": data.dates[3], "rate": 100}]
    action_inputs(store, data, events)
    actual, audit = cash_events(store, data)
    assert list(actual) == [5]
    assert actual[5][0]["adjusted_cash_per_unit"] == 150
    assert actual[5][0]["ids"] == ["first", "second"]
    assert audit["rejected_records"] == [{"ticker": "BBB", "id": "missing", "reason": "payment_date_missing"}]


def test_conflicting_cash_rates_never_settle(store):
    from us_market.research_actions import cash_events
    data = panel(15)
    base = {"acquiree_symbol": "AAA", "acquiree_cusip": "000000001", "rate": 75, "process_date": data.dates[4], "payable_date": data.dates[4]}
    action_inputs(store, data, [{**base, "id": "one"}, {**base, "id": "two", "rate": 80}])
    events, audit = cash_events(store, data)
    assert not events
    assert audit["rejected_records"][0]["reason"] == "conflicting_cash_events"
