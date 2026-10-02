#!/usr/bin/env python3
"""Export the frozen, complete daily signal population as an offline explorer.

This publishes existing signal-time evidence. It neither runs a new strategy nor
puts future trade outcomes into the signal list. Compact price rows are decoded
only for the stock currently displayed by the standalone HTML template.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RANK_REPORT = "deead8ba56001627867a0af188fa0ad4a2201fbbbac881b2a8f83dae0e6f68bb"
EXPECTED_MANIFEST = "6cd6ef3cbf9ebfced4741b2e60fea34212a2dc5605c6b6b5895c003a34bb65bf"
PRICE_COLUMNS = ["date", "raw_open", "raw_high", "raw_low", "raw_close", "volume",
                 "adjustment_factor", "adjusted_close", "quality_issue", "eligible"]
HISTORICAL_END = "2026-09-09"
SIGNAL_MARKER = "<!-- SIGNAL_DATA -->"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_evidence(bundle, rank_dir):
    """Check the sealed rank report, every direct source/output, and its manifest.

    Ancestor manifests remain bound as files; this is not another audit of every
    official market source table and cannot promote the data's certification.
    """
    report_file = rank_dir / "report.json"
    if digest(report_file) != EXPECTED_RANK_REPORT:
        raise ValueError("Rank report differs from the frozen rank-v2 study")
    if (rank_dir / "report.sha256").read_text().strip() != EXPECTED_RANK_REPORT:
        raise ValueError("Rank report sidecar mismatch")
    report = json.loads(report_file.read_text())
    if report["schema"] != "frozen_signal_rank_research_v1" or report["live_qualified"] is not False:
        raise ValueError("Unexpected rank evidence schema or qualification")
    refs = dict(report["source_sha256"])
    refs.update(report["output_sha256"])
    refs[str(report_file.relative_to(ROOT))] = EXPECTED_RANK_REPORT
    refs[str((rank_dir / "report.sha256").relative_to(ROOT))] = digest(rank_dir / "report.sha256")
    for name, expected in refs.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
            raise ValueError("Missing or changed sealed source: " + name)
    if digest(bundle / "manifest.json") != EXPECTED_MANIFEST:
        raise ValueError("Unexpected input manifest")
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if digest(bundle / name) != expected:
            raise ValueError("Changed manifest input: " + name)
    return refs, manifest


def number(value):
    """JSON has no NaN; absent or nonfinite observations stay explicitly null."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def source_scope(day):
    return "frozen_repaired_history" if day <= HISTORICAL_END else "finmind_extension_unverified"


def build_signal_rows(ranks, features, calendar, year, data_as_of):
    """Join frozen T0 fields by identity, never by a future outcome or exit date."""
    if ranks.signal_id.duplicated().any() or features.event_id.duplicated().any():
        raise ValueError("Duplicate signal identity")
    frozen = features.set_index("event_id")
    if set(frozen.index) != set(ranks.signal_id):
        raise ValueError("Rank and feature populations differ")
    selected = ranks.loc[(ranks.signal_date >= f"{year}-01-01") &
                         (ranks.signal_date <= data_as_of)].copy()
    result = []
    day_positions = {str(d.date()): i for i, d in enumerate(calendar)}
    for original in selected.to_dict("records"):
        key, day, sid = original["signal_id"], original["signal_date"], original["stock_id"]
        f = frozen.loc[key]
        if str(f.stock_id) != sid or f.signal_date != day or day not in day_positions:
            raise ValueError("Signal feature identity/calendar mismatch")
        priority = number(original["rank_priority"])
        if priority is None or not math.isclose(priority, f.relative20, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Signal score mismatch")
        i = day_positions[day]
        entry = str(calendar[i + 1].date()) if i + 1 < len(calendar) else None
        if original["entry_date"] != entry or f.entry_date != entry:
            raise ValueError("Entry is not the next observed market session")
        prior = number(f.previous60_high_adjusted)
        close = number(f.signal_close_adjusted)
        result.append(dict(signal_id=key, stock_id=sid, name=original["name"],
            market=original["market"], signal_date=day, entry_date=entry,
            daily_rank=int(original["daily_rank"]), candidate_count=int(original["daily_candidate_count"]),
            priority=priority, stock_return20=number(original["rank_stock_return20"]),
            benchmark_return20=number(original["rank_benchmark_return20"]),
            volume_ratio=number(f.volume_ratio), previous60_high_adjusted=prior,
            signal_close_adjusted=close, signal_close_raw=number(f.signal_close_raw),
            close_to_prior60_high_pct=(close / prior - 1) if close and prior else None,
            turnover_mean20=number(f.mean20_turnover), turnover_median20=number(f.median20_turnover),
            trend_state=f.trend_state, available_at=day + " 收盤資料完成後",
            source_scope=source_scope(day)))
    result.sort(key=lambda r: (r["signal_date"], r["daily_rank"]))
    for day, group in pd.DataFrame(result).groupby("signal_date") if result else []:
        if set(group.candidate_count) != {len(group)} or sorted(group.daily_rank) != list(range(1, len(group) + 1)):
            raise ValueError("Incomplete same-day original rank population: " + day)
    return result


def pack_prices(days, quotes, adjusted, quality, eligible):
    """One stock on the market calendar: no forward filling or invented candles.

    O/H/L use the same adjusted-close/raw-close factor as C. Known daily
    adjustment anomalies follow the existing independent-observer thresholds;
    missing raw bars, zero volume and unknown eligibility remain explicit.
    """
    if quotes.date.duplicated().any():
        raise ValueError("Duplicate raw stock/date quote")
    q = quotes.set_index("date").reindex(days)
    arrays = {field: q[field].to_numpy(float) for field in ("open", "high", "low", "close", "volume")}
    a = adjusted.reindex(days).to_numpy(float)
    b = quality.reindex(days).to_numpy(float)
    e = eligible.reindex(days).fillna(False).to_numpy(bool)
    raw = arrays["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = a / raw
        ar, br = a[1:] / a[:-1] - 1, b[1:] / b[:-1] - 1
    conflicts = np.r_[False, (np.abs(ar) > .20) | (np.abs(br) > .20) | (np.abs(ar - br) > .005)]
    rows, counts = [], Counter()
    for i, day in enumerate(days):
        opened, high, low, close, volume = [arrays[f][i] for f in ("open", "high", "low", "close", "volume")]
        issue = None
        if not all(np.isfinite(v) and v > 0 for v in (opened, high, low, close, a[i], b[i])):
            issue = "missing_or_invalid_price"
        elif not low <= min(opened, close) <= max(opened, close) <= high:
            issue = "raw_ohlc_conflict"
        elif not np.isfinite(volume) or volume <= 0:
            issue = "missing_or_nonpositive_volume"
        elif not e[i]:
            issue = "historical_identity_or_eligibility"
        elif conflicts[i]:
            issue = "daily_adjustment_conflict"
        if issue:
            counts[issue] += 1
        rows.append([str(day.date()), *[number(v) for v in (opened, high, low, close, volume)],
                     number(factors[i]), number(a[i]), issue, bool(e[i])])
    return rows, dict(counts)


def unpack_price(row):
    """Reference decoding contract used by tests and the offline UI."""
    bar = dict(zip(PRICE_COLUMNS, row))
    factor = bar["adjustment_factor"]
    for field in ("open", "high", "low"):
        raw = bar["raw_" + field]
        bar[field] = raw * factor if raw is not None and factor is not None else None
    bar["close"] = bar["adjusted_close"]
    bar["source_scope"] = source_scope(bar["date"])
    return bar


def build_payload(ranks, features, adjusted, quality, quotes, eligibility, *, year=2026,
                  data_as_of="2026-10-02", warmup_sessions=120):
    calendar = pd.DatetimeIndex(adjusted.index)
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Require unique ordered market calendar")
    if str(calendar[-1].date()) != data_as_of:
        raise ValueError("Input calendar and data-as-of differ")
    if not quality.index.equals(adjusted.index) or not eligibility.index.equals(adjusted.index):
        raise ValueError("Input price/quality/eligibility calendars differ")
    signals = build_signal_rows(ranks, features, calendar, year, data_as_of)
    market_days = calendar[(calendar >= pd.Timestamp(f"{year}-01-01")) & (calendar <= pd.Timestamp(data_as_of))]
    if market_days.empty:
        raise ValueError("No completed market sessions in requested year")
    first = int(calendar.get_loc(market_days[0]))
    price_days = calendar[max(0, first - warmup_sessions):]
    groups = {sid: group for sid, group in quotes.groupby("stock_id", sort=False)}
    stocks = {}
    invalid = Counter()
    by_day = {str(d.date()): [] for d in market_days}
    by_stock = {}
    for signal in signals:
        by_day[signal["signal_date"]].append(signal["signal_id"])
        by_stock.setdefault(signal["stock_id"], []).append(signal)
    for sid, stock_signals in sorted(by_stock.items()):
        if sid not in adjusted or sid not in quality or sid not in eligibility:
            raise ValueError("Missing price column for signalled stock " + sid)
        q = groups.get(sid, pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]))
        prices, issues = pack_prices(price_days, q, adjusted[sid], quality[sid], eligibility[sid])
        indexed = {row[0]: row for row in prices}
        for signal in stock_signals:
            candle = indexed[signal["signal_date"]]
            # The marker and feature both use the precise frozen adjusted close.
            if candle[7] != signal["signal_close_adjusted"] or candle[4] != signal["signal_close_raw"]:
                raise ValueError("Signal marker price basis mismatch")
        invalid.update(issues)
        stocks[sid] = dict(stock_id=sid, name=stock_signals[-1]["name"], market=stock_signals[-1]["market"],
                           prices=prices, signal_ids=[s["signal_id"] for s in stock_signals],
                           invalid_bar_count=sum(issues.values()), quality_issue_counts=issues)
    days = [dict(date=day, signal_count=len(ids), signal_ids=ids) for day, ids in by_day.items()]
    return dict(schema="offline_signal_explorer_v1", metadata=dict(year=year,
        date_start=str(market_days[0].date()), date_end=str(market_days[-1].date()), data_as_of=data_as_of,
        price_start=str(price_days[0].date()), warmup_sessions=min(first, warmup_sessions),
        signal_count=len(signals), stock_count=len(stocks), market_day_count=len(days),
        zero_signal_days=sum(d["signal_count"] == 0 for d in days),
        source_label="修復封存歷史；2026/09/10 起為尚未完成官方交叉核對的 FinMind 延伸",
        historical_end=HISTORICAL_END, price_basis="adjusted_ohlc_same_day_close_factor",
        invalid_bar_count=sum(invalid.values()), quality_issue_counts=dict(invalid),
        source_scope_labels={"frozen_repaired_history": "修復後封存歷史；非完整市場與成交認證",
                             "finmind_extension_unverified": "FinMind 延伸；尚未完成官方交叉核對"},
        strategy_label="60日收盤突破＋放量＋相對強勢＋流動性與市場趨勢",
        score_definition="個股近20個交易日報酬率 − 0050同期報酬率；分數不是獲利機率",
        marker_definition="當日收盤後確認的全部原始訊號，不代表實際持倉或已成交",
        limitations=["只展示封存策略的有限歷史股票池；不是當年全市場股票名單認證。",
                     "2026/09/10 後行情與身分為 FinMind／現有股票快照延伸，未完成官方交叉核對。",
                     "調整後K線供跨除權息／分割比較；原始價格保留於行情。異常或缺值不補價。",
                     "沒有新增策略回測、績效宣稱或實際下單；所有訊號不受三檔持股限制。",
                     "訊號只能在當日收盤資料完成後確認；最後一日的隔日入場仍未觀察。",
                     "公司名稱沿用封存名稱，可能不是每個歷史日期的當時名稱。"],
        live_qualified=False, cash_account=False, outcomes_included=False),
        price_columns=PRICE_COLUMNS, days=days, signals=signals, stocks=stocks)


def json_for_script(payload):
    # Escaping '<' also prevents malicious '</script>' and HTML comments from
    # terminating the inert JSON block. JS line separators stay escaped too.
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render_html(template, payload):
    if template.count(SIGNAL_MARKER) != 1:
        raise ValueError("Template requires exactly one SIGNAL_DATA marker")
    return template.replace(SIGNAL_MARKER, json_for_script(payload))


def run(bundle, rank_dir, template, output, payload_path=None, receipt_path=None):
    bundle, rank_dir, template, output = [Path(p).resolve() for p in (bundle, rank_dir, template, output)]
    if not all(p.is_relative_to(ROOT) for p in (bundle, rank_dir, template, output)):
        raise ValueError("Use repository-local evidence, template and output")
    refs, manifest = verify_evidence(bundle, rank_dir)
    ranks = pd.read_parquet(rank_dir / "signal-ranks.parquet")
    features = pd.read_parquet(bundle / "signal-features.parquet")
    if len(ranks) != 30188:
        raise ValueError("Require the full original 30188 signal population")
    ids = sorted(set(ranks.loc[ranks.signal_date.str.startswith("2026"), "stock_id"]))
    frames = {name: pd.read_parquet(bundle / (name + ".parquet"), columns=["date", *ids]).set_index("date")
              for name in ("close-official", "close-quality", "eligibility")}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    cal = frames["close-official"].index
    first = int(cal.searchsorted(pd.Timestamp("2026-01-01")))
    start = cal[max(0, first - 120)]
    quotes = pd.read_parquet(bundle / "quotes-unmasked.parquet", filters=[("date", ">=", start)])
    quotes = quotes.loc[quotes.stock_id.isin(ids)].copy()
    quotes.date = pd.to_datetime(quotes.date)
    payload = build_payload(ranks, features, frames["close-official"], frames["close-quality"], quotes,
                            frames["eligibility"], data_as_of=manifest["end"])
    refs[str(template.relative_to(ROOT))] = digest(template)
    for path in (Path(__file__).resolve(), ROOT / "tests/test_signal_explorer_data.py"):
        refs[str(path.relative_to(ROOT))] = digest(path)
    payload["metadata"]["source_sha256"] = refs
    encoded = json_for_script(payload)
    html = render_html(template.read_text(), payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    outputs = {str(output.relative_to(ROOT)): digest(output)}
    if payload_path:
        payload_path = Path(payload_path).resolve()
        if not payload_path.is_relative_to(ROOT):
            raise ValueError("Payload output must remain in repository")
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(encoded)
        outputs[str(payload_path.relative_to(ROOT))] = digest(payload_path)
    receipt = dict(schema="signal_explorer_export_receipt_v1", created_at=datetime.now(timezone.utc).isoformat(),
                   source_sha256=refs, output_sha256=outputs,
                   metadata={k: v for k, v in payload["metadata"].items() if k != "source_sha256"},
                   html_bytes=output.stat().st_size, live_qualified=False, recomputed_performance=False)
    receipt_path = Path(receipt_path).resolve() if receipt_path else output.with_suffix(".receipt.json")
    if not receipt_path.is_relative_to(ROOT):
        raise ValueError("Receipt output must remain in repository")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    receipt_path.with_suffix(".sha256").write_text(digest(receipt_path) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / ".cache/all-signals-2019-20261002/inputs")
    parser.add_argument("--rank-dir", type=Path, default=ROOT / ".cache/signal-rank-20261003/rank-v2")
    parser.add_argument("--template", type=Path, default=ROOT / "ui/signal_explorer.html")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports/signal_explorer_2026.html")
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = run(args.bundle, args.rank_dir, args.template, args.output, args.payload, args.receipt)
    print(json.dumps({"signals": result["metadata"]["signal_count"], "stocks": result["metadata"]["stock_count"],
                      "market_days": result["metadata"]["market_day_count"], "html_bytes": result["html_bytes"],
                      "outputs": result["output_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
