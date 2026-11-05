"""Run the independent, frozen US three-strategy experiment."""
import argparse
import csv
import io
import json
from pathlib import Path

from .research_backtest import benchmark, blackhorse_labels, simulate
from .research_actions import acquire_actions, cash_events
from .research_data import acquire, protocol
from .research_signals import ARMS, features, load_panel
from .store import Store, encoded, sha

NAMES = {"breakout_control": "單純突破對照", "rs_breakout": "相對強勢＋突破",
         "obv_breakout": "OBV 突破", "contraction_breakout": "波動收斂＋突破"}


def csv_bytes(rows):
    if not rows:
        return b""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def report(result):
    audit, spec = result["audit"], result["protocol"]
    lines = ["# 美股三策略固定規則比較", "", "研究用途；live_qualified=false。以下結果沒有啟用實盤或紙上下單。", "",
             "本報告是資料與策略的初步診斷。公司行動、代碼延續及退市對帳尚未完全通過，績效不能當成已驗證的可實現報酬。", "",
             f"資料：{audit['cohort_stocks']:,} 檔 2018-12-31 當時上市普通股，另加 SPY；{audit['first_session']} 至 {audit['last_session']}，{audit['calendar_sessions']:,} 個交易日。",
             f"每種調整口徑 {audit['rows_by_adjustment']['all']:,} 根日 K；split 與 all 調整資料皆保留。",
             "", "## 固定規則", "",
             "- 共用門檻：連續 200 個有效交易日、收盤高於 200 日均線、20 日平均成交額至少 500 萬美元。",
             "- 相對強勢＋突破：收盤突破前 20 日最高價，且 63 日報酬高於 SPY。",
             "- OBV：OBV 突破前 20 日 OBV 高點。",
             "- 波動收斂：前一日布林帶寬落在其前 120 日最低 20% 區間，當日突破前 20 日高價，成交量至少為前 20 日平均的 1.5 倍。",
             "- 各條件僅在由假轉真時發訊；共同按 63 日相對強度、成交額排序。沒有搜尋最佳參數。",
             "- 訊號收盤後成立，下一交易日開盤價代理成交；下一日無開盤價就跳過。初始研究資金 10 萬美元、最多 10 檔，不借款。",
             "- 每筆最多前一日淨值 10% 與平均成交額 1%；新買入只用前一日已知現金。",
             "- 收盤跌破進價 10%、前 10 日最低價，或持有 60 個交易日後，下一個有開盤價的交易日出場。非盤中停損。",
             "- 基本成本單邊 15 bps，壓力成本 30 bps，為假設的費用與滑價合計；未計稅負。",
             "- 每個時段重新從現金開始。2019–2022 開發、2023 驗證、2024–2025 留存比較；這些時段均未用來調參。", "",
             "## 淨績效（基本成本）", "",
             "| 時段 | 策略 | 累計報酬 | CAGR | 最大回撤 | 平均投入 | 平倉筆數 | 年單向換手 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for period, rows in result["comparisons"].items():
        for arm in ARMS:
            row = rows[arm]["base"]
            lines.append(f"| {period} | {NAMES[arm]} | {row['net_return']:.2%} | {row['cagr']:.2%} | {row['max_drawdown']:.2%} | {row['mean_exposure']:.2%} | {row['closed_trades']} | {row['annual_one_way_turnover']:.2f} |")
        base = rows["SPY"]["base"]
        lines.append(f"| {period} | SPY 買入持有 | {base['net_return']:.2%} | {base['cagr']:.2%} | {base['max_drawdown']:.2%} | 約 100% | — | — |")
    lines += ["", "## 留存時段成本與缺價敏感度", "",
              "| 策略 | 15 bps 淨報酬 | 30 bps 淨報酬 | 期末缺價持倉歸零後報酬 | 缺價持倉日數 |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = result["comparisons"]["holdout"][arm]
        lines.append(f"| {NAMES[arm]} | {row['base']['net_return']:.2%} | {row['stress']['net_return']:.2%} | {row['base']['final_return_stale_marked_zero']:.2%} | {row['base']['stale_position_days']} |")
    if result.get("cash_action_audit"):
        action_audit = result["cash_action_audit"]
        lines += ["", "## 現金併購帳務補正", "",
                  "選股與進出場規則未變；另以供應商明確付款日、清算單價及未還原價核對現金併購。現金在付款日、處理日與生效日三者較晚者之後的第一個交易日入帳，隔日才可用於買入。清算不收模擬交易費。",
                  "原始未結算版本保留在既有報告目錄；本版不會把所有缺價股票自動視為退市歸零或賣出。",
                  f"可核對的清算事件共 {action_audit['eligible_settlements']} 件；來源現金併購依處理年筆數：{action_audit['cash_records_by_process_year']}。",
                  "供應商此次回傳沒有 2018–2019 現金併購紀錄，因此開發期仍不完整。缺付款日、換股合併及改名未自動結算。逐筆拒絕原因見 cash-action-audit.json；實際入帳見各策略 settlements.csv。", "",
                  "| 策略 | 留存期現金清算筆數 | 期末待出場 | 期末缺價市值（美元） |", "|---|---:|---:|---:|"]
        for arm in ARMS:
            row = result["comparisons"]["holdout"][arm]["base"]
            lines.append(f"| {NAMES[arm]} | {row['cash_merger_settlements']} | {row['final_pending_exits']} | {row['final_stale_value']:.2f} |")
    lines += ["", "## 黑馬辨識（留存時段）", "",
              "定義：訊號隔日開盤至第 60 個持有交易日收盤，扣雙邊基本成本後漲幅至少 30%，並超越同段 SPY 至少 15 個百分點。這是事後評估標籤，完全不參與選股。",
              "每條策略取每日前 10 個候選；同股跨日訊號可重複且期間重疊，因此不是獨立統計樣本。持有期內缺價不當成失敗或成功，另列未知；期間尾端不足 60 日列未成熟。", "",
              "| 策略 | 成熟可評估 | 命中 | 命中率 | 未知缺價 | 未成熟 |", "|---|---:|---:|---:|---:|---:|"]
    for arm, row in result["blackhorse"]["holdout"].items():
        name = NAMES.get(arm, "合格股票日母體（不取前 10）")
        rate = f"{row['hit_rate']:.2%}" if row["hit_rate"] is not None else "N/A"
        lines.append(f"| {name} | {row['evaluated']} | {row['hits']} | {rate} | {row['unknown']} | {row['immature']} |")
    lines += ["", "## 資料品質與解讀界線", "",
              f"- 全期完全無日 K 的股票：{len(audit['symbols_without_bars'])}；期末缺價：{len(audit['symbols_missing_final_session'])}。完整名單在 data-audit.json / coverage.csv。未按未來資料完整度刪除股票。",
              f"- 歷史名單中未映射的股別：{json.dumps(audit['excluded_security_variants'], ensure_ascii=False)}。",
              f"- 相鄰交易日絕對漲跌幅超過 100% 的觀測：{audit['large_one_day_moves_over_100pct']}。這些異常未依事後報酬篩掉，需人工核對公司事件及代碼續存；詳見 large-moves.csv。",
              "- 名單為 2018 年底固定群組，不包含後來 IPO，無法代表全部美股黑馬。已退市代碼的缺價、改名、代碼重用與最後退市收益未完全對帳。",
              "- 全市場 historical SIP 日 K；使用供應商目前版本的調整價，並非逐日留存的當時版本。all-adjusted 報酬是還原價代理，不是實際股息現金、扣繳稅與公司行動帳本。",
              "- daily open 是日 K 開盤代理，沒有證明能以相同價格在常規盤開盤成交。訊號沒有偷用隔日價格；仍需分鐘資料驗證執行。",
              "- 缺價持倉沿用最後價格、無價不成交；報告揭露缺價日及期末歸零敏感度。缺少退市現金清算不能視為零風險。",
              "- SPY 在各段第一天開盤買入，策略最早第二天進場；期末皆以收盤估值、不強制賣出，未實現持倉未收出場費。",
              "- OBV 增益、相對強勢增益與收斂增益應對照單純突破、投入比重及成本，不能只以三者中最高報酬宣稱已找出最強策略。",
              "- 台股流程、資料庫、策略及既有回測均未接入此研究流程。", "",
              "官方來源：[Alpaca 日 K](https://docs.alpaca.markets/us/reference/stockbars)、[Massive 歷史名單](https://massive.com/docs/rest/stocks/tickers/all-tickers)、[Alpaca 交易日曆](https://docs.alpaca.markets/us/reference/legacycalendar)。", "",
              f"重現識別：`{result['run_id']}`；完整參數在 protocol.json，輸入與程式校驗碼在 results.json。", ""]
    return "\n".join(lines).encode()


def compare(store, *, settle_cash=False):
    spec = protocol()
    sources = {p.name: sha(p.read_bytes()) for p in Path(__file__).parent.glob("research_*.py")}
    bundle_hash = sha(encoded(store.read_json("inputs/bundle.json")))
    raw_inventory = []
    for path in sorted((store.root / "requests").glob("*.json")):
        body = path.read_bytes()
        receipt = json.loads(body)
        store.read_object(receipt["source"])
        raw_inventory.append({"request": str(path.relative_to(store.root)), "receipt_sha256": sha(body),
                              "source": receipt["source"]})
    raw_hash = sha(encoded(raw_inventory))
    cash_hash = sha(encoded(store.read_json("actions/cash-inputs.json"))) if settle_cash else None
    run_id = sha(encoded({"source": sources, "bundle": bundle_hash, "raw_sources": raw_hash, "cash_actions": cash_hash, "protocol": spec}))[:20]
    folder = f"reports/{run_id}"
    panel = load_panel(store)
    panel.audit["excluded_security_variants"] = store.read_json("inputs/universe.json")["excluded_security_variants"]
    cash_schedule, cash_audit = cash_events(store, panel) if settle_cash else (None, None)
    feat = features(panel)
    result = {"run_id": run_id, "source_sha256": sources, "bundle_sha256": bundle_hash,
              "raw_source_inventory_sha256": raw_hash,
              "cash_action_audit": cash_audit,
              "protocol": spec, "audit": panel.audit, "comparisons": {}, "blackhorse": {}, "live_qualified": False}

    def write(path, body):
        with store.lock():
            return store.put(f"{folder}/{path}", body)

    coverage = []
    for symbol in panel.universe:
        observed = panel.fields["close"][symbol].dropna()
        coverage.append({"ticker": symbol, "sessions": len(observed),
                         "first": observed.index[0] if len(observed) else "", "last": observed.index[-1] if len(observed) else "",
                         "eligible_days": int(feat["eligible"][symbol].sum())})
    write("coverage.csv", csv_bytes(coverage))
    moves = panel.fields["close"][panel.universe].pct_change(fill_method=None)
    unusual = moves.where(moves.abs() > 1).stack()
    write("large-moves.csv", csv_bytes([{"date": day, "ticker": symbol, "return": value} for (day, symbol), value in unusual.items()]))
    for period, (first, last) in spec["periods"].items():
        comparison = result["comparisons"][period] = {}
        for arm in ARMS:
            comparison[arm] = {}
            for cost_name, cost in (("base", spec["base_cost_bps_each_side"]), ("stress", spec["stress_cost_bps_each_side"])):
                output = simulate(panel, feat, arm, first, last, cost, cash_actions=cash_schedule)
                comparison[arm][cost_name] = output["metrics"]
                for kind in ("equity", "fills", "trades", "holdings", "settlements"):
                    write(f"{period}-{arm}-{cost_name}-{kind}.csv", csv_bytes(output[kind]))
            print(json.dumps({"period": period, "arm": arm, "net_return": comparison[arm]["base"]["net_return"]}), flush=True)
        comparison["SPY"] = {name: benchmark(panel, first, last, cost) for name, cost in
                             (("base", spec["base_cost_bps_each_side"]), ("stress", spec["stress_cost_bps_each_side"]))}
        result["blackhorse"][period] = blackhorse_labels(panel, feat, first, last)
    write("data-audit.json", encoded(panel.audit))
    if cash_audit:
        write("cash-action-audit.json", encoded(cash_audit))
    write("raw-source-inventory.json", encoded(raw_inventory))
    write("protocol.json", encoded(spec))
    write("results.json", encoded(result))
    ref = write("report.md", report(result))
    return {"report": str(store.root / ref["path"]), "run_id": run_id, "live_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "actions", "compare"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--fetch", action="store_true", help="Explicitly permit bounded historical HTTP reads")
    parser.add_argument("--max-requests", type=int, default=2500)
    parser.add_argument("--cash-actions", action="store_true", help="Require the explicitly acquired cash-merger settlement overlay")
    args = parser.parse_args()
    if not 0 < args.max_requests <= 2500:
        parser.error("Request budget must be between 1 and 2500")
    if args.command == "compare" and args.fetch:
        parser.error("Comparison is offline; --fetch is only valid for acquisition")
    if args.command != "compare" and args.cash_actions:
        parser.error("--cash-actions is only valid for comparison")
    store = Store(args.root, max_bytes=protocol()["max_storage_gib"] * 1024**3)
    if args.command == "acquire":
        output = acquire(store, fetch=args.fetch, max_requests=args.max_requests)
    elif args.command == "actions":
        output = acquire_actions(store, fetch=args.fetch, max_requests=args.max_requests)
    else:
        output = compare(store, settle_cash=args.cash_actions)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
