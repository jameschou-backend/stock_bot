#!/usr/bin/env python3
"""
generate_review_pack.py — 回測復盤腳本

Usage:
    python scripts/generate_review_pack.py --input artifacts/backtest/duckdb_verify.json

Output:
    artifacts/review_pack.md
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_PATH = Path("artifacts/review_pack.md")


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def load_backtest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _parse_date(s):
    if s is None:
        return None
    from datetime import date
    if isinstance(s, date):
        return s
    return datetime.fromisoformat(str(s)).date()


def annual_returns(periods: list) -> dict:
    """月頻 periods → 逐年複合報酬。"""
    yearly: dict = defaultdict(list)
    for p in periods:
        yr = str(p.get("rebalance_date", ""))[:4]
        yearly[yr].append(float(p.get("return", 0.0)))
    return {yr: float(np.prod([1 + r for r in rets])) - 1
            for yr, rets in sorted(yearly.items())}


def annual_bench_returns(periods: list) -> dict:
    yearly: dict = defaultdict(list)
    for p in periods:
        yr = str(p.get("rebalance_date", ""))[:4]
        yearly[yr].append(float(p.get("benchmark_return", 0.0)))
    return {yr: float(np.prod([1 + r for r in rets])) - 1
            for yr, rets in sorted(yearly.items())}


def consecutive_loss_streak(periods: list) -> tuple:
    """回傳 (最長連虧期數, 那段 periods 列表)。"""
    best, best_periods = 0, []
    curr, curr_periods = 0, []
    for p in periods:
        if float(p.get("return", 0.0)) < 0:
            curr += 1
            curr_periods.append(p)
            if curr > best:
                best, best_periods = curr, curr_periods.copy()
        else:
            curr, curr_periods = 0, []
    return best, best_periods


def worst_n_periods(periods: list, n: int = 5) -> list:
    return sorted(periods, key=lambda p: float(p.get("return", 0.0)))[:n]


# ─────────────────────────────────────────────
# Feature loading
# ─────────────────────────────────────────────

def load_features_for_trades(trades: list) -> pd.DataFrame:
    """從 parquet cache 載入交易對應的特徵。失敗回傳空 DataFrame。"""
    try:
        from app.db import get_session
        from skills import data_store

        entry_dates = sorted({
            _parse_date(t["entry_date"]) for t in trades if t.get("entry_date")
        })
        if not entry_dates:
            return pd.DataFrame()

        with get_session() as session:
            feat_df = data_store.get_features(session, entry_dates[0], entry_dates[-1])

        if feat_df.empty:
            return pd.DataFrame()

        feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
        feat_df["stock_id"] = feat_df["stock_id"].astype(str)
        return feat_df

    except Exception as e:
        print(f"  [warning] 無法載入特徵資料: {e}", file=sys.stderr)
        return pd.DataFrame()


def build_trade_feature_df(trades: list, feat_df: pd.DataFrame) -> pd.DataFrame:
    """把 trades_log 每筆交易與特徵對接，回傳合併表。"""
    if feat_df.empty:
        return pd.DataFrame()

    records = []
    for t in trades:
        sid = str(t.get("stock_id", ""))
        entry_dt = _parse_date(t.get("entry_date"))
        pnl = float(t.get("realized_pnl_pct", 0.0))
        score = float(t.get("score", 0.0))

        row = feat_df[
            (feat_df["stock_id"] == sid) &
            (feat_df["trading_date"] == entry_dt)
        ]
        if row.empty:
            continue
        feat_row = row.iloc[0].to_dict()
        feat_row["_pnl"] = pnl
        feat_row["_score"] = score
        feat_row["_winner"] = pnl > 0
        records.append(feat_row)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def feature_diff_table(tfd: pd.DataFrame) -> pd.DataFrame:
    """獲利 vs 虧損的特徵均值差異表（依標準化差異排序）。"""
    feat_cols = [
        c for c in tfd.columns
        if not c.startswith("_") and c not in ("stock_id", "trading_date")
    ]
    winners = tfd[tfd["_winner"]][feat_cols]
    losers = tfd[~tfd["_winner"]][feat_cols]

    rows = []
    for col in feat_cols:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        if pd.isna(w_mean) and pd.isna(l_mean):
            continue
        std = tfd[col].std()
        if std == 0 or pd.isna(std):
            continue
        rows.append({
            "feature": col,
            "winner_mean": w_mean,
            "loser_mean": l_mean,
            "diff_normalized": (w_mean - l_mean) / std,
        })

    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values("diff_normalized", key=abs, ascending=False)
            .reset_index(drop=True))


# ─────────────────────────────────────────────
# Markdown builder
# ─────────────────────────────────────────────

def build_markdown(data: dict, input_path: str) -> str:
    summary = data.get("summary", {})
    periods = data.get("periods", [])
    trades = data.get("trades_log", [])
    config = summary.get("config", {})

    fname = Path(input_path).name
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    L: list[str] = []

    def h(n, text):
        L.append(f"{'#' * n} {text}")
        L.append("")

    def row(*cells):
        L.append("| " + " | ".join(str(c) for c in cells) + " |")

    def sep(*cols):  # table separator
        L.append("|" + "|".join("------" for _ in cols) + "|")

    # ── Header ────────────────────────────────
    L += [
        "# 回測復盤報告",
        "",
        f"> 來源檔案：`{fname}`  ",
        f"> 產出時間：{now_str}",
        "",
    ]

    # ── §1 基本資訊 ───────────────────────────
    h(2, "1. 回測基本資訊")

    total_ret = float(summary.get("total_return", 0))
    ann_ret = summary.get("annualized_return")
    bench_ret = float(summary.get("benchmark_total_return", 0))
    mdd = float(summary.get("max_drawdown", 0))
    sharpe = summary.get("sharpe_ratio")
    calmar = summary.get("calmar_ratio")
    win_rate = float(summary.get("win_rate", 0))
    pf = summary.get("profit_factor")
    total_trades_n = int(summary.get("total_trades", len(trades)))
    total_periods_n = int(summary.get("total_periods", len(periods)))
    sl_triggered = int(summary.get("stoploss_triggered", 0))
    start = summary.get("backtest_start",
                        periods[0]["rebalance_date"] if periods else "")
    end = summary.get("backtest_end",
                      periods[-1].get("exit_date", "") if periods else "")

    h(3, "績效摘要")
    row("指標", "數值")
    sep("指標", "數值")
    row("回測區間", f"{start} ~ {end}")
    row("總報酬", f"{total_ret:+.2%}")
    row("年化報酬", f"{ann_ret:+.2%}" if ann_ret is not None else "N/A")
    row("大盤總報酬", f"{bench_ret:+.2%}")
    row("超額報酬", f"{total_ret - bench_ret:+.2%}")
    row("最大回撤 (MDD)", f"{mdd:.2%}")
    row("Sharpe Ratio", f"{sharpe:.4f}" if sharpe is not None else "N/A")
    row("Calmar Ratio", f"{calmar:.4f}" if calmar is not None else "N/A")
    row("勝率", f"{win_rate:.2%}")
    row("Profit Factor", f"{pf:.3f}" if pf is not None else "N/A")
    row("總交易次數", total_trades_n)
    row("再平衡期數", total_periods_n)
    row("停損觸發次數", sl_triggered)
    L.append("")

    h(3, "回測參數")
    row("參數", "值")
    sep("參數", "值")
    row("再平衡頻率", config.get("rebalance_freq", "M"))
    row("進場延遲 (天)", config.get("entry_delay_days", 0))
    row("倉位計算", config.get("position_sizing", "equal"))
    row("固定停損", config.get("stoploss_pct", "N/A"))
    row("移動停利", config.get("trailing_stop_pct", "N/A"))
    row("ATR 停損倍數", config.get("atr_stoploss_multiplier", "N/A"))
    L.append("")

    # ── §2 分年績效 ───────────────────────────
    h(2, "2. 分年績效表")

    yr_rets = annual_returns(periods)
    yr_bench = annual_bench_returns(periods)

    row("年份", "策略報酬", "大盤報酬", "超額", "標記")
    sep("年份", "策略報酬", "大盤報酬", "超額", "標記")
    for yr, r in yr_rets.items():
        b = yr_bench.get(yr, 0)
        flag = ""
        if r < -0.10:
            flag = "⚠️ 重大虧損"
        elif r < -0.05:
            flag = "⚠️ 虧損"
        elif r < 0:
            flag = "▼ 小虧"
        row(yr, f"{r:+.2%}", f"{b:+.2%}", f"{r - b:+.2%}", flag)
    L.append("")

    # ── §3 虧損期分析 ──────────────────────────
    h(2, "3. 虧損期分析")

    max_streak, streak_ps = consecutive_loss_streak(periods)
    L.append(f"- **最長連續虧損**：{max_streak} 期")
    if streak_ps:
        s_cum = float(np.prod([1 + float(p.get("return", 0)) for p in streak_ps])) - 1
        L.append(
            f"  - 區間：{streak_ps[0]['rebalance_date']} ~ "
            f"{streak_ps[-1]['rebalance_date']}，累積 {s_cum:+.2%}"
        )
    L.append("")

    h(3, "虧損最慘的 5 個 period")
    row("日期", "期間報酬", "交易檔數", "虧損最重前 3 股（報酬）")
    sep("日期", "期間報酬", "交易檔數", "虧損最重前 3 股（報酬）")
    for p in worst_n_periods(periods, 5):
        rb = p.get("rebalance_date", "")
        ret = float(p.get("return", 0))
        sr = p.get("stock_returns", {})
        n_tr = p.get("trades", len(sr))
        worst_3 = ", ".join(
            f"{sid}({float(r):+.1%})"
            for sid, r in sorted(sr.items(), key=lambda x: float(x[1]))[:3]
        )
        row(rb, f"{ret:+.2%}", n_tr, worst_3 or "—")
    L.append("")

    # ── §4 特徵分佈比較 ──────────────────────────
    h(2, "4. 特徵分佈比較")

    print("  載入特徵資料（從 parquet cache）...", file=sys.stderr)
    feat_df = load_features_for_trades(trades)
    tfd = build_trade_feature_df(trades, feat_df)
    diff_df = feature_diff_table(tfd) if not tfd.empty else pd.DataFrame()

    if diff_df.empty:
        L.append("*無法載入特徵資料，略過此節。請確認 parquet cache 存在（`make pipeline` 後可用）。*")
        L.append("")
    else:
        h(3, "獲利交易 vs 虧損交易特徵對比（差異最大前 15）")
        L.append("> 標準化差異 > 0 表示獲利組該特徵較高")
        L.append("")
        row("特徵", "獲利均值", "虧損均值", "標準化差異")
        sep("特徵", "獲利均值", "虧損均值", "標準化差異")
        for _, r in diff_df.head(15).iterrows():
            arrow = "▲" if r["diff_normalized"] > 0 else "▼"
            row(
                f"`{r['feature']}`",
                f"{r['winner_mean']:.4f}",
                f"{r['loser_mean']:.4f}",
                f"{r['diff_normalized']:+.3f} {arrow}",
            )
        L.append("")

        # High-score losers
        if not tfd.empty:
            score_80th = tfd["_score"].quantile(0.80)
            hs_losers = (
                tfd[(tfd["_score"] >= score_80th) & (~tfd["_winner"])]
                .sort_values("_score", ascending=False)
            )
            h(3, "模型分數高（前20%）但最終虧損的案例（前10）")
            if hs_losers.empty:
                L.append("*無此案例*")
                L.append("")
            else:
                row("股票", "進場日", "模型分數", "虧損幅度")
                sep("股票", "進場日", "模型分數", "虧損幅度")
                for _, r in hs_losers.head(10).iterrows():
                    row(
                        r.get("stock_id", ""),
                        r.get("trading_date", ""),
                        f"{float(r['_score']):.4f}",
                        f"{float(r['_pnl']):+.2%}",
                    )
                L.append("")

    # ── §5 固定 Q&A ───────────────────────────
    h(2, "5. 固定問題分析")

    # Q1
    h(3, "Q1：虧損最集中的年份或市場環境是什麼？")
    worst_yr = min(yr_rets, key=yr_rets.get)
    yr_loss_cnt: dict = defaultdict(int)
    for p in periods:
        if float(p.get("return", 0)) < 0:
            yr_loss_cnt[str(p.get("rebalance_date", ""))[:4]] += 1
    most_loss_yr = max(yr_loss_cnt, key=yr_loss_cnt.get) if yr_loss_cnt else "N/A"

    L += [
        f"- **年報酬最差年份**：{worst_yr}（{yr_rets[worst_yr]:+.2%}）",
        f"- **虧損期數最多年份**：{most_loss_yr}（{yr_loss_cnt.get(most_loss_yr, 0)} 期虧損）",
        "",
    ]

    yr_loss_detail: dict = defaultdict(list)
    for p in periods:
        ret = float(p.get("return", 0))
        if ret < 0:
            yr_loss_detail[str(p.get("rebalance_date", ""))[:4]].append(ret)

    row("年份", "虧損期數", "平均虧損幅度")
    sep("年份", "虧損期數", "平均虧損幅度")
    for yr, rets_list in sorted(yr_loss_detail.items()):
        row(yr, len(rets_list), f"{np.mean(rets_list):+.2%}")
    L.append("")

    # Q2
    h(3, "Q2：模型分數高但虧損的股票有什麼共同點？")
    if trades:
        tdf = pd.DataFrame(trades)
        tdf["_score"] = tdf["score"].astype(float)
        tdf["_pnl"] = tdf["realized_pnl_pct"].astype(float)
        tdf["_sl"] = tdf["stoploss_triggered"].astype(bool)
        score_80th = tdf["_score"].quantile(0.80)
        hs = tdf[tdf["_score"] >= score_80th]
        hs_loss = hs[hs["_pnl"] < 0]
        hs_win = hs[hs["_pnl"] >= 0]

        sl_rate = hs_loss["_sl"].mean() if not hs_loss.empty else 0
        L += [
            f"- 前20% 高分交易：{len(hs)} 筆，其中虧損 {len(hs_loss)} 筆"
            f"（{len(hs_loss)/max(len(hs),1):.1%}）",
            f"- 高分虧損平均幅度：{hs_loss['_pnl'].mean():+.2%}" if not hs_loss.empty else "",
            f"- 高分獲利平均幅度：{hs_win['_pnl'].mean():+.2%}" if not hs_win.empty else "",
            f"- 高分虧損中停損觸發比例：{sl_rate:.1%}",
        ]

        if not hs_loss.empty:
            hs_loss_c = hs_loss.copy()
            hs_loss_c["yr"] = pd.to_datetime(hs_loss_c["entry_date"]).dt.year.astype(str)
            yr_hs_cnt = hs_loss_c.groupby("yr").size()
            top_yr = yr_hs_cnt.idxmax()
            L.append(f"- 高分虧損最集中年份：{top_yr}（{yr_hs_cnt[top_yr]} 筆）")
        L.append("")

    # Q3
    h(3, "Q3：獲利組和虧損組哪個特徵差異最大？")
    if not diff_df.empty:
        top = diff_df.iloc[0]
        direction = "獲利組較高" if top["diff_normalized"] > 0 else "虧損組較高"
        L += [
            f"- **差異最大特徵**：`{top['feature']}`（{direction}，diff={top['diff_normalized']:+.3f}）",
            f"  - 獲利組均值 {top['winner_mean']:.4f}，虧損組均值 {top['loser_mean']:.4f}",
            "",
            "**前 5 大差異特徵：**",
            "",
        ]
        for i, (_, r) in enumerate(diff_df.head(5).iterrows()):
            d = "獲利組較高 ▲" if r["diff_normalized"] > 0 else "虧損組較高 ▼"
            L.append(f"{i + 1}. `{r['feature']}` — {d}（diff={r['diff_normalized']:+.3f}）")
        L.append("")
    else:
        L += ["*需要特徵資料，略過。*", ""]

    # Q4
    h(3, "Q4：目前策略最需要改善的一個環節是什麼？")
    if trades:
        sl_trades = tdf[tdf["_sl"]]
        non_sl_loss = tdf[~tdf["_sl"] & (tdf["_pnl"] < 0)]
        sl_avg = sl_trades["_pnl"].mean() if not sl_trades.empty else 0
        non_sl_avg = non_sl_loss["_pnl"].mean() if not non_sl_loss.empty else 0

        worst5_yrs = [str(p["rebalance_date"])[:4] for p in worst_n_periods(periods, 5)]
        top_bad_yr = Counter(worst5_yrs).most_common(1)[0][0] if worst5_yrs else "N/A"

        L += [
            "根據以上分析，以下幾點值得優先關注：",
            "",
            f"1. **停損機制效果**：停損觸發 {len(sl_trades)} 次，平均虧損 {sl_avg:+.2%}；"
            f"非停損虧損 {len(non_sl_loss)} 筆，平均 {non_sl_avg:+.2%}。"
            f"若停損虧損遠大於非停損虧損，可考慮調整停損點位。",
            "",
            f"2. **大虧集中期**：最差 5 期集中在 {top_bad_yr} 年附近。"
            f"可針對該時期進行個股歸因，確認是系統性（大盤）還是選股誤判。",
            "",
        ]
        if not diff_df.empty:
            top3 = diff_df.head(3)["feature"].tolist()
            L += [
                f"3. **特徵優化方向**：`{'`、`'.join(top3)}` "
                f"是獲利/虧損分群差異最顯著的特徵，"
                f"可審視是否有更好的替代或組合方式。",
                "",
            ]
        L += [
            f"4. **勝率 vs 賠率平衡**：目前勝率 {win_rate:.1%}，"
            f"可觀察是否有辦法在不犧牲 Sharpe 的前提下提升勝率，"
            f"或改善虧損交易的 exit timing。",
            "",
        ]

    L += ["---", "", "*本報告由 `generate_review_pack.py` 自動產出*"]

    return "\n".join(L)


# ─────────────────────────────────────────────
# Plain text builder
# ─────────────────────────────────────────────

def _md_table_to_plain(lines: list[str]) -> list[str]:
    """將 markdown 表格行轉換為對齊的純文字表格。"""
    # 收集表格行（以 | 開頭），跳過分隔行
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # 跳過分隔行 |------|------|
        if stripped.replace("|", "").replace("-", "").strip() == "":
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return []

    # 計算每欄最大寬度
    n_cols = max(len(r) for r in rows)
    widths = [0] * n_cols
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    # 格式化輸出
    result = []
    for idx, r in enumerate(rows):
        parts = []
        for i in range(n_cols):
            val = r[i] if i < len(r) else ""
            parts.append(val.ljust(widths[i]))
        result.append("  ".join(parts))
        # 在表頭後加分隔線
        if idx == 0:
            result.append("  ".join("-" * w for w in widths))

    return result


def md_to_plain_text(md: str) -> str:
    """將 markdown 報告轉換為純文字格式。"""
    lines = md.split("\n")
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 標題：移除 # 符號
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            out.append("")
            out.append(title)
            out.append("=" * len(title) if stripped.startswith("# ") and not stripped.startswith("## ") else "-" * len(title))
            i += 1
            continue

        # 表格：收集連續的表格行一次處理
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append("")
            out.extend(_md_table_to_plain(table_lines))
            out.append("")
            continue

        # 引用行：移除 > 前綴
        if stripped.startswith(">"):
            cleaned = stripped.lstrip("> ").strip().replace("**", "").replace("`", "")
            out.append(cleaned)
            i += 1
            continue

        # 粗體/code：移除 ** 和 `
        cleaned = stripped.replace("**", "").replace("`", "")

        # 列表項：移除 - 前綴，改為空格縮排
        if cleaned.startswith("- "):
            cleaned = "  " + cleaned[2:]

        # 編號列表：保留原樣
        # 水平線
        if cleaned == "---":
            out.append("")
            i += 1
            continue

        # 斜體
        if cleaned.startswith("*") and cleaned.endswith("*"):
            cleaned = cleaned.strip("*")

        out.append(cleaned)
        i += 1

    # 清除開頭多餘空行
    while out and out[0] == "":
        out.pop(0)

    return "\n".join(out) + "\n"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="回測復盤報告產生器")
    parser.add_argument(
        "--input", required=True,
        help="回測 JSON 檔案路徑（例如 artifacts/backtest/duckdb_verify.json）",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help=f"輸出 Markdown 路徑（預設：{OUTPUT_PATH}）",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = Path(args.output)

    if not Path(input_path).exists():
        print(f"[error] 找不到輸入檔案：{input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[generate_review_pack] 載入：{input_path}", file=sys.stderr)
    data = load_backtest(input_path)

    summary = data.get("summary", {})
    periods = data.get("periods", [])
    trades = data.get("trades_log", [])
    print(
        f"  → {len(periods)} 期，{len(trades)} 筆交易，"
        f"總報酬 {float(summary.get('total_return', 0)):+.2%}",
        file=sys.stderr,
    )

    print("[generate_review_pack] 分析中...", file=sys.stderr)
    md = build_markdown(data, input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"[generate_review_pack] ✅ 已輸出：{output_path}", file=sys.stderr)

    # 額外輸出純文字版
    txt_path = output_path.with_suffix(".txt")
    txt = md_to_plain_text(md)
    txt_path.write_text(txt, encoding="utf-8")
    print(f"[generate_review_pack] ✅ 已輸出：{txt_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
