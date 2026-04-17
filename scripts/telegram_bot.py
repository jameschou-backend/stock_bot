"""Telegram Bot：Strategy C 交易通知與持倉管理。

模式：
  --push      讀取今日 strategy_c 訊號，發送到 Telegram
  --listen    啟動長輪詢，等待使用者指令
  --dry-run   印出訊息內容，不實際發送

指令（Listen 模式）：
  /signal     今日選股建議
  /portfolio  持倉與損益
  /buy XXXX 價格 股數   記錄買進
  /sell XXXX 價格       記錄賣出
  /why XXXX             多智能體矛盾分析
  /help       指令說明

環境變數（.env）：
  TELEGRAM_BOT_TOKEN=
  TELEGRAM_CHAT_ID=
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config

# ─────────────────────────────────────────────
# 路徑常數
# ─────────────────────────────────────────────
SIGNAL_DIR     = ROOT / "artifacts" / "daily_signal"
PORTFOLIO_FILE = SIGNAL_DIR / "portfolio.json"

# ─────────────────────────────────────────────
# Telegram API 封裝
# ─────────────────────────────────────────────
class TelegramBot:
    BASE = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str, chat_id: str, dry_run: bool = False):
        self.token   = token
        self.chat_id = chat_id
        self.dry_run = dry_run
        self._offset = 0

    def _url(self, method: str) -> str:
        return self.BASE.format(token=self.token, method=method)

    def send(self, text: str, chat_id: str | None = None) -> bool:
        """發送訊息。dry_run 時只印出。"""
        if self.dry_run:
            print("─" * 50)
            print("[DRY RUN] 訊息內容：")
            print(text)
            print("─" * 50)
            return True
        try:
            resp = requests.post(
                self._url("sendMessage"),
                json={
                    "chat_id": chat_id or self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                },
                timeout=15,
            )
            if not resp.ok:
                print(f"[Telegram] 發送失敗：{resp.status_code} {resp.text[:200]}")
                return False
            return True
        except Exception as e:
            print(f"[Telegram] 連線錯誤：{e}")
            return False

    def get_updates(self) -> List[Dict]:
        """長輪詢取得新訊息。"""
        try:
            resp = requests.get(
                self._url("getUpdates"),
                params={"offset": self._offset, "timeout": 30},
                timeout=35,
            )
            if not resp.ok:
                return []
            updates = resp.json().get("result", [])
            if updates:
                self._offset = updates[-1]["update_id"] + 1
            return updates
        except Exception as e:
            print(f"[Telegram] getUpdates 錯誤：{e}")
            time.sleep(5)
            return []


# ─────────────────────────────────────────────
# 讀取今日最新訊號
# ─────────────────────────────────────────────
def _load_latest_signal(strategy: str = "c") -> Optional[Dict]:
    """找最新的 strategy_{c|d}_YYYY-MM-DD.json。"""
    files = sorted(SIGNAL_DIR.glob(f"strategy_{strategy}_20*.json"), reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text(encoding="utf-8"))


def _get_latest_prices(stock_ids: List[str]) -> Dict[str, float]:
    """從 DB 查最新收盤價。"""
    if not stock_ids:
        return {}
    try:
        from app.db import get_session
        from app.models import RawPrice
        from sqlalchemy import select, func

        with get_session() as session:
            subq = (
                select(
                    RawPrice.stock_id,
                    func.max(RawPrice.trading_date).label("max_date"),
                )
                .where(RawPrice.stock_id.in_(stock_ids))
                .group_by(RawPrice.stock_id)
                .subquery()
            )
            rows = session.execute(
                select(RawPrice.stock_id, RawPrice.close)
                .join(subq, (RawPrice.stock_id == subq.c.stock_id) &
                             (RawPrice.trading_date == subq.c.max_date))
            ).fetchall()
            return {str(r.stock_id): float(r.close) for r in rows}
    except Exception as e:
        print(f"[DB] 價格查詢失敗：{e}")
        return {}


# ─────────────────────────────────────────────
# 訊號格式化
# ─────────────────────────────────────────────
def _format_push_message(sig: Dict) -> str:
    """
    以 portfolio.json（使用者實際持倉）為主：
    - 持有 + 排名掉出 → 賣出警示
    - 持有 + 排名保持 → 目前持倉（維持）
    - 未持有 + 模型推薦 → 買進建議（依空位數量）
    """
    d       = sig["date"]
    capital = sig.get("capital", 1_000_000)

    # 今日模型資料
    above_threshold = set(sig.get("above_threshold_stocks", []))
    top_candidates  = sig.get("top_candidates", [])   # 前 20 名，含名稱分數
    score_cutoff    = sig.get("meta", {}).get("score_cutoff", 0)

    # 使用者真實持倉
    pf       = _load_portfolio()
    held     = pf.get("positions", [])
    held_ids = {p["stock_id"] for p in held}

    # 查今日收盤價（計算持倉損益）
    prices = _get_latest_prices(list(held_ids)) if held_ids else {}

    sell_list = []
    hold_list = []

    for pos in held:
        sid        = pos["stock_id"]
        name       = pos.get("stock_name", sid)
        entry_px   = float(pos["entry_price"])
        shares     = int(pos["shares"])
        entry_date = pos.get("entry_date", "?")
        cur_px     = prices.get(sid, entry_px)
        pnl_pct    = (cur_px / entry_px - 1) * 100

        if sid not in above_threshold:
            sell_list.append({
                "stock_id": sid, "name": name,
                "entry_price": entry_px, "cur_price": cur_px,
                "pnl_pct": pnl_pct, "shares": shares,
                "entry_date": entry_date,
            })
        else:
            hold_list.append({
                "stock_id": sid, "name": name,
                "entry_price": entry_px, "cur_price": cur_px,
                "pnl_pct": pnl_pct, "shares": shares,
                "entry_date": entry_date,
            })

    # 買進建議：模型 top 候選中，使用者尚未持有的（空位上限 6）
    empty_slots = max(0, 6 - len(hold_list))
    buy_list    = []
    for c in top_candidates:
        if len(buy_list) >= empty_slots:
            break
        if c["stock_id"] not in held_ids:
            buy_list.append(c)

    amt = capital // 6

    # 解析日期
    try:
        exec_label = f"（{date.fromisoformat(d).month}/{date.fromisoformat(d).day + 1} 執行）"
    except Exception:
        exec_label = ""

    n_total = len(hold_list) + len(buy_list)
    lines = [
        f"📊 <b>Strategy C 每日選股 {d}</b>",
        f"建議明日 {exec_label}執行｜每檔 ${amt:,}",
        f"買進 +{len(buy_list)}  賣出 -{len(sell_list)}  維持 {len(hold_list)}",
        "",
    ]

    if sell_list:
        lines.append("🔴 <b>賣出警示</b>（持倉排名掉出，建議出場）")
        for h in sell_list:
            sign = "📈" if h["pnl_pct"] >= 0 else "📉"
            pnl_amt = (h["cur_price"] - h["entry_price"]) * h["shares"]
            lines.append(
                f"  {h['stock_id']} {h['name']}｜"
                f"成本 {h['entry_price']:.1f}×{h['shares']}股｜現價 {h['cur_price']:.1f}｜"
                f"{sign} {h['pnl_pct']:+.1f}%（{pnl_amt:+,.0f}）"
            )
        lines.append("")

    if hold_list:
        lines.append("📋 <b>目前持倉</b>（排名正常，繼續持有）")
        for h in hold_list:
            sign = "📈" if h["pnl_pct"] >= 0 else "📉"
            pnl_amt = (h["cur_price"] - h["entry_price"]) * h["shares"]
            lines.append(
                f"  {h['stock_id']} {h['name']}｜"
                f"成本 {h['entry_price']:.1f}×{h['shares']}股｜現價 {h['cur_price']:.1f}｜"
                f"{sign} {h['pnl_pct']:+.1f}%（{pnl_amt:+,.0f}）✅"
            )
        lines.append("")

    if buy_list:
        # 分「已突破」與「等待突破」兩類
        ready  = [h for h in buy_list if h.get("breakthrough_ready")]
        waiting = [h for h in buy_list if not h.get("breakthrough_ready")]

        if ready:
            lines.append("🟢 <b>買進建議</b>（今日已突破，可直接進場）")
            for h in ready:
                bt_type = h.get("breakthrough_type", "")
                if bt_type == "institutional":
                    bt_label = "外資籌碼 ✅"
                elif bt_type == "price":
                    bt_label = "價格突破 ✅"
                else:
                    bt_label = "突破 ✅"
                vol = h.get("vol_ratio", 0)
                lines.append(
                    f"  {h['stock_id']} {h['name']}｜{bt_label}｜"
                    f"量比 {vol:.1f}x｜分數 {h['score_today']:.4f}"
                )
            lines.append("")

        if waiting:
            lines.append("⏳ <b>等待突破</b>（模型看好，尚未現量價訊號）")
            for h in waiting:
                bt_px  = h.get("close_max_20", 0)
                pct    = h.get("pct_to_price_bt", 0) * 100
                vol    = h.get("vol_ratio", 0)
                lines.append(
                    f"  {h['stock_id']} {h['name']}｜"
                    f"突破點 >{bt_px:.0f} 且量>均×1.5｜"
                    f"距突破 {pct:+.1f}%｜分數 {h['score_today']:.4f}"
                )
    elif not held:
        lines.append("（目前無持倉，請用 /buy 代號 價格 股數 記錄買進後，明日起會顯示持倉狀態）")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Portfolio 管理
# ─────────────────────────────────────────────
def _load_portfolio() -> Dict:
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    return {"positions": []}


def _save_portfolio(pf: Dict) -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_FILE.write_text(
        json.dumps(pf, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _format_portfolio() -> str:
    pf        = _load_portfolio()
    positions = pf.get("positions", [])
    if not positions:
        return "📂 目前無持倉記錄\n\n用 /buy 代號 價格 股數 記錄買進"

    stock_ids = [p["stock_id"] for p in positions]
    prices    = _get_latest_prices(stock_ids)

    today   = date.today()
    lines   = ["📋 <b>目前持倉</b>", ""]
    total_cost = 0.0
    total_mkt  = 0.0

    for pos in positions:
        sid        = pos["stock_id"]
        name       = pos.get("stock_name", sid)
        entry_px   = float(pos["entry_price"])
        shares     = int(pos["shares"])
        entry_date = pos.get("entry_date", "?")
        cur_px     = prices.get(sid, entry_px)
        pnl_pct    = (cur_px / entry_px - 1) * 100
        days_held  = (today - date.fromisoformat(entry_date)).days if entry_date != "?" else 0
        cost       = entry_px * shares
        mkt_val    = cur_px * shares
        total_cost += cost
        total_mkt  += mkt_val
        sign       = "🟢" if pnl_pct >= 0 else "🔴"
        lines.append(
            f"{sign} {sid} {name}\n"
            f"   進場 {entry_date}（{days_held}天）\n"
            f"   成本 ${entry_px:.1f}｜現價 ${cur_px:.1f}｜"
            f"{pnl_pct:+.1f}%｜{shares}股"
        )

    total_pnl = (total_mkt / total_cost - 1) * 100 if total_cost > 0 else 0.0
    sign = "🟢" if total_pnl >= 0 else "🔴"
    lines += [
        "",
        f"{sign} <b>整體損益：{total_pnl:+.1f}%</b>",
        f"成本：${total_cost:,.0f}｜市值：${total_mkt:,.0f}",
    ]
    return "\n".join(lines)


def _cmd_buy(args: List[str]) -> str:
    """處理 /buy 2330 855 1000"""
    if len(args) < 3:
        return "❌ 格式錯誤\n用法：/buy 代號 價格 股數\n例：/buy 2330 855 1000"
    stock_id = args[0].strip()
    try:
        price  = float(args[1])
        shares = int(args[2])
    except ValueError:
        return "❌ 價格或股數格式錯誤"

    # 查股票名稱
    name = stock_id
    try:
        from app.db import get_session
        from app.models import Stock
        with get_session() as session:
            row = session.query(Stock.name).filter(Stock.stock_id == stock_id).one_or_none()
            if row:
                name = str(row.name or stock_id)
    except Exception:
        pass

    pf = _load_portfolio()
    # 若同代號已存在則更新（加碼）
    existing = next((p for p in pf["positions"] if p["stock_id"] == stock_id), None)
    if existing:
        old_cost   = existing["entry_price"] * existing["shares"]
        new_cost   = price * shares
        total_sh   = existing["shares"] + shares
        avg_px     = (old_cost + new_cost) / total_sh
        existing["entry_price"] = round(avg_px, 2)
        existing["shares"]      = total_sh
        existing["stock_name"]  = name
        action_msg = f"已加碼，均價更新為 ${avg_px:.2f}"
    else:
        pf["positions"].append({
            "stock_id":   stock_id,
            "stock_name": name,
            "entry_date": date.today().isoformat(),
            "entry_price": price,
            "shares":     shares,
        })
        action_msg = "買進記錄完成"

    _save_portfolio(pf)
    cost = price * shares
    return (
        f"✅ {action_msg}\n"
        f"{stock_id} {name}｜${price:.1f} × {shares:,} 股\n"
        f"金額：${cost:,.0f}"
    )


def _cmd_sell(args: List[str]) -> str:
    """處理 /sell 2330 900"""
    if len(args) < 2:
        return "❌ 格式錯誤\n用法：/sell 代號 價格\n例：/sell 2330 900"
    stock_id = args[0].strip()
    try:
        sell_price = float(args[1])
    except ValueError:
        return "❌ 價格格式錯誤"

    pf       = _load_portfolio()
    existing = next((p for p in pf["positions"] if p["stock_id"] == stock_id), None)
    if not existing:
        return f"❌ 持倉中找不到 {stock_id}"

    entry_px = float(existing["entry_price"])
    shares   = int(existing["shares"])
    pnl_pct  = (sell_price / entry_px - 1) * 100
    pnl_amt  = (sell_price - entry_px) * shares
    name     = existing.get("stock_name", stock_id)

    pf["positions"] = [p for p in pf["positions"] if p["stock_id"] != stock_id]
    _save_portfolio(pf)

    sign = "🟢" if pnl_pct >= 0 else "🔴"
    return (
        f"{sign} 賣出成功\n"
        f"{stock_id} {name}\n"
        f"成本 ${entry_px:.1f} → 賣出 ${sell_price:.1f}｜{shares:,} 股\n"
        f"損益：{pnl_pct:+.1f}%（${pnl_amt:+,.0f}）"
    )


def _cmd_signal() -> str:
    sig = _load_latest_signal()
    if not sig:
        return "❌ 今日尚無選股訊號，請先執行 make daily-c"
    return _format_push_message(sig)


def _cmd_why(args: List[str]) -> str:
    """處理 /why 2330 — 顯示多智能體矛盾分析。"""
    if not args:
        return "❌ 請提供股票代號\n用法：/why 2330"

    stock_id = args[0].strip()

    try:
        from skills.feature_store import FeatureStore
        from skills.multi_agent_selector import explain_stock, _zscore
        import pandas as pd
        from datetime import date as _date

        fs = FeatureStore()
        max_date = fs.get_max_date()
        if max_date is None:
            return "❌ FeatureStore 無資料"

        # 讀取最近 5 天，取最新一筆此股票的特徵
        from datetime import timedelta
        start = max_date - timedelta(days=10)
        feat_df = fs.read(start, max_date)
        feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
        feat_df["stock_id"] = feat_df["stock_id"].astype(str)
        stock_rows = feat_df[feat_df["stock_id"] == stock_id].sort_values("trading_date")

        if stock_rows.empty:
            return f"❌ {stock_id} 無近期特徵資料（FeatureStore max_date={max_date}）"

        row = stock_rows.iloc[-1].copy()
        row_date = row["trading_date"]

        # 計算當日截面的 z_map（用全市場當日資料）
        day_df = feat_df[feat_df["trading_date"] == row_date].reset_index(drop=True)
        z_map = {
            "ret_20":         _zscore(day_df.get("ret_20",         pd.Series(0.0, index=day_df.index))),
            "foreign_net_20": _zscore(day_df.get("foreign_net_20", pd.Series(0.0, index=day_df.index))),
        }
        # 找到 row 在 day_df 中的 index（用於 z_map 對齊）
        match_idx = day_df[day_df["stock_id"] == stock_id].index
        if len(match_idx) == 0:
            row_in_day = row.copy()
            row_in_day.name = 0
            z_map_single = {k: pd.Series([float(v.iloc[0]) if len(v) > 0 else 0.0], index=[0])
                            for k, v in z_map.items()}
        else:
            row_in_day = day_df.loc[match_idx[0]].copy()
            z_map_single = z_map

        ratio_median = float(
            pd.to_numeric(day_df.get("margin_short_ratio"), errors="coerce").median(skipna=True) or 0.2
        )

        result = explain_stock(row_in_day, dq_ctx={}, z_map=z_map_single, ratio_median=ratio_median)
        sv     = result["supervisor"]
        agents = result["agents"]

        # 組裝訊息
        verdict_emoji = {"PASS": "✅", "WATCH": "⚠️", "CONFLICT": "❌"}.get(sv["verdict"], "❓")
        lines = [
            f"🔍 <b>{stock_id}</b> 多智能體分析（{row_date}）",
            f"{verdict_emoji} Supervisor：{sv['verdict']}（衝突分 {sv['conflict_score']:.3f}）",
            f"說明：{sv['explanation']}",
            "",
            "── 各 Agent 訊號 ──",
        ]

        agent_label = {
            "tech":   "📈 技術面",
            "flow":   "🏦 法人買賣",
            "margin": "📊 融資融券",
            "fund":   "💰 基本面",
            "theme":  "🔥 主題熱度",
        }
        sig_emoji = {-2: "🔴🔴", -1: "🔴", 0: "⚪", 1: "🟢", 2: "🟢🟢"}
        for name, a in agents.items():
            label = agent_label.get(name, name)
            if bool(a.get("unavailable")):
                reason = a["reasons"][0] if a.get("reasons") else "無資料"
                lines.append(f"{label}：⚫ 無資料（{reason}）")
            else:
                sig = int(a.get("signal", 0))
                conf = float(a.get("confidence", 0.0))
                reasons = "、".join(str(r) for r in a.get("reasons", []))
                lines.append(f"{label}：{sig_emoji.get(sig,'?')} 信號={sig} 信心={conf:.2f}")
                if reasons:
                    lines.append(f"   {reasons}")

        return "\n".join(lines)

    except Exception as exc:
        import traceback
        return f"❌ /why 執行失敗：{exc}\n{traceback.format_exc()[-400:]}"


HELP_TEXT = """\
📖 <b>Strategy C Bot 指令</b>

/signal      今日選股建議
/portfolio   持倉狀況與損益
/buy 代號 價格 股數
             記錄買進（例：/buy 2330 855 1000）
/sell 代號 價格
             記錄賣出（例：/sell 2330 900）
/why 代號    多智能體矛盾分析（例：/why 2330）
/help        顯示本說明"""


# ─────────────────────────────────────────────
# 指令路由
# ─────────────────────────────────────────────
def _dispatch(bot: TelegramBot, text: str, chat_id: str) -> None:
    text = text.strip()
    if not text.startswith("/"):
        return

    parts   = text.split()
    cmd     = parts[0].split("@")[0].lower()   # 去掉 @botname
    args    = parts[1:]

    if cmd == "/help":
        reply = HELP_TEXT
    elif cmd == "/signal":
        reply = _cmd_signal()
    elif cmd == "/portfolio":
        reply = _format_portfolio()
    elif cmd == "/buy":
        reply = _cmd_buy(args)
    elif cmd == "/sell":
        reply = _cmd_sell(args)
    elif cmd == "/why":
        reply = _cmd_why(args)
    else:
        reply = f"❓ 未知指令：{cmd}\n輸入 /help 查看所有指令"

    bot.send(reply, chat_id=chat_id)


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram 交易通知 Bot")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--push",    action="store_true", help="推送今日訊號")
    group.add_argument("--listen",  action="store_true", help="啟動監聽模式")
    group.add_argument("--dry-run", action="store_true", help="印出訊息（不發送）")
    parser.add_argument("--strategy", type=str, default="c", choices=["c", "d"],
                        help="使用哪個策略的訊號（c=Strategy C, d=Strategy D，預設 c）")
    args = parser.parse_args()

    # 載入 token
    load_config()
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token and not args.dry_run:
        print("❌ 請在 .env 設定 TELEGRAM_BOT_TOKEN")
        sys.exit(1)

    if not chat_id and args.push:
        print("❌ 請在 .env 設定 TELEGRAM_CHAT_ID")
        sys.exit(1)

    dry_run = args.dry_run or (not token)
    bot     = TelegramBot(token or "DRYRUN", chat_id or "DRYRUN", dry_run=dry_run)

    # ── 推送模式 ──
    if args.push or args.dry_run:
        sig = _load_latest_signal(strategy=args.strategy)
        if not sig:
            pick_script = f"python scripts/strategy_{args.strategy}_pick.py"
            print(f"❌ 找不到訊號檔案，請先執行 {pick_script}")
            sys.exit(1)
        msg = _format_push_message(sig)
        ok  = bot.send(msg)
        if ok and not dry_run:
            print(f"✅ 訊號已推送（{sig['date']}）")
        return

    # ── 監聽模式 ──
    print(f"🤖 Bot 啟動，監聽指令中... (Ctrl+C 停止)")
    if not chat_id:
        print("⚠️  TELEGRAM_CHAT_ID 未設定，將接受任意聊天室的指令")

    while True:
        updates = bot.get_updates()
        for upd in updates:
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            text    = msg.get("text", "")
            from_id = str(msg["chat"]["id"])
            user    = msg.get("from", {}).get("username", from_id)

            # 若設定了 CHAT_ID 只處理指定聊天
            if chat_id and from_id != str(chat_id):
                print(f"[ignore] 非授權 chat_id={from_id} ({user}): {text[:40]}")
                continue

            print(f"[cmd] {user}: {text[:60]}")
            _dispatch(bot, text, from_id)


if __name__ == "__main__":
    main()
