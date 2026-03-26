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
def _load_latest_signal() -> Optional[Dict]:
    """找最新的 strategy_c_YYYY-MM-DD.json。"""
    files = sorted(SIGNAL_DIR.glob("strategy_c_20*.json"), reverse=True)
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
            lines.append(
                f"  {h['stock_id']} {h['name']}｜"
                f"進場 {h['entry_date']}｜"
                f"{sign} {h['pnl_pct']:+.1f}%"
            )
        lines.append("")

    if hold_list:
        lines.append("📋 <b>目前持倉</b>（排名正常，繼續持有）")
        for h in hold_list:
            sign = "📈" if h["pnl_pct"] >= 0 else "📉"
            lines.append(
                f"  {h['stock_id']} {h['name']}｜"
                f"進場 {h['entry_date']}｜"
                f"{sign} {h['pnl_pct']:+.1f}% ✅"
            )
        lines.append("")

    if buy_list:
        # 區分今日已突破 vs 等待突破
        ready_buys   = [h for h in buy_list if h.get("breakthrough_ready")]
        waiting_buys = [h for h in buy_list if not h.get("breakthrough_ready") and "breakthrough_ready" in h]
        no_bt_buys   = [h for h in buy_list if "breakthrough_ready" not in h]  # 無突破資料（舊格式相容）

        if ready_buys:
            lines.append("🟢 <b>買進建議</b>（今日已突破，可直接進場）")
            for h in ready_buys:
                bt_type = "價格突破" if h.get("breakthrough_type") == "price" else "外資籌碼"
                vol_r   = f"量比 {h['vol_ratio']:.1f}x" if h.get("vol_ratio") else ""
                lines.append(
                    f"  {h['stock_id']} {h['name']}｜"
                    f"<b>{bt_type}</b> ✅｜"
                    f"{vol_r}｜建議 ${amt:,}"
                )
            lines.append("")

        if waiting_buys:
            lines.append("⏳ <b>等待突破</b>（模型看好，尚未出現量價訊號）")
            for h in waiting_buys:
                close_max = h.get("close_max_20", 0)
                pct_gap   = h.get("pct_to_price_bt")
                gap_str   = f"距突破 +{pct_gap*100:.1f}%" if pct_gap and pct_gap > 0 else ""
                lines.append(
                    f"  {h['stock_id']} {h['name']}｜"
                    f"突破點 >{close_max:.1f} 且量>均×1.5｜"
                    f"{gap_str}｜建議 ${amt:,}"
                )
            lines.append("")

        if no_bt_buys:
            lines.append("🟢 <b>買進建議</b>（模型評分最高，尚未持有）")
            for h in no_bt_buys:
                lines.append(
                    f"  {h['stock_id']} {h['name']}｜"
                    f"分數 {h['score_today']:.4f}｜建議金額 ${amt:,}"
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


HELP_TEXT = """\
📖 <b>Strategy C Bot 指令</b>

/signal      今日選股建議
/portfolio   持倉狀況與損益
/buy 代號 價格 股數
             記錄買進（例：/buy 2330 855 1000）
/sell 代號 價格
             記錄賣出（例：/sell 2330 900）
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
        sig = _load_latest_signal()
        if not sig:
            print("❌ 找不到訊號檔案，請先執行 python scripts/strategy_c_pick.py")
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
