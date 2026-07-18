"""Telegram Bot：誠實日報推播 + Strategy C / D 訊號（手動）與持倉管理。

模式：
  --push      發送到 Telegram（--strategy daily-brief=誠實日報；c/d=策略訊號，手動用）
  --listen    啟動長輪詢，等待使用者指令
  --dry-run   印出訊息內容，不實際發送
  --strategy  daily-brief（每日排程預設，誠實日報）/ c / d（降級手動用）

誠實日報（daily-brief）內容依序：
  ① A 線今日 picks 前 10（紙上追蹤——各口徑均無可執行 alpha，詳 docs/prereg_*）
  ② paper NAV 最新淨值與近 30 日變化（artifacts/paper_nav/nav.jsonl）
  ③ 今日申購機會（artifacts/ipo_lottery/scan_*.json，折價>10%；唯一可行動項）
  ④ 處置股新增警示（artifacts/disposition/*.json 最新 vs 前一份）
  ⑤ 哨兵狀態（pick 特徵一致性抽驗 + pipeline 最新 job）

指令（Listen 模式）：
  /signal     今日 Strategy C 選股建議
  /signal c   今日 Strategy C 選股建議
  /signal d   今日 Strategy D 選股建議（label=5d + trailing stop -25%）
  /portfolio  持倉與損益
  /buy XXXX 價格 股數   記錄買進
  /sell XXXX 價格       記錄賣出
  /why XXXX             多智能體矛盾分析
  /help       指令說明

環境變數（.env）：
  TELEGRAM_BOT_TOKEN=
  TELEGRAM_CHAT_ID=
  TELEGRAM_USER_ID=   （選填）額外比對發訊者 user id，未設定時只比 chat id
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from skills.io_utils import atomic_write_json, safe_read_json

# ─────────────────────────────────────────────
# 路徑常數
# ─────────────────────────────────────────────
SIGNAL_DIR     = ROOT / "artifacts" / "daily_signal"
PORTFOLIO_FILE = SIGNAL_DIR / "portfolio.json"
NAV_FILE        = ROOT / "artifacts" / "paper_nav" / "nav.jsonl"
IPO_DIR         = ROOT / "artifacts" / "ipo_lottery"
DISPOSITION_DIR = ROOT / "artifacts" / "disposition"

# 誠實橫幅（與 dashboard 總覽頁同一句，單一真相源級別的文案）
HONEST_BANNER = "A 線各口徑均無可執行 alpha（v2.2），picks 僅紙上追蹤——詳 docs/prereg_*"

# 申購機會門檻：折價 > 10% 才列入日報（唯一可行動項）
IPO_MIN_DISCOUNT = 0.10

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

    def _sanitize(self, msg: object) -> str:
        """遮罩訊息中的 bot token（連線例外的 URL 會含完整 token，不可進 log）。"""
        text = str(msg)
        if self.token and self.token != "DRYRUN":
            text = text.replace(self.token, "[TOKEN]")
        return text

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
                print(f"[Telegram] 發送失敗：{resp.status_code} {self._sanitize(resp.text[:200])}")
                return False
            return True
        except Exception as e:
            print(f"[Telegram] 連線錯誤：{self._sanitize(e)}")
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
            print(f"[Telegram] getUpdates 錯誤：{self._sanitize(e)}")
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
        name       = html.escape(str(pos.get("stock_name", sid)))  # 進 HTML parse_mode 前 escape
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
        _exec_dt = date.fromisoformat(d) + timedelta(days=1)
        while _exec_dt.weekday() >= 5:  # 跳週末，避免顯示非交易日（含修掉月底 6/31 越界）
            _exec_dt += timedelta(days=1)
        exec_label = f"（{_exec_dt.month}/{_exec_dt.day} 執行）"
    except Exception:
        exec_label = ""

    n_total = len(hold_list) + len(buy_list)
    _strat = str(sig.get("strategy", "c")).upper()
    lines = [
        f"📊 <b>Strategy {_strat} 每日選股 {d}</b>",
        f"建議明日 {exec_label}執行｜每檔 ${amt:,}",
        f"買進 +{len(buy_list)}  賣出 -{len(sell_list)}  維持 {len(hold_list)}",
        "",
    ]
    if _strat == "D":
        # 2026-07-10 預登記重驗裁決 FAIL（誠實時序臂 MDD -60.9% 觸 -50% 條件，
        # docs/prereg_d_revalidation_20260710.md）——訊號降級紙上追蹤，勿實單跟隨
        lines.insert(1, "📄 <b>紙上訊號</b>（D 重驗 FAIL：誠實時序 MDD -61%）— 勿實單跟隨")

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
                    f"  {h['stock_id']} {html.escape(str(h['name']))}｜{bt_label}｜"
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
                    f"  {h['stock_id']} {html.escape(str(h['name']))}｜"
                    f"突破點 >{bt_px:.0f} 且量>均×1.5｜"
                    f"距突破 {pct:+.1f}%｜分數 {h['score_today']:.4f}"
                )
    elif not held:
        lines.append("（目前無持倉，請用 /buy 代號 價格 股數 記錄買進後，明日起會顯示持倉狀態）")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 輸入驗證
# ─────────────────────────────────────────────
def _validate_stock_id(raw: str) -> Optional[str]:
    """驗證使用者輸入的股票代號：只允許四碼台股（與 risk/build_features 一致）。

    不合法回 None（呼叫端回覆錯誤訊息）。避免任意字串進入 DB 查詢 /
    portfolio.json / HTML 回顯。
    """
    sid = raw.strip()
    if re.fullmatch(r"\d{4}", sid):
        return sid
    return None


# ─────────────────────────────────────────────
# Portfolio 管理
# ─────────────────────────────────────────────
def _load_portfolio() -> Dict:
    # 原子讀取：損毀時試 .bak，全失敗 raise（不靜默回空持倉而誤判使用者無部位）
    return safe_read_json(PORTFOLIO_FILE, default={"positions": []})


def _save_portfolio(pf: Dict) -> None:
    # 原子寫入：/buy /sell 寫入真實持倉，避免中途中斷造成損毀
    atomic_write_json(PORTFOLIO_FILE, pf)


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
        name       = html.escape(str(pos.get("stock_name", sid)))  # 進 HTML parse_mode 前 escape
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
    stock_id = _validate_stock_id(args[0])
    if stock_id is None:
        return "❌ 股票代號格式錯誤（只接受四碼台股，例：2330）"
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
    # 股名來自 DB / 外部資料，回顯到 HTML parse_mode 前必須 escape
    return (
        f"✅ {action_msg}\n"
        f"{stock_id} {html.escape(name)}｜${price:.1f} × {shares:,} 股\n"
        f"金額：${cost:,.0f}"
    )


def _cmd_sell(args: List[str]) -> str:
    """處理 /sell 2330 900"""
    if len(args) < 2:
        return "❌ 格式錯誤\n用法：/sell 代號 價格\n例：/sell 2330 900"
    stock_id = _validate_stock_id(args[0])
    if stock_id is None:
        return "❌ 股票代號格式錯誤（只接受四碼台股，例：2330）"
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
    # 股名來自 portfolio.json（源自 DB），回顯到 HTML parse_mode 前必須 escape
    return (
        f"{sign} 賣出成功\n"
        f"{stock_id} {html.escape(name)}\n"
        f"成本 ${entry_px:.1f} → 賣出 ${sell_price:.1f}｜{shares:,} 股\n"
        f"損益：{pnl_pct:+.1f}%（${pnl_amt:+,.0f}）"
    )


def _cmd_signal(strategy: str = "c") -> str:
    sig = _load_latest_signal(strategy=strategy)
    if not sig:
        make_cmd = f"make daily-{strategy}"
        return f"❌ 今日尚無 Strategy {strategy.upper()} 訊號，請先執行 {make_cmd}"
    return _format_push_message(sig)


def _cmd_why(args: List[str]) -> str:
    """處理 /why 2330 — 顯示多智能體矛盾分析。"""
    if not args:
        return "❌ 請提供股票代號\n用法：/why 2330"

    stock_id = _validate_stock_id(args[0])
    if stock_id is None:
        return "❌ 股票代號格式錯誤（只接受四碼台股，例：/why 2330）"

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


# ─────────────────────────────────────────────
# 誠實日報（daily-brief）
# ─────────────────────────────────────────────
def _load_nav_records(path: Path = NAV_FILE) -> List[Dict]:
    """讀 nav.jsonl（每行一筆 {"date","nav","holdings_n","config_version","notes"}）。

    檔案缺失回空 list；損毀行跳過（日報缺一節不該擋整份推播）。
    """
    if not path.exists():
        return []
    records: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "date" in rec and "nav" in rec:
            records.append(rec)
    return records


def _summarize_nav(records: List[Dict]) -> Optional[Dict]:
    """最新 NAV + 近 30 日（日曆天）變化。基準取「最新日 -30 天以前最近一筆」，
    歷史不足 30 天時退回最早一筆（chg 標注實際基準日）。"""
    if not records:
        return None
    recs = sorted(records, key=lambda r: str(r["date"]))
    latest = recs[-1]
    latest_date = date.fromisoformat(str(latest["date"]))
    cutoff = latest_date - timedelta(days=30)
    base = None
    for r in recs:
        d = date.fromisoformat(str(r["date"]))
        if d <= cutoff:
            base = r
        else:
            break
    if base is None:
        base = recs[0]
    base_nav = float(base["nav"])
    chg = float(latest["nav"]) / base_nav - 1 if base_nav else 0.0
    return {
        "nav": float(latest["nav"]),
        "date": str(latest["date"]),
        "chg_30d": chg,
        "base_date": str(base["date"]),
        "holdings_n": latest.get("holdings_n"),
        "config_version": str(latest.get("config_version", "?")),
    }


def _load_latest_ipo_scan(scan_dir: Path = IPO_DIR) -> Optional[Dict]:
    """最新 scan_YYYY-MM-DD.json；無檔 / 損毀回 None。"""
    files = sorted(scan_dir.glob("scan_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _select_ipo_actionable(
    items: List[Dict], today: date, min_discount: float = IPO_MIN_DISCOUNT
) -> List[Dict]:
    """折價 > min_discount 且尚可申購（sub_end >= 今天）的案子，折價降冪。

    折價未知（無市價）不列入——無法證明達標。sub_end 缺失視為不可申購（保守）。
    """
    out = []
    for it in items:
        disc = it.get("discount")
        if disc is None or float(disc) < min_discount:
            continue
        sub_end = it.get("sub_end")
        try:
            if sub_end is None or date.fromisoformat(str(sub_end)) < today:
                continue
        except ValueError:
            continue
        out.append(it)
    return sorted(out, key=lambda it: float(it["discount"]), reverse=True)


def _load_disposition_pair(dispo_dir: Path = DISPOSITION_DIR) -> tuple:
    """（最新, 前一份）處置股快取；不足時對應位置 None。"""
    files = sorted(dispo_dir.glob("20??-??-??.json"))

    def _read(p: Path) -> Optional[Dict]:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    latest = _read(files[-1]) if files else None
    prev = _read(files[-2]) if len(files) >= 2 else None
    return latest, prev


def _new_disposition_ids(latest: Optional[Dict], prev: Optional[Dict]) -> List[str]:
    """最新快取相對前一份的新增處置股（四碼）。無前一份可比 → 空（首日不誤報全量）。"""
    if not latest or not prev:
        return []
    return sorted(set(latest.get("disposition", [])) - set(prev.get("disposition", [])))


def _disposition_names(dispo: Optional[Dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rec in (dispo or {}).get("records", []):
        sid = str(rec.get("stock_id", ""))
        if re.fullmatch(r"\d{4}", sid):
            out.setdefault(sid, str(rec.get("name", "")))
    return out


def _fetch_today_picks(limit: int = 10) -> Dict:
    """DB 最新 pick_date 的 picks（分數降冪，前 limit 檔）+ 總檔數。"""
    from app.db import get_session
    from sqlalchemy import text as _text

    with get_session() as session:
        rows = session.execute(_text("""
            SELECT p.pick_date, p.stock_id, p.score, s.name
            FROM picks p LEFT JOIN stocks s ON s.stock_id = p.stock_id
            WHERE p.pick_date = (SELECT MAX(pick_date) FROM picks)
            ORDER BY p.score DESC
        """)).fetchall()
    picks = [
        {"pick_date": str(r.pick_date), "stock_id": str(r.stock_id),
         "score": float(r.score), "name": str(r.name or r.stock_id)}
        for r in rows
    ]
    return {"picks": picks[:limit], "total": len(picks)}


def _sentinel_status() -> Dict:
    """哨兵：pick 特徵一致性抽驗（P0-1 型錯位偵測）+ pipeline 最新 job 狀態。"""
    status: Dict[str, object] = {"sanity_ok": None, "mismatch": None,
                                 "error": None, "last_job": None}
    try:
        from app.db import get_session
        from sqlalchemy import text as _text
        from scripts.reconcile_live_vs_backtest import run_sanity

        with get_session() as session:
            bad = run_sanity(session)
            row = session.execute(_text(
                "SELECT job_name, status, started_at FROM jobs "
                "ORDER BY started_at DESC LIMIT 1"
            )).fetchone()
        status["sanity_ok"] = (bad == 0)
        status["mismatch"] = int(bad)
        if row:
            status["last_job"] = f"{row.job_name} {row.status}（{row.started_at}）"
    except Exception as exc:
        status["error"] = str(exc)
    return status


def _render_daily_brief(
    today: date,
    picks_info: Optional[Dict],
    nav: Optional[Dict],
    ipo_items: List[Dict],
    ipo_scan_date: Optional[str],
    dispo_new: List[str],
    dispo_names: Dict[str, str],
    dispo_total: Optional[int],
    sentinel: Dict,
) -> str:
    """組裝誠實日報（純格式化，可測試；股名進 HTML parse_mode 前 escape）。"""
    lines = [
        f"📰 <b>誠實日報 {today.isoformat()}</b>",
        f"⚠️ {HONEST_BANNER}",
        "",
    ]

    # ① A 線今日 picks 前 10（紙上追蹤）
    lines.append("① 🧾 <b>A 線今日 picks 前 10</b>（📄 紙上追蹤，非投資建議）")
    if picks_info and picks_info["picks"]:
        p0 = picks_info["picks"][0]
        lines.append(f"pick_date {p0['pick_date']}｜共 {picks_info['total']} 檔"
                     "（大盤過濾時有效 topN 會 <30）")
        for i, p in enumerate(picks_info["picks"], 1):
            lines.append(f"  {i}. {p['stock_id']} {html.escape(p['name'])}｜{p['score']:.4f}")
    else:
        lines.append("  （picks 表無資料——pipeline 可能未跑）")
    lines.append("")

    # ② paper NAV
    lines.append("② 📈 <b>Paper NAV</b>（訊號價、無成本、raw close → 系統性低估）")
    if nav:
        sign = "🟢" if nav["chg_30d"] >= 0 else "🔴"
        lines.append(
            f"  {nav['nav']:.4f}（{nav['date']}）｜{sign} 近 30 日 {nav['chg_30d']:+.2%}"
            f"（基準 {nav['base_date']}）｜持股 {nav['holdings_n']} 檔｜{nav['config_version']}"
        )
    else:
        lines.append("  （無 NAV 紀錄——paper_nav 可能未跑）")
    lines.append("")

    # ③ 今日申購機會（唯一可行動項，放醒目）
    lines.append(f"③ 🎯🎯 <b>今日申購機會（唯一可行動項）</b>"
                 f"｜掃描日 {ipo_scan_date or '—'}")
    if ipo_items:
        for it in ipo_items:
            name = html.escape(str(it.get("name", it.get("stock_id", "?"))))
            price = it.get("effective_price")
            mkt = it.get("market_price")
            price_part = f"｜承銷 {price:,.1f} / 市價 {mkt:,.1f}" if price and mkt else ""
            lines.append(
                f"  🔥 {it.get('stock_id', '?')} {name}｜折價 <b>{float(it['discount']):+.1%}</b>"
                f"｜申購 {it.get('sub_start', '?')}~{it.get('sub_end', '?')}"
                f"｜抽籤 {it.get('draw_date', '?')}{price_part}"
            )
        lines.append("  （申購處理費 20 元；中籤另約 50 元；折價為掃描時點市價，申購前自行再確認）")
    else:
        lines.append(f"  （今日無折價 >{IPO_MIN_DISCOUNT:.0%} 且可申購的案子）")
    lines.append("")

    # ④ 處置股新增警示
    lines.append("④ 🚧 <b>處置股</b>")
    if dispo_total is None:
        lines.append("  （無處置股快取）")
    elif dispo_new:
        names = "、".join(
            f"{s} {html.escape(dispo_names.get(s, ''))}".rstrip() for s in dispo_new)
        lines.append(f"  🆕 新增 {len(dispo_new)} 檔：{names}（總數 {dispo_total} 檔）")
    else:
        lines.append(f"  無新增（總數 {dispo_total} 檔）")
    lines.append("")

    # ⑤ 哨兵狀態
    lines.append("⑤ 🛡️ <b>哨兵</b>")
    if sentinel.get("error"):
        lines.append(f"  ⚠️ 哨兵無法執行：{html.escape(str(sentinel['error'])[:120])}")
    elif sentinel.get("sanity_ok"):
        lines.append("  ✅ pick 特徵一致性抽驗通過")
    else:
        lines.append(f"  🚨 pick 特徵 MISMATCH {sentinel.get('mismatch')} 筆"
                     "（P0-1 型錯位，今日 picks 分數不可信）")
    if sentinel.get("last_job"):
        lines.append(f"  pipeline 最新 job：{html.escape(str(sentinel['last_job']))}")

    return "\n".join(lines)


def _build_daily_brief() -> str:
    """收集各節資料並組裝日報。單節失敗不擋整份（該節顯示缺料原因）。"""
    today = date.today()

    try:
        picks_info = _fetch_today_picks(limit=10)
    except Exception as exc:
        print(f"[daily-brief] picks 查詢失敗：{exc}")
        picks_info = None

    nav = _summarize_nav(_load_nav_records())

    scan = _load_latest_ipo_scan()
    ipo_items: List[Dict] = []
    ipo_scan_date = None
    if scan:
        ipo_scan_date = scan.get("scan_date")
        ipo_items = _select_ipo_actionable(scan.get("items", []), today)

    dispo, dispo_prev = _load_disposition_pair()
    dispo_new = _new_disposition_ids(dispo, dispo_prev)
    dispo_names = _disposition_names(dispo)
    dispo_total = len(dispo.get("disposition", [])) if dispo else None

    sentinel = _sentinel_status()

    return _render_daily_brief(
        today=today,
        picks_info=picks_info,
        nav=nav,
        ipo_items=ipo_items,
        ipo_scan_date=ipo_scan_date,
        dispo_new=dispo_new,
        dispo_names=dispo_names,
        dispo_total=dispo_total,
        sentinel=sentinel,
    )


HELP_TEXT = """\
📖 <b>Strategy C/D Bot 指令</b>

/signal      今日 Strategy C 選股建議
/signal c    今日 Strategy C 選股建議（日頻、Rank Drop 出場）
/signal d    今日 Strategy D 選股建議（label=5d + trailing stop -25%）
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
        # /signal       → Strategy C（預設）
        # /signal c     → Strategy C
        # /signal d     → Strategy D
        strat = args[0].lower() if args and args[0].lower() in ("c", "d") else "c"
        reply = _cmd_signal(strategy=strat)
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
    parser.add_argument("--strategy", type=str, default="c",
                        choices=["c", "d", "daily-brief"],
                        help="daily-brief=誠實日報（每日排程用）；"
                             "c/d=策略訊號（已降級紙上，手動用；預設 c）")
    args = parser.parse_args()

    # 載入 token
    load_config()
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    # 選填的第二層授權：除了 chat id 外同時比對發訊者 user id。
    # 未設定時退回只比 chat id（向後相容）。群組 chat 中任何成員都共享 chat id，
    # 設定 TELEGRAM_USER_ID 可防止群組其他成員操作 /buy /sell。
    user_id = os.environ.get("TELEGRAM_USER_ID", "")

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
        # 誠實日報（每日排程預設；單節缺料不擋整份）
        if args.strategy == "daily-brief":
            msg = _build_daily_brief()
            ok = bot.send(msg)
            if ok and not dry_run:
                print(f"✅ 誠實日報已推送（{date.today().isoformat()}）")
            elif not ok:
                sys.exit(1)
            return

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
    # fail-closed：listen 會處理 /buy /sell 等操作真實持倉的指令，CHAT_ID 是唯一授權邊界。
    # 未設則拒絕啟動（而非降級成接受任意聊天室），避免任何知道 bot @username 的人下指令。
    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID 未設定，listen 模式拒絕啟動（避免接受任意聊天室指令）")
        sys.exit(1)
    print(f"🤖 Bot 啟動，監聽指令中... (Ctrl+C 停止)")

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

            # 若設定了 TELEGRAM_USER_ID，額外比對發訊者 user id（防群組內其他成員下指令）
            if user_id:
                from_user_id = str((msg.get("from") or {}).get("id", ""))
                if from_user_id != str(user_id):
                    print(f"[ignore] 非授權 user_id={from_user_id} ({user}): {text[:40]}")
                    continue

            print(f"[cmd] {user}: {text[:60]}")
            _dispatch(bot, text, from_id)


if __name__ == "__main__":
    main()
