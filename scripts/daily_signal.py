"""每日選股訊號輸出腳本。

從 picks 表讀取今日選股結果，與前一交易日比較，
輸出買進/賣出/維持清單與建議部位金額。

輸出：artifacts/daily_signal/YYYY-MM-DD.json

用法：
    python scripts/daily_signal.py                         # 使用最新有效日期
    python scripts/daily_signal.py --date 2026-03-13       # 指定日期
    python scripts/daily_signal.py --capital 2000000       # 自訂資金
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

# 確保專案根目錄在 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import get_session
from app.models import Pick, Stock


OUTPUT_DIR = ROOT / "artifacts" / "daily_signal"


def _load_picks_for_date(session, target_date: date) -> List[Dict]:
    """從 picks 表讀取指定日期的選股結果。"""
    rows = (
        session.query(Pick)
        .filter(Pick.pick_date == target_date)
        .order_by(Pick.score.desc())
        .all()
    )
    return [
        {
            "stock_id": str(r.stock_id),
            "score": float(r.score),
            "model_id": str(r.model_id),
            "reason_json": r.reason_json if isinstance(r.reason_json, dict) else {},
        }
        for r in rows
    ]


def _get_available_pick_dates(session, limit: int = 60) -> List[date]:
    """取得 picks 表中有資料的日期清單（最新在前）。"""
    rows = (
        session.query(Pick.pick_date)
        .distinct()
        .order_by(Pick.pick_date.desc())
        .limit(limit)
        .all()
    )
    return [r[0] for r in rows]


def _load_stock_names(session, stock_ids: List[str]) -> Dict[str, str]:
    """查詢股票中文名稱。"""
    if not stock_ids:
        return {}
    rows = (
        session.query(Stock.stock_id, Stock.name)
        .filter(Stock.stock_id.in_(stock_ids))
        .all()
    )
    result = {str(r.stock_id): str(r.name or r.stock_id) for r in rows}
    for sid in stock_ids:
        result.setdefault(sid, sid)
    return result


def generate_signal(target_date: date, capital: int) -> Dict:
    """產生每日訊號：載入選股、比較前日、輸出結構化結果。"""
    with get_session() as session:
        available_dates = _get_available_pick_dates(session)

        if not available_dates:
            print("❌ picks 表無資料，請先執行 make pipeline")
            sys.exit(1)

        # 找目標日期
        if target_date not in available_dates:
            nearest = available_dates[0]
            print(f"⚠️  {target_date} 無選股資料，使用最近有效日期：{nearest}")
            target_date = nearest

        today_picks = _load_picks_for_date(session, target_date)
        if not today_picks:
            print(f"❌ {target_date} 無選股資料")
            sys.exit(1)

        # 找前一交易日
        today_idx = available_dates.index(target_date)
        prev_date = available_dates[today_idx + 1] if today_idx + 1 < len(available_dates) else None
        prev_picks = _load_picks_for_date(session, prev_date) if prev_date else []

        # 載入股票名稱
        all_sids = list({p["stock_id"] for p in today_picks + prev_picks})
        names = _load_stock_names(session, all_sids)

    # 計算建議部位金額
    n = len(today_picks)
    amount_per_pos = capital // n if n > 0 else 0

    today_ids = {p["stock_id"] for p in today_picks}
    prev_ids = {p["stock_id"] for p in prev_picks}

    buy_ids = today_ids - prev_ids
    sell_ids = prev_ids - today_ids
    hold_ids = today_ids & prev_ids

    # 建立今日持倉清單（含 action）
    holdings = []
    for p in today_picks:
        sid = p["stock_id"]
        action = "buy" if sid in buy_ids else "hold"
        holdings.append({
            "stock_id": sid,
            "name": names.get(sid, sid),
            "score": round(p["score"], 6),
            "action": action,
            "amount": amount_per_pos,
        })

    # 賣出清單（前日有、今日無）
    sell_list = []
    for p in prev_picks:
        sid = p["stock_id"]
        if sid in sell_ids:
            sell_list.append({
                "stock_id": sid,
                "name": names.get(sid, sid),
                "score": round(p["score"], 6),
            })

    result = {
        "date": target_date.isoformat(),
        "previous_date": prev_date.isoformat() if prev_date else None,
        "capital": capital,
        "num_positions": n,
        "amount_per_position": amount_per_pos,
        "holdings": holdings,
        "changes": {
            "buy":  [h for h in holdings if h["action"] == "buy"],
            "sell": sell_list,
            "hold": [h for h in holdings if h["action"] == "hold"],
        },
        "summary": {
            "buy_count": len(buy_ids),
            "sell_count": len(sell_ids),
            "hold_count": len(hold_ids),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return result


def _print_signal(sig: Dict) -> None:
    """輸出人類可讀的訊號摘要。"""
    d = sig["date"]
    prev = sig["previous_date"] or "（無）"
    cap = sig["capital"]
    n = sig["num_positions"]
    amt = sig["amount_per_position"]
    s = sig["summary"]

    print()
    print("=" * 56)
    print(f"  每日選股訊號  {d}  （前日：{prev}）")
    print("=" * 56)
    print(f"  資金：{cap:,} 元  /  持倉：{n} 檔  /  每檔：{amt:,} 元")
    print(f"  買進 +{s['buy_count']}  賣出 -{s['sell_count']}  維持 {s['hold_count']}")

    if sig["changes"]["buy"]:
        print("\n  ▶ 買進（新進場）")
        for h in sig["changes"]["buy"]:
            print(f"    {h['stock_id']}  {h['name']:<10}  分數 {h['score']:.4f}  金額 {h['amount']:>9,}")

    if sig["changes"]["sell"]:
        print("\n  ◀ 賣出（出場）")
        for h in sig["changes"]["sell"]:
            print(f"    {h['stock_id']}  {h['name']:<10}  分數 {h['score']:.4f}")

    if sig["changes"]["hold"]:
        print("\n  ─ 維持持有")
        for h in sig["changes"]["hold"]:
            print(f"    {h['stock_id']}  {h['name']:<10}  分數 {h['score']:.4f}  金額 {h['amount']:>9,}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="每日選股訊號輸出")
    parser.add_argument(
        "--date",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="指定日期（YYYY-MM-DD），預設今日",
    )
    parser.add_argument(
        "--capital",
        type=int,
        default=1_000_000,
        help="總資金（元），預設 1,000,000",
    )
    args = parser.parse_args()

    signal = generate_signal(args.date, args.capital)
    _print_signal(signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{signal['date']}.json"
    out_path.write_text(json.dumps(signal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已儲存：{out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
