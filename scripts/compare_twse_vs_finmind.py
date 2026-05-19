"""對照腳本：TWSE/TPEx 官方 API vs FinMind 同一天同一資料的差異。

用法：
    # 純驗 TWSE（不需要 FinMind token）
    python scripts/compare_twse_vs_finmind.py 2026-05-15

    # 同時 diff FinMind（需要 FINMIND_TOKEN）
    python scripts/compare_twse_vs_finmind.py 2026-05-15 --with-finmind

    # 只測單一資料類別
    python scripts/compare_twse_vs_finmind.py 2026-05-15 --only prices

驗證內容：
1. TWSE/TPEx 抓得到該日資料（行數、樣本）
2. 與 FinMind 同日同 stock_id 的關鍵欄位數值是否一致
3. 列出兩邊缺漏的股票（可能是 ETF、興櫃、停牌差異）

不會寫入任何 DB，純輸出 stdout / artifacts/twse_finmind_diff/<date>.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 確保能 import 專案模組
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.twse_client import TWSEClient, TWSEError  # noqa: E402

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "twse_finmind_diff"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt_row(row: Dict[str, Any], keys: List[str]) -> str:
    return " | ".join(f"{k}={row.get(k)}" for k in keys)


# ──────────────────────────────────────────────
# FinMind side（可選）
# ──────────────────────────────────────────────

def _fetch_finmind_prices(d: date) -> List[Dict[str, Any]]:
    from app.finmind import fetch_dataset
    token = os.environ.get("FINMIND_TOKEN")
    df = fetch_dataset("TaiwanStockPrice", start_date=d, end_date=d, token=token)
    if df is None or df.empty:
        return []
    return df.to_dict("records")


def _fetch_finmind_institutional(d: date) -> List[Dict[str, Any]]:
    from app.finmind import fetch_dataset
    token = os.environ.get("FINMIND_TOKEN")
    df = fetch_dataset("TaiwanStockInstitutionalInvestorsBuySell", start_date=d, end_date=d, token=token)
    if df is None or df.empty:
        return []
    return df.to_dict("records")


def _fetch_finmind_margin(d: date) -> List[Dict[str, Any]]:
    from app.finmind import fetch_dataset
    token = os.environ.get("FINMIND_TOKEN")
    df = fetch_dataset("TaiwanStockMarginPurchaseShortSale", start_date=d, end_date=d, token=token)
    if df is None or df.empty:
        return []
    return df.to_dict("records")


def _fetch_finmind_per(d: date) -> List[Dict[str, Any]]:
    from app.finmind import fetch_dataset
    token = os.environ.get("FINMIND_TOKEN")
    df = fetch_dataset("TaiwanStockPER", start_date=d, end_date=d, token=token)
    if df is None or df.empty:
        return []
    return df.to_dict("records")


# ──────────────────────────────────────────────
# Diff helpers
# ──────────────────────────────────────────────

def _index_by_stock(rows: List[Dict[str, Any]], stock_key: str = "stock_id") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = r.get(stock_key)
        if isinstance(sid, str):
            out[sid] = r
    return out


def _diff_field(twse_val: Any, finmind_val: Any, tolerance: float = 0.01) -> Tuple[bool, str]:
    """回傳 (matches, message)。對浮點數允許 tolerance 容差。"""
    if twse_val is None and finmind_val in (None, "", 0, "0"):
        return True, "both None/empty"
    if finmind_val is None and twse_val in (None, "", 0, "0"):
        return True, "both None/empty"
    try:
        twse_f = float(twse_val) if twse_val is not None else None
        fm_f = float(finmind_val) if finmind_val is not None else None
        if twse_f is None or fm_f is None:
            return False, f"one is None: twse={twse_val} finmind={finmind_val}"
        diff = abs(twse_f - fm_f)
        rel = diff / max(abs(fm_f), 1e-9)
        if rel < tolerance:
            return True, f"≈ {twse_f}"
        return False, f"twse={twse_f} finmind={fm_f} rel_diff={rel:.4f}"
    except (TypeError, ValueError):
        if str(twse_val).strip() == str(finmind_val).strip():
            return True, f"={twse_val}"
        return False, f"twse={twse_val} finmind={finmind_val}"


def _compare(
    name: str,
    twse_rows: List[Dict[str, Any]],
    finmind_rows: List[Dict[str, Any]],
    field_map: Dict[str, str],
    sample_size: int = 5,
) -> Dict[str, Any]:
    """對照兩邊資料。field_map 是 TWSE 欄位 -> FinMind 欄位。"""
    twse_idx = _index_by_stock(twse_rows)
    fm_idx = _index_by_stock(finmind_rows)

    only_twse = sorted(set(twse_idx) - set(fm_idx))
    only_fm = sorted(set(fm_idx) - set(twse_idx))
    common = sorted(set(twse_idx) & set(fm_idx))

    field_mismatches: List[Dict[str, Any]] = []
    sample_compares: List[Dict[str, Any]] = []

    for sid in common:
        twse_r = twse_idx[sid]
        fm_r = fm_idx[sid]
        row_mismatches = []
        for twse_field, fm_field in field_map.items():
            ok, msg = _diff_field(twse_r.get(twse_field), fm_r.get(fm_field))
            if not ok:
                row_mismatches.append({"field": twse_field, "vs": fm_field, "msg": msg})
        if row_mismatches:
            field_mismatches.append({"stock_id": sid, "mismatches": row_mismatches})
        if len(sample_compares) < sample_size:
            sample_compares.append({
                "stock_id": sid,
                "twse": {k: twse_r.get(k) for k in field_map.keys()},
                "finmind": {v: fm_r.get(v) for v in field_map.values()},
            })

    return {
        "dataset": name,
        "twse_rows": len(twse_rows),
        "finmind_rows": len(finmind_rows),
        "common": len(common),
        "only_twse": len(only_twse),
        "only_finmind": len(only_fm),
        "mismatches": len(field_mismatches),
        "match_rate": (len(common) - len(field_mismatches)) / max(len(common), 1),
        "only_twse_sample": only_twse[:10],
        "only_finmind_sample": only_fm[:10],
        "mismatch_sample": field_mismatches[:5],
        "sample_compares": sample_compares,
    }


# ──────────────────────────────────────────────
# Main flow
# ──────────────────────────────────────────────

def _print_summary(label: str, rows: List[Dict[str, Any]], sample_keys: List[str], n_sample: int = 3) -> None:
    print(f"\n=== {label}：{len(rows)} 筆 ===")
    if not rows:
        print("  （空）")
        return
    twse_count = sum(1 for r in rows if r.get("market") == "TWSE")
    tpex_count = sum(1 for r in rows if r.get("market") == "TPEx")
    print(f"  TWSE: {twse_count}, TPEx: {tpex_count}")
    print(f"  樣本（前 {n_sample} 筆）:")
    for r in rows[:n_sample]:
        print(f"    {_fmt_row(r, sample_keys)}")


def run(target_date: date, with_finmind: bool, only: Optional[str]) -> int:
    print(f"目標日期: {target_date}")
    print(f"FinMind 對照: {'ON' if with_finmind else 'OFF（只跑 TWSE 自我驗證）'}")

    client = TWSEClient()
    summaries: Dict[str, Any] = {"date": target_date.isoformat(), "with_finmind": with_finmind}

    datasets = {
        "prices": {
            "twse_fn": lambda: client.fetch_prices_history(target_date),
            "fm_fn": _fetch_finmind_prices,
            "sample_keys": ["stock_id", "trading_date", "open", "high", "low", "close", "volume", "market"],
            "field_map": {
                "open": "open",
                "high": "max",
                "low": "min",
                "close": "close",
                "volume": "Trading_Volume",
            },
        },
        "institutional": {
            "twse_fn": lambda: client.fetch_institutional_history(target_date),
            "fm_fn": _fetch_finmind_institutional,
            "sample_keys": ["stock_id", "foreign_buy", "foreign_sell", "foreign_net", "trust_net", "dealer_net", "market"],
            "field_map": {
                "foreign_net": "Foreign_Investor",
                "trust_net": "Investment_Trust",
            },
        },
        "margin": {
            "twse_fn": lambda: client.fetch_margin_short_history(target_date),
            "fm_fn": _fetch_finmind_margin,
            "sample_keys": ["stock_id", "margin_purchase_buy", "margin_purchase_today_balance", "short_sale_today_balance", "market"],
            "field_map": {
                "margin_purchase_buy": "MarginPurchaseBuy",
                "margin_purchase_today_balance": "MarginPurchaseTodayBalance",
                "short_sale_today_balance": "ShortSaleTodayBalance",
            },
        },
        "per": {
            "twse_fn": lambda: client.fetch_per_history(target_date),
            "fm_fn": _fetch_finmind_per,
            "sample_keys": ["stock_id", "per", "pbr", "dividend_yield", "market"],
            "field_map": {
                "per": "PER",
                "pbr": "PBR",
                "dividend_yield": "dividend_yield",
            },
        },
    }

    selected = [only] if only else list(datasets.keys())
    for unknown in [k for k in selected if k not in datasets]:
        print(f"⚠ 未知 dataset: {unknown}（可選: {list(datasets.keys())}）", file=sys.stderr)
        return 2

    for key in selected:
        cfg = datasets[key]
        print(f"\n──── {key} ────────────────────────────────────────")
        try:
            twse_rows = cfg["twse_fn"]()
        except TWSEError as exc:
            print(f"❌ TWSE {key} 取資料失敗: {exc}")
            summaries[key] = {"error": str(exc)}
            continue
        _print_summary(f"TWSE/TPEx {key}", twse_rows, cfg["sample_keys"])

        if not with_finmind:
            summaries[key] = {
                "twse_rows": len(twse_rows),
                "twse_count": sum(1 for r in twse_rows if r.get("market") == "TWSE"),
                "tpex_count": sum(1 for r in twse_rows if r.get("market") == "TPEx"),
            }
            continue

        try:
            fm_rows = cfg["fm_fn"](target_date)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ FinMind {key} 取資料失敗: {exc}")
            summaries[key] = {"twse_rows": len(twse_rows), "finmind_error": str(exc)}
            continue

        summary = _compare(key, twse_rows, fm_rows, cfg["field_map"])
        summaries[key] = summary
        print(
            f"\n  比對結果：common={summary['common']} | "
            f"only_twse={summary['only_twse']} | only_finmind={summary['only_finmind']} | "
            f"mismatches={summary['mismatches']} | match_rate={summary['match_rate']:.2%}"
        )
        if summary["mismatch_sample"]:
            print(f"  前 5 筆不一致樣本:")
            for m in summary["mismatch_sample"]:
                print(f"    {m['stock_id']}: {m['mismatches']}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"{target_date.isoformat()}.json"
    out_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n結果存檔: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="對照 TWSE/TPEx 官方 API 與 FinMind 同日資料")
    parser.add_argument("target_date", help="目標日期 YYYY-MM-DD")
    parser.add_argument("--with-finmind", action="store_true", help="同時呼叫 FinMind 做欄位 diff")
    parser.add_argument("--only", choices=["prices", "institutional", "margin", "per"], help="只測單一類別")
    args = parser.parse_args()

    target_date = _parse_date(args.target_date)
    return run(target_date, with_finmind=args.with_finmind, only=args.only)


if __name__ == "__main__":
    sys.exit(main())
