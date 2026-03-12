#!/usr/bin/env python3
"""
generate_trade_log.py
從回測 JSON 提取 trades_log，JOIN stocks 表取名稱，輸出 trade_log.csv。

Usage:
    # 直接用現有 JSON（秒完成，推薦）
    python scripts/generate_trade_log.py \
        --backtest-json artifacts/backtest/test_v1a_with_sf.json \
        --output artifacts/trade_log.csv

    # 或依參數自動尋找/重跑
    python scripts/generate_trade_log.py \
        --months 120 --train-lookback 1825 --seasonal-filter \
        --output artifacts/trade_log.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.db import get_session
from app.models import Stock

# 預設優先使用的回測 JSON（與 --months 120 --train-lookback 1825 --seasonal-filter 相符）
_DEFAULT_CANDIDATES = [
    ROOT / "artifacts" / "backtest" / "test_v1a_with_sf.json",
]


def _load_stock_names() -> dict[str, str]:
    """從 stocks 表取 {stock_id: name} 字典"""
    with get_session() as session:
        rows = session.query(Stock.stock_id, Stock.name).all()
    return {r.stock_id: (r.name or "") for r in rows}


def _generate_from_json(json_path: Path, output_path: Path) -> None:
    print(f"讀取回測 JSON: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    trades = data.get("trades_log", [])
    if not trades:
        print("ERROR: trades_log 為空，請確認回測 JSON 格式正確", file=sys.stderr)
        sys.exit(1)
    print(f"  交易筆數: {len(trades)}")

    print("查詢股票名稱...")
    name_map = _load_stock_names()
    print(f"  股票主檔: {len(name_map)} 支")

    df = pd.DataFrame(trades)
    df["name"] = df["stock_id"].map(name_map).fillna("")

    # 欄位順序
    ordered_cols = [
        "stock_id", "name", "entry_date", "exit_date",
        "entry_price", "exit_price", "realized_pnl_pct",
        "stoploss_triggered", "exit_reason", "score",
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ 已儲存: {output_path}（{len(df)} 筆）")


def _run_new_backtest(args: argparse.Namespace) -> Path:
    """跑新的回測，回傳輸出 JSON 路徑"""
    from datetime import datetime

    from app.config import load_config
    from skills.backtest import run_backtest

    config = load_config()  # noqa: F841
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = ROOT / "artifacts" / "backtest" / f"trade_log_bt_{ts}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"執行回測（months={args.months}, train_lookback={args.train_lookback}, "
          f"seasonal_filter={args.seasonal_filter}）…")
    with get_session() as session:
        result = run_backtest(
            session=session,
            months=args.months,
            train_lookback_days=args.train_lookback,
            enable_seasonal_filter=args.seasonal_filter,
            clip_loss_pct=-0.50,
        )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, default=str)
    print(f"回測完成，JSON: {out_json}")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description="產生 trade_log.csv")
    parser.add_argument("--backtest-json", type=Path, default=None,
                        help="指定現有回測 JSON（優先使用）")
    parser.add_argument("--months", type=int, default=120,
                        help="回測月數（找不到現有 JSON 時才用）")
    parser.add_argument("--train-lookback", type=int, default=1825,
                        dest="train_lookback",
                        help="訓練視窗天數（找不到現有 JSON 時才用）")
    parser.add_argument("--seasonal-filter", action="store_true",
                        dest="seasonal_filter",
                        help="啟用季節性降倉（找不到現有 JSON 時才用）")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/trade_log.csv"))
    args = parser.parse_args()

    # 決定 JSON 來源
    if args.backtest_json is not None:
        json_path = args.backtest_json
        if not json_path.exists():
            print(f"ERROR: 指定的 JSON 不存在: {json_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # 自動找現有 candidate
        found = next((p for p in _DEFAULT_CANDIDATES if p.exists()), None)
        if found:
            print(f"找到現有回測 JSON: {found}")
            json_path = found
        else:
            print("找不到現有回測 JSON，開始重新回測（約需 90 分鐘）…")
            json_path = _run_new_backtest(args)

    _generate_from_json(json_path, args.output)


if __name__ == "__main__":
    main()
