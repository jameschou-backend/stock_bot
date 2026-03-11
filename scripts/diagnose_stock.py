#!/usr/bin/env python
"""股票診斷工具：分析指定股票在指定日期的選股排名與過濾原因。

用法：
    python scripts/diagnose_stock.py --stock 2408 --date 2025-03-01
    python scripts/diagnose_stock.py --stock 2408 --date 2025-03-01 --show-all
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# 確保可 import app/skills
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from app.config import load_config
from app.db import get_session
from app.models import Feature, ModelVersion, RawPrice, Stock
from skills import risk
from skills import tradability_filter
from skills.build_features import FEATURE_COLUMNS

# 顯示的關鍵特徵（含新增強勢訊號特徵）
KEY_FEATURES = [
    "foreign_buy_streak",
    "volume_surge_ratio",
    "foreign_buy_intensity",
    "ret_20",
    "amt_ratio_20",
    "rsi_14",
    "foreign_buy_ratio_5",
    "foreign_buy_consecutive_days",
]


def _find_feature_date(session, target: date) -> date | None:
    """找到 <= target 最近的特徵日期。"""
    row = (
        session.query(Feature.trading_date)
        .filter(Feature.trading_date <= target)
        .order_by(Feature.trading_date.desc())
        .limit(1)
        .one_or_none()
    )
    return row[0] if row else None


def _load_features_on_date(session, feature_date: date) -> pd.DataFrame:
    """讀取指定日期所有 4 碼股票的特徵。"""
    stmt = (
        select(Feature.stock_id, Feature.features_json)
        .where(Feature.trading_date == feature_date)
    )
    df = pd.read_sql(stmt, session.get_bind())
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]
    return df


def _parse_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    parsed = [
        json.loads(v) if isinstance(v, str) else v
        for v in df["features_json"]
    ]
    feat_df = pd.json_normalize(parsed)
    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = np.nan
    return feat_df[feature_names].copy()


def _load_price_window(session, feature_date: date, stock_ids: list[str]) -> pd.DataFrame:
    start = feature_date - timedelta(days=40)
    stmt = (
        select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close, RawPrice.volume)
        .where(RawPrice.trading_date.between(start, feature_date))
        .where(RawPrice.stock_id.in_(stock_ids))
        .order_by(RawPrice.stock_id, RawPrice.trading_date)
    )
    df = pd.read_sql(stmt, session.get_bind())
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df.dropna(subset=["close", "volume"])


def _get_stock_name(session, stock_id: str) -> str:
    row = session.query(Stock.name).filter(Stock.stock_id == stock_id).one_or_none()
    return row[0] if row else "未知"


def _impute(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = feat_df.copy().replace([np.inf, -np.inf], np.nan)
    medians = feat_df.median(skipna=True)
    for col in feat_df.columns:
        if feat_df[col].isna().all():
            feat_df[col] = feat_df[col].fillna(0)
        else:
            feat_df[col] = feat_df[col].fillna(medians[col])
    return feat_df


def _effective_topn(config, feature_date: date) -> int:
    topn = config.topn
    month = feature_date.month
    weak_months = tuple(getattr(config, "seasonal_weak_months", (3, 10)))
    mult = float(getattr(config, "seasonal_topn_multiplier", 0.5))
    if month in weak_months:
        topn = max(5, int(topn * mult))
    return topn


def run_diagnose(stock_id: str, target_date: date, show_all: bool = False) -> None:
    config = load_config()

    with get_session() as session:
        _run_diagnose_inner(stock_id, target_date, show_all, config, session)


def _run_diagnose_inner(
    stock_id: str,
    target_date: date,
    show_all: bool,
    config,
    session,
) -> None:
    try:
        # ── 1. 找特徵日期 ──
        feature_date = _find_feature_date(session, target_date)
        if feature_date is None:
            print(f"❌ 找不到 <= {target_date} 的特徵資料。")
            return
        if feature_date != target_date:
            print(f"⚠️  指定日期 {target_date} 無特徵，使用最近特徵日：{feature_date}")

        # ── 2. 載入模型 ──
        mv = (
            session.query(ModelVersion)
            .order_by(ModelVersion.created_at.desc())
            .limit(1)
            .one_or_none()
        )
        if mv is None:
            print("❌ 找不到模型。")
            return
        artifact = joblib.load(mv.artifact_path)
        model = artifact["model"]
        feature_names = artifact["feature_names"]

        # ── 3. 載入所有特徵 ──
        raw_df = _load_features_on_date(session, feature_date)
        if raw_df.empty:
            print(f"❌ {feature_date} 無特徵資料。")
            return

        # ── 4. 取得 universe（可交易上市普通股）──
        all_stocks = raw_df["stock_id"].tolist()
        universe_df = risk.get_universe(session, feature_date, config)
        tradability_logs: dict = {}
        if getattr(config, "enable_tradability_filter", True):
            universe_df, tradability_logs = tradability_filter.filter_universe(
                session, universe_df, feature_date, return_stats=True
            )
        valid_stocks = set(universe_df["stock_id"].astype(str).tolist())
        if valid_stocks:
            raw_df = raw_df[raw_df["stock_id"].isin(valid_stocks)]

        # ── 5. 流動性過濾 ──
        price_df = _load_price_window(session, feature_date, raw_df["stock_id"].tolist())
        avg_turnover_df = risk.apply_liquidity_filter(price_df, config)
        liquid_stocks = set(avg_turnover_df["stock_id"].tolist())

        # 目標股流動性資訊
        target_turnover_row = avg_turnover_df[avg_turnover_df["stock_id"] == stock_id]
        target_avg_turnover = (
            float(target_turnover_row["avg_turnover"].iloc[0])
            if not target_turnover_row.empty
            else None
        )
        liquidity_threshold = (
            float(getattr(config, "min_amt_20", 0.0) or 0.0)
            or float(getattr(config, "min_avg_turnover", 0.0)) * 1e8
        )

        # ── 6. 目標股最新股價 ──
        target_price_rows = price_df[price_df["stock_id"] == stock_id].sort_values("trading_date")
        target_close = float(target_price_rows["close"].iloc[-1]) if not target_price_rows.empty else None

        # ── 7. 準備特徵矩陣 ──
        scored_df = raw_df[raw_df["stock_id"].isin(liquid_stocks)].copy().reset_index(drop=True)
        if scored_df.empty:
            print("❌ 流動性過濾後無候選股。")
            return

        feat_df = _parse_features(scored_df, feature_names)
        feat_df_imputed = _impute(feat_df)

        # ── 8. 打分 ──
        scores = model.predict(feat_df_imputed.values)
        scored_df["score"] = scores
        scored_df = scored_df.sort_values("score", ascending=False).reset_index(drop=True)
        scored_df["rank"] = scored_df.index + 1

        # ── 9. 有效 topN（含季節性降倉）──
        eff_topn = _effective_topn(config, feature_date)
        threshold_score = (
            float(scored_df.iloc[eff_topn - 1]["score"])
            if len(scored_df) >= eff_topn
            else float(scored_df["score"].min())
        )

        # ── 10. 目標股資訊 ──
        stock_name = _get_stock_name(session, stock_id)
        target_row = scored_df[scored_df["stock_id"] == stock_id]
        in_liquid = stock_id in liquid_stocks
        in_universe = stock_id in valid_stocks if valid_stocks else True
        in_raw = stock_id in set(raw_df["stock_id"].tolist())

        # 目標股特徵值（原始，未 impute）
        target_raw_features: dict = {}
        if in_raw:
            raw_feat_df = _parse_features(
                raw_df[raw_df["stock_id"] == stock_id].reset_index(drop=True),
                FEATURE_COLUMNS,
            )
            if not raw_feat_df.empty:
                target_raw_features = raw_feat_df.iloc[0].to_dict()

        # 特徵覆蓋率
        n_valid = int(sum(
            1 for col in FEATURE_COLUMNS
            if col in target_raw_features and not pd.isna(target_raw_features.get(col))
        ))
        n_total = len(FEATURE_COLUMNS)

        # ── 輸出報告 ──
        print()
        print(f"=== {stock_id} {stock_name} 診斷報告（{feature_date}）===")
        print()

        # 結果
        if not in_raw:
            print(f"【結果】❌ 無特徵資料（該日無記錄）")
        elif not in_universe and valid_stocks:
            print(f"【結果】❌ 不在可交易 universe（ETF/非普通股/已下市）")
        elif not in_liquid:
            print(f"【結果】❌ 流動性不足（未通過過濾）")
        elif target_row.empty:
            print(f"【結果】❌ 未入選（打分後無記錄）")
        else:
            rank = int(target_row.iloc[0]["rank"])
            score = float(target_row.iloc[0]["score"])
            if rank <= eff_topn:
                print(f"【結果】✅ 入選第 {rank} 名（有效 TopN={eff_topn}）")
            else:
                print(f"【結果】⚠️  排名第 {rank}（未進前 {eff_topn}）")
        print()

        # 模型評分
        if not target_row.empty:
            score = float(target_row.iloc[0]["score"])
            rank = int(target_row.iloc[0]["rank"])
            print("【模型評分】")
            print(f"  本股得分：{score:.4f}")
            print(f"  本股排名：第 {rank} / {len(scored_df)} 名")
            print(f"  第 {eff_topn} 名門檻：{threshold_score:.4f}")
            diff = score - threshold_score
            diff_sym = "▲" if diff >= 0 else "▼"
            print(f"  差距門檻：{diff_sym}{abs(diff):.4f}")
        elif in_raw and in_liquid:
            print("【模型評分】（資料異常，無法取得分數）")
        print()

        # 過濾關卡
        print("【過濾關卡】")
        # Universe
        if valid_stocks:
            uni_sym = "✅" if in_universe else "❌"
            print(f"  {uni_sym} Universe：{'上市普通股' if in_universe else '非上市普通股 / ETF / 已下市'}")

        # 流動性
        liq_sym = "✅" if in_liquid else "❌"
        if target_avg_turnover is not None:
            avg_vol_lots = target_avg_turnover / max(target_close or 1, 1) / 1000  # 張
            thresh_lots = liquidity_threshold / max(target_close or 1, 1) / 1000
            print(
                f"  {liq_sym} 流動性：近20日均成交額 "
                f"{target_avg_turnover/1e6:.1f}萬元"
                f"（門檻 {liquidity_threshold/1e6:.0f}萬元）"
            )
        else:
            if not in_raw:
                print(f"  ❌ 流動性：無價格資料")
            else:
                print(f"  ❌ 流動性：近20日均成交額不足（門檻 {liquidity_threshold/1e6:.0f}萬元）")

        # 股價
        if target_close is not None:
            price_ok = target_close >= 5
            p_sym = "✅" if price_ok else "❌"
            print(f"  {p_sym} 股價：{target_close:.2f} 元（參考門檻 > 5 元）")
        else:
            print(f"  ⚠️  股價：無資料")

        # 特徵覆蓋率
        cov_sym = "✅" if n_valid >= int(n_total * 0.8) else "❌"
        print(f"  {cov_sym} 特徵覆蓋率：{n_valid}/{n_total} 個特徵完整")
        print()

        # 關鍵特徵值
        print("【關鍵特徵值】")
        for feat in KEY_FEATURES:
            val = target_raw_features.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                print(f"  {feat}：N/A")
            else:
                print(f"  {feat}：{val:.4f}" if isinstance(val, float) else f"  {feat}：{val}")
        # 季節性降倉提示
        month = feature_date.month
        if month in tuple(getattr(config, "seasonal_weak_months", (3, 10))):
            print(f"  （⚠️  {month}月季節性降倉：topN {config.topn}→{eff_topn}）")
        print()

        # 同期前 N 名
        topn_display = 5 if not show_all else len(scored_df)
        print(f"【同期前 {'5' if not show_all else str(len(scored_df))} 名股票】")
        name_cache: dict[str, str] = {}
        for _, row in scored_df.head(topn_display).iterrows():
            sid = str(row["stock_id"])
            if sid not in name_cache:
                name_cache[sid] = _get_stock_name(session, sid)
            mark = " ← 目標股" if sid == stock_id else ""
            print(f"  {int(row['rank']):>3}. {sid} {name_cache[sid]:<8s}（得分 {float(row['score']):.4f}）{mark}")

        if not show_all and not target_row.empty:
            rank = int(target_row.iloc[0]["rank"])
            if rank > 5:
                print(f"  ...")
                print(f"  {rank:>3}. {stock_id} {stock_name:<8s}（得分 {float(target_row.iloc[0]['score']):.4f}） ← 目標股")
        print()

    except Exception:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="診斷指定股票在指定日期的選股狀況。"
    )
    parser.add_argument("--stock", required=True, help="股票代號（例：2408）")
    parser.add_argument(
        "--date",
        required=True,
        help="診斷日期 YYYY-MM-DD（使用當日或最近一個特徵日）",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        default=False,
        help="輸出完整排名表",
    )
    args = parser.parse_args()

    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print(f"❌ 日期格式錯誤：{args.date}，請使用 YYYY-MM-DD")
        sys.exit(1)

    run_diagnose(
        stock_id=args.stock,
        target_date=target_date,
        show_all=args.show_all,
    )


if __name__ == "__main__":
    main()
