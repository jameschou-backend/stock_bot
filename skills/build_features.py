from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import (
    Feature,
    RawFundamental,
    RawInstitutional,
    RawMarginShort,
    RawPrice,
    RawThemeFlow,
    Stock,
)


# ── 核心特徵（必須存在才保留該筆資料）──────────────────────────────
CORE_FEATURE_COLUMNS = [
    # 動能
    "ret_5", "ret_10", "ret_20", "ret_60",
    # 均線
    "ma_5", "ma_20", "ma_60",
    # 技術指標（基礎）
    "bias_20", "vol_20", "vol_ratio_20",
    # 法人
    "foreign_net_5", "foreign_net_20",
    "trust_net_5", "trust_net_20",
    "dealer_net_5", "dealer_net_20",
]

# ── 擴充特徵（允許 NaN，用 0 填補）──────────────────────────────
EXTENDED_FEATURE_COLUMNS = [
    # 經典技術指標
    "rsi_14",
    "macd_hist",
    "kd_k", "kd_d",
    # 籌碼面（融資融券）
    "margin_balance_chg_5", "margin_balance_chg_20",
    "short_balance_chg_5", "short_balance_chg_20",
    "margin_short_ratio",
    # 大盤相對強弱
    "market_rel_ret_20",
    # 技術面（擴充）
    "breakout_20",
    "drawdown_60",
    # 籌碼面（擴充）
    "foreign_buy_streak_5",
    "chip_flow_intensity_20",
    # 基本面（月營收）
    "fund_revenue_mom",
    "fund_revenue_yoy",
    "fund_revenue_trend_3m",
    # 題材/金流（產業聚合）
    "theme_turnover_ratio",
    "theme_return_20",
    "theme_hot_score",
]

# 完整特徵列表（供 daily_pick / train_ranker 使用）
FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + EXTENDED_FEATURE_COLUMNS


def _fetch_data(session: Session, start_date: date, end_date: date) -> pd.DataFrame:
    """讀取 raw_prices + raw_institutional + raw_margin_short 並合併"""
    # ── 價格 ──
    price_stmt = (
        select(
            RawPrice.stock_id,
            RawPrice.trading_date,
            RawPrice.open,
            RawPrice.high,
            RawPrice.low,
            RawPrice.close,
            RawPrice.volume,
        )
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.stock_id, RawPrice.trading_date)
    )
    price_df = pd.read_sql(price_stmt, session.get_bind())
    if price_df.empty:
        return price_df

    for col in ["open", "high", "low", "close"]:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    price_df["volume"] = pd.to_numeric(price_df["volume"], errors="coerce")

    # ── 法人 ──
    inst_stmt = (
        select(
            RawInstitutional.stock_id,
            RawInstitutional.trading_date,
            RawInstitutional.foreign_net,
            RawInstitutional.trust_net,
            RawInstitutional.dealer_net,
        )
        .where(RawInstitutional.trading_date.between(start_date, end_date))
        .order_by(RawInstitutional.stock_id, RawInstitutional.trading_date)
    )
    inst_df = pd.read_sql(inst_stmt, session.get_bind())

    if inst_df.empty:
        price_df["foreign_net"] = 0
        price_df["trust_net"] = 0
        price_df["dealer_net"] = 0
    else:
        for col in ["foreign_net", "trust_net", "dealer_net"]:
            inst_df[col] = pd.to_numeric(inst_df[col], errors="coerce").fillna(0)
        price_df = price_df.merge(inst_df, on=["stock_id", "trading_date"], how="left")
        for col in ["foreign_net", "trust_net", "dealer_net"]:
            price_df[col] = price_df[col].fillna(0)

    # ── 融資融券 ──
    margin_stmt = (
        select(
            RawMarginShort.stock_id,
            RawMarginShort.trading_date,
            RawMarginShort.margin_purchase_balance,
            RawMarginShort.short_sale_balance,
        )
        .where(RawMarginShort.trading_date.between(start_date, end_date))
        .order_by(RawMarginShort.stock_id, RawMarginShort.trading_date)
    )
    margin_df = pd.read_sql(margin_stmt, session.get_bind())

    if margin_df.empty:
        price_df["margin_purchase_balance"] = np.nan
        price_df["short_sale_balance"] = np.nan
    else:
        for col in ["margin_purchase_balance", "short_sale_balance"]:
            margin_df[col] = pd.to_numeric(margin_df[col], errors="coerce")
        price_df = price_df.merge(margin_df, on=["stock_id", "trading_date"], how="left")

    # ── 股票主檔（產業） ──
    stock_stmt = (
        select(
            Stock.stock_id,
            Stock.industry_category,
            Stock.is_listed,
            Stock.security_type,
        )
        .where(Stock.security_type == "stock")
        .where(Stock.is_listed == True)
    )
    stock_df = pd.read_sql(stock_stmt, session.get_bind())
    if not stock_df.empty:
        stock_df["stock_id"] = stock_df["stock_id"].astype(str)
        price_df = price_df.merge(stock_df[["stock_id", "industry_category"]], on="stock_id", how="left")
    else:
        price_df["industry_category"] = None

    # ── 基本面（月營收）──
    fund_stmt = (
        select(
            RawFundamental.stock_id,
            RawFundamental.trading_date,
            RawFundamental.revenue_mom,
            RawFundamental.revenue_yoy,
        )
        .where(RawFundamental.trading_date.between(start_date - timedelta(days=370), end_date))
        .order_by(RawFundamental.stock_id, RawFundamental.trading_date)
    )
    fund_df = pd.read_sql(fund_stmt, session.get_bind())
    if fund_df.empty:
        price_df["fund_revenue_mom"] = np.nan
        price_df["fund_revenue_yoy"] = np.nan
    else:
        for col in ["revenue_mom", "revenue_yoy"]:
            fund_df[col] = pd.to_numeric(fund_df[col], errors="coerce")
        fund_df = fund_df.rename(columns={"revenue_mom": "fund_revenue_mom", "revenue_yoy": "fund_revenue_yoy"})
        fund_df = fund_df.sort_values(["stock_id", "trading_date"])
        price_df = price_df.sort_values(["stock_id", "trading_date"])
        merged = []
        for sid, sub in price_df.groupby("stock_id", sort=False):
            sub_f = fund_df[fund_df["stock_id"] == sid]
            if sub_f.empty:
                sub = sub.copy()
                sub["fund_revenue_mom"] = np.nan
                sub["fund_revenue_yoy"] = np.nan
                merged.append(sub)
                continue
            aligned = pd.merge_asof(
                sub.sort_values("trading_date"),
                sub_f.sort_values("trading_date")[["trading_date", "fund_revenue_mom", "fund_revenue_yoy"]],
                on="trading_date",
                direction="backward",
            )
            merged.append(aligned)
        price_df = pd.concat(merged, ignore_index=True)

    # ── 題材/金流（產業聚合）──
    theme_stmt = (
        select(
            RawThemeFlow.theme_id,
            RawThemeFlow.trading_date,
            RawThemeFlow.turnover_ratio,
            RawThemeFlow.theme_return_20,
            RawThemeFlow.hot_score,
        )
        .where(RawThemeFlow.trading_date.between(start_date, end_date))
        .order_by(RawThemeFlow.trading_date, RawThemeFlow.theme_id)
    )
    theme_df = pd.read_sql(theme_stmt, session.get_bind())
    if theme_df.empty:
        price_df["theme_turnover_ratio"] = np.nan
        price_df["theme_return_20"] = np.nan
        price_df["theme_hot_score"] = np.nan
    else:
        theme_df = theme_df.rename(
            columns={
                "theme_id": "industry_category",
                "turnover_ratio": "theme_turnover_ratio",
                "hot_score": "theme_hot_score",
            }
        )
        for col in ["theme_turnover_ratio", "theme_return_20", "theme_hot_score"]:
            theme_df[col] = pd.to_numeric(theme_df[col], errors="coerce")
        price_df = price_df.merge(
            theme_df[["industry_category", "trading_date", "theme_turnover_ratio", "theme_return_20", "theme_hot_score"]],
            on=["industry_category", "trading_date"],
            how="left",
        )

    return price_df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算所有特徵（逐股分組）"""
    if df.empty:
        return df

    df = df.sort_values(["stock_id", "trading_date"]).copy()

    def apply_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trading_date").copy()
        close = group["close"]
        volume = group["volume"]
        high = group["high"]
        low = group["low"]

        # ── 動能 ──
        group["ret_5"] = close.pct_change(5)
        group["ret_10"] = close.pct_change(10)
        group["ret_20"] = close.pct_change(20)
        group["ret_60"] = close.pct_change(60)

        # ── 均線 ──
        group["ma_5"] = close.rolling(5).mean()
        group["ma_20"] = close.rolling(20).mean()
        group["ma_60"] = close.rolling(60).mean()

        # ── 技術指標（基礎）──
        group["bias_20"] = close / group["ma_20"] - 1
        daily_ret = close.pct_change(1)
        group["vol_20"] = daily_ret.rolling(20).std()
        group["vol_ratio_20"] = volume / volume.rolling(20).mean()
        rolling_max20 = close.rolling(20).max()
        rolling_max60 = close.rolling(60).max()
        group["breakout_20"] = close / rolling_max20 - 1
        group["drawdown_60"] = close / rolling_max60 - 1

        # ── 法人 ──
        group["foreign_net_5"] = group["foreign_net"].rolling(5).sum()
        group["foreign_net_20"] = group["foreign_net"].rolling(20).sum()
        group["trust_net_5"] = group["trust_net"].rolling(5).sum()
        group["trust_net_20"] = group["trust_net"].rolling(20).sum()
        group["dealer_net_5"] = group["dealer_net"].rolling(5).sum()
        group["dealer_net_20"] = group["dealer_net"].rolling(20).sum()
        group["foreign_buy_streak_5"] = (
            (group["foreign_net"] > 0).astype(int).rolling(5).sum()
        )
        group["chip_flow_intensity_20"] = (
            (group["foreign_net"] + group["trust_net"] + group["dealer_net"]).rolling(20).sum()
            / volume.rolling(20).sum().replace(0, np.nan)
        )

        # ── RSI 14 ──
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        # loss=0 → rs=inf → RSI=100（純上漲）; gain=0 → rs=0 → RSI=0（純下跌）
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = gain / loss
        group["rsi_14"] = (100 - (100 / (1 + rs))).clip(0, 100)

        # ── MACD Histogram (12, 26, 9) ──
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        group["macd_hist"] = macd_line - signal_line

        # ── KD 隨機指標 (9, 3, 3) ──
        low_min = low.rolling(9).min()
        high_max = high.rolling(9).max()
        denom = high_max - low_min
        rsv = ((close - low_min) / denom.replace(0, np.nan)) * 100
        group["kd_k"] = rsv.ewm(com=2, adjust=False).mean()
        group["kd_d"] = group["kd_k"].ewm(com=2, adjust=False).mean()

        # ── 融資融券特徵 ──
        if "margin_purchase_balance" in group.columns:
            mpb = group["margin_purchase_balance"]
            ssb = group["short_sale_balance"]
            group["margin_balance_chg_5"] = mpb.pct_change(5, fill_method=None)
            group["margin_balance_chg_20"] = mpb.pct_change(20, fill_method=None)
            group["short_balance_chg_5"] = ssb.pct_change(5, fill_method=None)
            group["short_balance_chg_20"] = ssb.pct_change(20, fill_method=None)
            group["margin_short_ratio"] = ssb / mpb.replace(0, np.nan)
        else:
            for col in [
                "margin_balance_chg_5", "margin_balance_chg_20",
                "short_balance_chg_5", "short_balance_chg_20",
                "margin_short_ratio",
            ]:
                group[col] = np.nan

        # ── 基本面（月營收）──
        if "fund_revenue_mom" in group.columns:
            group["fund_revenue_mom"] = pd.to_numeric(group["fund_revenue_mom"], errors="coerce")
            group["fund_revenue_yoy"] = pd.to_numeric(group["fund_revenue_yoy"], errors="coerce")
            group["fund_revenue_trend_3m"] = group["fund_revenue_yoy"].rolling(60, min_periods=20).mean()
        else:
            group["fund_revenue_mom"] = np.nan
            group["fund_revenue_yoy"] = np.nan
            group["fund_revenue_trend_3m"] = np.nan

        # ── 題材/金流（產業）──
        if "theme_turnover_ratio" in group.columns:
            group["theme_turnover_ratio"] = pd.to_numeric(group["theme_turnover_ratio"], errors="coerce")
            group["theme_return_20"] = pd.to_numeric(group["theme_return_20"], errors="coerce")
            group["theme_hot_score"] = pd.to_numeric(group["theme_hot_score"], errors="coerce")
        else:
            group["theme_turnover_ratio"] = np.nan
            group["theme_return_20"] = np.nan
            group["theme_hot_score"] = np.nan

        return group

    featured = df.groupby("stock_id", group_keys=False).apply(apply_group)

    # ── 大盤相對強弱（跨股票計算）──
    if "ret_20" in featured.columns and not featured["ret_20"].isna().all():
        market_avg = featured.groupby("trading_date")["ret_20"].transform("mean")
        featured["market_rel_ret_20"] = featured["ret_20"] - market_avg
    else:
        featured["market_rel_ret_20"] = np.nan

    return featured


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "build_features")
    logs: Dict[str, object] = {}
    try:
        max_price_date = db_session.query(func.max(RawPrice.trading_date)).scalar()
        if max_price_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        max_feature_date = db_session.query(func.max(Feature.trading_date)).scalar()
        if max_feature_date is None:
            target_start = db_session.query(func.min(RawPrice.trading_date)).scalar()
        else:
            target_start = max_feature_date + timedelta(days=1)

        if target_start is None or target_start > max_price_date:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        # 往前多拉 120 天以確保 60 日指標可計算
        calc_start = target_start - timedelta(days=120)
        merged = _fetch_data(db_session, calc_start, max_price_date)
        if merged.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        featured = _compute_features(merged)
        featured = featured[featured["trading_date"] >= target_start]

        # 核心特徵必須存在；擴充特徵允許 NaN，用 0 填補
        featured = featured.dropna(subset=CORE_FEATURE_COLUMNS)
        featured = featured.replace([np.inf, -np.inf], np.nan)
        featured = featured.dropna(subset=CORE_FEATURE_COLUMNS)
        for col in EXTENDED_FEATURE_COLUMNS:
            if col in featured.columns:
                featured[col] = featured[col].fillna(0)

        if featured.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        records: List[Dict] = []
        for _, row in featured.iterrows():
            features = {col: float(row[col]) for col in FEATURE_COLUMNS if col in row.index}
            records.append(
                {
                    "stock_id": row["stock_id"],
                    "trading_date": row["trading_date"],
                    "features_json": features,
                }
            )

        # 分批寫入：26 個特徵的 JSON 比 16 個大很多，縮小批次避免超過 MySQL max_allowed_packet
        BATCH_SIZE = 1000
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            stmt = insert(Feature).values(batch)
            stmt = stmt.on_duplicate_key_update(features_json=stmt.inserted.features_json)
            db_session.execute(stmt)
            db_session.commit()

        logs = {
            "rows": len(records),
            "feature_count": len(FEATURE_COLUMNS),
            "start_date": target_start.isoformat(),
            "end_date": max_price_date.isoformat(),
        }
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
