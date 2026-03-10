"""特徵工程模組：從 raw_prices、raw_institutional、raw_margin_short、raw_fundamental、
raw_theme_flow 等原始資料計算技術/籌碼/基本面特徵，寫入 features 表。

採增量建置模式，每次只補算尚未存在的日期。支援 schema 自動偵測補算與 force_recompute_days 手動補算。
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAS_PSUTIL = False

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _tqdm = None  # type: ignore
    _HAS_TQDM = False

from app.job_utils import finish_job, start_job
from app.models import (
    Feature,
    PriceAdjustFactor,
    RawFundamental,
    RawInstitutional,
    RawMarginShort,
    RawPrice,
    RawThemeFlow,
    Stock,
)

logger = logging.getLogger(__name__)

# ── 核心特徵（必須存在才保留該筆資料）──────────────────────────────
CORE_FEATURE_COLUMNS = [
    # 流動性
    "amt_20",
    # 波動率
    "vol_20",
    "vol_ratio_20",
    # 反向特徵（IC 為負但穩定，取負號讓模型方向一致）
    "vol_20_inv",             # -vol_20：高波動預期負報酬（IC=-0.056，ICIR=-0.34）
    "atr_inv",                # -atr_14_pct：高 ATR% 預期負報酬（IC=-0.058，ICIR=-0.34）
    "trend_persistence_inv",  # -trend_persistence：整份報告最強信號（ICIR=-1.07）
    "trust_net_5_inv",        # -trust_net_5：投信反向有效（ICIR=-0.87）
    # 法人
    "foreign_net_5", "foreign_net_20",
    "trust_net_5", "trust_net_20",
]

# ── 擴充特徵（允許 NaN，用 0 填補）──────────────────────────────
EXTENDED_FEATURE_COLUMNS = [
    # 動能（由 CORE 降級：IC 有效但 ICIR < 0.5）
    "ret_5", "ret_10", "ret_20", "ret_60",
    # 均線（由 CORE 降級）
    "ma_5", "ma_20", "ma_60",
    # 技術指標基礎（由 CORE 降級）
    "bias_20",
    # 經典技術指標
    "rsi_14",
    "macd_hist",
    "kd_k", "kd_d",
    # 籌碼面（融資融券）
    "short_balance_chg_5", "short_balance_chg_20",
    "margin_short_ratio",
    # 技術面（擴充）
    "drawdown_60",
    "amt",
    "amt_ratio_20",
    # 布林帶位置百分位（0=下軌, 1=上軌）
    "boll_pct",
    # 價量背離信號（+1=正背離/量縮價低, -1=負背離/量縮價高, 0=中性）
    "price_volume_divergence",
    # 近 60 日報酬分布特徵
    "ret_60_skew",
    "ret_60_kurt",
    # 籌碼面（擴充）
    "foreign_buy_streak_5",
    "foreign_buy_consecutive_days",
    "chip_flow_intensity_20",
    "foreign_buy_ratio_5",
    "foreign_buy_ratio_20",
    # 趨勢與型態
    "ma_alignment",
    "trend_persistence",
    # 基本面（月營收）
    "fund_revenue_mom",
    "fund_revenue_yoy",
    "fund_revenue_yoy_accel",
    # 題材/金流（產業聚合）
    "theme_turnover_ratio",
    "theme_return_20",
    # 新增技術指標（pandas-ta 對應，手動計算確保無洩漏）
    "willr_14",             # Williams %R 14 日（-100~0，越低越超賣）
    "cci_20",               # CCI 20 日（偏離正常范圍信號）
    "cmf_20",               # Chaikin Money Flow 20 日（資金流入/流出強度）
    # 市場環境特徵（全市場層面，ProcessPoolExecutor 外計算後 merge，2026-03-08 新增）
    "market_trend_20",      # 市場等權指數近 20 日報酬（衡量市場短期動能）
    "market_trend_60",      # 市場等權指數近 60 日報酬（衡量市場中期動能）
    "market_above_200ma",   # 市場等權指數是否在 200 日均線以上（0/1 市場多空環境）
    "market_volatility_20", # 市場等權日報酬近 20 日波動率（衡量市場恐慌程度）
    "sector_momentum",      # 個股所屬產業近 20 日報酬 - 市場近 20 日報酬（產業相對動能）
]

# 完整特徵列表（供 daily_pick / train_ranker 使用）
FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + EXTENDED_FEATURE_COLUMNS

# ── 10y 逐步優化實驗用特徵子集 ─────────────────────────────────
# 原始基準（≈ commit aa978b8 可用欄位，不含 IC 優化新增特徵）
# 包含：動能/均線/技術/籌碼/法人/基本面（fund_revenue_yoy）
# 排除：trust_net_5_inv（Change A 新增）、theme_turnover_ratio（Change A）、fund_revenue_mom（Change A）
# 排除：市場環境特徵、IC 衍生特徵（vol_20_inv, atr_inv, trend_persistence_inv）、新特徵
BASELINE_FEATURE_COLS: List[str] = [
    # 動能
    "ret_5", "ret_10", "ret_20", "ret_60",
    # 均線
    "ma_5", "ma_20", "ma_60", "bias_20",
    # 波動/流動性
    "vol_20", "vol_ratio_20", "amt_20", "amt", "amt_ratio_20",
    # 法人
    "foreign_net_5", "foreign_net_20",
    "trust_net_5", "trust_net_20",
    # 技術指標
    "rsi_14", "macd_hist", "kd_k", "kd_d",
    # 籌碼
    "short_balance_chg_5", "short_balance_chg_20",
    "margin_short_ratio",
    "foreign_buy_streak_5", "chip_flow_intensity_20",
    "foreign_buy_ratio_5", "foreign_buy_ratio_20",
    # 技術面
    "drawdown_60",
    "ma_alignment", "trend_persistence",
    # 基本面（YoY，無公告延遲問題）
    "fund_revenue_yoy",
]

# Change A：在 BASELINE 上加入 IC 分析中最有效的 3 個特徵
# theme_turnover_ratio（ICIR 強正）、fund_revenue_mom（MoM 訊號）、trust_net_5_inv（ICIR -0.87 → 取反）
CHANGE_A_FEATURE_COLS: List[str] = BASELINE_FEATURE_COLS + [
    "theme_turnover_ratio",
    "fund_revenue_mom",
    "trust_net_5_inv",
]

# ProcessPoolExecutor 每個 task 包含的股票數（降低序列化次數）
_CHUNK_SIZE = 50


# ── 系統資源監控 ────────────────────────────────────────────────

def log_system_resources(stage: str) -> None:
    """輸出 CPU 與記憶體使用狀況（需 psutil）"""
    if not _HAS_PSUTIL:
        return
    cpu = _psutil.cpu_percent(interval=1)
    mem = _psutil.virtual_memory()
    logger.info(
        f"[RESOURCE] {stage} | "
        f"CPU: {cpu}% | "
        f"RAM: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB"
    )


# ── Module-level 單股特徵計算（供 ProcessPoolExecutor pickle）────

def _apply_group_impl(
    group: pd.DataFrame,
    use_adjusted_price: bool = True,
) -> tuple:
    """單股特徵計算，回傳 (result_df, timing_dict)。

    timing_dict 的 key 對應 FEATURE_COLUMNS 中的特徵名稱（或特徵群），
    value 為該特徵的計算時間（秒）。
    """
    _t = time.perf_counter

    timing: Dict[str, float] = {}

    group = group.sort_values("trading_date").copy()
    close = group["adj_close"] if use_adjusted_price and "adj_close" in group.columns else group["close"]
    raw_close = group["close"]
    volume = group["volume"]
    high = group["high"]
    low = group["low"]

    # ── 動能 ──
    t0 = _t()
    group["ret_5"] = close.pct_change(5)
    timing["ret_5"] = _t() - t0
    t0 = _t()
    group["ret_10"] = close.pct_change(10)
    timing["ret_10"] = _t() - t0
    t0 = _t()
    group["ret_20"] = close.pct_change(20)
    timing["ret_20"] = _t() - t0
    t0 = _t()
    group["ret_60"] = close.pct_change(60)
    timing["ret_60"] = _t() - t0

    # ── 均線 ──
    t0 = _t()
    group["ma_5"] = close.rolling(5).mean()
    timing["ma_5"] = _t() - t0
    t0 = _t()
    group["ma_20"] = close.rolling(20).mean()
    timing["ma_20"] = _t() - t0
    t0 = _t()
    group["ma_60"] = close.rolling(60).mean()
    timing["ma_60"] = _t() - t0

    # ── 技術指標（基礎）──
    t0 = _t()
    group["bias_20"] = close / group["ma_20"] - 1
    timing["bias_20"] = _t() - t0

    t0 = _t()
    daily_ret = close.pct_change(1)
    group["vol_20"] = daily_ret.rolling(20).std()
    timing["vol_20"] = _t() - t0

    t0 = _t()
    group["vol_ratio_20"] = volume / volume.rolling(20).mean()
    timing["vol_ratio_20"] = _t() - t0

    t0 = _t()
    rolling_max20 = close.rolling(20).max()
    rolling_max60 = close.rolling(60).max()
    group["breakout_20"] = close / rolling_max20 - 1
    group["drawdown_60"] = close / rolling_max60 - 1
    group["amt"] = raw_close * volume
    timing["drawdown_60"] = _t() - t0

    # ── ATR 14 ──
    t0 = _t()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    group["atr_14_pct"] = atr_14 / close.replace(0, np.nan)
    group["amt_20"] = group["amt"].rolling(20).mean()
    group["amt_ratio_20"] = group["amt"] / group["amt_20"].replace(0, np.nan)
    timing["atr_inv"] = _t() - t0   # atr_14_pct → atr_inv

    # ── 法人 ──
    t0 = _t()
    group["foreign_net_5"] = group["foreign_net"].rolling(5).sum()
    group["foreign_net_20"] = group["foreign_net"].rolling(20).sum()
    group["trust_net_5"] = group["trust_net"].rolling(5).sum()
    group["trust_net_20"] = group["trust_net"].rolling(20).sum()
    group["dealer_net_5"] = group["dealer_net"].rolling(5).sum()
    group["dealer_net_20"] = group["dealer_net"].rolling(20).sum()
    group["foreign_buy_streak_5"] = (
        (group["foreign_net"] > 0).astype(int).rolling(5).sum()
    )
    timing["foreign_net_5"] = _t() - t0
    timing["foreign_net_20"] = timing["foreign_net_5"]
    timing["trust_net_5"] = timing["foreign_net_5"]
    timing["trust_net_20"] = timing["foreign_net_5"]
    timing["foreign_buy_streak_5"] = timing["foreign_net_5"]

    # 外資連續淨買超天數
    t0 = _t()
    is_buy = (group["foreign_net"] > 0).astype(int)
    run_id = (is_buy != is_buy.shift()).cumsum()
    group["foreign_buy_consecutive_days"] = is_buy * (
        is_buy.groupby(run_id).cumcount() + 1
    )
    timing["foreign_buy_consecutive_days"] = _t() - t0

    t0 = _t()
    group["chip_flow_intensity_20"] = (
        (group["foreign_net"] + group["trust_net"] + group["dealer_net"]).rolling(20).sum()
        / volume.rolling(20).sum().replace(0, np.nan)
    )
    timing["chip_flow_intensity_20"] = _t() - t0

    t0 = _t()
    group["foreign_buy_ratio_5"] = (
        group["foreign_net"].rolling(5).sum() / volume.rolling(5).sum().replace(0, np.nan)
    ).clip(-1, 1)
    group["foreign_buy_ratio_20"] = (
        group["foreign_net"].rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    ).clip(-1, 1)
    timing["foreign_buy_ratio_5"] = _t() - t0
    timing["foreign_buy_ratio_20"] = timing["foreign_buy_ratio_5"]

    # ── 趨勢與型態 ──
    t0 = _t()
    group["ma_alignment"] = (
        (close > group["ma_5"]) & (group["ma_5"] > group["ma_20"]) & (group["ma_20"] > group["ma_60"])
    ).astype(int)
    timing["ma_alignment"] = _t() - t0

    t0 = _t()
    group["trend_persistence"] = (close > group["open"]).astype(int).rolling(20).mean()
    timing["trend_persistence"] = _t() - t0

    # ── 布林帶位置百分位（0=下軌，1=上軌）──
    t0 = _t()
    boll_mid = close.rolling(20).mean()
    boll_std = close.rolling(20).std()
    boll_upper = boll_mid + 2 * boll_std
    boll_lower = boll_mid - 2 * boll_std
    boll_range = (boll_upper - boll_lower).replace(0, np.nan)
    group["boll_pct"] = ((close - boll_lower) / boll_range).clip(0, 1)
    timing["boll_pct"] = _t() - t0

    # ── 近 60 日報酬偏態與峰態 ──
    t0 = _t()
    group["ret_60_skew"] = daily_ret.rolling(60, min_periods=30).skew()
    timing["ret_60_skew"] = _t() - t0
    t0 = _t()
    group["ret_60_kurt"] = daily_ret.rolling(60, min_periods=30).kurt()
    timing["ret_60_kurt"] = _t() - t0

    # ── 價量背離信號 ──
    t0 = _t()
    vol_ma_10 = volume.rolling(10).mean()
    price_near_high = close >= close.rolling(10).max() * 0.98
    price_near_low = close <= close.rolling(10).min() * 1.02
    vol_shrink = volume < vol_ma_10 * 0.8
    group["price_volume_divergence"] = (
        price_near_low & vol_shrink
    ).astype(int) - (price_near_high & vol_shrink).astype(int)
    timing["price_volume_divergence"] = _t() - t0

    # ── RSI 14 ──
    t0 = _t()
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
    group["rsi_14"] = (100 - (100 / (1 + rs))).clip(0, 100)
    timing["rsi_14"] = _t() - t0

    # ── MACD Histogram (12, 26, 9) ──
    t0 = _t()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    group["macd_hist"] = macd_line - signal_line
    timing["macd_hist"] = _t() - t0

    # ── KD 隨機指標 (9, 3, 3) ──
    t0 = _t()
    low_min = low.rolling(9).min()
    high_max = high.rolling(9).max()
    denom = high_max - low_min
    rsv = ((close - low_min) / denom.replace(0, np.nan)) * 100
    group["kd_k"] = rsv.ewm(com=2, adjust=False).mean()
    group["kd_d"] = group["kd_k"].ewm(com=2, adjust=False).mean()
    timing["kd_k"] = _t() - t0
    timing["kd_d"] = timing["kd_k"]

    # ── 融資融券特徵 ──
    t0 = _t()
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
    timing["short_balance_chg_5"] = _t() - t0
    timing["short_balance_chg_20"] = timing["short_balance_chg_5"]
    timing["margin_short_ratio"] = timing["short_balance_chg_5"]

    # ── 基本面（月營收）──
    t0 = _t()
    if "fund_revenue_mom" in group.columns:
        group["fund_revenue_mom"] = pd.to_numeric(group["fund_revenue_mom"], errors="coerce")
        group["fund_revenue_yoy"] = pd.to_numeric(group["fund_revenue_yoy"], errors="coerce")
        if "fund_revenue_yoy_accel" not in group.columns:
            group["fund_revenue_yoy_accel"] = np.nan
        else:
            group["fund_revenue_yoy_accel"] = pd.to_numeric(group["fund_revenue_yoy_accel"], errors="coerce")
    else:
        group["fund_revenue_mom"] = np.nan
        group["fund_revenue_yoy"] = np.nan
        group["fund_revenue_yoy_accel"] = np.nan
    timing["fund_revenue_mom"] = _t() - t0
    timing["fund_revenue_yoy"] = timing["fund_revenue_mom"]
    timing["fund_revenue_yoy_accel"] = timing["fund_revenue_mom"]

    # ── 題材/金流（產業）──
    t0 = _t()
    if "theme_turnover_ratio" in group.columns:
        group["theme_turnover_ratio"] = pd.to_numeric(group["theme_turnover_ratio"], errors="coerce")
        group["theme_return_20"] = pd.to_numeric(group["theme_return_20"], errors="coerce")
    else:
        group["theme_turnover_ratio"] = np.nan
        group["theme_return_20"] = np.nan
    timing["theme_turnover_ratio"] = _t() - t0
    timing["theme_return_20"] = timing["theme_turnover_ratio"]

    # ── 反向特徵（負 IC 穩定，取負號讓模型方向一致）──
    t0 = _t()
    group["vol_20_inv"] = -group["vol_20"]
    group["atr_inv"] = -group["atr_14_pct"]
    group["trend_persistence_inv"] = -group["trend_persistence"]
    group["trust_net_5_inv"] = -group["trust_net_5"]
    timing["vol_20_inv"] = _t() - t0
    timing["trend_persistence_inv"] = timing["vol_20_inv"]
    timing["trust_net_5_inv"] = timing["vol_20_inv"]

    # ── Williams %R 14 日 ──
    t0 = _t()
    high_14 = high.rolling(14).max()
    low_14 = low.rolling(14).min()
    hl_range = (high_14 - low_14).replace(0, np.nan)
    group["willr_14"] = (high_14 - close) / hl_range * -100
    timing["willr_14"] = _t() - t0

    # ── CCI 20 日（Commodity Channel Index）──
    t0 = _t()
    tp = (high + low + close) / 3.0
    tp_ma = tp.rolling(20).mean()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN")
        tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    group["cci_20"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    timing["cci_20"] = _t() - t0

    # ── CMF 20 日（Chaikin Money Flow）──
    t0 = _t()
    hl_diff = (high - low).replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / hl_diff
    mf_volume = mf_multiplier * volume
    vol_sum_20 = volume.rolling(20).sum().replace(0, np.nan)
    group["cmf_20"] = mf_volume.rolling(20).sum() / vol_sum_20
    timing["cmf_20"] = _t() - t0

    # amt_ratio_20 timing（歸屬到 ATR block）
    timing["amt_ratio_20"] = timing["atr_inv"]
    timing["amt"] = timing["atr_inv"]

    return group, timing


def _compute_chunk(args: tuple) -> tuple:
    """ProcessPoolExecutor worker：處理一批股票（CHUNK_SIZE 檔）。

    Returns:
        (concatenated_df, aggregated_timing_dict)
    """
    chunk_df, use_adjusted_price = args
    results: List[pd.DataFrame] = []
    agg_timing: Dict[str, float] = {}

    for _, grp in chunk_df.groupby("stock_id", sort=False):
        result_df, timing = _apply_group_impl(grp, use_adjusted_price)
        results.append(result_df)
        for feat, t in timing.items():
            agg_timing[feat] = agg_timing.get(feat, 0.0) + t

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), agg_timing


def _fetch_data(session: Session, start_date: date, end_date: date) -> pd.DataFrame:
    """讀取 raw_prices + raw_institutional + raw_margin_short 並合併"""
    # ── 價格 ──
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    n_stocks = price_df["stock_id"].nunique() if not price_df.empty else 0
    n_days = price_df["trading_date"].nunique() if not price_df.empty else 0
    logger.info(f"[PERF] fetch_prices: {elapsed:.2f}s（{n_stocks}檔 × {n_days}天，{len(price_df):,}列）")

    if price_df.empty:
        return price_df

    price_df["stock_id"] = price_df["stock_id"].astype(str)
    price_df["trading_date"] = pd.to_datetime(price_df["trading_date"], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    price_df["volume"] = pd.to_numeric(price_df["volume"], errors="coerce")

    # ── 法人 ──
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_institutional: {elapsed:.2f}s（{len(inst_df):,}列）")

    if inst_df.empty:
        price_df["foreign_net"] = 0
        price_df["trust_net"] = 0
        price_df["dealer_net"] = 0
    else:
        inst_df["stock_id"] = inst_df["stock_id"].astype(str)
        inst_df["trading_date"] = pd.to_datetime(inst_df["trading_date"], errors="coerce")
        for col in ["foreign_net", "trust_net", "dealer_net"]:
            inst_df[col] = pd.to_numeric(inst_df[col], errors="coerce").fillna(0)
        price_df = price_df.merge(inst_df, on=["stock_id", "trading_date"], how="left")
        for col in ["foreign_net", "trust_net", "dealer_net"]:
            price_df[col] = price_df[col].fillna(0)

    # ── 融資融券 ──
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_margin_short: {elapsed:.2f}s（{len(margin_df):,}列）")

    if margin_df.empty:
        price_df["margin_purchase_balance"] = np.nan
        price_df["short_sale_balance"] = np.nan
    else:
        margin_df["stock_id"] = margin_df["stock_id"].astype(str)
        margin_df["trading_date"] = pd.to_datetime(margin_df["trading_date"], errors="coerce")
        for col in ["margin_purchase_balance", "short_sale_balance"]:
            margin_df[col] = pd.to_numeric(margin_df[col], errors="coerce")
        price_df = price_df.merge(margin_df, on=["stock_id", "trading_date"], how="left")

    # ── 還原因子（公司行為）──
    t0 = time.perf_counter()
    try:
        factor_stmt = (
            select(
                PriceAdjustFactor.stock_id,
                PriceAdjustFactor.trading_date,
                PriceAdjustFactor.adj_factor,
            )
            .where(PriceAdjustFactor.trading_date.between(start_date, end_date))
            .order_by(PriceAdjustFactor.stock_id, PriceAdjustFactor.trading_date)
        )
        factor_df = pd.read_sql(factor_stmt, session.get_bind())
    except Exception:
        factor_df = pd.DataFrame()
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_adj_factor: {elapsed:.2f}s（{len(factor_df):,}列）")

    if factor_df.empty:
        price_df["adj_factor"] = 1.0
        price_df["factor_missing"] = 1
    else:
        factor_df["stock_id"] = factor_df["stock_id"].astype(str)
        factor_df["trading_date"] = pd.to_datetime(factor_df["trading_date"], errors="coerce")
        factor_df["adj_factor"] = pd.to_numeric(factor_df["adj_factor"], errors="coerce")
        price_df = price_df.merge(factor_df, on=["stock_id", "trading_date"], how="left")
        price_df["factor_missing"] = price_df["adj_factor"].isna().astype(int)
        price_df["adj_factor"] = price_df["adj_factor"].fillna(1.0)

    price_df["adj_close"] = pd.to_numeric(price_df["close"], errors="coerce") * price_df["adj_factor"]

    # ── 股票主檔（產業） ──
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_stock_master: {elapsed:.2f}s（{len(stock_df):,}檔）")

    if not stock_df.empty:
        stock_df["stock_id"] = stock_df["stock_id"].astype(str)
        price_df = price_df.merge(stock_df[["stock_id", "industry_category"]], on="stock_id", how="left")
    else:
        price_df["industry_category"] = None

    # ── 基本面（月營收）──
    t0 = time.perf_counter()
    fund_stmt = (
        select(
            RawFundamental.stock_id,
            RawFundamental.trading_date,
            RawFundamental.revenue_current_month,
            RawFundamental.revenue_last_month,
            RawFundamental.revenue_last_year,
            RawFundamental.revenue_mom,
            RawFundamental.revenue_yoy,
        )
        .where(RawFundamental.trading_date.between(start_date - timedelta(days=370), end_date))
        .order_by(RawFundamental.stock_id, RawFundamental.trading_date)
    )
    fund_df = pd.read_sql(fund_stmt, session.get_bind())
    elapsed_fetch = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_fundamental: {elapsed_fetch:.2f}s（{len(fund_df):,}列）")

    t0 = time.perf_counter()
    if fund_df.empty:
        price_df["fund_revenue_mom"] = np.nan
        price_df["fund_revenue_yoy"] = np.nan
        price_df["fund_revenue_yoy_accel"] = np.nan
    else:
        fund_df["stock_id"] = fund_df["stock_id"].astype(str)
        fund_df["trading_date"] = pd.to_datetime(fund_df["trading_date"], errors="coerce")
        for col in ["revenue_current_month", "revenue_last_month", "revenue_last_year", "revenue_mom", "revenue_yoy"]:
            fund_df[col] = pd.to_numeric(fund_df[col], errors="coerce")
        fund_df = fund_df.sort_values(["stock_id", "trading_date"])
        fund_df["rev_prev_1m"] = fund_df.groupby("stock_id")["revenue_current_month"].shift(1)
        fund_df["rev_prev_12m"] = fund_df.groupby("stock_id")["revenue_current_month"].shift(12)
        prev_month = fund_df["revenue_last_month"].where(fund_df["revenue_last_month"].notna(), fund_df["rev_prev_1m"])
        prev_year = fund_df["revenue_last_year"].where(fund_df["revenue_last_year"].notna(), fund_df["rev_prev_12m"])
        mom_fallback = fund_df["revenue_current_month"] / prev_month.replace(0, np.nan) - 1.0
        yoy_fallback = fund_df["revenue_current_month"] / prev_year.replace(0, np.nan) - 1.0
        fund_df["revenue_mom"] = fund_df["revenue_mom"].fillna(mom_fallback)
        fund_df["revenue_yoy"] = fund_df["revenue_yoy"].fillna(yoy_fallback)
        fund_df = fund_df.rename(columns={"revenue_mom": "fund_revenue_mom", "revenue_yoy": "fund_revenue_yoy"})
        fund_df["fund_revenue_yoy_accel"] = fund_df.groupby("stock_id")["fund_revenue_yoy"].diff(1)
        # 月營收公告約在報告月結束後 45 天才公開，加入公告延遲避免前向洩漏
        fund_df["available_date"] = fund_df["trading_date"] + pd.Timedelta(days=45)
        fund_df = fund_df.sort_values(["stock_id", "available_date"])
        price_df = price_df.sort_values(["stock_id", "trading_date"])
        # 使用 per-stock groupby loop 避免 merge_asof(by=) 全域排序限制
        merged = []
        for sid, sub in price_df.groupby("stock_id", sort=False):
            sub_f = fund_df[fund_df["stock_id"] == sid]
            if sub_f.empty:
                sub = sub.copy()
                sub["fund_revenue_mom"] = np.nan
                sub["fund_revenue_yoy"] = np.nan
                sub["fund_revenue_yoy_accel"] = np.nan
                merged.append(sub)
                continue
            aligned = pd.merge_asof(
                sub.sort_values("trading_date"),
                sub_f.sort_values("available_date")[
                    ["available_date", "fund_revenue_mom", "fund_revenue_yoy", "fund_revenue_yoy_accel"]
                ],
                left_on="trading_date",
                right_on="available_date",
                direction="backward",
            )
            aligned = aligned.drop(columns=["available_date"], errors="ignore")
            merged.append(aligned)
        price_df = pd.concat(merged, ignore_index=True)
    elapsed_merge = time.perf_counter() - t0
    logger.info(f"[PERF] merge_fundamental: {elapsed_merge:.2f}s（per-stock merge_asof）")

    # ── 題材/金流（產業聚合）──
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_theme_flow: {elapsed:.2f}s（{len(theme_df):,}列）")

    if theme_df.empty:
        price_df["theme_turnover_ratio"] = np.nan
        price_df["theme_return_20"] = np.nan
        price_df["theme_hot_score"] = np.nan
    else:
        theme_df["trading_date"] = pd.to_datetime(theme_df["trading_date"], errors="coerce")
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


def _compute_market_context_features(
    df: pd.DataFrame,
    use_adjusted_price: bool = True,
) -> pd.DataFrame:
    """計算全市場環境特徵（等權市場指數，無洩漏：僅使用歷史滾動計算）。

    Args:
        df: merged price DataFrame，包含 stock_id, trading_date, adj_close/close
        use_adjusted_price: 是否使用還原收盤價

    Returns:
        DataFrame with columns: trading_date, market_trend_20, market_trend_60,
            market_above_200ma, market_volatility_20
    """
    if df.empty:
        return pd.DataFrame()

    price_col = "adj_close" if (use_adjusted_price and "adj_close" in df.columns) else "close"

    # 只使用四碼台股計算等權市場指數
    mkt = (
        df[df["stock_id"].str.fullmatch(r"\d{4}", na=False)]
        .groupby("trading_date")[price_col]
        .mean()
        .sort_index()
    )

    if len(mkt) < 2:
        return pd.DataFrame()

    mkt_ret = mkt.pct_change()  # 日報酬（無洩漏）

    result = pd.DataFrame({"trading_date": mkt.index})
    result["market_trend_20"] = mkt.pct_change(20).values
    result["market_trend_60"] = mkt.pct_change(60).values
    result["market_above_200ma"] = (
        mkt.gt(mkt.rolling(200, min_periods=40).mean())
    ).astype(float).values
    result["market_volatility_20"] = mkt_ret.rolling(20, min_periods=5).std().values

    return result


def _compute_sector_momentum(
    df: pd.DataFrame,
    mkt_ctx_df: pd.DataFrame,
    use_adjusted_price: bool = True,
) -> pd.DataFrame:
    """計算個股相對大盤的產業動能（sector_momentum）。

    sector_momentum = 產業等權近 20 日報酬 - 市場等權近 20 日報酬
    無資料洩漏：均使用歷史收盤價滾動計算。

    Args:
        df: merged price DataFrame，包含 stock_id, trading_date, industry_category, adj_close/close
        mkt_ctx_df: _compute_market_context_features 輸出，含 trading_date, market_trend_20
        use_adjusted_price: 是否使用還原收盤價

    Returns:
        DataFrame with columns: stock_id, trading_date, sector_momentum
    """
    if df.empty or mkt_ctx_df.empty or "industry_category" not in df.columns:
        return pd.DataFrame()

    price_col = "adj_close" if (use_adjusted_price and "adj_close" in df.columns) else "close"

    # 只使用四碼台股
    sub = df[df["stock_id"].str.fullmatch(r"\d{4}", na=False)].copy()
    sub = sub.dropna(subset=["industry_category", price_col])

    if sub.empty:
        return pd.DataFrame()

    # 產業等權平均收盤價
    sector_close = (
        sub.groupby(["industry_category", "trading_date"])[price_col]
        .mean()
        .reset_index()
        .sort_values(["industry_category", "trading_date"])
    )

    # 產業近 20 日報酬（無洩漏）
    sector_close["sector_trend_20"] = sector_close.groupby("industry_category")[price_col].transform(
        lambda x: x.pct_change(20)
    )

    # 合併市場 20 日報酬
    mkt_trend = mkt_ctx_df[["trading_date", "market_trend_20"]].copy()
    sector_close = sector_close.merge(mkt_trend, on="trading_date", how="left")
    sector_close["sector_momentum_val"] = sector_close["sector_trend_20"] - sector_close["market_trend_20"]

    # 每檔股票對應其產業 sector_momentum
    stock_sector = sub[["stock_id", "trading_date", "industry_category"]].drop_duplicates()
    result = stock_sector.merge(
        sector_close[["industry_category", "trading_date", "sector_momentum_val"]],
        on=["industry_category", "trading_date"],
        how="left",
    )
    result = result.rename(columns={"sector_momentum_val": "sector_momentum"})
    return result[["stock_id", "trading_date", "sector_momentum"]]


def _compute_features(
    df: pd.DataFrame,
    use_adjusted_price: bool = True,
    perf_out: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """計算所有特徵（逐股分組，使用 ProcessPoolExecutor 平行加速）

    Args:
        df: merged price+institutional DataFrame
        use_adjusted_price: 是否使用還原收盤價
        perf_out: 若不為 None，將 per-feature timing 資訊 append 進去
    """
    if df.empty:
        return df

    df = df.sort_values(["stock_id", "trading_date"]).copy()

    # ── 拆分成 chunks，每個 chunk 含 _CHUNK_SIZE 檔股票 ──
    stock_ids = df["stock_id"].unique().tolist()
    n_stocks = len(stock_ids)
    n_days = df["trading_date"].nunique()
    chunks: List[pd.DataFrame] = []
    for i in range(0, n_stocks, _CHUNK_SIZE):
        batch_ids = stock_ids[i: i + _CHUNK_SIZE]
        chunks.append(df[df["stock_id"].isin(batch_ids)])

    workers = int(os.environ.get("BUILD_FEATURES_WORKERS",
                                 max(1, min(os.cpu_count() or 1, 8) - 1)))
    chunk_args = [(chunk, use_adjusted_price) for chunk in chunks]

    agg_timing: Dict[str, float] = {}
    result_parts: List[pd.DataFrame] = []

    # ── 進度條 ──
    desc = f"建立特徵 ({n_stocks}檔 × {n_days}天)"
    pbar = _tqdm(total=len(chunks), desc=desc, unit="chunk") if _HAS_TQDM else None

    if workers > 1:
        logger.info(f"[PERF] ProcessPoolExecutor workers={workers}，chunks={len(chunks)}")
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_compute_chunk, arg): idx for idx, arg in enumerate(chunk_args)}
                for fut in as_completed(futures):
                    part_df, chunk_timing = fut.result()
                    result_parts.append(part_df)
                    for feat, t in chunk_timing.items():
                        agg_timing[feat] = agg_timing.get(feat, 0.0) + t
                    if pbar is not None:
                        pbar.update(1)
        except Exception as exc:  # pragma: no cover
            # ProcessPoolExecutor 失敗時 fallback 到序列計算
            logger.warning(f"[PERF] ProcessPoolExecutor 失敗（{exc}），fallback 序列計算")
            result_parts.clear()
            agg_timing.clear()
            for arg in chunk_args:
                part_df, chunk_timing = _compute_chunk(arg)
                result_parts.append(part_df)
                for feat, t in chunk_timing.items():
                    agg_timing[feat] = agg_timing.get(feat, 0.0) + t
                if pbar is not None:
                    pbar.update(1)
    else:
        # 單核序列計算
        for arg in chunk_args:
            part_df, chunk_timing = _compute_chunk(arg)
            result_parts.append(part_df)
            for feat, t in chunk_timing.items():
                agg_timing[feat] = agg_timing.get(feat, 0.0) + t
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    featured = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()

    # ── 市場環境特徵（全市場層面，ProcessPoolExecutor worker 無法計算，於此合併）──
    if not featured.empty:
        _t_mkt = time.perf_counter()
        mkt_ctx_df = _compute_market_context_features(df, use_adjusted_price=use_adjusted_price)
        if not mkt_ctx_df.empty:
            featured = featured.merge(mkt_ctx_df, on="trading_date", how="left")
            sec_mom_df = _compute_sector_momentum(df, mkt_ctx_df, use_adjusted_price=use_adjusted_price)
            if not sec_mom_df.empty:
                featured = featured.merge(sec_mom_df, on=["stock_id", "trading_date"], how="left")
            else:
                featured["sector_momentum"] = np.nan
        else:
            for _mkt_col in ["market_trend_20", "market_trend_60", "market_above_200ma",
                             "market_volatility_20", "sector_momentum"]:
                featured[_mkt_col] = np.nan
        logger.info(f"[PERF] market_context_features: {time.perf_counter() - _t_mkt:.2f}s")

    # ── 彙整 per-feature perf 資訊 ──
    if perf_out is not None:
        for feat in FEATURE_COLUMNS:
            perf_out.append({
                "feature": feat,
                "total_ms": round(agg_timing.get(feat, 0.0) * 1000, 3),
                "n_stocks": n_stocks,
                "n_days": n_days,
                "avg_ms_per_stock": round(agg_timing.get(feat, 0.0) * 1000 / max(n_stocks, 1), 4),
            })

    return featured


def _detect_schema_outdated(db_session: Session) -> bool:
    """檢查 DB 中最新一筆 features_json 的欄位數是否低於預期。
    若 < 80% 視為 schema 過時，需要補算。"""
    import json as _json
    row = db_session.query(Feature).order_by(Feature.trading_date.desc()).first()
    if row is None:
        return False
    existing = row.features_json if isinstance(row.features_json, dict) else _json.loads(row.features_json)
    return len(existing) < len(FEATURE_COLUMNS) * 0.8


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "build_features")
    logs: Dict[str, object] = {}
    try:
        log_system_resources("build_features start")

        max_price_date = db_session.query(func.max(RawPrice.trading_date)).scalar()
        if max_price_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        max_feature_date = db_session.query(func.max(Feature.trading_date)).scalar()

        # force_recompute_days：可由 config 指定，強制補算最近 N 天的特徵
        force_days = int(getattr(config, "force_recompute_days", 0))

        # schema 自動檢測：若現有特徵欄位數不足，自動補算 180 天
        schema_outdated = _detect_schema_outdated(db_session)
        if schema_outdated and force_days == 0:
            force_days = 180
            logs["schema_recompute_triggered"] = True

        if force_days > 0 and max_feature_date is not None:
            recompute_from = max_price_date - timedelta(days=force_days)
            db_session.query(Feature).filter(Feature.trading_date >= recompute_from).delete()
            db_session.commit()
            logs["force_recompute_from"] = recompute_from.isoformat()
            logs["force_recompute_days"] = force_days
            max_feature_date = db_session.query(func.max(Feature.trading_date)).scalar()

        if max_feature_date is None:
            target_start = db_session.query(func.min(RawPrice.trading_date)).scalar()
        else:
            target_start = max_feature_date + timedelta(days=1)

        if target_start is None or target_start > max_price_date:
            finish_job(db_session, job_id, "success", logs={"rows": 0, **logs})
            return {"rows": 0, **logs}

        # 往前多拉 250 天以確保 200 日均線可計算（新增 market_above_200ma 特徵）
        calc_start = target_start - timedelta(days=250)

        t_fetch = time.perf_counter()
        merged = _fetch_data(db_session, calc_start, max_price_date)
        elapsed_fetch = time.perf_counter() - t_fetch
        n_stocks_total = merged["stock_id"].nunique() if not merged.empty else 0
        n_days_total = merged["trading_date"].nunique() if not merged.empty else 0
        logger.info(
            f"[PERF] fetch_data total: {elapsed_fetch:.2f}s"
            f"（{n_stocks_total}檔 × {n_days_total}天，{len(merged):,}列）"
        )

        if merged.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0, **logs})
            return {"rows": 0, **logs}

        log_system_resources("before compute_features")

        use_adjusted_price = bool(getattr(config, "use_adjusted_price", True))

        perf_records: List[Dict] = []
        t_compute = time.perf_counter()
        featured = _compute_features(merged, use_adjusted_price=use_adjusted_price, perf_out=perf_records)
        elapsed_compute = time.perf_counter() - t_compute
        logger.info(
            f"[PERF] compute_features: {elapsed_compute:.2f}s"
            f"（{n_stocks_total}檔 × {n_days_total}天）"
        )

        log_system_resources("after compute_features")

        target_start_ts = pd.Timestamp(target_start)
        featured = featured[featured["trading_date"] >= target_start_ts]

        # 核心特徵必須存在；擴充特徵允許 NaN，用 0 填補
        featured = featured.dropna(subset=CORE_FEATURE_COLUMNS)
        featured = featured.replace([np.inf, -np.inf], np.nan)
        featured = featured.dropna(subset=CORE_FEATURE_COLUMNS)
        for col in EXTENDED_FEATURE_COLUMNS:
            if col in featured.columns:
                featured[col] = featured[col].fillna(0)

        if featured.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0, **logs})
            return {"rows": 0, **logs}

        # ── 寫入 DB（向量化建構，避免 iterrows 逐行 Python 迴圈）──
        t_save = time.perf_counter()
        feat_cols_in_df = [col for col in FEATURE_COLUMNS if col in featured.columns]
        # numpy 一次性轉換，比 iterrows 快 10x 以上
        _feat_arr = featured[feat_cols_in_df].to_numpy(dtype=float, na_value=0.0)
        _stock_ids = featured["stock_id"].tolist()
        _trading_dates = [pd.Timestamp(d).date() for d in featured["trading_date"].tolist()]
        records = [
            {
                "stock_id": sid,
                "trading_date": td,
                "features_json": dict(zip(feat_cols_in_df, row.tolist())),
            }
            for sid, td, row in zip(_stock_ids, _trading_dates, _feat_arr)
        ]

        BATCH_SIZE = 5000  # 從 1000 提升至 5000，減少 commit 次數 5x
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i: i + BATCH_SIZE]
            stmt = insert(Feature).values(batch)
            stmt = stmt.on_duplicate_key_update(features_json=stmt.inserted.features_json)
            db_session.execute(stmt)
            db_session.commit()

        elapsed_save = time.perf_counter() - t_save
        logger.info(
            f"[PERF] save_features: {elapsed_save:.2f}s"
            f"（{len(records):,}列，batch_size={BATCH_SIZE}）"
        )

        # ── 輸出 feature_perf.csv ──
        if perf_records:
            import pathlib
            artifacts_dir = pathlib.Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            perf_df = pd.DataFrame(perf_records).sort_values("total_ms", ascending=False)
            perf_path = artifacts_dir / "feature_perf.csv"
            perf_df.to_csv(perf_path, index=False)
            logger.info(f"[PERF] feature_perf.csv 已輸出: {perf_path}")

        logs = {
            **logs,
            "rows": len(records),
            "feature_count": len(FEATURE_COLUMNS),
            "start_date": target_start.isoformat(),
            "end_date": max_price_date.isoformat(),
            "use_adjusted_price": use_adjusted_price,
            "factor_missing_ratio": float(merged["factor_missing"].mean()) if "factor_missing" in merged.columns else 1.0,
            "price_adjustment_mode": "adjusted" if use_adjusted_price else "unadjusted",
            "elapsed_fetch_s": round(elapsed_fetch, 2),
            "elapsed_compute_s": round(elapsed_compute, 2),
            "elapsed_save_s": round(elapsed_save, 2),
        }
        if not use_adjusted_price:
            logs["warning"] = "Feature price series is unadjusted (USE_ADJUSTED_PRICE=false)."

        log_system_resources("build_features end")
        finish_job(db_session, job_id, "success", logs=logs)
        return logs

    except Exception as exc:  # pragma: no cover - exercised by pipeline
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
