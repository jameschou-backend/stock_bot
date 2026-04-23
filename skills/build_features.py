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
    RawBrokerTrade,
    RawFearGreed,
    RawFundamental,
    RawGovBank,
    RawHoldingDist,
    RawInstitutional,
    RawKBarDaily,
    RawMarginShort,
    RawPER,
    RawPrice,
    RawQuarterlyFundamental,
    RawSecuritiesLending,
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
    # 強勢訊號過濾特徵（外資異常買超 + 放量，2026-03-11 新增）
    "foreign_buy_streak",       # 外資連續「高於20日均買量」天數（嚴格版連買天數）
    "volume_surge_ratio",       # 近5日均量/近20日均量（週成交量放大比例，>2代表異常放量）
    "foreign_buy_intensity",    # 近5日外資淨買超 / 近20日均量（外資買進力道vs流動性）
    # 相對強弱排名（全市場截面，ProcessPoolExecutor 外計算，2026-03-30 新增）
    "rs_rank_20",               # ret_20 在全市場當日百分位排名（0=最弱, 1=最強）
    "rs_rank_60",               # ret_60 在全市場當日百分位排名（0=最弱, 1=最強）
    # ── Sponsor 資料集特徵（2026-04-22 新增，資料存在時有值，否則 NaN）──
    # 分點券商聚合（TaiwanStockTradingDailyReport）
    "broker_top5_conc",         # 當日前5分點淨買超集中度（0~1，越高代表籌碼越集中）
    "broker_net_5d",            # 近5日分點合計淨買超 / 近20日均量（外資以外籌碼動向）
    "broker_buy_days_5",        # 近5日分點淨買超天數（0~5，連買信號）
    # 持股分級（TaiwanStockHoldingSharesPer，週資料前向填充）
    "large_holder_pct",         # 大戶（>=1000張）持股比例（0~1）
    "large_holder_chg_4w",      # 大戶持股比例 4 週變動（籌碼集散速度）
    # 分鐘K線日內特徵（TaiwanStockKBar）
    "kbar_morning_ret",         # 開盤後30分鐘報酬（09:01-09:30，強弱信號）
    "kbar_intraday_pos",        # 收盤在當日振幅中的位置（0=最低, 1=最高）
    # 官股銀行（TaiwanstockGovernmentBankBuySell）
    "gov_bank_net_5d",          # 近5日官股銀行合計淨買超 / 近20日均量（政府資金方向）
    # CNN 恐懼貪婪指數（市場層面，ProcessPoolExecutor 外計算）
    "fear_greed_norm",          # CNN 恐懼貪婪分數 / 100（0=極度恐懼, 1=極度貪婪）
    "fear_greed_chg_5d",        # 恐懼貪婪指數近5日變動（情緒轉變速度）
    # ── 價值因子（TaiwanStockPER，2026-04-23 新增，資料存在時有值，否則 NaN）──
    "per_ratio",                # 本益比（越低越便宜；< 0 代表虧損）
    "pbr_ratio",                # 本淨比（< 1 = 市價低於帳面，潛在低估）
    "dividend_yield",           # 現金殖利率（%，台灣高殖利率股防禦性強）
    "earnings_yield",           # 盈利收益率 = 1/PER（比 PER 更線性，適合模型訓練）
    # ── 借券（TaiwanStockSecuritiesLending，2026-04-23 新增）──
    "lending_balance_ratio",    # 借券餘額 / 近20日均量（放空壓力相對成交量）
    "lending_fee_rate",         # 借券費率（%年率，高費率=稀缺放空標的）
    # ── 季報財務（TaiwanStockBalanceSheet 等，2026-04-23 新增，60天公告延遲）──
    "roe_ttm",                  # 股東權益報酬率（%，Fama-French 品質因子）
    "debt_ratio_q",             # 負債比率（%，越低財務越健全）
    "operating_margin_q",       # 營業利益率（%，業務獲利能力）
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

# SHAP 剪枝特徵集（2026-03-18，SHAP 分析後移除低重要性特徵）
# 從 FEATURE_COLUMNS 移除 8 個低重要性特徵（gain < 0.3% AND shap < 0.2%）：
# price_volume_divergence(shap=0.005%), vol_ratio_20(0.047%), boll_pct(0.076%),
# kd_k(0.084%), foreign_buy_consecutive_days(0.056%), willr_14(0.141%),
# amt_ratio_20(0.114%), foreign_buy_intensity(0.135%)
_PRUNE_SET = {
    "price_volume_divergence", "vol_ratio_20", "boll_pct",
    "kd_k", "foreign_buy_consecutive_days", "willr_14",
    "amt_ratio_20", "foreign_buy_intensity",
    # broker_trades / kbar_daily 尚無歷史資料（每日增量，資料累積中）
    # 待資料足夠（建議 60+ 交易日）後移回一般特徵池並重新 SHAP 剪枝
    "broker_top5_conc", "broker_net_5d", "broker_buy_days_5",
    "kbar_morning_ret", "kbar_intraday_pos",
    # 2026-04-23 新增：PER/借券/季報特徵 — 回補中，暫入 _PRUNE_SET
    # 待 10y 回補完成 + SHAP IC 分析後決定是否移回主特徵池
    "per_ratio", "pbr_ratio", "dividend_yield", "earnings_yield",
    "lending_balance_ratio", "lending_fee_rate",
    "roe_ttm", "debt_ratio_q", "operating_margin_q",
}

# IC 衰減剪枝集（2026-04-15，近 2 年 IC 分析，10y WF 驗證後放棄）
# 結論：IC 衰減的特徵（ma_5/20/60、foreign_buy_streak 等）在強勢多頭（2023 +141%→+57%）仍不可或缺
# 10y WF: 42feat ret=1784% Sharpe=1.07 vs 50feat ret=1863% Sharpe=0.98
# Sharpe 微升 +0.09 但 2023 績效大幅退化（-83pp），Calmar 也從 1.27→1.20 惡化 → 不採用
_IC_DECAY_PRUNE_SET: set = set()  # 保留所有 50 個 SHAP 剪枝後特徵

# Sponsor 資料集特徵集合（2026-04-22）
# 這些特徵當來源資料不存在時應保留 NaN（null），而非用 0 填補。
# 理由：LightGBM 可原生處理 NaN（透過缺失值分支路由），
# 但 0 會被誤判為有效信號（如 fear_greed_norm=0 等於「極度恐懼」，large_holder_pct=0 等於無大戶）。
_SPONSOR_FEATURES: set = {
    "broker_top5_conc", "broker_net_5d", "broker_buy_days_5",
    "large_holder_pct", "large_holder_chg_4w",
    "kbar_morning_ret", "kbar_intraday_pos",
    "gov_bank_net_5d",
    "fear_greed_norm", "fear_greed_chg_5d",
    # 2026-04-23 新增：PER/借券/季報特徵（NaN = 無資料，LightGBM 缺失分支路由）
    "per_ratio", "pbr_ratio", "dividend_yield", "earnings_yield",
    "lending_balance_ratio", "lending_fee_rate",
    "roe_ttm", "debt_ratio_q", "operating_margin_q",
}

# SHAP 剪枝（2026-03-18，58 → 50 特徵）
PRUNED_FEATURE_COLS: List[str] = [
    f for f in FEATURE_COLUMNS
    if f not in _PRUNE_SET and f not in _IC_DECAY_PRUNE_SET
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

    # ── 強勢訊號特徵（外資異常買超 + 放量，2026-03-11 新增）──
    # foreign_buy_streak: 外資連續「高於20日均買量」天數（比 foreign_buy_consecutive_days 更嚴格）
    # foreign_buy_consecutive_days 只要 > 0 即算，這裡要求超過 20 日均值才算「強勢」
    t0 = _t()
    _foreign_avg20 = group["foreign_net"].rolling(20, min_periods=5).mean().clip(lower=0)
    _is_strong_buy = (group["foreign_net"] > _foreign_avg20).astype(int)
    _run_id_s = (_is_strong_buy != _is_strong_buy.shift()).cumsum()
    group["foreign_buy_streak"] = _is_strong_buy * (
        _is_strong_buy.groupby(_run_id_s).cumcount() + 1
    )
    timing["foreign_buy_streak"] = _t() - t0

    # volume_surge_ratio: 近5日均量 / 近20日均量（捕捉週成交量異常放大，> 2 代表異常放量）
    t0 = _t()
    _vol_5avg = volume.rolling(5, min_periods=1).mean()
    _vol_20avg = volume.rolling(20, min_periods=5).mean()
    group["volume_surge_ratio"] = (_vol_5avg / _vol_20avg.replace(0, np.nan)).clip(0, 10)
    timing["volume_surge_ratio"] = _t() - t0

    # foreign_buy_intensity: 近5日外資買超張數 / 近20日平均成交量（外資買進力道 vs 流動性）
    # 與 foreign_buy_ratio_5 的區別：分母是 20 日均量（流動性基準），而非 5 日成交量
    t0 = _t()
    _foreign_5sum = group["foreign_net"].rolling(5, min_periods=1).sum()
    group["foreign_buy_intensity"] = (
        _foreign_5sum / volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
    ).clip(-1, 1)
    timing["foreign_buy_intensity"] = _t() - t0

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

    # ── Sponsor 特徵（資料存在時計算；若無則 NaN）──────────────────
    # 分點券商（broker_trades）
    t0 = _t()
    if "broker_top5_concentration" in group.columns:
        group["broker_top5_conc"] = pd.to_numeric(group["broker_top5_concentration"], errors="coerce")
    else:
        group["broker_top5_conc"] = np.nan
    # broker_net_5d / broker_buy_days_5：只在該股確實有分點資料時計算；
    # 若欄位存在但全為 NaN（源資料尚未回補），保留 NaN 而非填 0，
    # 讓 LightGBM 以缺失值分支路由處理，避免假零信號污染訓練資料。
    # 即使部分日期有資料，無資料日仍應保留 NaN（避免 fillna(0)+rolling 造成歷史假零）。
    if "broker_total_net" in group.columns and group["broker_total_net"].notna().any():
        _broker_net = pd.to_numeric(group["broker_total_net"], errors="coerce").fillna(0)
        _has_broker_data = group["broker_total_net"].notna()
        group["broker_net_5d"] = (
            _broker_net.rolling(5, min_periods=1).sum()
            / group["amt_20"].replace(0, np.nan)
        ).clip(-1, 1).where(_has_broker_data, other=np.nan)
        group["broker_buy_days_5"] = (
            (_broker_net > 0).astype(int).rolling(5, min_periods=1).sum()
        ).where(_has_broker_data, other=np.nan)
    else:
        group["broker_net_5d"] = np.nan
        group["broker_buy_days_5"] = np.nan
    timing["broker_top5_conc"] = _t() - t0
    timing["broker_net_5d"] = timing["broker_top5_conc"]
    timing["broker_buy_days_5"] = timing["broker_top5_conc"]

    # 持股分級（holding_dist，已 forward-fill 為週資料 → 每日）
    t0 = _t()
    if "large_holder_pct_raw" in group.columns:
        _lhp = pd.to_numeric(group["large_holder_pct_raw"], errors="coerce")
        group["large_holder_pct"] = _lhp
        # 4 週 ≈ 28 天；持股分級每週一筆，向前 4 筆即為 4 週前的值
        group["large_holder_chg_4w"] = _lhp - _lhp.shift(4)
    else:
        group["large_holder_pct"] = np.nan
        group["large_holder_chg_4w"] = np.nan
    timing["large_holder_pct"] = _t() - t0
    timing["large_holder_chg_4w"] = timing["large_holder_pct"]

    # 分鐘K線日內特徵（kbar_daily）
    t0 = _t()
    if "kbar_morning_ret_raw" in group.columns:
        group["kbar_morning_ret"] = pd.to_numeric(group["kbar_morning_ret_raw"], errors="coerce")
    else:
        group["kbar_morning_ret"] = np.nan
    if "kbar_intraday_pos_raw" in group.columns:
        group["kbar_intraday_pos"] = pd.to_numeric(group["kbar_intraday_pos_raw"], errors="coerce")
    else:
        group["kbar_intraday_pos"] = np.nan
    timing["kbar_morning_ret"] = _t() - t0
    timing["kbar_intraday_pos"] = timing["kbar_morning_ret"]

    # 官股銀行（gov_bank）：只在有實際資料時計算，全 NaN 時保留 NaN
    t0 = _t()
    if "gov_net_raw" in group.columns and group["gov_net_raw"].notna().any():
        _gov = pd.to_numeric(group["gov_net_raw"], errors="coerce").fillna(0)
        group["gov_bank_net_5d"] = (
            _gov.rolling(5, min_periods=1).sum()
            / group["amt_20"].replace(0, np.nan)
        ).clip(-1, 1)
    else:
        group["gov_bank_net_5d"] = np.nan
    timing["gov_bank_net_5d"] = _t() - t0

    # CNN 恐懼貪婪（market-level，同日所有股票值相同）
    t0 = _t()
    if "fear_greed_score_raw" in group.columns:
        _fg = pd.to_numeric(group["fear_greed_score_raw"], errors="coerce")
        group["fear_greed_norm"] = _fg / 100.0
        group["fear_greed_chg_5d"] = _fg.diff(5) / 100.0
    else:
        group["fear_greed_norm"] = np.nan
        group["fear_greed_chg_5d"] = np.nan
    timing["fear_greed_norm"] = _t() - t0
    timing["fear_greed_chg_5d"] = timing["fear_greed_norm"]

    # ── 價值因子（PER/PBR/殖利率，每日資料，Sponsor）──
    t0 = _t()
    if "per_raw" in group.columns:
        _per = pd.to_numeric(group["per_raw"], errors="coerce")
        _pbr = pd.to_numeric(group.get("pbr_raw", pd.Series(np.nan, index=group.index)), errors="coerce")
        _div = pd.to_numeric(group.get("dividend_yield_raw", pd.Series(np.nan, index=group.index)), errors="coerce")
        group["per_ratio"] = _per
        group["pbr_ratio"] = _pbr
        group["dividend_yield"] = _div
        # earnings_yield = 1/PER（負 PER 代表虧損，保留負值）
        group["earnings_yield"] = (1.0 / _per.replace(0, np.nan)).clip(-10, 50)
    else:
        group["per_ratio"] = np.nan
        group["pbr_ratio"] = np.nan
        group["dividend_yield"] = np.nan
        group["earnings_yield"] = np.nan
    timing["per_ratio"] = _t() - t0
    timing["pbr_ratio"] = timing["per_ratio"]
    timing["dividend_yield"] = timing["per_ratio"]
    timing["earnings_yield"] = timing["per_ratio"]

    # ── 借券特徵（TaiwanStockSecuritiesLending，Sponsor）──
    t0 = _t()
    if "lending_balance_raw" in group.columns and group["lending_balance_raw"].notna().any():
        _lb = pd.to_numeric(group["lending_balance_raw"], errors="coerce").fillna(0)
        _lfr = pd.to_numeric(group.get("lending_fee_rate_raw", pd.Series(np.nan, index=group.index)), errors="coerce")
        _has_lending = group["lending_balance_raw"].notna()
        group["lending_balance_ratio"] = (
            _lb / group["amt_20"].replace(0, np.nan)
        ).clip(0, 5).where(_has_lending, other=np.nan)
        group["lending_fee_rate"] = _lfr
    else:
        group["lending_balance_ratio"] = np.nan
        group["lending_fee_rate"] = np.nan
    timing["lending_balance_ratio"] = _t() - t0
    timing["lending_fee_rate"] = timing["lending_balance_ratio"]

    # ── 季報財務（ROE/debt_ratio/operating_margin，Sponsor，60天延遲）──
    t0 = _t()
    if "roe_raw" in group.columns:
        group["roe_ttm"] = pd.to_numeric(group["roe_raw"], errors="coerce")
    else:
        group["roe_ttm"] = np.nan
    if "debt_ratio_raw" in group.columns:
        group["debt_ratio_q"] = pd.to_numeric(group["debt_ratio_raw"], errors="coerce")
    else:
        group["debt_ratio_q"] = np.nan
    if "operating_margin_raw" in group.columns:
        group["operating_margin_q"] = pd.to_numeric(group["operating_margin_raw"], errors="coerce")
    else:
        group["operating_margin_q"] = np.nan
    timing["roe_ttm"] = _t() - t0
    timing["debt_ratio_q"] = timing["roe_ttm"]
    timing["operating_margin_q"] = timing["roe_ttm"]

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

    # ── 分點券商聚合（Sponsor）──
    t0 = time.perf_counter()
    try:
        broker_stmt = (
            select(
                RawBrokerTrade.stock_id,
                RawBrokerTrade.trading_date,
                RawBrokerTrade.top5_concentration,
                RawBrokerTrade.total_net,
            )
            .where(RawBrokerTrade.trading_date.between(start_date, end_date))
        )
        broker_df = pd.read_sql(broker_stmt, session.get_bind())
    except Exception:
        broker_df = pd.DataFrame()
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_broker_trades: {elapsed:.2f}s（{len(broker_df):,}列）")

    if broker_df.empty:
        price_df["broker_top5_concentration"] = np.nan
        price_df["broker_total_net"] = np.nan
    else:
        broker_df["stock_id"] = broker_df["stock_id"].astype(str)
        broker_df["trading_date"] = pd.to_datetime(broker_df["trading_date"], errors="coerce")
        for col in ["top5_concentration", "total_net"]:
            broker_df[col] = pd.to_numeric(broker_df[col], errors="coerce")
        broker_df = broker_df.rename(columns={
            "top5_concentration": "broker_top5_concentration",
            "total_net": "broker_total_net",
        })
        price_df = price_df.merge(
            broker_df[["stock_id", "trading_date", "broker_top5_concentration", "broker_total_net"]],
            on=["stock_id", "trading_date"],
            how="left",
        )

    # ── 持股分級週報（Sponsor，週資料→前向填充）──
    t0 = time.perf_counter()
    try:
        holding_stmt = (
            select(
                RawHoldingDist.stock_id,
                RawHoldingDist.trading_date,
                RawHoldingDist.large_holder_pct,
            )
            .where(RawHoldingDist.trading_date.between(start_date - timedelta(days=30), end_date))
            .order_by(RawHoldingDist.stock_id, RawHoldingDist.trading_date)
        )
        holding_df = pd.read_sql(holding_stmt, session.get_bind())
    except Exception:
        holding_df = pd.DataFrame()
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_holding_dist: {elapsed:.2f}s（{len(holding_df):,}列）")

    if holding_df.empty:
        price_df["large_holder_pct_raw"] = np.nan
    else:
        holding_df["stock_id"] = holding_df["stock_id"].astype(str)
        holding_df["trading_date"] = pd.to_datetime(holding_df["trading_date"], errors="coerce")
        holding_df["large_holder_pct"] = pd.to_numeric(holding_df["large_holder_pct"], errors="coerce")
        holding_df = holding_df.sort_values(["stock_id", "trading_date"])
        # 使用 merge_asof per-stock 前向填充週資料到每日
        price_df = price_df.sort_values(["stock_id", "trading_date"])
        merged_holding = []
        for sid, sub in price_df.groupby("stock_id", sort=False):
            sub_h = holding_df[holding_df["stock_id"] == sid]
            if sub_h.empty:
                sub = sub.copy()
                sub["large_holder_pct_raw"] = np.nan
                merged_holding.append(sub)
                continue
            aligned = pd.merge_asof(
                sub.sort_values("trading_date"),
                sub_h.sort_values("trading_date")[["trading_date", "large_holder_pct"]],
                on="trading_date",
                direction="backward",
            )
            aligned = aligned.rename(columns={"large_holder_pct": "large_holder_pct_raw"})
            merged_holding.append(aligned)
        price_df = pd.concat(merged_holding, ignore_index=True)

    # ── 分鐘K線日內特徵（Sponsor）──
    t0 = time.perf_counter()
    try:
        kbar_stmt = (
            select(
                RawKBarDaily.stock_id,
                RawKBarDaily.trading_date,
                RawKBarDaily.morning_ret,
                RawKBarDaily.intraday_high_pos,
            )
            .where(RawKBarDaily.trading_date.between(start_date, end_date))
        )
        kbar_df = pd.read_sql(kbar_stmt, session.get_bind())
    except Exception:
        kbar_df = pd.DataFrame()
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_kbar_daily: {elapsed:.2f}s（{len(kbar_df):,}列）")

    if kbar_df.empty:
        price_df["kbar_morning_ret_raw"] = np.nan
        price_df["kbar_intraday_pos_raw"] = np.nan
    else:
        kbar_df["stock_id"] = kbar_df["stock_id"].astype(str)
        kbar_df["trading_date"] = pd.to_datetime(kbar_df["trading_date"], errors="coerce")
        for col in ["morning_ret", "intraday_high_pos"]:
            kbar_df[col] = pd.to_numeric(kbar_df[col], errors="coerce")
        kbar_df = kbar_df.rename(columns={
            "morning_ret": "kbar_morning_ret_raw",
            "intraday_high_pos": "kbar_intraday_pos_raw",
        })
        price_df = price_df.merge(
            kbar_df[["stock_id", "trading_date", "kbar_morning_ret_raw", "kbar_intraday_pos_raw"]],
            on=["stock_id", "trading_date"],
            how="left",
        )

    # ── 官股銀行（Sponsor）──
    t0 = time.perf_counter()
    try:
        gov_stmt = (
            select(
                RawGovBank.stock_id,
                RawGovBank.trading_date,
                RawGovBank.gov_net,
            )
            .where(RawGovBank.trading_date.between(start_date, end_date))
        )
        gov_df = pd.read_sql(gov_stmt, session.get_bind())
    except Exception:
        gov_df = pd.DataFrame()
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_gov_bank: {elapsed:.2f}s（{len(gov_df):,}列）")

    if gov_df.empty:
        price_df["gov_net_raw"] = np.nan
    else:
        gov_df["stock_id"] = gov_df["stock_id"].astype(str)
        gov_df["trading_date"] = pd.to_datetime(gov_df["trading_date"], errors="coerce")
        gov_df["gov_net"] = pd.to_numeric(gov_df["gov_net"], errors="coerce").fillna(0)
        gov_df = gov_df.rename(columns={"gov_net": "gov_net_raw"})
        price_df = price_df.merge(
            gov_df[["stock_id", "trading_date", "gov_net_raw"]],
            on=["stock_id", "trading_date"],
            how="left",
        )

    # ── CNN 恐懼貪婪指數（Sponsor，市場層面 → 按 trading_date join）──
    t0 = time.perf_counter()
    try:
        fg_stmt = (
            select(
                RawFearGreed.date.label("trading_date"),
                RawFearGreed.score,
            )
            .where(RawFearGreed.date.between(start_date - timedelta(days=10), end_date))
            .order_by(RawFearGreed.date)
        )
        fg_df = pd.read_sql(fg_stmt, session.get_bind())
    except Exception:
        fg_df = pd.DataFrame()
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_fear_greed: {elapsed:.2f}s（{len(fg_df):,}列）")

    if fg_df.empty:
        price_df["fear_greed_score_raw"] = np.nan
    else:
        fg_df["trading_date"] = pd.to_datetime(fg_df["trading_date"], errors="coerce")
        fg_df["score"] = pd.to_numeric(fg_df["score"], errors="coerce")
        fg_df = fg_df.rename(columns={"score": "fear_greed_score_raw"})
        # 使用 merge_asof 填補缺日（週末/假日）
        price_df = price_df.sort_values("trading_date")
        price_df = pd.merge_asof(
            price_df,
            fg_df.sort_values("trading_date")[["trading_date", "fear_greed_score_raw"]],
            on="trading_date",
            direction="backward",
        )

    # ── 本益比/殖利率/本淨比（PER，Sponsor，每日）──
    t0 = time.perf_counter()
    try:
        per_stmt = (
            select(
                RawPER.stock_id,
                RawPER.trading_date,
                RawPER.per,
                RawPER.pbr,
                RawPER.dividend_yield,
            )
            .where(RawPER.trading_date.between(start_date, end_date))
        )
        per_df = pd.read_sql(per_stmt, session.get_bind())
    except Exception:
        per_df = pd.DataFrame()
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_per: {elapsed:.2f}s（{len(per_df):,}列）")

    if per_df.empty:
        price_df["per_raw"] = np.nan
        price_df["pbr_raw"] = np.nan
        price_df["dividend_yield_raw"] = np.nan
    else:
        per_df["stock_id"] = per_df["stock_id"].astype(str)
        per_df["trading_date"] = pd.to_datetime(per_df["trading_date"], errors="coerce")
        for col in ["per", "pbr", "dividend_yield"]:
            per_df[col] = pd.to_numeric(per_df[col], errors="coerce")
        per_df = per_df.rename(columns={
            "per": "per_raw",
            "pbr": "pbr_raw",
            "dividend_yield": "dividend_yield_raw",
        })
        price_df = price_df.merge(
            per_df[["stock_id", "trading_date", "per_raw", "pbr_raw", "dividend_yield_raw"]],
            on=["stock_id", "trading_date"],
            how="left",
        )

    # ── 借券餘額（SecuritiesLending，Sponsor，每日）──
    t0 = time.perf_counter()
    try:
        lending_stmt = (
            select(
                RawSecuritiesLending.stock_id,
                RawSecuritiesLending.trading_date,
                RawSecuritiesLending.lending_balance,
                RawSecuritiesLending.lending_fee_rate,
            )
            .where(RawSecuritiesLending.trading_date.between(start_date, end_date))
        )
        lending_df = pd.read_sql(lending_stmt, session.get_bind())
    except Exception:
        lending_df = pd.DataFrame()
    elapsed = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_securities_lending: {elapsed:.2f}s（{len(lending_df):,}列）")

    if lending_df.empty:
        price_df["lending_balance_raw"] = np.nan
        price_df["lending_fee_rate_raw"] = np.nan
    else:
        lending_df["stock_id"] = lending_df["stock_id"].astype(str)
        lending_df["trading_date"] = pd.to_datetime(lending_df["trading_date"], errors="coerce")
        for col in ["lending_balance", "lending_fee_rate"]:
            lending_df[col] = pd.to_numeric(lending_df[col], errors="coerce")
        lending_df = lending_df.rename(columns={
            "lending_balance": "lending_balance_raw",
            "lending_fee_rate": "lending_fee_rate_raw",
        })
        price_df = price_df.merge(
            lending_df[["stock_id", "trading_date", "lending_balance_raw", "lending_fee_rate_raw"]],
            on=["stock_id", "trading_date"],
            how="left",
        )

    # ── 季報財務（QuarterlyFundamental，Sponsor，60天公告延遲）──
    t0 = time.perf_counter()
    try:
        qfund_stmt = (
            select(
                RawQuarterlyFundamental.stock_id,
                RawQuarterlyFundamental.report_date,
                RawQuarterlyFundamental.roe,
                RawQuarterlyFundamental.debt_ratio,
                RawQuarterlyFundamental.operating_margin,
            )
            .where(RawQuarterlyFundamental.report_date.between(
                start_date - timedelta(days=450),   # 往前 15 個月確保有最新季報
                end_date,
            ))
            .order_by(RawQuarterlyFundamental.stock_id, RawQuarterlyFundamental.report_date)
        )
        qfund_df = pd.read_sql(qfund_stmt, session.get_bind())
    except Exception:
        qfund_df = pd.DataFrame()
    elapsed_fetch = time.perf_counter() - t0
    logger.info(f"[PERF] fetch_quarterly_fundamental: {elapsed_fetch:.2f}s（{len(qfund_df):,}列）")

    t0 = time.perf_counter()
    if qfund_df.empty:
        price_df["roe_raw"] = np.nan
        price_df["debt_ratio_raw"] = np.nan
        price_df["operating_margin_raw"] = np.nan
    else:
        qfund_df["stock_id"] = qfund_df["stock_id"].astype(str)
        qfund_df["report_date"] = pd.to_datetime(qfund_df["report_date"], errors="coerce")
        for col in ["roe", "debt_ratio", "operating_margin"]:
            qfund_df[col] = pd.to_numeric(qfund_df[col], errors="coerce")
        # 加入 60 天公告延遲（同月營收的 45 天機制）
        qfund_df["available_date"] = qfund_df["report_date"] + pd.Timedelta(days=60)
        qfund_df = qfund_df.sort_values(["stock_id", "available_date"])
        price_df = price_df.sort_values(["stock_id", "trading_date"])
        qmerged = []
        for sid, sub in price_df.groupby("stock_id", sort=False):
            sub_q = qfund_df[qfund_df["stock_id"] == sid]
            if sub_q.empty:
                sub = sub.copy()
                sub["roe_raw"] = np.nan
                sub["debt_ratio_raw"] = np.nan
                sub["operating_margin_raw"] = np.nan
                qmerged.append(sub)
                continue
            aligned = pd.merge_asof(
                sub.sort_values("trading_date"),
                sub_q.sort_values("available_date")[
                    ["available_date", "roe", "debt_ratio", "operating_margin"]
                ],
                left_on="trading_date",
                right_on="available_date",
                direction="backward",
            )
            aligned = aligned.drop(columns=["available_date"], errors="ignore")
            aligned = aligned.rename(columns={
                "roe": "roe_raw",
                "debt_ratio": "debt_ratio_raw",
                "operating_margin": "operating_margin_raw",
            })
            qmerged.append(aligned)
        price_df = pd.concat(qmerged, ignore_index=True)
    elapsed_merge = time.perf_counter() - t0
    logger.info(f"[PERF] merge_quarterly_fundamental: {elapsed_merge:.2f}s（per-stock merge_asof）")

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

    # ── 相對強弱排名（全市場截面，ProcessPoolExecutor 外計算）──────────────────
    if not featured.empty:
        _t_rs = time.perf_counter()
        # rs_rank_20 / rs_rank_60：個股報酬在同日全市場的百分位（0=最弱, 1=最強）
        for _rs_col, _src_col in [("rs_rank_20", "ret_20"), ("rs_rank_60", "ret_60")]:
            if _src_col in featured.columns:
                featured[_rs_col] = featured.groupby("trading_date")[_src_col].transform(
                    lambda x: x.rank(pct=True, na_option="keep")
                )
            else:
                featured[_rs_col] = np.nan
        logger.info(f"[PERF] rs_rank: {time.perf_counter() - _t_rs:.3f}s")


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
    """檢查最新特徵資料的欄位數是否低於預期。
    優先查 Parquet Feature Store；若 Parquet 無資料則 fallback 至 MySQL。
    若 < 95% 視為 schema 過時，需要補算。"""
    import json as _json

    # ── 優先查 Parquet Feature Store ──
    try:
        from skills.feature_store import FeatureStore
        _fs = FeatureStore()
        _max_date = _fs.get_max_date()
        if _max_date is not None:
            _sample = _fs.read(_max_date, _max_date)
            if not _sample.empty:
                _feat_cols = [
                    c for c in _sample.columns
                    if c not in ("stock_id", "trading_date")
                ]
                return len(_feat_cols) < len(FEATURE_COLUMNS) * 0.95
    except Exception:
        pass  # fallback to MySQL

    # ── Fallback：MySQL ──
    row = db_session.query(Feature).order_by(Feature.trading_date.desc()).first()
    if row is None:
        return False
    existing = (
        row.features_json
        if isinstance(row.features_json, dict)
        else _json.loads(row.features_json)
    )
    # 0.95 閾值：新增 3 個特徵時 53 < 56×0.95=53.2 → 可觸發自動補算
    return len(existing) < len(FEATURE_COLUMNS) * 0.95


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "build_features")
    logs: Dict[str, object] = {}
    try:
        log_system_resources("build_features start")

        max_price_date = db_session.query(func.max(RawPrice.trading_date)).scalar()
        if max_price_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        # ── 讀取最新特徵日期：優先 Parquet，fallback MySQL ──
        from skills.feature_store import FeatureStore as _FeatureStore
        _feature_store = _FeatureStore()
        _parquet_max = _feature_store.get_max_date()
        max_feature_date = (
            _parquet_max
            if _parquet_max is not None
            else db_session.query(func.max(Feature.trading_date)).scalar()
        )

        # force_recompute_days：可由 config 指定，強制補算最近 N 天的特徵
        force_days = int(getattr(config, "force_recompute_days", 0))

        # schema 自動檢測：若現有特徵欄位數不足，自動補算 180 天
        schema_outdated = _detect_schema_outdated(db_session)
        if schema_outdated and force_days == 0:
            force_days = 180
            logs["schema_recompute_triggered"] = True

        if force_days > 0 and max_feature_date is not None:
            recompute_from = max_price_date - timedelta(days=force_days)
            # 同步刪除 Parquet 和 MySQL 中 >= recompute_from 的資料
            _feature_store.delete_from(recompute_from)
            db_session.query(Feature).filter(Feature.trading_date >= recompute_from).delete()
            db_session.commit()
            logs["force_recompute_from"] = recompute_from.isoformat()
            logs["force_recompute_days"] = force_days
            # 重新查最新日期（Parquet 優先）
            _parquet_max = _feature_store.get_max_date()
            max_feature_date = (
                _parquet_max
                if _parquet_max is not None
                else db_session.query(func.max(Feature.trading_date)).scalar()
            )

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

        # 核心特徵必須存在；擴充特徵允許 NaN，用 0 填補（Sponsor 特徵除外）
        # Sponsor 特徵（broker / kbar / holding_dist / gov_bank / fear_greed）：
        #   來源資料缺失時應保留 NaN，讓 LightGBM 透過缺失值分支路由處理。
        #   fillna(0) 會產生假零信號（如 fear_greed_norm=0 等於「極度恐懼」），
        #   污染 4+ 年訓練資料（Sponsor API 資料只有近期有值）。
        featured = featured.dropna(subset=CORE_FEATURE_COLUMNS)
        featured = featured.replace([np.inf, -np.inf], np.nan)
        featured = featured.dropna(subset=CORE_FEATURE_COLUMNS)
        for col in EXTENDED_FEATURE_COLUMNS:
            if col in featured.columns and col not in _SPONSOR_FEATURES:
                featured[col] = featured[col].fillna(0)

        if featured.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0, **logs})
            return {"rows": 0, **logs}

        # ── 寫入 DB（向量化建構，避免 iterrows 逐行 Python 迴圈）──
        t_save = time.perf_counter()
        feat_cols_in_df = [col for col in FEATURE_COLUMNS if col in featured.columns]
        # numpy 一次性轉換，比 iterrows 快 10x 以上
        # na_value=float('nan') 保留 NaN（Sponsor 特徵來源缺失時），
        # 寫 JSON 時將 Sponsor NaN 轉為 None（MySQL JSON null），
        # 讓後續讀取端（json.loads）得到 None → 轉 np.nan → LightGBM 缺失值路由。
        _feat_arr = featured[feat_cols_in_df].to_numpy(dtype=float, na_value=float('nan'))
        _stock_ids = featured["stock_id"].tolist()
        _trading_dates = [pd.Timestamp(d).date() for d in featured["trading_date"].tolist()]
        # 找出 Sponsor 特徵在 feat_cols_in_df 中的索引，供快速 NaN→None 替換
        _sponsor_idx_set = {
            i for i, c in enumerate(feat_cols_in_df) if c in _SPONSOR_FEATURES
        }
        records = []
        for sid, td, row in zip(_stock_ids, _trading_dates, _feat_arr):
            flist = row.tolist()
            fdict = dict(zip(feat_cols_in_df, flist))
            # Sponsor NaN → None（JSON null）；非 Sponsor NaN → 0（已在 fillna 階段處理）
            for idx in _sponsor_idx_set:
                v = flist[idx]
                if isinstance(v, float) and (v != v):  # fast NaN check (NaN != NaN)
                    fdict[feat_cols_in_df[idx]] = None
            records.append({
                "stock_id": sid,
                "trading_date": td,
                "features_json": fdict,
            })

        # BATCH_SIZE 必須讓 INSERT 封包 < max_allowed_packet（預設 4MB）
        # 每筆 features_json ≈ 1,775 bytes（56特徵）；500×1775 = 0.88MB，安全邊際充足
        # 2026-03-18：改為 500 避免 Broken pipe（大 JSON 時 1500 筆可能觸發斷線）
        BATCH_SIZE = 500
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

        # ── 同步寫入 Parquet Feature Store（Dual-write，確保 Parquet 始終最新）──
        # Parquet 儲存預解析的數值欄位，供 train_ranker / daily_pick / data_store 直接讀取，
        # 無需 JSON 解析，DuckDB predicate pushdown 讓日期範圍查詢快 3-5×。
        try:
            from skills.feature_store import FeatureStore
            _fs = FeatureStore()
            _parquet_df = featured[["stock_id", "trading_date"] + feat_cols_in_df].copy()
            _parquet_df["trading_date"] = pd.to_datetime(
                _parquet_df["trading_date"]
            ).dt.date
            _t_parquet = time.perf_counter()
            _fs.write(_parquet_df)
            logger.info(
                f"[PERF] save_features_parquet: {time.perf_counter()-_t_parquet:.2f}s"
                f"（{len(_parquet_df):,}列）"
            )
            del _parquet_df
        except Exception as _exc:  # pragma: no cover
            # Parquet 寫入失敗不中斷主流程（MySQL 仍完整保留），僅記錄 warning
            logger.warning(f"[build_features] FeatureStore.write 失敗（不影響 MySQL）：{_exc}")

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
