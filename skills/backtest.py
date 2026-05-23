"""Walk-forward backtest framework for the stock ranking system.

策略：
- 每月初（首個交易日）重新平衡
- 等權重持有 Top-N 檔股票
- 持有至月底或觸發停損
- 每季重新訓練模型（walk-forward）

交易成本：
- 買入手續費 0.1425%
- 賣出手續費 0.1425% + 證交稅 0.3%
- 來回合計約 0.585%

輸出指標：
- 年化報酬率、最大回撤、Sharpe Ratio
- 勝率、盈虧比
- 月度報酬分布
- 與大盤比較
"""

from __future__ import annotations

import collections
import gc
import os
import time
from dataclasses import dataclass as _module_dataclass
from datetime import date, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Feature, Label, RawPrice, Stock
from skills import data_store, risk
from skills.breakthrough import (
    precompute_stats as _precompute_breakthrough_stats,
    compute_breakthrough_map as _compute_breakthrough_map,
)
from skills.feature_utils import (
    parse_features_json as _parse_features_json_shared,
    impute_features as _impute_features_shared,
    filter_schema_valid_rows as _filter_schema_valid_rows,
    cross_section_normalize as _cross_section_normalize,
)

# ── 模型訓練（複用 train_ranker 邏輯）──
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
from sklearn.ensemble import GradientBoostingRegressor


def _get_process_memory_gb() -> float:
    """取得目前 process 的 RSS 記憶體使用量（GB）。"""
    try:
        import os
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    except ImportError:
        try:
            import platform
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            # macOS ru_maxrss 單位為 bytes；Linux 為 kilobytes
            if platform.system() == "Darwin":
                return ru.ru_maxrss / 1024 ** 3
            return ru.ru_maxrss / 1024 ** 2 / 1024
        except Exception:
            return 0.0


_log_t0 = time.time()


def _log(label: str) -> None:
    """計時 + 記憶體監控 log（統一格式 [TIMER] label | mem=X.XGB | t=X.Xs）。"""
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    elapsed = time.time() - _log_t0
    print(f"[TIMER] {label} | mem={mem:.1f}GB | t={elapsed:.1f}s", flush=True)


def _format_eta(seconds: float) -> str:
    """將秒數轉為易讀時間格式。"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds / 60)}分鐘"
    return f"{seconds / 3600:.1f}小時"


def _parse_features(series: pd.Series) -> pd.DataFrame:
    """解析 features_json；委派給 feature_utils.parse_features_json（統一實作）。"""
    return _parse_features_json_shared(series)


def _train_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    fast_mode: bool = False,
    sample_weight: Optional[np.ndarray] = None,
    train_groups: Optional[np.ndarray] = None,
):
    """訓練一個輕量級模型供回測使用。fast_mode=True 時減少樹數以加速。

    train_groups（非 None）時啟用 LambdaRank 模式：
      - 資料須按 trading_date 排序
      - train_groups 為每個 query（日期）的樣本數陣列
      - 直接優化 top-20 排名一致性（NDCG@20），比 regression 更貼近選股目標
    """
    n_est = 150 if fast_mode else 500
    if _HAS_LGBM:
        if train_groups is not None:
            # LambdaRank：直接優化截面排名（NDCG@20）
            model = lgb.LGBMRanker(
                n_estimators=n_est,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=10,       # ranking 每 query 樣本數少，需放寬
                lambdarank_truncation_level=20,  # 只關心 top-20 排名
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(train_X, train_y, group=train_groups, sample_weight=sample_weight)
        else:
            # Regression 模式（現行預設）
            model = lgb.LGBMRegressor(
                n_estimators=n_est,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=50,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(train_X, train_y, sample_weight=sample_weight)
    else:
        n_est_gbr = 100 if fast_mode else 300
        model = GradientBoostingRegressor(
            n_estimators=n_est_gbr, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42,
        )
        model.fit(train_X, train_y, sample_weight=sample_weight)
    return model


@_module_dataclass
class _StackingAdapter:
    """讓 StackingEnsemble 能與 backtest 內 `model.predict(X_ndarray)` 介面相容。

    `is_stacking=False` 時降級為一般 LightGBM regressor（小樣本 fallback）。
    """
    model: object
    feature_names: List[str]
    is_stacking: bool

    def predict(self, X) -> np.ndarray:
        if not self.is_stacking:
            return self.model.predict(X)
        # StackingEnsemble 需要 DataFrame：把 ndarray 包成 DataFrame（features 順序與訓練一致）
        import pandas as _pd  # noqa
        if not isinstance(X, _pd.DataFrame):
            X = _pd.DataFrame(np.asarray(X), columns=self.feature_names)
        return self.model.predict(X)


def _load_prices(
    db_session: Session,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """從 DB 載入 price 資料（不含 features_json，無 JSON 解析負擔）。"""
    price_stmt = (
        select(
            RawPrice.stock_id, RawPrice.trading_date,
            RawPrice.open, RawPrice.high, RawPrice.low, RawPrice.close, RawPrice.volume,
        )
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.trading_date, RawPrice.stock_id)
    )
    price_df = pd.read_sql(price_stmt, db_session.get_bind())
    for col in ["open", "high", "low", "close", "volume"]:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    return price_df


def _load_features_labels(
    db_session: Session,
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """從 DB 載入 features（raw JSON）和 labels，供滾動視窗按需使用。"""
    feat_stmt = (
        select(Feature.stock_id, Feature.trading_date, Feature.features_json)
        .where(Feature.trading_date.between(start_date, end_date))
        .order_by(Feature.trading_date, Feature.stock_id)
    )
    label_stmt = (
        select(Label.stock_id, Label.trading_date, Label.future_ret_h)
        .where(Label.trading_date.between(start_date, end_date))
        .order_by(Label.trading_date, Label.stock_id)
    )
    feat_df = pd.read_sql(feat_stmt, db_session.get_bind())
    label_df = pd.read_sql(label_stmt, db_session.get_bind())
    return feat_df, label_df


def _load_all_data(
    db_session: Session,
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """從 DB 載入 features, labels, prices（向後相容封裝）。"""
    feat_df, label_df = _load_features_labels(db_session, start_date, end_date)
    price_df = _load_prices(db_session, start_date, end_date)
    return feat_df, label_df, price_df


def _parse_and_filter_features(
    raw_feat_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """解析 features_json、套用欄位過濾與 schema 遷移保護。

    Args:
        raw_feat_df: 含 stock_id / trading_date / features_json 欄位的原始 DataFrame。
        feature_columns: 若指定，只保留該子集（None = 保留全部）。

    Returns:
        展開後的特徵 DataFrame（stock_id, trading_date, feature_cols...）。
    """
    parsed = _parse_features(raw_feat_df["features_json"])
    parsed = parsed.replace([np.inf, -np.inf], np.nan)
    feat_df = raw_feat_df[["stock_id", "trading_date"]].reset_index(drop=True)
    feat_df = pd.concat([feat_df, parsed.reset_index(drop=True)], axis=1)
    del parsed

    # feature_columns 子集過濾
    if feature_columns is not None:
        _avail = [c for c in feature_columns if c in feat_df.columns]
        _missing = [c for c in feature_columns if c not in feat_df.columns]
        if _missing:
            print(f"  [feature_columns] 警告：DB 中缺少 {_missing}，已忽略")
        feat_df = feat_df[["stock_id", "trading_date"] + _avail]
        print(f"  [feature_columns] 使用 {len(_avail)} 個特徵（共 {len(feature_columns)} 指定）")

    # schema 遷移保護：過濾 feature 覆蓋率不足的舊版資料（門檻 50%）
    _fc = [c for c in feat_df.columns if c not in ("stock_id", "trading_date")]
    if _fc:
        _thr = max(1, int(len(_fc) * 0.50))
        _mask = feat_df[_fc].notna().sum(axis=1) >= _thr
        _n_drop = int((~_mask).sum())
        if _n_drop > 0:
            print(
                f"  [schema filter] 過濾 {_n_drop:,} 筆舊版特徵資料"
                f" (threshold={_thr}/{len(_fc)} features)",
                flush=True,
            )
            feat_df = feat_df.loc[_mask].reset_index(drop=True)

    return feat_df


def _get_rebalance_dates(trading_dates: List[date], freq: str = "W") -> List[date]:
    """依據頻率找出再平衡日（W:每週第一個交易日, M:每月第一個交易日）"""
    dates = sorted(trading_dates)
    rebalance = []
    if freq == "W":
        prev_week = None
        for d in dates:
            yw = d.isocalendar()[:2]
            if yw != prev_week:
                rebalance.append(d)
                prev_week = yw
    else:
        prev_month = None
        for d in dates:
            ym = (d.year, d.month)
            if ym != prev_month:
                rebalance.append(d)
                prev_month = ym
    return rebalance


def _precompute_liquidity_eligible_map(
    price_df: pd.DataFrame,
    min_avg_turnover: float,
) -> Dict[date, set[str]]:
    """預先計算每個交易日符合 20 日平均成交值門檻的股票集合。"""
    if min_avg_turnover <= 0 or price_df.empty:
        return {}

    threshold = float(min_avg_turnover) * 1e8  # 向後相容：舊參數單位為「億元」
    if threshold <= 0:
        return {}

    df = price_df[["stock_id", "trading_date", "close", "volume"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["stock_id", "trading_date", "close", "volume"])
    if df.empty:
        return {}

    df["stock_id"] = df["stock_id"].astype(str)
    df = df.sort_values(["stock_id", "trading_date"])
    df["turnover"] = df["close"] * df["volume"]
    df["avg_turnover_20"] = df.groupby("stock_id")["turnover"].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )

    eligible = df[df["avg_turnover_20"] >= threshold][["trading_date", "stock_id"]]
    if eligible.empty:
        return {}

    result: Dict[date, set[str]] = {}
    for td, sub in eligible.groupby("trading_date"):
        result[td] = set(sub["stock_id"].tolist())
    return result


def _precompute_market_median_ret20(price_df: pd.DataFrame) -> Dict[date, float]:
    """預先計算每個交易日全市場 20 日報酬中位數（供大盤環境濾網）。"""
    if price_df.empty:
        return {}

    df = price_df[["stock_id", "trading_date", "close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["stock_id", "trading_date", "close"])
    if df.empty:
        return {}

    df["stock_id"] = df["stock_id"].astype(str)
    df = df.sort_values(["stock_id", "trading_date"])
    df["ret_20"] = df.groupby("stock_id")["close"].pct_change(20)
    med = df.groupby("trading_date")["ret_20"].median().dropna()
    return {d: float(v) for d, v in med.items()}


def _precompute_market_weekly_drop(price_df: pd.DataFrame) -> Dict[date, float]:
    """預先計算每個交易日全市場 5 日報酬中位數（供週跌幅危機偵測）。"""
    if price_df.empty:
        return {}
    df = price_df[["stock_id", "trading_date", "close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["stock_id", "trading_date", "close"])
    if df.empty:
        return {}
    df["stock_id"] = df["stock_id"].astype(str)
    df = df.sort_values(["stock_id", "trading_date"])
    df["ret_5"] = df.groupby("stock_id")["close"].pct_change(5)
    med = df.groupby("trading_date")["ret_5"].median().dropna()
    return {d: float(v) for d, v in med.items()}


def _precompute_market_200ma_bear(price_df: pd.DataFrame) -> Dict[date, bool]:
    """預先計算每個交易日等權市場均價是否低於 200 日均線（True=空頭）。
    用於現金保留機制：空頭時保留 30% 現金。min_periods=40 確保資料不足時不誤判。
    """
    if price_df.empty:
        return {}
    df = price_df[["trading_date", "close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["trading_date", "close"])
    if df.empty:
        return {}
    mkt = df.groupby("trading_date")["close"].mean().sort_index()
    ma200 = mkt.rolling(200, min_periods=40).mean()
    bear_series = (mkt < ma200).dropna()
    return {d: bool(v) for d, v in bear_series.items()}


def _get_entry_positions(
    stock_ids: list,
    price_df: pd.DataFrame,
    period_prices: pd.DataFrame,
    entry_date: date,
    exit_date: date,
    entry_delay_days: int,
    per_stock_entry_dates: Optional[Dict[str, date]],
) -> Tuple[Optional[pd.DataFrame], date, pd.DataFrame]:
    """確定實際進場日及每股進場價，回傳 (positions_input, actual_entry_date, entry_prices)。

    若無法取得任何進場價，回傳 (None, entry_date, empty_df)。
    """
    if per_stock_entry_dates:
        # 各股獨立進場日（突破確認進場模式）
        positions_list = []
        for sid in stock_ids:
            sid_str = str(sid)
            stock_entry = per_stock_entry_dates.get(sid_str, entry_date)
            ep_row = price_df[
                (price_df["stock_id"].astype(str) == sid_str)
                & (price_df["trading_date"] == stock_entry)
            ]
            if ep_row.empty:
                continue
            ep = float(ep_row["close"].iloc[0])
            if ep <= 0:
                continue
            positions_list.append({
                "stock_id": sid_str,
                "entry_date": stock_entry,
                "planned_exit_date": exit_date,
                "entry_price": ep,
            })
        if not positions_list:
            return None, entry_date, pd.DataFrame()
        positions_input = pd.DataFrame(positions_list)
        actual_entry_date = min(per_stock_entry_dates.values())  # 供 ATR/滑價 lookup 參考
        entry_prices = positions_input[["stock_id", "entry_price"]]
    else:
        # 原有邏輯：統一進場日
        all_trading_dates = sorted(period_prices["trading_date"].unique())
        if entry_delay_days > 0:
            future_dates = [d for d in all_trading_dates if d > entry_date]
            if len(future_dates) < entry_delay_days:
                return None, entry_date, pd.DataFrame()
            actual_entry_date = future_dates[entry_delay_days - 1]
        else:
            actual_entry_date = entry_date

        entry_prices = period_prices[
            period_prices["trading_date"] == actual_entry_date
        ][["stock_id", "close"]].rename(columns={"close": "entry_price"})

        if entry_prices.empty:
            return None, actual_entry_date, pd.DataFrame()

        positions_input = entry_prices.assign(
            entry_date=actual_entry_date,
            planned_exit_date=exit_date,
        )[["stock_id", "entry_date", "planned_exit_date", "entry_price"]]

    return positions_input, actual_entry_date, entry_prices


def _compute_slippage_map(
    entry_prices: pd.DataFrame,
    atr_df: Optional[pd.DataFrame],
    actual_entry_date: date,
    enable_slippage: bool,
    tiered_slippage_map: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """計算每股來回滑價（ATR 的 10%，上限 0.3%，進出場各一次）。

    優先使用 tiered_slippage_map（外部預計算的分級滑價），
    其次使用 ATR 模型推算，最後回傳空 dict（無滑價）。
    """
    if tiered_slippage_map is not None:
        return tiered_slippage_map
    if not enable_slippage or atr_df is None or atr_df.empty:
        return {}

    atr_for_slippage = (
        atr_df[atr_df["trading_date"] < actual_entry_date]
        .groupby("stock_id")["atr"]
        .last()
    )
    slippage_map: Dict[str, float] = {}
    for _, ep_row in entry_prices.iterrows():
        sid = str(ep_row["stock_id"])
        ep = float(ep_row["entry_price"])
        if sid in atr_for_slippage.index and ep > 0:
            atr_pct = float(atr_for_slippage[sid]) / ep
            slippage_one_way = min(atr_pct * 0.1, 0.003)  # 單邊上限 0.3%
            slippage_map[sid] = slippage_one_way * 2       # 來回 × 2
        else:
            slippage_map[sid] = 0.0
    return slippage_map


def _calc_stock_return(
    entry_px: float,
    exit_px: float,
    transaction_cost_pct: float,
    slippage_pct: float,
    clip_loss_pct: float,
) -> float:
    """計算單筆股票報酬（扣除交易成本與滑價，並套用 clip 防止退市股拖垮組合）。"""
    ret = exit_px / entry_px - 1 - transaction_cost_pct - slippage_pct
    return max(ret, clip_loss_pct)


def _simulate_period(
    picks: pd.DataFrame,
    price_df: pd.DataFrame,
    entry_date: date,
    exit_date: date,
    stoploss_pct: float,
    transaction_cost_pct: float,
    entry_delay_days: int = 1,
    position_weights: Optional[pd.DataFrame] = None,
    trailing_stop_pct: Optional[float] = None,
    atr_df: Optional[pd.DataFrame] = None,
    atr_stoploss_multiplier: Optional[float] = None,
    enable_slippage: bool = True,
    clip_loss_pct: float = -0.50,
    per_stock_entry_dates: Optional[Dict[str, date]] = None,
    per_stock_stoploss_override: Optional[Dict[str, float]] = None,
    tiered_slippage_map: Optional[Dict[str, float]] = None,  # 預計算的分級滑價（來回），優先於 ATR 模型
    portfolio_circuit_breaker_pct: Optional[float] = None,  # 投資組合熔斷：月中等權報酬跌破此值時全出場
) -> Dict:
    """模擬一個持有期間的績效。

    Args:
        picks: DataFrame with 'stock_id' and 'score'
        price_df: full price data（含 high/low/close）
        entry_date: 選股決策日（收盤後決策）
        exit_date: 預計出場日
        stoploss_pct: 全局固定停損比例（如 -0.07）
        transaction_cost_pct: 來回交易成本
        entry_delay_days: 進場延遲（1 = 次一交易日收盤，更符合實際執行）
        position_weights: 各股倉位比例 DataFrame（stock_id, weight），None 時等權
        trailing_stop_pct: 移動停利比例（如 -0.12），None 時不啟用
        atr_df: 預先計算的 ATR DataFrame（stock_id, trading_date, atr）
        atr_stoploss_multiplier: ATR 倍數停損（如 2.5），覆蓋全局 stoploss_pct
        enable_slippage: 是否套用滑價模型（ATR 的 10%，上限 0.3%，適用進出場各一次）
        clip_loss_pct: 單筆最大損失 clip（預設 -50%；診斷可傳 -1.01 停用）
        per_stock_stoploss_override: 外部預算的個股停損 dict（覆蓋 stoploss_pct）

    Returns:
        Dict with period results
    """
    _stoploss_time = 0.0  # 計時佔位符

    if picks.empty:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0, "_stoploss_time": 0.0}

    stock_ids = picks["stock_id"].tolist()

    # 若有 per-stock 進場日，以最早進場日為 price 載入起點
    _price_start = entry_date
    if per_stock_entry_dates:
        _ps_starts = [per_stock_entry_dates.get(str(s), entry_date) for s in stock_ids]
        if _ps_starts:
            _price_start = min(_ps_starts)

    period_prices = price_df[
        (price_df["stock_id"].isin(stock_ids)) &
        (price_df["trading_date"] >= _price_start) &
        (price_df["trading_date"] <= exit_date)
    ].copy()

    if period_prices.empty:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0, "_stoploss_time": 0.0}

    # ── 1 & 2. 確定進場日並取進場價 ──
    positions_input, actual_entry_date, entry_prices = _get_entry_positions(
        stock_ids, price_df, period_prices, entry_date, exit_date,
        entry_delay_days, per_stock_entry_dates,
    )
    if positions_input is None:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0, "_stoploss_time": 0.0}

    # ── 3. ATR-based 個股動態停損 ──
    per_stock_stop: Optional[Dict[str, float]] = None
    if per_stock_stoploss_override:
        per_stock_stop = per_stock_stoploss_override
    atr_at_entry: Optional[pd.Series] = None
    if atr_stoploss_multiplier is not None and atr_df is not None and not atr_df.empty:
        atr_at_entry = (
            atr_df[atr_df["trading_date"] < actual_entry_date]
            .groupby("stock_id")["atr"]
            .last()
        )
        per_stock_stop = {}
        for _, row in entry_prices.iterrows():
            sid = str(row["stock_id"])
            ep = float(row["entry_price"])
            if sid in atr_at_entry.index and ep > 0:
                atr_val = float(atr_at_entry[sid])
                dynamic = -(atr_stoploss_multiplier * atr_val / ep)
                per_stock_stop[sid] = max(dynamic, -0.30)  # 最大停損 30%
            else:
                per_stock_stop[sid] = stoploss_pct

    # ── 4. 執行出場邏輯 ──
    _t_sl = time.time()
    if trailing_stop_pct is not None or atr_stoploss_multiplier is not None:
        stoploss_result = risk.apply_trailing_stop(
            positions_input, price_df=period_prices,
            trailing_stop_pct=trailing_stop_pct if trailing_stop_pct is not None else -0.15,
            stoploss_pct=stoploss_pct,
            per_stock_stoploss=per_stock_stop,
            atr_stoploss_multiplier=atr_stoploss_multiplier,
            atr_at_entry=atr_at_entry,
        )
    else:
        stoploss_result = risk.apply_stoploss(
            positions_input, period_prices, stoploss_pct, per_stock_stop
        )
    _stoploss_time = time.time() - _t_sl

    # ── P0-3: 投資組合熔斷（月中等權累積報酬 < threshold 時全出場）──
    if portfolio_circuit_breaker_pct is not None and portfolio_circuit_breaker_pct < 0 and not stoploss_result.empty:
        px_cb = period_prices[["stock_id", "trading_date", "close"]].copy()
        px_cb["stock_id"] = px_cb["stock_id"].astype(str)
        ep_cb = entry_prices.copy()
        ep_cb["stock_id"] = ep_cb["stock_id"].astype(str)
        cb_merged = px_cb.merge(ep_cb, on="stock_id", how="inner")
        cb_merged = cb_merged[cb_merged["trading_date"] > actual_entry_date].copy()
        if not cb_merged.empty:
            cb_merged["daily_ret"] = cb_merged["close"] / cb_merged["entry_price"] - 1
            daily_port = cb_merged.groupby("trading_date")["daily_ret"].mean().sort_index()
            breach = daily_port[daily_port < portfolio_circuit_breaker_pct]
            if not breach.empty:
                cb_date = breach.index[0]
                # 取熔斷日各股收盤價
                cb_prices = px_cb[px_cb["trading_date"] == cb_date].set_index("stock_id")["close"]
                for idx, row in stoploss_result.iterrows():
                    sid = str(row["stock_id"])
                    # 僅覆蓋尚未觸發停損、且月底出場日晚於熔斷日的部位
                    if not row["stoploss_triggered"] and row["exit_date"] > cb_date:
                        cb_px = cb_prices.get(sid)
                        if cb_px is not None and cb_px > 0:
                            stoploss_result.at[idx, "exit_date"] = cb_date
                            stoploss_result.at[idx, "exit_price"] = float(cb_px)
                            stoploss_result.at[idx, "stoploss_triggered"] = True  # 計入熔斷計數

    # ── 預先計算個股滑價（ATR 的 10%，上限 0.3%，進出場各一次）──
    slippage_map = _compute_slippage_map(
        entry_prices, atr_df, actual_entry_date,
        enable_slippage, tiered_slippage_map,
    )

    stoploss_count = 0
    stock_returns: Dict[str, float] = {}
    trades_log = []
    if not stoploss_result.empty:
        stoploss_count = int(stoploss_result["stoploss_triggered"].sum())
        for _, row in stoploss_result.iterrows():
            sid = str(row["stock_id"])
            entry_px = float(row["entry_price"])
            exit_px = float(row["exit_price"])
            exit_date_val = str(row["exit_date"])
            sl_triggered = bool(row["stoploss_triggered"])
            if entry_px > 0:
                ret = _calc_stock_return(
                    entry_px, exit_px, transaction_cost_pct,
                    slippage_map.get(sid, 0.0), clip_loss_pct,
                )
                stock_returns[sid] = ret

                # 收錄完整交易紀錄
                reason = "Stop Loss / Trailing Stop" if sl_triggered else "Normal Rebalance Exit"
                pick_row = picks[picks["stock_id"].astype(str) == sid]
                score_val = float(pick_row["score"].iloc[0]) if not pick_row.empty else 0.0
                trade_record = {
                    "stock_id": sid,
                    "entry_date": str(actual_entry_date),
                    "exit_date": exit_date_val,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "realized_pnl_pct": ret,
                    "stoploss_triggered": sl_triggered,
                    "exit_reason": reason,
                    "score": score_val,
                    "slippage_pct": slippage_map.get(sid, 0.0),
                }
                trades_log.append(trade_record)

    if not stock_returns:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0, "trades_log": [], "_stoploss_time": _stoploss_time}

    # ── 5. 計算組合報酬（支援分級倉位）──
    if position_weights is not None and not position_weights.empty:
        pw = position_weights.set_index("stock_id")["weight"]
        n_total = len(stock_returns)
        weighted_ret, total_w = 0.0, 0.0
        for sid, ret in stock_returns.items():
            w = float(pw.get(sid, 1.0 / n_total))
            weighted_ret += w * ret
            total_w += w
        portfolio_return = weighted_ret / total_w if total_w > 0 else 0.0
    else:
        portfolio_return = sum(stock_returns.values()) / len(stock_returns)

    wins = sum(1 for r in stock_returns.values() if r > 0)
    losses = sum(1 for r in stock_returns.values() if r <= 0)

    return {
        "return": portfolio_return,
        "trades": len(stock_returns),
        "wins": wins,
        "losses": losses,
        "stoploss_triggered": stoploss_count,
        "stock_returns": stock_returns,
        "actual_entry_date": actual_entry_date.isoformat() if hasattr(actual_entry_date, "isoformat") else str(actual_entry_date),
        "trades_log": trades_log,
        "_stoploss_time": _stoploss_time,
    }


# ── Walk-Forward 回測參數封裝 ────────────────────────────────────────────────
from dataclasses import dataclass, field


@dataclass
class WalkForwardConfig:
    """Walk-forward 回測所有參數的型別安全封裝。

    取代 run_backtest() 30+ 個零散 kwargs，提升可讀性與可維護性。
    向後相容：run_backtest() 仍接受原有的 kwargs，並在內部自動建構此物件。

    使用方式：
        cfg = WalkForwardConfig(backtest_months=120, enable_seasonal_filter=True)
        results = run_backtest(config, session, wf_config=cfg)
    """
    # ── 基本設定 ──
    backtest_months: int = 24
    retrain_freq_months: int = 3
    topn: int = 30  # Stage 10.1（2026-05-23）: 20→30 提升 Sharpe +0.18 / MDD +6pp（10y WF 驗證）
    stoploss_pct: float = -0.07
    transaction_cost_pct: float = 0.00585
    min_train_days: int = 500
    min_avg_turnover: float = 0.0
    eval_start: Optional[date] = None
    eval_end: Optional[date] = None
    train_lookback_days: Optional[int] = None

    # ── 進出場設定 ──
    entry_delay_days: int = 0
    risk_free_rate: float = 0.015
    benchmark_with_cost: bool = True
    position_sizing: str = "equal"
    position_sizing_method: str = "risk_parity"
    trailing_stop_pct: Optional[float] = None
    atr_stoploss_multiplier: Optional[float] = None
    atr_period: int = 14
    rebalance_freq: str = "M"
    label_horizon_buffer: int = 20
    enable_slippage: bool = False
    enable_tiered_slippage: bool = False
    fast_mode: bool = False
    clip_loss_pct: float = -0.50

    # ── 特徵與訓練設定 ──
    feature_columns: Optional[List[str]] = None
    time_weighting: bool = False
    liquidity_weighting: bool = False
    momentum_penalty_cols: Optional[Dict[str, float]] = None

    # ── 過濾器設定 ──
    enable_complex_filter: bool = False
    enable_seasonal_filter: bool = False
    topn_floor: int = 0
    enable_breakthrough_entry: bool = False
    breakthrough_max_wait: int = 10
    atr_dynamic_stoploss: bool = False
    market_filter: bool = False
    market_filter_tiers: Optional[List[tuple]] = None
    market_filter_min_positions: int = 1
    entry_signal_filter: Optional[Dict[str, object]] = None
    # Stage 10.4 D1：近 20 日報酬 < threshold 的 candidate 排除
    # （0.0=disabled；< 0 啟用，例如 -0.15）。DD attribution 顯示 5301 在 -38% 後仍被選中
    recent_dd_skip_pct: float = 0.0
    # Stage 10.5 D2：同產業最大持股 N 檔（0=disabled）。DD attribution 顯示 2025-03
    # 觀光餐旅 2 檔同時崩 -10.92%。利用 risk.apply_sector_constraint
    max_per_sector: int = 0
    # ── 投資組合熔斷 ──
    portfolio_circuit_breaker_pct: Optional[float] = None  # 月中累積虧損觸發全出場（如 -0.15）
    # ── Label 設定 ──
    label_type: str = "abs"  # "abs"=絕對報酬（現行）；"excess"=等權超額報酬（P1-2）
    # ── LambdaRank ──
    use_lambdarank: bool = False  # True=優化 NDCG@20 截面排名；False=regression（現行）
    # ── 截面正規化 ──
    cross_section_normalize: bool = False  # True=每個再平衡期特徵截面 Z-score 正規化
    # ── Ensemble ──
    ensemble_n_checkpoints: int = 1  # 保留最近 N 次重訓 checkpoint 並平均排名分數（1=停用）
    # ── Volatility Targeting（Stage 7.2）──
    # > 0 啟用：每月 _apply_market_regime_filter 後估 picks 60d realized vol，
    # 若 > target → cash_ratio = max(原值, 1 - target/realized)。0 = disabled。
    # 10y WF 驗證：Sharpe Δ +0.078, MDD Δ +4.29pp（target=0.30 時）
    vol_target_pct: float = 0.0
    vol_target_lookback_days: int = 60
    # ── Stacking Ensemble（Stage 6.1）──
    # True 啟用 LightGBM + XGBoost + CatBoost rank-averaged stacking。
    # Quick eval IC lift +7.1%。與 use_lambdarank 互斥（stacking 使用 regressor）。
    use_stacking: bool = False
    stacking_val_frac: float = 0.20  # 末段切多少比例做 early-stopping 驗證


class BacktestPipeline:
    """Walk-forward 回測管線。

    將 run_backtest() 的大型流程拆分為可獨立測試的階段：
      1. __init__：把 WalkForwardConfig 攤平為實例屬性（保留向後相容欄位名稱）
      2. prepare()：載入價格 / 特徵 / 標籤；預先計算 ATR / 流動性 / 市場環境地圖
      3. _train_model_for_period(rb_date)：訓練週期模型，更新 ensemble buffer
      4. _simulate_one_period(...)：對單一再平衡期執行選股、過濾、加權與模擬
      5. run()：主迴圈協調 + 結果彙整

    向後相容：模組層 `run_backtest()` 為 thin wrapper，行為與舊版相同。
    """

    def __init__(self, config, db_session: Session, wf_config: "WalkForwardConfig") -> None:
        # AppConfig / DB session
        self.config = config
        self.db_session = db_session
        self.wf_config = wf_config

        # ── wf_config 攤平為實例屬性（沿用原 kwargs 名稱，便於遷移）──
        self.backtest_months        = wf_config.backtest_months
        self.retrain_freq_months    = wf_config.retrain_freq_months
        self.topn                   = wf_config.topn
        self.stoploss_pct           = wf_config.stoploss_pct
        self.transaction_cost_pct   = wf_config.transaction_cost_pct
        self.min_train_days         = wf_config.min_train_days
        self.min_avg_turnover       = wf_config.min_avg_turnover
        self.eval_start             = wf_config.eval_start
        self.eval_end               = wf_config.eval_end
        self.train_lookback_days    = wf_config.train_lookback_days
        self.entry_delay_days       = wf_config.entry_delay_days
        self.risk_free_rate         = wf_config.risk_free_rate
        self.benchmark_with_cost    = wf_config.benchmark_with_cost
        self.position_sizing        = wf_config.position_sizing
        self.position_sizing_method = wf_config.position_sizing_method
        self.trailing_stop_pct      = wf_config.trailing_stop_pct
        self.atr_stoploss_multiplier = wf_config.atr_stoploss_multiplier
        self.atr_period             = wf_config.atr_period
        self.rebalance_freq         = wf_config.rebalance_freq
        self.label_horizon_buffer   = wf_config.label_horizon_buffer
        self.enable_slippage        = wf_config.enable_slippage
        self.enable_tiered_slippage = wf_config.enable_tiered_slippage
        self.fast_mode              = wf_config.fast_mode
        self.clip_loss_pct          = wf_config.clip_loss_pct
        self.feature_columns        = wf_config.feature_columns
        self.time_weighting         = wf_config.time_weighting
        self.liquidity_weighting    = wf_config.liquidity_weighting
        self.momentum_penalty_cols  = wf_config.momentum_penalty_cols
        self.enable_complex_filter  = wf_config.enable_complex_filter
        self.enable_seasonal_filter = wf_config.enable_seasonal_filter
        self.topn_floor             = wf_config.topn_floor
        self.enable_breakthrough_entry = wf_config.enable_breakthrough_entry
        self.breakthrough_max_wait  = wf_config.breakthrough_max_wait
        self.atr_dynamic_stoploss   = wf_config.atr_dynamic_stoploss
        self.market_filter          = wf_config.market_filter
        self.market_filter_tiers    = wf_config.market_filter_tiers
        self.market_filter_min_positions = wf_config.market_filter_min_positions
        self.entry_signal_filter    = wf_config.entry_signal_filter
        # Stage 10.4 D1
        self.recent_dd_skip_pct     = wf_config.recent_dd_skip_pct
        # Stage 10.5 D2 — 同產業 max 持股；sector_map 在 prepare() 載入
        self.max_per_sector         = wf_config.max_per_sector
        self.sector_map: Dict[str, str] = {}
        self.portfolio_circuit_breaker_pct = wf_config.portfolio_circuit_breaker_pct
        self.label_type             = wf_config.label_type
        self.use_lambdarank         = wf_config.use_lambdarank
        self.cross_section_normalize = wf_config.cross_section_normalize
        self.ensemble_n_checkpoints = wf_config.ensemble_n_checkpoints
        # Stage 7.2 Vol Targeting
        self.vol_target_pct         = wf_config.vol_target_pct
        self.vol_target_lookback_days = wf_config.vol_target_lookback_days
        # Stage 6.1 Stacking
        self.use_stacking           = wf_config.use_stacking
        self.stacking_val_frac      = wf_config.stacking_val_frac
        if self.use_stacking and self.use_lambdarank:
            raise ValueError(
                "use_stacking 與 use_lambdarank 互斥：stacking base models "
                "為 regressor，無法接受 LambdaRank 的 int label。"
            )

        # ── prepare() 階段填入的執行狀態 ──
        self.price_df: pd.DataFrame = pd.DataFrame()
        self.feat_df: pd.DataFrame = pd.DataFrame()
        self.label_df: pd.DataFrame = pd.DataFrame()
        self.atr_df: Optional[pd.DataFrame] = None
        self.bt_stats_df: Optional[pd.DataFrame] = None
        self.liquidity_eligible_map: Dict[date, set] = {}
        self.market_median_ret20_map: Dict[date, float] = {}
        self.market_weekly_drop_map: Dict[date, float] = {}
        self.market_200ma_bear_map: Dict[date, bool] = {}
        self.emerging_ids: set = set()
        self.rebalance_dates: List[date] = []
        self.bt_trading_dates: List[date] = []
        self.data_start: Optional[date] = None
        self.data_end: Optional[date] = None
        self.backtest_start: Optional[date] = None
        self.benchmark_tc: float = 0.0

        # ── 計時統計（prepare/run 間共用）──
        self._timer_load_prices = 0.0
        self._timer_load_features = 0.0
        self._timer_load_labels = 0.0
        self._timer_train_model = 0.0
        self._timer_predict = 0.0
        self._timer_breakthrough = 0.0
        self._timer_apply_stoploss = 0.0
        self._timer_simulate = 0.0
        self._count_train_model = 0
        self._count_predict = 0
        self._count_breakthrough = 0
        self._count_apply_stoploss = 0
        self._count_simulate = 0

        # ── run() 中累積的訓練 / 模型狀態 ──
        self._use_rolling_window: bool = self.train_lookback_days is not None
        self.current_model = None
        self.current_feature_names: Optional[List[str]] = None
        self.last_train_date: Optional[date] = None
        self._model_buf: Deque = collections.deque(maxlen=max(1, self.ensemble_n_checkpoints))
        # 滾動視窗暫存
        self._rw_feat_df: Optional[pd.DataFrame] = None
        self._rw_label_df: Optional[pd.DataFrame] = None
        self._rw_range: Optional[Tuple[date, date]] = None

    # ── 1. 預先載入資料 + 計算靜態地圖 ───────────────────────────────────────
    def prepare(self) -> None:
        """載入價格 / 特徵 / 標籤；預計算 ATR / 流動性 / 市場環境地圖 / 突破指標。"""
        print("\n" + "=" * 60)
        print("Walk-Forward Backtest")
        print("=" * 60)

        # ── 1. 確認可用資料範圍 ──
        max_feat_date = self.db_session.query(func.max(Feature.trading_date)).scalar()
        min_feat_date = self.db_session.query(func.min(Feature.trading_date)).scalar()
        max_label_date = self.db_session.query(func.max(Label.trading_date)).scalar()

        if max_feat_date is None or max_label_date is None:
            raise ValueError("features 或 labels 表為空，請先跑 pipeline-build")

        data_end = min(max_feat_date, max_label_date)
        if self.eval_end is not None:
            data_end = min(data_end, self.eval_end)
        backtest_start = (
            self.eval_start
            if self.eval_start is not None
            else data_end - timedelta(days=30 * self.backtest_months)
        )
        data_start = min_feat_date
        if self.train_lookback_days:
            # 有滾動訓練視窗：以 backtest_start 為錨點往前推，縮小資料抓取範圍
            warmup_days = max(60, self.atr_period * 3)
            _anchor = self.eval_start if self.eval_start is not None else backtest_start
            bounded_start = _anchor - timedelta(days=self.train_lookback_days + warmup_days)
            data_start = max(min_feat_date, bounded_start)

        exit_strategy = (
            "trailing"
            if self.trailing_stop_pct is not None
            else (
                f"ATR×{self.atr_stoploss_multiplier}"
                if self.atr_stoploss_multiplier is not None
                else f"fixed {self.stoploss_pct:.0%}"
            )
        )
        print(f"  資料範圍: {data_start} ~ {data_end}")
        print(f"  回測期間: {backtest_start} ~ {data_end}")
        print(f"  模型重訓: 每 {self.retrain_freq_months} 個月")
        print(f"  選股數量: {self.topn}  倉位: {self.position_sizing}")
        print(
            f"  停損策略: {exit_strategy}"
            + (f"  移動停利: {self.trailing_stop_pct:.0%}" if self.trailing_stop_pct else "")
        )
        print(
            f"  交易成本: {self.transaction_cost_pct:.3%}（來回）  進場延遲: {self.entry_delay_days} 交易日"
        )
        print(f"  無風險利率: {self.risk_free_rate:.1%}  Benchmark含成本: {self.benchmark_with_cost}")
        if self.train_lookback_days:
            print(f"  訓練窗長: {self.train_lookback_days} 日")
        print()

        # ── 2. 載入資料（DuckDB parquet cache via data_store）──
        # 第一次執行：MySQL 全量載入 → parquet（features 含 JSON 解析 + float32 壓縮）
        # 後續執行：DuckDB predicate pushdown 直接讀 parquet，無 MySQL 往返
        # TTL: 24 小時，超時自動重建
        self._use_rolling_window = self.train_lookback_days is not None

        # ── 載入價格資料 ──
        _log("load_prices start")
        _t = time.time()
        price_df = data_store.get_prices(self.db_session, data_start, data_end)
        self._timer_load_prices = time.time() - _t
        _log(f"load_prices done {self._timer_load_prices:.1f}s")

        if price_df.empty:
            raise ValueError("資料不足，無法進行回測")

        # ── 預熱 features/labels 快取（若尚未建立或已失效）──
        # _ensure() 在 get_features/get_labels 內自動呼叫；
        # 此處提前呼叫是為了讓 timer 能分開計量 price vs feature 建立時間
        self._timer_load_features = 0.0
        self._timer_load_labels = 0.0

        # 非滾動視窗：一次用 DuckDB 讀全量（訓練資料隨時間累積增長）
        # 滾動視窗：佔位符，每 fold 在迴圈內用 DuckDB 按日期範圍讀取
        if not self._use_rolling_window:
            _log("load_all_features start (non-rolling, DuckDB)")
            _t = time.time()
            feat_df = data_store.get_features(self.db_session, data_start, data_end, None)  # 全欄載入，過濾延至訓練步驟
            label_df = data_store.get_labels(self.db_session, data_start, data_end)
            self._timer_load_features = time.time() - _t
            _log(f"load_all_features done {self._timer_load_features:.1f}s")
            # ── P2-1: 截面 Z-score 正規化（在 label 轉換前完成，保留 date/id 欄位）──
            if self.cross_section_normalize and not feat_df.empty:
                _t = time.time()
                feat_df = _cross_section_normalize(feat_df, date_col="trading_date")
                print(f"  [cs_norm] 截面 Z-score 正規化完成 ({time.time()-_t:.1f}s)", flush=True)
            # ── P1-2: 超額報酬 label 轉換（label_type="excess"）──
            if self.label_type == "excess" and not label_df.empty:
                mkt_ret = label_df.groupby("trading_date")["future_ret_h"].mean().rename("_mkt_ret")
                label_df = label_df.join(mkt_ret, on="trading_date")
                label_df["future_ret_h"] = label_df["future_ret_h"] - label_df["_mkt_ret"]
                label_df = label_df.drop(columns=["_mkt_ret"])
                print(f"  [label_type=excess] 已將 future_ret_h 轉換為等權超額報酬", flush=True)
        else:
            feat_df = pd.DataFrame()
            label_df = pd.DataFrame()
            _log(f"rolling_window_mode: per-fold DuckDB load (window={self.train_lookback_days}d)")

        # ── 3. 預計算 ATR（若需要）──
        atr_df: Optional[pd.DataFrame] = None
        if (
            self.atr_stoploss_multiplier is not None
            or self.position_sizing == "vol_inverse"
            or self.enable_slippage
        ):
            print("  預計算 ATR ...", flush=True)
            atr_df = risk.compute_atr(price_df, period=self.atr_period)
        liquidity_eligible_map = _precompute_liquidity_eligible_map(price_df, self.min_avg_turnover)
        market_median_ret20_map = _precompute_market_median_ret20(price_df)
        # ── P3-2: 載入興櫃股 ID，回測 universe 與 production 一致 ──
        _emerging_ids: set = {
            str(r.stock_id)
            for r in self.db_session.query(Stock.stock_id).filter(Stock.market == "EMERGING").all()
        }
        if _emerging_ids:
            print(f"  EMERGING 過濾：排除 {len(_emerging_ids)} 支興櫃股", flush=True)
        market_weekly_drop_map = _precompute_market_weekly_drop(price_df)
        market_200ma_bear_map = _precompute_market_200ma_bear(price_df)

        # ── 3b. 突破進場：一次性預計算所有股票的 rolling 突破指標 ──
        # 避免在主迴圈中每個再平衡期重算（117 次 rolling → 1 次）
        bt_stats_df: Optional[pd.DataFrame] = None
        if self.enable_breakthrough_entry:
            print("  預計算突破 rolling 指標（close_max_20 / vol_avg_20 / ma_20）...", flush=True)
            bt_stats_df = _precompute_breakthrough_stats(price_df, lookback=20)

        # ── 4. 找出回測期間的再平衡日 ──
        bt_trading_dates = sorted(price_df[price_df["trading_date"] >= backtest_start]["trading_date"].unique())
        rebalance_dates = _get_rebalance_dates(bt_trading_dates, freq=self.rebalance_freq)
        print(f"  再平衡次數: {len(rebalance_dates)} (頻率: {self.rebalance_freq})")

        # Stage 10.5 D2：若啟用 max_per_sector，預先載入 industry mapping
        if self.max_per_sector > 0:
            try:
                from app.models import Stock as _Stk
                _rows = self.db_session.query(_Stk.stock_id, _Stk.industry_category).all()
                self.sector_map = {str(sid): (ind or "未分類") for sid, ind in _rows}
                print(f"  [max_per_sector] 載入 {len(self.sector_map)} 檔 industry mapping，"
                      f"限制每產業 ≤ {self.max_per_sector} 檔")
            except Exception as _exc:
                print(f"  [max_per_sector] 載入失敗（{_exc}），停用 sector constraint")
                self.sector_map = {}

        # ── 將所有結果回寫至 self ──
        self.price_df = price_df
        self.feat_df = feat_df
        self.label_df = label_df
        self.atr_df = atr_df
        self.bt_stats_df = bt_stats_df
        self.liquidity_eligible_map = liquidity_eligible_map
        self.market_median_ret20_map = market_median_ret20_map
        self.market_weekly_drop_map = market_weekly_drop_map
        self.market_200ma_bear_map = market_200ma_bear_map
        self.emerging_ids = _emerging_ids
        self.rebalance_dates = rebalance_dates
        self.bt_trading_dates = bt_trading_dates
        self.data_start = data_start
        self.data_end = data_end
        self.backtest_start = backtest_start
        self.benchmark_tc = self.transaction_cost_pct if self.benchmark_with_cost else 0.0

    # ── 2. 單期訓練 ──────────────────────────────────────────────────────
    def _train_model_for_period(
        self,
        rb_date: date,
        feat_df: pd.DataFrame,
        label_df: pd.DataFrame,
        liquidity_eligible_map: Dict[date, set],
        model_buf: Deque,
    ) -> Optional[Dict]:
        """訓練單一再平衡日的模型。

        - 若資料不足或全部過濾掉，回傳 None（呼叫端應 continue）。
        - 成功時回傳 dict（model, feature_names, last_train_date, train_secs, n_samples）。
        - model_buf：傳入的 ensemble buffer（in-place 更新）。
        """
        label_horizon_buffer   = self.label_horizon_buffer
        train_lookback_days    = self.train_lookback_days
        min_avg_turnover       = self.min_avg_turnover
        min_train_days         = self.min_train_days
        feature_columns        = self.feature_columns
        momentum_penalty_cols  = self.momentum_penalty_cols
        time_weighting         = self.time_weighting
        liquidity_weighting    = self.liquidity_weighting
        use_lambdarank         = self.use_lambdarank
        fast_mode              = self.fast_mode

        label_cutoff = rb_date - timedelta(days=label_horizon_buffer)
        train_feat = feat_df[feat_df["trading_date"] < rb_date]
        train_label = label_df[label_df["trading_date"] < label_cutoff]
        if train_lookback_days:
            train_start = rb_date - timedelta(days=train_lookback_days)
            train_feat = train_feat[train_feat["trading_date"] >= train_start]
            train_label = train_label[train_label["trading_date"] >= train_start]

        if train_feat.empty or train_label.empty:
            print(f"  [{rb_date}] 訓練資料不足，跳過")
            return None

        merged = train_feat.merge(train_label, on=["stock_id", "trading_date"], how="inner")

        # 訓練資料流動性過濾（與評分階段保持一致）
        if min_avg_turnover > 0 and liquidity_eligible_map:
            _train_tds = pd.to_datetime(merged["trading_date"]).dt.date.values
            _train_sids = merged["stock_id"].astype(str).values
            _liq_ok = np.array([
                str(sid) in liquidity_eligible_map.get(td, set())
                for sid, td in zip(_train_sids, _train_tds)
            ])
            merged = merged[_liq_ok]

        if len(merged) < min_train_days:
            print(f"  [{rb_date}] 訓練資料 {len(merged)} 筆 < {min_train_days}，跳過")
            return None

        # feat_df 已預解析（2b 步驟），直接取特徵欄位（排除 meta 欄與 label）
        _meta_cols = {"stock_id", "trading_date", "future_ret_h"}
        fmat = merged.drop(columns=[c for c in _meta_cols if c in merged.columns])
        # 若指定 feature_columns，訓練時只使用該子集（day_feat 仍保有全欄供 entry_signal_filter 使用）
        if feature_columns is not None:
            _avail_fc = [c for c in feature_columns if c in fmat.columns]
            fmat = fmat[_avail_fc]
        fmat = fmat.replace([np.inf, -np.inf], np.nan)
        for col in fmat.columns:
            if fmat[col].isna().all():
                fmat[col] = 0
            else:
                fmat[col] = fmat[col].fillna(fmat[col].median())

        valid = fmat.notna().all(axis=1)
        fmat = fmat.loc[valid]
        merged = merged.loc[fmat.index]

        if fmat.empty:
            return None

        current_feature_names = list(fmat.columns)

        # 動能懲罰：對指定特徵乘以縮放係數（訓練前）
        if momentum_penalty_cols:
            for _pc, _ps in momentum_penalty_cols.items():
                if _pc in fmat.columns:
                    fmat[_pc] = fmat[_pc] * _ps

        y = merged["future_ret_h"].astype(float).values

        # 時間加權：近 1 年 × 2.0，1~2 年 × 1.0，>2 年 × 0.5（近期市場規律更重要）
        if time_weighting:
            _merged_dates = pd.to_datetime(merged["trading_date"]).dt.date.values
            _days_ago = np.array([(rb_date - d).days for d in _merged_dates], dtype=float)
            _sample_weight = np.where(_days_ago <= 365, 2.0, np.where(_days_ago <= 730, 1.0, 0.5))
        else:
            _sample_weight = None  # 等權樣本（baseline 模式）

        # 流動性加權：sample_weight ∝ log(1+amt_20)，讓模型學偏大型股模式
        if liquidity_weighting and "amt_20" in merged.columns:
            _amt20_vals = merged["amt_20"].fillna(0).clip(lower=0).values.astype(float)
            _liq_w = np.log1p(_amt20_vals)
            _liq_mean = _liq_w.mean()
            if _liq_mean > 0:
                _liq_w = _liq_w / _liq_mean  # 歸一化：平均權重 = 1.0
            if _sample_weight is not None:
                _sample_weight = _sample_weight * _liq_w  # 與時間加權相乘
            else:
                _sample_weight = _liq_w

        # ── LambdaRank：按 trading_date 排序並計算 group sizes ──
        if use_lambdarank:
            _td_vals = pd.to_datetime(merged["trading_date"]).values
            _sort_idx = np.argsort(_td_vals, kind="stable")
            _fmat_arr = fmat.values[_sort_idx]
            _y_cont = y[_sort_idx]           # 連續 forward return
            _sw_sorted = _sample_weight[_sort_idx] if _sample_weight is not None else None
            # group = 每個 trading_date 的股票數（LightGBM LambdaRank 需要）
            _td_sorted = _td_vals[_sort_idx]
            _unique_td, _gcounts = np.unique(_td_sorted, return_counts=True)
            _train_groups = _gcounts.astype(np.int32)
            # LGBMRanker 需要 int label：將各截面的連續 return 轉為 quintile 排名（0~4）
            _y_sorted_int = np.empty(len(_y_cont), dtype=np.int32)
            _pos = 0
            for _gc in _gcounts:
                _seg = _y_cont[_pos: _pos + _gc]
                # 截面內按分位數分組：0(最低)~4(最高)
                _ranks = _seg.argsort().argsort()  # 0~(gc-1)
                _n_bins = min(5, _gc)
                _bins = (_ranks * _n_bins / _gc).astype(np.int32).clip(0, _n_bins - 1)
                _y_sorted_int[_pos: _pos + _gc] = _bins
                _pos += _gc
            _y_sorted = _y_sorted_int
        else:
            _fmat_arr = fmat.values
            _y_sorted = y
            _sw_sorted = _sample_weight
            _train_groups = None

        _t = time.time()
        if self.use_stacking:
            # Stage 6.1: LightGBM + XGBoost + CatBoost rank-averaged
            current_model = self._train_stacking_for_period(
                fmat, y, current_feature_names,
            )
        else:
            current_model = _train_model(
                _fmat_arr, _y_sorted,
                fast_mode=fast_mode,
                sample_weight=_sw_sorted,
                train_groups=_train_groups,
            )
        _dt = time.time() - _t
        model_buf.append(current_model)  # P2-2: 加入 ensemble buffer
        return {
            "model": current_model,
            "feature_names": current_feature_names,
            "last_train_date": rb_date,
            "train_secs": _dt,
            "n_samples": len(y),
        }

    # ── Stage 6.1 Helper：訓練 stacking ensemble + ndarray-compatible 適配 ──
    def _train_stacking_for_period(
        self,
        fmat: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str],
    ) -> "_StackingAdapter":
        """切尾段 val 給 early-stopping，訓 LGBM+XGB+CatBoost，回傳 ndarray-compat adapter。

        注意：不傳 sample_weight 給 base trainers，因為 xgb / catboost early-stopping
        對權重 + small val set 有時不穩；stacking 增益主要來自演算法 diversity。
        """
        from skills.stacking import train_stacking_ensemble
        _val_frac = float(self.stacking_val_frac)
        if not 0.05 <= _val_frac <= 0.5:
            _val_frac = 0.20
        _n = len(fmat)
        _n_val = max(1, int(_n * _val_frac))
        _n_tr = _n - _n_val
        if _n_tr < 50 or _n_val < 10:
            # 樣本太少時降級到 LightGBM 單模型，避免 stacking 在小樣本失敗
            print(f"  [stacking] 樣本不足 (train={_n_tr}, val={_n_val})，降級至 LGBM regression")
            _model = _train_model(
                fmat.values, y, fast_mode=self.fast_mode,
                sample_weight=None, train_groups=None,
            )
            return _StackingAdapter(model=_model, feature_names=feature_names, is_stacking=False)
        _tr_X, _val_X = fmat.iloc[:_n_tr], fmat.iloc[_n_tr:]
        _tr_y, _val_y = y[:_n_tr], y[_n_tr:]
        _ens = train_stacking_ensemble(
            train_X=_tr_X, train_y=_tr_y,
            val_X=_val_X, val_y=_val_y,
        )
        return _StackingAdapter(model=_ens, feature_names=feature_names, is_stacking=True)

    # ── Helper：當日特徵評分（含 momentum penalty + ensemble averaging）──
    def _score_day_features(
        self,
        rb_date: date,
        day_feat: pd.DataFrame,
        current_model,
        current_feature_names: List[str],
        model_buf: Deque,
    ) -> Tuple[pd.DataFrame, float]:
        """對 day_feat 評分並寫入 `score` 欄；回傳 (day_feat, predict_secs)。"""
        # feat_df 已預解析，直接從展開後的 DataFrame 取特徵欄位
        fmat = day_feat.drop(columns=["stock_id", "trading_date"], errors="ignore")
        for col in current_feature_names:
            if col not in fmat.columns:
                fmat[col] = 0
        fmat = fmat[current_feature_names]
        fmat = fmat.replace([np.inf, -np.inf], np.nan)
        for col in fmat.columns:
            if fmat[col].isna().all():
                fmat[col] = 0
            else:
                fmat[col] = fmat[col].fillna(fmat[col].median())

        # 動能懲罰：對指定特徵乘以縮放係數（預測前）
        if self.momentum_penalty_cols:
            for _pc, _ps in self.momentum_penalty_cols.items():
                if _pc in fmat.columns:
                    fmat[_pc] = fmat[_pc] * _ps

        _t = time.time()
        if self.ensemble_n_checkpoints > 1 and len(model_buf) > 1:
            # P2-2: Ensemble：各模型預測的「截面排名百分位」取平均，再轉為最終分數
            _raw_preds = np.array([m.predict(fmat.values) for m in model_buf])  # (n_models, n_stocks)
            # rank-based averaging：每個模型的預測分數轉換為截面排名百分位（0~1）
            _n = _raw_preds.shape[1]
            _ranked = np.argsort(np.argsort(_raw_preds, axis=1), axis=1).astype(float) / max(_n - 1, 1)
            scores = _ranked.mean(axis=0)
        else:
            scores = current_model.predict(fmat.values)
        _dt = time.time() - _t
        day_feat = day_feat.reset_index(drop=True)
        day_feat["score"] = scores
        return day_feat, _dt

    # ── Helper：停損冷卻過濾 ──
    def _apply_cooldown_filter(
        self,
        day_feat: pd.DataFrame,
        rb_date: date,
        cooldown_until: Dict[str, date],
    ) -> pd.DataFrame:
        """根據冷卻表排除停損後仍在冷卻期的股票。"""
        if cooldown_until:
            excluded = {sid for sid, expiry in cooldown_until.items() if expiry > rb_date}
            if excluded:
                before_n = len(day_feat)
                day_feat = day_feat[~day_feat["stock_id"].astype(str).isin(excluded)]
                n_excl = before_n - len(day_feat)
                if n_excl > 0:
                    print(f"  [{rb_date}] 停損冷卻：排除 {n_excl} 檔")
        return day_feat

    # ── Helper：大盤環境濾網（複雜 / 簡單分支 + 季節性 + topN floor + RSI/低波動加權 + 動態現金）──
    def _apply_market_regime_filter(
        self,
        day_feat: pd.DataFrame,
        rb_date: date,
        effective_topn: int,
    ) -> Tuple[pd.DataFrame, int, float, bool]:
        """套用大盤環境濾網；回傳 (day_feat, effective_topn, cash_ratio, day_feat_empty_flag)。

        當 RSI 過濾後 day_feat 為空時 day_feat_empty_flag=True，caller 應 continue。
        """
        config = self.config
        topn = self.topn
        enable_complex_filter = self.enable_complex_filter
        enable_seasonal_filter = self.enable_seasonal_filter
        topn_floor = self.topn_floor
        market_median_ret20_map = self.market_median_ret20_map
        market_weekly_drop_map = self.market_weekly_drop_map
        market_200ma_bear_map = self.market_200ma_bear_map

        _cash_ratio = 0.0
        median_ret = market_median_ret20_map.get(rb_date)
        weekly_drop = market_weekly_drop_map.get(rb_date)

        if enable_complex_filter:
            # ── 複雜過濾（v2-v7 最佳化設定）──
            crisis_threshold = getattr(config, "market_filter_weekly_drop_threshold", -0.03)
            crisis_topn = getattr(config, "market_filter_crisis_topn", max(2, topn // 4))
            bear_topn = getattr(config, "market_filter_bear_topn", max(3, topn // 3))

            if weekly_drop is not None and weekly_drop < crisis_threshold:
                effective_topn = min(effective_topn, crisis_topn)
                print(f"  [{rb_date}] 危機模式 (週跌 {weekly_drop:.2%})，topN → {effective_topn}")
            elif median_ret is not None and median_ret < -0.03:
                effective_topn = min(effective_topn, bear_topn)
                print(f"  [{rb_date}] 空頭市場 (20日中位數 {median_ret:.2%})，topN → {effective_topn}")

            # ── 季節性降倉（弱勢月份）──
            _s_weak = tuple(getattr(config, "seasonal_weak_months", (3, 10)))
            _s_mult = float(getattr(config, "seasonal_topn_multiplier", 0.5))
            effective_topn, _reduced = risk.apply_seasonal_topn_reduction(
                effective_topn, rb_date.month, weak_months=_s_weak, multiplier=_s_mult, topn_floor=1
            )
            if _reduced:
                print(f"  [{rb_date}] 弱勢月份 {rb_date.month}月（complex_filter），topN → {effective_topn}")

            # ── topN 絕對下限保護 ──
            _extreme_bear = weekly_drop is not None and weekly_drop < -0.10
            _min_topn = 3 if _extreme_bear else 5
            if effective_topn < _min_topn:
                print(f"  [{rb_date}] [topN-floor] {effective_topn}→{_min_topn} (min_topn={_min_topn})")
                effective_topn = _min_topn

            # ── 判斷是否為空頭環境 ──
            _is_bear_env = (
                (weekly_drop is not None and weekly_drop < crisis_threshold) or
                (median_ret is not None and median_ret < -0.03)
            )

            # ── RSI 自適應過濾 ──
            if _is_bear_env and "rsi_14" in day_feat.columns:
                _5d_bounce = market_weekly_drop_map.get(rb_date, 0.0)
                _20d = median_ret if median_ret is not None else 0.0
                if _5d_bounce > 0.03:
                    _rsi_threshold = None
                elif _20d < -0.15:
                    _rsi_threshold = 75
                else:
                    _rsi_threshold = 80
                if _rsi_threshold is not None:
                    _rsi_before = len(day_feat)
                    day_feat = day_feat[day_feat["rsi_14"].fillna(0) <= _rsi_threshold]
                    _rsi_removed = _rsi_before - len(day_feat)
                    if _rsi_removed > 0:
                        print(f"  [{rb_date}] 空頭RSI過濾: 移除{_rsi_removed}檔(rsi>{_rsi_threshold})")
                    if day_feat.empty:
                        return day_feat, effective_topn, _cash_ratio, True

            # ── 低波動加權（atr_inv z-score 加分）──
            if _is_bear_env and "atr_inv" in day_feat.columns:
                _atr_inv = day_feat["atr_inv"]
                _atr_std = float(_atr_inv.std())
                if _atr_std > 0:
                    day_feat = day_feat.copy()
                    day_feat["score"] = (
                        day_feat["score"] + 0.3 * (_atr_inv - _atr_inv.mean()) / _atr_std
                    )

            # ── 動態現金保留機制 ──
            _200ma_bear = market_200ma_bear_map.get(rb_date, False)
            if _200ma_bear:
                _5d_bounce = market_weekly_drop_map.get(rb_date, 0.0)
                _cash_ratio = 0.10 if _5d_bounce > 0.03 else 0.30

        else:
            # ── 簡單過濾（baseline 模式）：20d 中位數跌幅 > 4% 才縮減 topN ──
            if median_ret is not None and median_ret < -0.04:
                effective_topn = max(2, topn // 3)
                print(f"  [{rb_date}] 大盤環境不佳 (20日中位數 {median_ret:.2%})，topN → {effective_topn}")

        # ── 獨立季節性降倉（enable_complex_filter=False 時也可啟用，對應 daily_pick 行為）──
        if not enable_complex_filter and enable_seasonal_filter:
            _s_weak = tuple(getattr(config, "seasonal_weak_months", (3, 10)))
            _s_mult = float(getattr(config, "seasonal_topn_multiplier", 0.5))
            effective_topn, _reduced = risk.apply_seasonal_topn_reduction(
                effective_topn, rb_date.month, weak_months=_s_weak, multiplier=_s_mult, topn_floor=5
            )
            if _reduced:
                print(f"  [{rb_date}] seasonal_filter: 月份{rb_date.month} topN → {effective_topn}")

        # ── 顯式 topN floor（Change B 實驗用，enable_complex_filter 無關）──
        if topn_floor > 0 and effective_topn < topn_floor:
            print(f"  [{rb_date}] [topN-floor] {effective_topn}→{topn_floor}")
            effective_topn = topn_floor

        # ── Stage 7.2 Vol Targeting：若 picks 60d realized vol > target，
        # 拉高 cash_ratio 減少總部位。disabled (vol_target_pct=0) 時跳過。
        if self.vol_target_pct > 0 and effective_topn > 0 and not day_feat.empty:
            _vt_extra_cash = self._compute_vol_target_cash_share(
                day_feat, rb_date, effective_topn
            )
            if _vt_extra_cash > _cash_ratio:
                _cash_ratio = _vt_extra_cash

        return day_feat, effective_topn, _cash_ratio, False

    # ── Stage 7.2 Helper：vol-target cash share ──
    def _compute_vol_target_cash_share(
        self,
        day_feat: pd.DataFrame,
        rb_date: date,
        effective_topn: int,
    ) -> float:
        """估 picks 60d realized vol，若 > target 計算需要 hold 多少 cash。

        從 day_feat 估 top-N picks（依 score / model_score），用
        skills.vol_targeting.compute_vol_scaler_for_picks 算 scaler。
        任何錯誤 silently fallback 0（不縮減）。
        """
        try:
            from skills.vol_targeting import compute_vol_scaler_for_picks
            _sc = None
            for c in ("score", "model_score"):
                if c in day_feat.columns:
                    _sc = c
                    break
            if _sc:
                _top = day_feat.nlargest(effective_topn, _sc)
            else:
                _top = day_feat.head(effective_topn)
            _pick_sids = _top["stock_id"].astype(str).tolist()
            if len(_pick_sids) < 2:
                return 0.0
            vt = compute_vol_scaler_for_picks(
                _pick_sids, self.price_df, rb_date,
                target_vol=float(self.vol_target_pct),
                lookback_days=int(self.vol_target_lookback_days),
            )
            return float(vt.get("cash_share", 0.0))
        except Exception:
            return 0.0

    # ── Helper：進場訊號過濾 + topN 選股 ──
    def _apply_entry_signal_filter(
        self,
        day_feat: pd.DataFrame,
        effective_topn: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """套用進場訊號過濾並選取 topN；回傳 (picks, bt_candidate_pool)。"""
        entry_signal_filter = self.entry_signal_filter
        market_filter_min_positions = self.market_filter_min_positions

        # Stage 10.4 D1：上 20 日報酬 < threshold 排除（已暴雷股下月不再 pick）
        if self.recent_dd_skip_pct < 0 and "ret_20" in day_feat.columns:
            _before_n = len(day_feat)
            day_feat = day_feat[day_feat["ret_20"] >= self.recent_dd_skip_pct]
            _n_excl = _before_n - len(day_feat)
            if _n_excl > 0:
                # 限制 log 頻率：只在過濾數 > 5 時印
                if _n_excl > 5:
                    print(f"  [recent_dd_skip] 排除 {_n_excl} 檔 (ret_20 < {self.recent_dd_skip_pct:.2%})")

        if entry_signal_filter:
            _esf = entry_signal_filter
            _filtered = day_feat.copy()
            # 依序套用各條件
            if "foreign_buy_streak_max" in _esf and "foreign_buy_streak" in _filtered.columns:
                _filtered = _filtered[_filtered["foreign_buy_streak"] <= _esf["foreign_buy_streak_max"]]
            if "rsi_min" in _esf and "rsi_14" in _filtered.columns:
                _filtered = _filtered[_filtered["rsi_14"] >= _esf["rsi_min"]]
            if "rsi_max" in _esf and "rsi_14" in _filtered.columns:
                _filtered = _filtered[_filtered["rsi_14"] <= _esf["rsi_max"]]
            if "bias_20_max" in _esf and "bias_20" in _filtered.columns:
                _filtered = _filtered[_filtered["bias_20"] <= _esf["bias_20_max"]]
            if "volume_surge_ratio_min" in _esf and "volume_surge_ratio" in _filtered.columns:
                _filtered = _filtered[_filtered["volume_surge_ratio"] >= _esf["volume_surge_ratio_min"]]
            # 新增過濾條件（2026-03-20 圓桌策略）
            if "ma_alignment_min" in _esf and "ma_alignment" in _filtered.columns:
                _filtered = _filtered[_filtered["ma_alignment"] >= _esf["ma_alignment_min"]]
            if "price_volume_divergence_min" in _esf and "price_volume_divergence" in _filtered.columns:
                _filtered = _filtered[_filtered["price_volume_divergence"] >= _esf["price_volume_divergence_min"]]
            if "foreign_buy_intensity_max_pct" in _esf and "foreign_buy_intensity" in _filtered.columns:
                _fi_thresh = _filtered["foreign_buy_intensity"].quantile(_esf["foreign_buy_intensity_max_pct"])
                _filtered = _filtered[_filtered["foreign_buy_intensity"] <= _fi_thresh]
            if ("ret_20_rank_min" in _esf or "ret_20_rank_max" in _esf) and "ret_20" in _filtered.columns:
                _r20_pct = _filtered["ret_20"].rank(pct=True)
                if "ret_20_rank_min" in _esf:
                    _filtered = _filtered[_r20_pct >= _esf["ret_20_rank_min"]]
                if "ret_20_rank_max" in _esf:
                    _filtered = _filtered[_r20_pct[_filtered.index] <= _esf["ret_20_rank_max"]]
            if "ret_60_rank_min" in _esf and "ret_60" in _filtered.columns:
                _r60_pct = _filtered["ret_60"].rank(pct=True)
                _filtered = _filtered[_r60_pct >= _esf["ret_60_rank_min"]]

            if len(_filtered) >= effective_topn:
                picks = risk.pick_topn(_filtered, effective_topn)
            elif len(_filtered) >= market_filter_min_positions:
                picks = risk.pick_topn(_filtered, min(effective_topn, len(_filtered)))
            else:
                # 逐步放寬：先放寬 foreign_buy_streak，再放寬 RSI
                _relaxed = day_feat.copy()
                if "rsi_min" in _esf and "rsi_14" in _relaxed.columns:
                    _relaxed = _relaxed[_relaxed["rsi_14"] >= max(_esf.get("rsi_min", 0) - 10, 30)]
                if "rsi_max" in _esf and "rsi_14" in _relaxed.columns:
                    _relaxed = _relaxed[_relaxed["rsi_14"] <= min(_esf.get("rsi_max", 100) + 10, 85)]
                picks = risk.pick_topn(_relaxed, min(effective_topn, max(market_filter_min_positions, len(_relaxed))))
            # breakthrough entry 擴展池也使用過濾後的候選（_filtered），確保過濾條件不被繞過
            _bt_candidate_pool = _filtered if len(_filtered) >= market_filter_min_positions else day_feat
        else:
            picks = risk.pick_topn(day_feat, effective_topn)
            _bt_candidate_pool = day_feat

        # Stage 10.5 D2：套用同產業最大持股限制（max_per_sector > 0 時啟用）
        if self.max_per_sector > 0 and self.sector_map and not picks.empty:
            # 從 day_feat（過濾後 universe）按 sector constraint 重新挑 picks，
            # 確保 sector 滿時能往後挑（picks 已是 topN，需要從 _filtered/day_feat 重挑）
            _src = _filtered if entry_signal_filter and len(_filtered) >= market_filter_min_positions else day_feat
            _src_with_score = _src if "score" in _src.columns else _src.copy()
            picks = risk.apply_sector_constraint(
                _src_with_score, effective_topn,
                sector_map=self.sector_map, max_per_sector=self.max_per_sector,
            )

        return picks, _bt_candidate_pool

    # ── Helper：突破確認進場 ──
    def _apply_breakthrough_filter(
        self,
        picks: pd.DataFrame,
        bt_candidate_pool: pd.DataFrame,
        rb_date: date,
        effective_topn: int,
        feat_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, date], float, int]:
        """套用突破進場過濾；回傳 (picks, per_stock_entry_dates, breakthrough_secs, count_inc)。"""
        per_stock_entry_dates: Dict[str, date] = {}
        _bt_secs = 0.0
        _bt_count = 0
        if not self.enable_breakthrough_entry:
            return picks, per_stock_entry_dates, _bt_secs, _bt_count

        # 取再平衡日後的交易日視窗（最多 breakthrough_max_wait 天）
        _bt_window = [
            d for d in self.bt_trading_dates if d > rb_date
        ][:self.breakthrough_max_wait]

        if not _bt_window:
            return picks, per_stock_entry_dates, _bt_secs, _bt_count

        # 擴展候補池（entry_signal_filter 啟用時從過濾後候選池取，否則從全量 day_feat 取）
        _slots_needed = effective_topn
        _ext_n = min(_slots_needed * 3, len(bt_candidate_pool)) if _slots_needed > 0 else 0

        _confirmed_rows: List[Dict] = []
        _entry_map: Dict[str, date] = {}
        _orig_topn_sids = set(picks["stock_id"].astype(str).tolist())

        if _ext_n > 0:
            _extended = risk.pick_topn(bt_candidate_pool, _ext_n)
            _ext_sids = _extended["stock_id"].astype(str).tolist()

            # 向量化一次算出所有新候選股的突破日（使用預計算的 rolling stats）
            _t = time.time()
            _bt_map = _compute_breakthrough_map(
                _ext_sids, _bt_window, self.bt_stats_df, feat_df
            )
            _bt_secs = time.time() - _t
            _bt_count = 1

            # 按 score 排序取前 _slots_needed 個有突破的新進股
            for _, _cand_row in _extended.iterrows():
                if len(_confirmed_rows) >= effective_topn:
                    break
                _sid = str(_cand_row["stock_id"])
                if _sid in _bt_map:
                    _confirmed_rows.append(_cand_row.to_dict())
                    _entry_map[_sid] = _bt_map[_sid]

        if _confirmed_rows:
            picks = pd.DataFrame(_confirmed_rows)
            per_stock_entry_dates = _entry_map
            _n_replaced = sum(1 for s in _entry_map if s not in _orig_topn_sids)
            if _n_replaced > 0:
                print(
                    f"  [{rb_date}] 突破進場：{len(_confirmed_rows)} 檔確認"
                    f"（{_n_replaced} 檔以候補替換）",
                    flush=True,
                )
        else:
            picks = pd.DataFrame(columns=picks.columns)
            print(f"  [{rb_date}] 突破進場：無訊號，本期持現金", flush=True)
        return picks, per_stock_entry_dates, _bt_secs, _bt_count

    # ── Helper：倉位權重計算 ──
    def _compute_position_weights(
        self,
        picks: pd.DataFrame,
        rb_date: date,
    ) -> pd.DataFrame:
        """根據 position_sizing / position_sizing_method 計算倉位權重。"""
        period_price_slice = self.price_df[self.price_df["trading_date"] <= rb_date]
        # position_sizing_method 優先（支援 mean_variance / risk_parity）
        _ps_method = self.position_sizing_method if self.position_sizing_method != "vol_inverse" else self.position_sizing
        if _ps_method in ("mean_variance", "risk_parity"):
            try:
                from skills import position_sizing as _pos_mod
                _pick_sids = picks["stock_id"].astype(str).tolist()
                # 只傳 picks 股票的價格（避免全市場股票含 0/inf 資料干擾協方差估計）
                _picks_prices = period_price_slice[
                    period_price_slice["stock_id"].isin(_pick_sids)
                ]
                _price_pivot = (
                    _picks_prices.pivot_table(
                        index="trading_date", columns="stock_id", values="close"
                    )
                    if not _picks_prices.empty else pd.DataFrame()
                )
                _scores_map = {str(r["stock_id"]): float(r["score"]) for _, r in picks.iterrows()}
                _weight_map = _pos_mod.compute_weights(_price_pivot, _scores_map, method=_ps_method)
                pos_weights = pd.DataFrame(
                    [{"stock_id": sid, "weight": w} for sid, w in _weight_map.items()]
                )
            except Exception as _ps_exc:
                print(f"  [{rb_date}] position_sizing {_ps_method} 失敗（{_ps_exc}），fallback vol_inverse")
                pos_weights = risk.compute_position_weights(
                    picks, method="vol_inverse",
                    price_df=period_price_slice,
                    atr_period=self.atr_period,
                )
        else:
            pos_weights = risk.compute_position_weights(
                picks, method=self.position_sizing,
                price_df=period_price_slice if self.position_sizing == "vol_inverse" else None,
                atr_period=self.atr_period,
            )
        return pos_weights

    # ── Helper：漸進式大盤過濾（market_filter_tiers / market_filter）──
    def _apply_market_filter_tiers(
        self,
        picks: pd.DataFrame,
        period_results: List[Dict],
        rb_date: date,
        effective_topn: int,
    ) -> Tuple[pd.DataFrame, bool, float]:
        """根據前期大盤報酬調整持倉；回傳 (picks, market_filter_skip, mf_multiplier)。"""
        _market_filter_skip = False
        _mf_multiplier = 1.0  # 記錄實際持倉倍率供 period_results 保存
        if len(period_results) > 0 and (self.market_filter or self.market_filter_tiers):
            _prev_bm = period_results[-1].get("benchmark_return", 0.0)
            if self.market_filter_tiers:
                # 漸進式：按 tiers 從深到淺匹配（tiers 須由淺到深排序）
                _applied = False
                for _thr, _mult in reversed(self.market_filter_tiers):
                    if _prev_bm < _thr:
                        _mf_multiplier = _mult
                        _mf_new = max(1, int(effective_topn * _mult))
                        if _mult <= 0:
                            _market_filter_skip = True
                            print(f"  [{rb_date}] market_filter_tiers: 前期大盤 {_prev_bm:+.2%} < {_thr:.0%}，本期持現金")
                        elif _mf_new < len(picks):
                            picks = picks.head(_mf_new)
                            print(f"  [{rb_date}] market_filter_tiers: 前期大盤 {_prev_bm:+.2%} < {_thr:.0%}，持股×{_mult}→{_mf_new}")
                        _applied = True
                        break
            elif self.market_filter:
                # 原始二階段過濾
                if _prev_bm < -0.10:
                    _market_filter_skip = True
                    _mf_multiplier = 0.0
                    print(f"  [{rb_date}] market_filter: 前期大盤 {_prev_bm:+.2%} < -10%，本期持現金")
                elif _prev_bm < -0.05:
                    _mf_multiplier = 0.5
                    _mf_new = max(1, effective_topn // 2)
                    if _mf_new < len(picks):
                        picks = picks.head(_mf_new)
                        print(f"  [{rb_date}] market_filter: 前期大盤 {_prev_bm:+.2%} < -5%，持股減半→{_mf_new}")
        return picks, _market_filter_skip, _mf_multiplier

    # ── Helper：大盤等權基準（向量化）──
    def _compute_benchmark_return(
        self,
        rb_date: date,
        exit_date: date,
    ) -> float:
        """計算大盤等權基準報酬（套用相同流動性門檻、不設停損、無滑價）。"""
        price_df = self.price_df
        min_avg_turnover = self.min_avg_turnover
        liquidity_eligible_map = self.liquidity_eligible_map
        benchmark_tc = self.benchmark_tc

        all_stocks_on_date = price_df[price_df["trading_date"] == rb_date]["stock_id"].unique()
        if min_avg_turnover > 0 and liquidity_eligible_map:
            eligible_set = liquidity_eligible_map.get(rb_date, set())
            all_stocks_on_date = [s for s in all_stocks_on_date if str(s) in eligible_set]

        # 向量化：直接計算 rb_date → exit_date 期間各股等權報酬均值
        _bm_mask = (
            price_df["stock_id"].isin(all_stocks_on_date) &
            price_df["trading_date"].isin([rb_date, exit_date])
        )
        _bm_prices = price_df[_bm_mask][["stock_id", "trading_date", "close"]].pivot_table(
            index="stock_id", columns="trading_date", values="close"
        )
        if rb_date in _bm_prices.columns and exit_date in _bm_prices.columns:
            _bm_valid = _bm_prices[[rb_date, exit_date]].dropna()
            # Bug-2 fix：排除 rb_date 或 exit_date 價格為 0 的股票，避免除以零產生 +inf%
            _bm_valid = _bm_valid[
                (_bm_valid[rb_date] > 0) & (_bm_valid[exit_date] > 0)
            ]
            if not _bm_valid.empty:
                _raw_bm_ret = _bm_valid[exit_date] / _bm_valid[rb_date] - 1 - benchmark_tc
                # 二次防禦：過濾殘留的 inf/nan（adj_factor 異常等邊緣情況）
                _raw_bm_ret = _raw_bm_ret.replace([np.inf, -np.inf], np.nan).dropna()
                benchmark_ret = float(_raw_bm_ret.mean()) if not _raw_bm_ret.empty else 0.0
            else:
                benchmark_ret = 0.0
        else:
            benchmark_ret = 0.0
        return benchmark_ret

    # ── 3. 主迴圈：逐期訓練 + 模擬 ─────────────────────────────────────────
    def run(self) -> Dict:
        """執行 walk-forward 主流程：prepare → loop → finalize。"""
        if not self.rebalance_dates:
            self.prepare()

        # 從 self 解出常用引用（沿用原 local 變數名，減少 diff）。P2-2 refactor 後，
        # 多個過濾邏輯已抽到 helper method，僅保留迴圈主體 / 摘要區段仍會用到的別名。
        config                 = self.config
        db_session             = self.db_session
        price_df               = self.price_df
        atr_df                 = self.atr_df
        liquidity_eligible_map = self.liquidity_eligible_map
        _emerging_ids          = self.emerging_ids
        rebalance_dates        = self.rebalance_dates
        bt_trading_dates       = self.bt_trading_dates
        data_start             = self.data_start
        data_end               = self.data_end

        # 迴圈主體 / 摘要區段仍需的設定別名
        retrain_freq_months    = self.retrain_freq_months
        stoploss_pct           = self.stoploss_pct
        transaction_cost_pct   = self.transaction_cost_pct
        min_avg_turnover       = self.min_avg_turnover
        train_lookback_days    = self.train_lookback_days
        entry_delay_days       = self.entry_delay_days
        risk_free_rate         = self.risk_free_rate
        benchmark_with_cost    = self.benchmark_with_cost
        position_sizing        = self.position_sizing
        trailing_stop_pct      = self.trailing_stop_pct
        atr_stoploss_multiplier = self.atr_stoploss_multiplier
        rebalance_freq         = self.rebalance_freq
        enable_slippage        = self.enable_slippage
        enable_tiered_slippage = self.enable_tiered_slippage
        clip_loss_pct          = self.clip_loss_pct
        atr_dynamic_stoploss   = self.atr_dynamic_stoploss
        market_filter_min_positions = self.market_filter_min_positions
        ensemble_n_checkpoints = self.ensemble_n_checkpoints
        _use_rolling_window    = self._use_rolling_window
        _wf_cb                 = self.portfolio_circuit_breaker_pct

        # ── 5. Walk-forward 執行 ──
        current_model = None
        current_feature_names = None
        last_train_date = None
        # Ensemble checkpoint buffer（P2-2）：保留最近 ensemble_n_checkpoints 個模型
        _model_buf: Deque = self._model_buf
        period_results: List[Dict] = []
        all_trades_log: List[Dict] = []
        cooldown_until: Dict[str, date] = {}  # stock_id -> 冷卻截止日（停損後 N 週不再選入）
        equity = 10000.0
        equity_curve = [{"date": rebalance_dates[0].isoformat() if rebalance_dates else data_start.isoformat(), "equity": equity}]

        benchmark_equity = 10000.0
        benchmark_curve = [{"date": equity_curve[0]["date"], "equity": benchmark_equity}]
        benchmark_tc = self.benchmark_tc

        # 仍保留 feat_df / label_df 區域引用（rolling 模式會在迴圈內覆寫）
        feat_df = self.feat_df
        label_df = self.label_df

        # ── 滾動視窗狀態 ──
        _rw_feat_df: Optional[pd.DataFrame] = None
        _rw_label_df: Optional[pd.DataFrame] = None
        _rw_range: Optional[Tuple[date, date]] = None

        # ── 計時器累計（load_* 由 prepare() 計算，於迴圈外彙整時引用）──
        _timer_load_prices    = self._timer_load_prices
        _timer_load_features  = self._timer_load_features
        _timer_load_labels    = self._timer_load_labels
        _timer_train_model    = 0.0
        _timer_predict        = 0.0
        _timer_breakthrough   = 0.0
        _timer_apply_stoploss = 0.0
        _timer_simulate       = 0.0
        _count_train_model    = 0
        _count_predict        = 0
        _count_breakthrough   = 0
        _count_apply_stoploss = 0
        _count_simulate       = 0

        # ── 進度計時 ──
        _loop_start = time.time()
        _n_periods = len(rebalance_dates)

        for i, rb_date in enumerate(rebalance_dates):
            # 決定退出日（下一個再平衡日的最後一個交易日，或資料末尾）
            if i + 1 < len(rebalance_dates):
                exit_candidates = [d for d in bt_trading_dates if d > rb_date and d <= rebalance_dates[i + 1]]
                exit_date = exit_candidates[-1] if exit_candidates else rb_date
            else:
                exit_candidates = [d for d in bt_trading_dates if d > rb_date]
                exit_date = exit_candidates[-1] if exit_candidates else rb_date

            if exit_date <= rb_date:
                continue

            # ── 是否需要重訓模型 ──
            need_retrain = (
                current_model is None or
                last_train_date is None or
                (rb_date - last_train_date).days >= retrain_freq_months * 30
            )

            # ── 滾動視窗：按需載入 features/labels ──
            if _use_rolling_window:
                # 訓練視窗起點
                _win_start = max(data_start, rb_date - timedelta(days=train_lookback_days))
                # 視窗終點：涵蓋本次再平衡 + 下一次再平衡前的評分日（約 retrain_freq 個月）
                _win_end = min(data_end, rb_date + timedelta(days=retrain_freq_months * 30 + 10))
                # 需要重新載入：(a) 尚未載入、(b) 視窗起點左移、(c) rb_date 超出已載入範圍
                _need_reload = (
                    _rw_feat_df is None
                    or _rw_range is None
                    or _win_start < _rw_range[0]
                    or rb_date > _rw_range[1]
                )
                if _need_reload:
                    # 釋放上一 fold 的特徵/標籤，再按日期範圍讀取新 fold
                    if _rw_feat_df is not None:
                        del _rw_feat_df, _rw_label_df
                        _rw_feat_df = None
                        _rw_label_df = None
                        gc.collect()
                    _log(f"fold_load {_win_start}~{_win_end}")
                    _t_fold = time.time()
                    _rw_feat_df = data_store.get_features(db_session, _win_start, _win_end, None)  # 全欄載入
                    _rw_label_df = data_store.get_labels(db_session, _win_start, _win_end)
                    _rw_range = (_win_start, _win_end)
                    _log(f"fold_loaded feat:{len(_rw_feat_df):,} dt={time.time()-_t_fold:.1f}s")
                feat_df = _rw_feat_df
                label_df = _rw_label_df

            if need_retrain:
                _train_res = self._train_model_for_period(
                    rb_date, feat_df, label_df, liquidity_eligible_map, _model_buf,
                )
                if _train_res is None:
                    continue
                current_model = _train_res["model"]
                current_feature_names = _train_res["feature_names"]
                last_train_date = _train_res["last_train_date"]
                _timer_train_model += _train_res["train_secs"]
                _count_train_model += 1
                print(f"  [{rb_date}] 模型重訓完成 (訓練筆數: {_train_res['n_samples']:,}) [TIMER] train_model: {_train_res['train_secs']:.2f}s"
                      + (f"  [ensemble={len(_model_buf)}/{ensemble_n_checkpoints}]" if ensemble_n_checkpoints > 1 else ""),
                      flush=True)

            if current_model is None:
                continue

            # ── 對當日股票評分 ──
            day_feat = feat_df[feat_df["trading_date"] == rb_date].copy()
            day_feat = day_feat[day_feat["stock_id"].str.fullmatch(r"\d{4}")]
            if _emerging_ids:
                day_feat = day_feat[~day_feat["stock_id"].isin(_emerging_ids)]

            if day_feat.empty:
                for fallback in range(1, 6):
                    fb_date = rb_date - timedelta(days=fallback)
                    day_feat = feat_df[feat_df["trading_date"] == fb_date].copy()
                    day_feat = day_feat[day_feat["stock_id"].str.fullmatch(r"\d{4}")]
                    if _emerging_ids:
                        day_feat = day_feat[~day_feat["stock_id"].isin(_emerging_ids)]
                    if not day_feat.empty:
                        break

            if day_feat.empty:
                continue

            day_feat, _dt = self._score_day_features(
                rb_date, day_feat, current_model, current_feature_names, _model_buf,
            )
            _timer_predict += _dt
            _count_predict += 1

            # 流動性過濾
            if min_avg_turnover > 0:
                keep_ids = liquidity_eligible_map.get(rb_date, set())
                day_feat = day_feat[day_feat["stock_id"].astype(str).isin(keep_ids)]
                if day_feat.empty:
                    continue

            # ── 停損冷卻過濾 ──
            day_feat = self._apply_cooldown_filter(day_feat, rb_date, cooldown_until)

            # ── 大盤環境濾網 (Market Regime Filter) + 季節性降倉 + topN floor ──
            day_feat, effective_topn, _cash_ratio, _day_feat_empty = self._apply_market_regime_filter(
                day_feat, rb_date, self.topn,
            )
            if _day_feat_empty:
                continue

            # ── 進場訊號過濾（entry_signal_filter）+ topN 選股 ──
            picks, _bt_candidate_pool = self._apply_entry_signal_filter(day_feat, effective_topn)

            # ── 突破確認進場（Breakthrough Entry Filter）──
            # 月底選股後不立即進場，等每檔個股出現突破訊號（最多等 breakthrough_max_wait 個交易日）。
            # 無訊號者以後排候選補位；仍無訊號者持現金。
            # 使用向量化批次計算，避免逐股逐日掃描 price_df（效能 O(n_stocks × window)）。
            picks, per_stock_entry_dates, _bt_dt, _bt_inc = self._apply_breakthrough_filter(
                picks, _bt_candidate_pool, rb_date, effective_topn, feat_df,
            )
            _timer_breakthrough += _bt_dt
            _count_breakthrough += _bt_inc

            # ── 計算倉位權重（支援 position_sizing_method）──
            pos_weights = self._compute_position_weights(picks, rb_date)

            # ── 大盤過濾（market_filter / market_filter_tiers）：根據前期大盤報酬調整持倉 ──
            picks, _market_filter_skip, _mf_multiplier = self._apply_market_filter_tiers(
                picks, period_results, rb_date, effective_topn,
            )

            # ── 最低持股數保護（market_filter_min_positions）──
            if not _market_filter_skip and market_filter_min_positions > 1 and len(picks) < market_filter_min_positions:
                _need = market_filter_min_positions - len(picks)
                _existing_sids = set(picks["stock_id"].astype(str).tolist())
                _candidates = day_feat[~day_feat["stock_id"].astype(str).isin(_existing_sids)].sort_values("score", ascending=False)
                _backfill = _candidates.head(_need)
                if not _backfill.empty:
                    picks = pd.concat([picks, _backfill[picks.columns]], ignore_index=True)
                    print(f"  [{rb_date}] min_positions: {len(picks)-len(_backfill)}→{len(picks)}（補{len(_backfill)}檔）")

            if _market_filter_skip:
                # 全現金：0% 報酬
                result = {"return": 0.0, "trades": 0, "stoploss_triggered": 0, "stock_returns": {}, "trades_log": [], "_stoploss_time": 0.0}
            else:
                # ── ATR 動態停損：以 atr_inv 中位數分界，低波動 -15%、高波動 -25% ──
                _per_stock_sl: Optional[Dict[str, float]] = None
                if atr_dynamic_stoploss and "atr_inv" in day_feat.columns:
                    _atr_inv_median = float(day_feat["atr_inv"].median())
                    _per_stock_sl = {}
                    for _, _p_row in picks.iterrows():
                        _sid = str(_p_row["stock_id"])
                        _feat_row = day_feat[day_feat["stock_id"].astype(str) == _sid]
                        if not _feat_row.empty:
                            _atr_inv_val = float(_feat_row["atr_inv"].iloc[0])
                            _per_stock_sl[_sid] = -0.15 if _atr_inv_val >= _atr_inv_median else -0.25
                        else:
                            _per_stock_sl[_sid] = -0.20  # fallback

                # ── 分級滑價計算（依 amt_20 決定流動性層級）──
                _tiered_slip_map: Optional[Dict[str, float]] = None
                if enable_tiered_slippage and "amt_20" in day_feat.columns:
                    _tiered_slip_map = {}
                    for _, _p_row in picks.iterrows():
                        _sid = str(_p_row["stock_id"])
                        _feat_row = day_feat[day_feat["stock_id"].astype(str) == _sid]
                        if not _feat_row.empty:
                            _amt20 = float(_feat_row["amt_20"].iloc[0])
                            if _amt20 >= 5e8:    # 大型股 > 5億
                                _slip = 0.002    # 0.2% 來回
                            elif _amt20 >= 1e8:  # 中型股 1~5億
                                _slip = 0.006    # 0.6% 來回
                            else:                # 小型股 < 1億
                                _slip = 0.010    # 1.0% 來回
                            _tiered_slip_map[_sid] = _slip
                        else:
                            _tiered_slip_map[_sid] = 0.006  # fallback 中型

                # ── 模擬持有 ──
                _t = time.time()
                result = _simulate_period(
                    picks, price_df, rb_date, exit_date,
                    stoploss_pct, transaction_cost_pct,
                    entry_delay_days=entry_delay_days,
                    position_weights=pos_weights,
                    trailing_stop_pct=trailing_stop_pct,
                    atr_df=atr_df,
                    atr_stoploss_multiplier=atr_stoploss_multiplier,
                    enable_slippage=enable_slippage,
                    clip_loss_pct=clip_loss_pct,
                    per_stock_entry_dates=per_stock_entry_dates if per_stock_entry_dates else None,
                    per_stock_stoploss_override=_per_stock_sl,
                    tiered_slippage_map=_tiered_slip_map,
                    portfolio_circuit_breaker_pct=_wf_cb,
                )
                _dt = time.time() - _t
                _timer_simulate += _dt
                _count_simulate += 1
                _timer_apply_stoploss += result.get("_stoploss_time", 0.0)
                _count_apply_stoploss += 1

            # ── 大盤基準（等權，向量化計算，不設停損／無滑價，與策略套用相同流動性門檻）──
            benchmark_ret = self._compute_benchmark_return(rb_date, exit_date)

            period_ret = result["return"]
            # 現金保留：_cash_ratio 部分以 0% 報酬計（等效縮減風險敞口）
            if _cash_ratio > 0.0:
                period_ret = period_ret * (1.0 - _cash_ratio)
            equity *= (1 + period_ret)
            benchmark_equity *= (1 + benchmark_ret)

            period_results.append({
                "rebalance_date": rb_date.isoformat(),
                "exit_date": exit_date.isoformat(),
                "actual_entry_date": result.get("actual_entry_date", rb_date.isoformat()),
                "return": period_ret,
                "benchmark_return": benchmark_ret,
                "excess_return": period_ret - benchmark_ret,
                "trades": result["trades"],
                "wins": result.get("wins", 0),
                "losses": result.get("losses", 0),
                "stock_returns": result.get("stock_returns", {}),
                "stoploss_triggered": result["stoploss_triggered"],
                "market_filter_multiplier": _mf_multiplier,
                "equity": equity,
                "benchmark_equity": benchmark_equity,
            })
            equity_curve.append({"date": exit_date.isoformat(), "equity": equity})
            benchmark_curve.append({"date": exit_date.isoformat(), "equity": benchmark_equity})
            all_trades_log.extend(result.get("trades_log", []))

            # ── 更新停損冷卻表 ──
            cooldown_weeks = getattr(config, "stoploss_cooldown_weeks", 3)
            for trade in result.get("trades_log", []):
                if trade.get("stoploss_triggered"):
                    expiry = rb_date + timedelta(days=cooldown_weeks * 7)
                    cooldown_until[str(trade["stock_id"])] = expiry

            sign = "+" if period_ret >= 0 else ""
            bm_sign = "+" if benchmark_ret >= 0 else ""
            _done = i + 1
            _elapsed = time.time() - _loop_start
            _eta = _elapsed / _done * (_n_periods - _done) if _done < _n_periods else 0.0
            _mem = _get_process_memory_gb()
            print(
                f"  [{rb_date} ~ {exit_date}] "
                f"組合: {sign}{period_ret:.2%}  大盤: {bm_sign}{benchmark_ret:.2%}  "
                f"持股: {result['trades']}  停損: {result['stoploss_triggered']}  "
                f"淨值: {equity:,.0f}  "
                f"[{_done}/{_n_periods} | 記憶體: {_mem:.1f}GB | 預估剩餘: {_format_eta(_eta)}]",
                flush=True,
            )

        # ── TIMER 瓶頸報告 ──
        _total_loop = time.time() - _loop_start
        print("\n" + "─" * 60)
        print("[TIMER] 效能瓶頸報告")
        print("─" * 60)
        print(f"  {'步驟':<28} {'次數':>5} {'總秒':>8} {'平均秒':>8}")
        print(f"  {'-'*54}")
        for _lbl, _cnt, _tot in [
            ("load_prices",        1,                    _timer_load_prices),
            ("load_features",      1,                    _timer_load_features),
            ("load_labels",        1,                    _timer_load_labels),
            ("train_model",        _count_train_model,   _timer_train_model),
            ("predict",            _count_predict,       _timer_predict),
            ("breakthrough_map",   _count_breakthrough,  _timer_breakthrough),
            ("simulate_period",    _count_simulate,      _timer_simulate),
            ("apply_stoploss",     _count_apply_stoploss,_timer_apply_stoploss),
        ]:
            _avg = _tot / max(_cnt, 1)
            print(f"  {_lbl:<28} {_cnt:>5} {_tot:>8.2f} {_avg:>8.3f}")
        _unaccounted = _total_loop - (
            _timer_load_prices + _timer_load_features + _timer_load_labels
            + _timer_train_model + _timer_predict + _timer_breakthrough + _timer_simulate
        )
        print(f"  {'其他（overhead）':<28} {'—':>5} {_unaccounted:>8.2f}")
        print(f"  {'總 loop 時間':<28} {'—':>5} {_total_loop:>8.2f}")
        print("─" * 60 + "\n")

        # ── 6. 計算總結指標 ──
        if not period_results:
            print("\n[WARN] 無有效回測期間")
            return {"error": "no valid backtest periods"}

        returns = [p["return"] for p in period_results]
        benchmark_returns = [p["benchmark_return"] for p in period_results]

        total_return = equity / 10000 - 1
        benchmark_total = benchmark_equity / 10000 - 1
        n_periods = len(returns)

        # Bug-1 fix：用實際回測期間（日曆天數）計算年化，不再以 n_periods/12 為分母
        # （舊版：n_periods/12 對週頻回測會把 102週 誤算成 8.5 年）
        _ec_start = pd.Timestamp(equity_curve[0]["date"]) if equity_curve else None
        _ec_end = pd.Timestamp(equity_curve[-1]["date"]) if equity_curve else None
        if _ec_start and _ec_end and _ec_end > _ec_start:
            years = (_ec_end - _ec_start).days / 365.25
        else:
            years = max(n_periods / 12, 0.01)  # fallback（月頻回測仍正確）

        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if total_return > -1 else -1
        benchmark_annualized = (1 + benchmark_total) ** (1 / max(years, 0.01)) - 1 if benchmark_total > -1 else -1

        # Max Drawdown
        peak = 10000.0
        max_dd = 0.0
        for ec in equity_curve:
            v = ec["equity"]
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # Sharpe（扣除無風險利率，月化後年化）
        rf_monthly = (1 + risk_free_rate) ** (1 / 12) - 1
        monthly_returns = np.array(returns)
        excess_monthly = monthly_returns - rf_monthly
        if len(excess_monthly) > 1 and monthly_returns.std() > 0:
            sharpe = (excess_monthly.mean() / monthly_returns.std()) * np.sqrt(12)
        else:
            sharpe = 0.0

        # Calmar Ratio（年化報酬 / |最大回撤|）
        calmar = annualized_return / abs(max_dd) if max_dd < 0 else float("inf")

        # 勝率 & 盈虧比
        total_wins = sum(p["wins"] for p in period_results)
        total_losses = sum(p["losses"] for p in period_results)
        total_trades = total_wins + total_losses
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        all_stock_returns = []
        for p in period_results:
            if "stock_returns" in p:
                all_stock_returns.extend(p["stock_returns"].values())

        avg_win = np.mean([r for r in all_stock_returns if r > 0]) if any(r > 0 for r in all_stock_returns) else 0
        avg_loss = abs(np.mean([r for r in all_stock_returns if r <= 0])) if any(r <= 0 for r in all_stock_returns) else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

        summary = {
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized_return, 4),
            "benchmark_total_return": round(benchmark_total, 4),
            "benchmark_annualized_return": round(benchmark_annualized, 4),
            "excess_return": round(total_return - benchmark_total, 4),
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": round(sharpe, 4),
            "calmar_ratio": round(calmar, 4) if calmar != float("inf") else None,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "total_trades": total_trades,
            "total_periods": n_periods,
            "stoploss_triggered": sum(p["stoploss_triggered"] for p in period_results),
            "backtest_start": period_results[0]["rebalance_date"],
            "backtest_end": period_results[-1]["exit_date"],
            # 回測設定紀錄
            "config": {
                "entry_delay_days": entry_delay_days,
                "risk_free_rate": risk_free_rate,
                "benchmark_with_cost": benchmark_with_cost,
                "position_sizing": position_sizing,
                "stoploss_pct": stoploss_pct,
                "trailing_stop_pct": trailing_stop_pct,
                "atr_stoploss_multiplier": atr_stoploss_multiplier,
                "rebalance_freq": rebalance_freq,
            },
        }

        # ── 7. 輸出報告 ──
        print("\n" + "=" * 60)
        print("回測結果摘要")
        print("=" * 60)
        print(f"  回測期間: {summary['backtest_start']} ~ {summary['backtest_end']}")
        print(f"  再平衡次數: {n_periods}  總交易次數: {total_trades}")
        print()
        print(f"  {'指標':<22} {'組合':>12} {'大盤':>12}")
        print(f"  {'-'*46}")
        print(f"  {'累積報酬':<20} {total_return:>11.2%} {benchmark_total:>11.2%}")
        print(f"  {'年化報酬':<20} {annualized_return:>11.2%} {benchmark_annualized:>11.2%}")
        print(f"  {'超額報酬':<20} {total_return - benchmark_total:>11.2%}")
        print(f"  {'最大回撤':<20} {max_dd:>11.2%}")
        print(f"  {f'Sharpe Ratio (rf={risk_free_rate:.1%})':<20} {sharpe:>11.2f}")
        print(f"  {'Calmar Ratio':<20} {calmar:>11.2f}" if calmar != float("inf") else f"  {'Calmar Ratio':<20} {'∞':>11}")
        print(f"  {'勝率':<20} {win_rate:>11.2%}")
        print(f"  {'盈虧比':<20} {profit_factor:>11.2f}")
        print(f"  {'停損觸發次數':<20} {summary['stoploss_triggered']:>11}")
        print()

        print("  月度報酬:")
        for p in period_results:
            ret = p["return"]
            bm = p["benchmark_return"]
            bar = "█" * max(1, int(abs(ret) * 200))
            sign = "+" if ret >= 0 else ""
            color_bar = f"{'↑' if ret >= 0 else '↓'} {bar}"
            print(f"    {p['rebalance_date'][:7]}  {sign}{ret:>7.2%}  (大盤 {bm:>+7.2%})  {color_bar}")

        print("=" * 60)

        # ── 8. quantstats HTML 報告 ──
        try:
            import quantstats as qs
            import matplotlib
            matplotlib.use("Agg")
            from pathlib import Path as _Path

            artifacts_dir = _Path(__file__).resolve().parent.parent / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            report_path = artifacts_dir / "backtest_report.html"

            # 依再平衡頻率設定 periods_per_year，避免 quantstats 預設 252 把週頻/月頻誤算成日頻
            _freq_to_periods = {"W": 52, "M": 12, "D": 252}
            _periods_per_year = _freq_to_periods.get(rebalance_freq.upper(), 52)

            # 建立淨值 Series（以再平衡結束日為索引）
            eq_dates = [ec["date"] for ec in equity_curve]
            eq_vals = [ec["equity"] for ec in equity_curve]
            bm_vals = [ec["equity"] for ec in benchmark_curve]

            eq_series = pd.Series(eq_vals, index=pd.to_datetime(eq_dates), name="strategy")
            bm_series = pd.Series(bm_vals, index=pd.to_datetime(eq_dates), name="benchmark")

            # 轉換為週期報酬（保持原始頻率，不插值成日頻以免失真）
            strat_rets = eq_series.pct_change().dropna()
            bench_rets = bm_series.pct_change().dropna()

            # quantstats 報告：明確指定每年期數，確保 CAGR / Sharpe 正確
            qs.reports.html(
                strat_rets,
                benchmark=bench_rets,
                periods_per_year=_periods_per_year,
                output=str(report_path),
                title=f"Stock Bot Walk-Forward Backtest ({rebalance_freq}, {_periods_per_year}p/yr)",
                download_filename=str(report_path),
            )
            print(f"\n[quantstats] 績效報告已輸出: {report_path}  (periods_per_year={_periods_per_year})")
            summary["quantstats_report"] = str(report_path)
        except Exception as _qs_exc:
            print(f"[quantstats] 報告產出失敗（跳過）: {_qs_exc}")

        return {
            "summary": summary,
            "periods": period_results,
            "equity_curve": equity_curve,
            "benchmark_curve": benchmark_curve,
            "trades_log": all_trades_log,
        }


# ── 向後相容 thin wrapper ───────────────────────────────────────────────────
def run_backtest(
    config,
    db_session: Session,
    backtest_months: int = 24,
    retrain_freq_months: int = 3,
    topn: int = 30,  # Stage 10.1: 20→30，提升 Sharpe / 降 MDD（10y WF 驗證）
    stoploss_pct: float = -0.07,
    transaction_cost_pct: float = 0.00585,
    min_train_days: int = 500,
    min_avg_turnover: float = 0.0,
    eval_start: Optional[date] = None,
    eval_end: Optional[date] = None,
    train_lookback_days: Optional[int] = None,
    entry_delay_days: int = 0,
    risk_free_rate: float = 0.015,
    benchmark_with_cost: bool = True,
    position_sizing: str = "equal",
    position_sizing_method: str = "risk_parity",
    trailing_stop_pct: Optional[float] = None,
    atr_stoploss_multiplier: Optional[float] = None,
    atr_period: int = 14,
    rebalance_freq: str = "M",
    label_horizon_buffer: int = 20,
    enable_slippage: bool = False,
    enable_tiered_slippage: bool = False,
    fast_mode: bool = False,
    feature_columns: Optional[List[str]] = None,
    time_weighting: bool = False,
    enable_complex_filter: bool = False,
    enable_seasonal_filter: bool = False,
    topn_floor: int = 0,
    clip_loss_pct: float = -0.50,
    enable_breakthrough_entry: bool = False,
    breakthrough_max_wait: int = 10,
    momentum_penalty_cols: Optional[Dict[str, float]] = None,
    atr_dynamic_stoploss: bool = False,
    market_filter: bool = False,
    market_filter_tiers: Optional[List[tuple]] = None,
    market_filter_min_positions: int = 1,
    entry_signal_filter: Optional[Dict[str, object]] = None,
    liquidity_weighting: bool = False,
    portfolio_circuit_breaker_pct: Optional[float] = None,
    label_type: str = "abs",
    use_lambdarank: bool = False,
    cross_section_normalize: bool = False,
    ensemble_n_checkpoints: int = 1,
    vol_target_pct: float = 0.0,
    vol_target_lookback_days: int = 60,
    use_stacking: bool = False,
    stacking_val_frac: float = 0.20,
    recent_dd_skip_pct: float = 0.0,
    max_per_sector: int = 0,
    wf_config: Optional["WalkForwardConfig"] = None,
) -> Dict:
    """執行 walk-forward 回測（向後相容 wrapper）。

    內部建立 `WalkForwardConfig` 並委派給 `BacktestPipeline(config, db_session, wf_config).run()`。
    若呼叫端已提供 `wf_config`，則直接沿用；否則用 kwargs 組裝。

    Args:
        config: AppConfig
        db_session: DB session
        wf_config: 型別安全封裝；若提供則優先於 kwargs（建議新程式碼使用）。
        其餘 kwargs 與舊版完全相同，向後相容。

    Returns:
        Dict 包含完整回測結果（summary, periods, equity_curve, benchmark_curve, trades_log）。
    """
    if wf_config is None:
        wf_config = WalkForwardConfig(
            backtest_months=backtest_months,
            retrain_freq_months=retrain_freq_months,
            topn=topn,
            stoploss_pct=stoploss_pct,
            transaction_cost_pct=transaction_cost_pct,
            min_train_days=min_train_days,
            min_avg_turnover=min_avg_turnover,
            eval_start=eval_start,
            eval_end=eval_end,
            train_lookback_days=train_lookback_days,
            entry_delay_days=entry_delay_days,
            risk_free_rate=risk_free_rate,
            benchmark_with_cost=benchmark_with_cost,
            position_sizing=position_sizing,
            position_sizing_method=position_sizing_method,
            trailing_stop_pct=trailing_stop_pct,
            atr_stoploss_multiplier=atr_stoploss_multiplier,
            atr_period=atr_period,
            rebalance_freq=rebalance_freq,
            label_horizon_buffer=label_horizon_buffer,
            enable_slippage=enable_slippage,
            enable_tiered_slippage=enable_tiered_slippage,
            fast_mode=fast_mode,
            clip_loss_pct=clip_loss_pct,
            feature_columns=feature_columns,
            time_weighting=time_weighting,
            liquidity_weighting=liquidity_weighting,
            momentum_penalty_cols=momentum_penalty_cols,
            enable_complex_filter=enable_complex_filter,
            enable_seasonal_filter=enable_seasonal_filter,
            topn_floor=topn_floor,
            enable_breakthrough_entry=enable_breakthrough_entry,
            breakthrough_max_wait=breakthrough_max_wait,
            atr_dynamic_stoploss=atr_dynamic_stoploss,
            market_filter=market_filter,
            market_filter_tiers=market_filter_tiers,
            market_filter_min_positions=market_filter_min_positions,
            entry_signal_filter=entry_signal_filter,
            portfolio_circuit_breaker_pct=portfolio_circuit_breaker_pct,
            label_type=label_type,
            use_lambdarank=use_lambdarank,
            cross_section_normalize=cross_section_normalize,
            ensemble_n_checkpoints=ensemble_n_checkpoints,
            vol_target_pct=vol_target_pct,
            vol_target_lookback_days=vol_target_lookback_days,
            use_stacking=use_stacking,
            stacking_val_frac=stacking_val_frac,
            recent_dd_skip_pct=recent_dd_skip_pct,
            max_per_sector=max_per_sector,
        )
    return BacktestPipeline(config, db_session, wf_config).run()
