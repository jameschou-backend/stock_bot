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

import gc
import os
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Feature, Label, RawPrice
from skills import data_store, risk
from skills.breakthrough import (
    precompute_stats as _precompute_breakthrough_stats,
    compute_breakthrough_map as _compute_breakthrough_map,
)
from skills.feature_utils import (
    parse_features_json as _parse_features_json_shared,
    impute_features as _impute_features_shared,
    filter_schema_valid_rows as _filter_schema_valid_rows,
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
):
    """訓練一個輕量級模型供回測使用。fast_mode=True 時減少樹數以加速。
    sample_weight 支援時間加權（近期樣本權重更高）。
    """
    n_est = 150 if fast_mode else 500
    if _HAS_LGBM:
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
    topn: int = 20
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


def run_backtest(
    config,
    db_session: Session,
    backtest_months: int = 24,
    retrain_freq_months: int = 3,
    topn: int = 20,
    stoploss_pct: float = -0.07,
    transaction_cost_pct: float = 0.00585,
    min_train_days: int = 500,
    min_avg_turnover: float = 0.0,
    eval_start: Optional[date] = None,
    eval_end: Optional[date] = None,
    train_lookback_days: Optional[int] = None,
    # ── 新增參數 ──
    entry_delay_days: int = 0,           # 原始基準：當日收盤進場（0）
    risk_free_rate: float = 0.015,
    benchmark_with_cost: bool = True,
    position_sizing: str = "equal",      # 原始基準：等權重
    position_sizing_method: str = "risk_parity",  # vol_inverse | mean_variance | risk_parity
    trailing_stop_pct: Optional[float] = None,    # 原始基準：無移動停利
    atr_stoploss_multiplier: Optional[float] = None,  # 原始基準：無 ATR 停損
    atr_period: int = 14,
    rebalance_freq: str = "M",
    label_horizon_buffer: int = 20,      # label horizon = 20 交易日，訓練截止往前 20 天避免標籤洩漏
    enable_slippage: bool = False,       # 原始基準：無滑價模型
    enable_tiered_slippage: bool = False,  # 分級滑價：依 amt_20 決定流動性層級（小型股更高滑價）
    fast_mode: bool = False,  # 加速模式：減少樹數（LightGBM 150 棵）
    # ── 實驗參數（10y 逐步優化用）──
    feature_columns: Optional[List[str]] = None,  # None=用 DB 所有特徵；指定時只用列出的欄位
    time_weighting: bool = False,        # 原始基準：等權樣本，不強調近期
    enable_complex_filter: bool = False, # 原始基準：無季節/RSI/200MA/空頭縮 topN
    enable_seasonal_filter: bool = False, # 獨立季節性降倉旗標（不啟用其他複雜過濾）
    topn_floor: int = 0,  # 0=不強制下限；>0 時 effective_topn 不低於此值（Change B 用）
    clip_loss_pct: float = -0.50,  # 單筆最大損失 clip（預設 -50%）；診斷用可傳 -1.01 停用
    enable_breakthrough_entry: bool = False,  # 突破確認進場：等待訊號後才進場
    breakthrough_max_wait: int = 10,         # 最多等待幾個交易日出現突破訊號
    momentum_penalty_cols: Optional[Dict[str, float]] = None,  # 動能懲罰：{col: scale}，訓練/預測前對指定特徵乘以 scale
    atr_dynamic_stoploss: bool = False,  # ATR 動態停損：低波動股 -15%，高波動股 -25%（以 atr_inv 中位數分界）
    market_filter: bool = False,  # 大盤過濾：前期大盤月跌>5% 持股減半，>10% 全現金
    market_filter_tiers: Optional[List[tuple]] = None,  # 漸進式大盤過濾：[(threshold, multiplier), ...] 由淺到深排序，如 [(-0.05,0.5),(-0.10,0.25),(-0.15,0.10)]
    market_filter_min_positions: int = 1,  # 大盤過濾後最低持股數（防止單押集中風險）
    entry_signal_filter: Optional[Dict[str, object]] = None,  # 進場訊號過濾：{"foreign_buy_streak_max":3, "rsi_min":45, "rsi_max":70, "bias_20_max":0.15, "volume_surge_ratio_min":1.0}
    liquidity_weighting: bool = False,   # 流動性加權訓練：sample_weight ∝ log(1+amt_20)，讓模型學偏大型股模式
    wf_config: Optional["WalkForwardConfig"] = None,  # 型別安全封裝（優先於上方 kwargs）
) -> Dict:
    """執行 walk-forward 回測。

    Args:
        config: AppConfig
        db_session: DB session
        wf_config: WalkForwardConfig 物件，若提供則覆蓋下方所有 kwargs（建議新程式碼使用）。
        （其餘 kwargs 為向後相容保留，當 wf_config=None 時生效）

    Returns:
        Dict 包含完整回測結果
    """
    # ── wf_config 覆蓋 kwargs（向後相容）──
    if wf_config is not None:
        backtest_months        = wf_config.backtest_months
        retrain_freq_months    = wf_config.retrain_freq_months
        topn                   = wf_config.topn
        stoploss_pct           = wf_config.stoploss_pct
        transaction_cost_pct   = wf_config.transaction_cost_pct
        min_train_days         = wf_config.min_train_days
        min_avg_turnover       = wf_config.min_avg_turnover
        eval_start             = wf_config.eval_start
        eval_end               = wf_config.eval_end
        train_lookback_days    = wf_config.train_lookback_days
        entry_delay_days       = wf_config.entry_delay_days
        risk_free_rate         = wf_config.risk_free_rate
        benchmark_with_cost    = wf_config.benchmark_with_cost
        position_sizing        = wf_config.position_sizing
        position_sizing_method = wf_config.position_sizing_method
        trailing_stop_pct      = wf_config.trailing_stop_pct
        atr_stoploss_multiplier= wf_config.atr_stoploss_multiplier
        atr_period             = wf_config.atr_period
        rebalance_freq         = wf_config.rebalance_freq
        label_horizon_buffer   = wf_config.label_horizon_buffer
        enable_slippage        = wf_config.enable_slippage
        enable_tiered_slippage = wf_config.enable_tiered_slippage
        fast_mode              = wf_config.fast_mode
        clip_loss_pct          = wf_config.clip_loss_pct
        feature_columns        = wf_config.feature_columns
        time_weighting         = wf_config.time_weighting
        liquidity_weighting    = wf_config.liquidity_weighting
        momentum_penalty_cols  = wf_config.momentum_penalty_cols
        enable_complex_filter  = wf_config.enable_complex_filter
        enable_seasonal_filter = wf_config.enable_seasonal_filter
        topn_floor             = wf_config.topn_floor
        enable_breakthrough_entry = wf_config.enable_breakthrough_entry
        breakthrough_max_wait  = wf_config.breakthrough_max_wait
        atr_dynamic_stoploss   = wf_config.atr_dynamic_stoploss
        market_filter          = wf_config.market_filter
        market_filter_tiers    = wf_config.market_filter_tiers
        market_filter_min_positions = wf_config.market_filter_min_positions
        entry_signal_filter    = wf_config.entry_signal_filter

    print("\n" + "=" * 60)
    print("Walk-Forward Backtest")
    print("=" * 60)

    # ── 1. 確認可用資料範圍 ──
    max_feat_date = db_session.query(func.max(Feature.trading_date)).scalar()
    min_feat_date = db_session.query(func.min(Feature.trading_date)).scalar()
    max_label_date = db_session.query(func.max(Label.trading_date)).scalar()

    if max_feat_date is None or max_label_date is None:
        raise ValueError("features 或 labels 表為空，請先跑 pipeline-build")

    data_end = min(max_feat_date, max_label_date)
    if eval_end is not None:
        data_end = min(data_end, eval_end)
    backtest_start = eval_start if eval_start is not None else data_end - timedelta(days=30 * backtest_months)
    data_start = min_feat_date
    if train_lookback_days:
        # 有滾動訓練視窗：以 backtest_start 為錨點往前推，縮小資料抓取範圍
        warmup_days = max(60, atr_period * 3)
        _anchor = eval_start if eval_start is not None else backtest_start
        bounded_start = _anchor - timedelta(days=train_lookback_days + warmup_days)
        data_start = max(min_feat_date, bounded_start)

    exit_strategy = "trailing" if trailing_stop_pct is not None else (
        f"ATR×{atr_stoploss_multiplier}" if atr_stoploss_multiplier is not None else f"fixed {stoploss_pct:.0%}"
    )
    print(f"  資料範圍: {data_start} ~ {data_end}")
    print(f"  回測期間: {backtest_start} ~ {data_end}")
    print(f"  模型重訓: 每 {retrain_freq_months} 個月")
    print(f"  選股數量: {topn}  倉位: {position_sizing}")
    print(f"  停損策略: {exit_strategy}" + (f"  移動停利: {trailing_stop_pct:.0%}" if trailing_stop_pct else ""))
    print(f"  交易成本: {transaction_cost_pct:.3%}（來回）  進場延遲: {entry_delay_days} 交易日")
    print(f"  無風險利率: {risk_free_rate:.1%}  Benchmark含成本: {benchmark_with_cost}")
    if train_lookback_days:
        print(f"  訓練窗長: {train_lookback_days} 日")
    print()

    # ── 2. 載入資料（DuckDB parquet cache via data_store）──
    # 第一次執行：MySQL 全量載入 → parquet（features 含 JSON 解析 + float32 壓縮）
    # 後續執行：DuckDB predicate pushdown 直接讀 parquet，無 MySQL 往返
    # TTL: 24 小時，超時自動重建
    _use_rolling_window = train_lookback_days is not None

    # ── 載入價格資料 ──
    _log("load_prices start")
    _t = time.time()
    price_df = data_store.get_prices(db_session, data_start, data_end)
    _timer_load_prices = time.time() - _t
    _log(f"load_prices done {_timer_load_prices:.1f}s")

    if price_df.empty:
        raise ValueError("資料不足，無法進行回測")

    # ── 預熱 features/labels 快取（若尚未建立或已失效）──
    # _ensure() 在 get_features/get_labels 內自動呼叫；
    # 此處提前呼叫是為了讓 timer 能分開計量 price vs feature 建立時間
    _timer_load_features = 0.0
    _timer_load_labels = 0.0

    # 非滾動視窗：一次用 DuckDB 讀全量（訓練資料隨時間累積增長）
    # 滾動視窗：佔位符，每 fold 在迴圈內用 DuckDB 按日期範圍讀取
    if not _use_rolling_window:
        _log("load_all_features start (non-rolling, DuckDB)")
        _t = time.time()
        feat_df = data_store.get_features(db_session, data_start, data_end, feature_columns)
        label_df = data_store.get_labels(db_session, data_start, data_end)
        _timer_load_features = time.time() - _t
        _log(f"load_all_features done {_timer_load_features:.1f}s")
    else:
        feat_df = pd.DataFrame()
        label_df = pd.DataFrame()
        _log(f"rolling_window_mode: per-fold DuckDB load (window={train_lookback_days}d)")

    # ── 3. 預計算 ATR（若需要）──
    atr_df: Optional[pd.DataFrame] = None
    if atr_stoploss_multiplier is not None or position_sizing == "vol_inverse" or enable_slippage:
        print("  預計算 ATR ...", flush=True)
        atr_df = risk.compute_atr(price_df, period=atr_period)
    liquidity_eligible_map = _precompute_liquidity_eligible_map(price_df, min_avg_turnover)
    market_median_ret20_map = _precompute_market_median_ret20(price_df)
    market_weekly_drop_map = _precompute_market_weekly_drop(price_df)
    market_200ma_bear_map = _precompute_market_200ma_bear(price_df)

    # ── 3b. 突破進場：一次性預計算所有股票的 rolling 突破指標 ──
    # 避免在主迴圈中每個再平衡期重算（117 次 rolling → 1 次）
    bt_stats_df: Optional[pd.DataFrame] = None
    if enable_breakthrough_entry:
        print("  預計算突破 rolling 指標（close_max_20 / vol_avg_20 / ma_20）...", flush=True)
        bt_stats_df = _precompute_breakthrough_stats(price_df, lookback=20)

    # ── 4. 找出回測期間的再平衡日 ──
    bt_trading_dates = sorted(price_df[price_df["trading_date"] >= backtest_start]["trading_date"].unique())
    rebalance_dates = _get_rebalance_dates(bt_trading_dates, freq=rebalance_freq)
    print(f"  再平衡次數: {len(rebalance_dates)} (頻率: {rebalance_freq})")

    # ── 5. Walk-forward 執行 ──
    current_model = None
    current_feature_names = None
    last_train_date = None
    period_results: List[Dict] = []
    all_trades_log: List[Dict] = []
    cooldown_until: Dict[str, date] = {}  # stock_id -> 冷卻截止日（停損後 N 週不再選入）
    equity = 10000.0
    equity_curve = [{"date": rebalance_dates[0].isoformat() if rebalance_dates else data_start.isoformat(), "equity": equity}]

    benchmark_equity = 10000.0
    benchmark_curve = [{"date": equity_curve[0]["date"], "equity": benchmark_equity}]
    benchmark_tc = transaction_cost_pct if benchmark_with_cost else 0.0

    # ── 滾動視窗狀態 ──
    _rw_feat_df: Optional[pd.DataFrame] = None
    _rw_label_df: Optional[pd.DataFrame] = None
    _rw_range: Optional[Tuple[date, date]] = None

    # ── 計時器累計 ──
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
                _rw_feat_df = data_store.get_features(db_session, _win_start, _win_end, feature_columns)
                _rw_label_df = data_store.get_labels(db_session, _win_start, _win_end)
                _rw_range = (_win_start, _win_end)
                _log(f"fold_loaded feat:{len(_rw_feat_df):,} dt={time.time()-_t_fold:.1f}s")
            feat_df = _rw_feat_df
            label_df = _rw_label_df

        if need_retrain:
            label_cutoff = rb_date - timedelta(days=label_horizon_buffer)
            train_feat = feat_df[feat_df["trading_date"] < rb_date]
            train_label = label_df[label_df["trading_date"] < label_cutoff]
            if train_lookback_days:
                train_start = rb_date - timedelta(days=train_lookback_days)
                train_feat = train_feat[train_feat["trading_date"] >= train_start]
                train_label = train_label[train_label["trading_date"] >= train_start]

            if train_feat.empty or train_label.empty:
                print(f"  [{rb_date}] 訓練資料不足，跳過")
                continue

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
                continue

            # feat_df 已預解析（2b 步驟），直接取特徵欄位（排除 meta 欄與 label）
            _meta_cols = {"stock_id", "trading_date", "future_ret_h"}
            fmat = merged.drop(columns=[c for c in _meta_cols if c in merged.columns])
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
                continue

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

            _t = time.time()
            current_model = _train_model(fmat.values, y, fast_mode=fast_mode, sample_weight=_sample_weight)
            _dt = time.time() - _t
            _timer_train_model += _dt
            _count_train_model += 1
            last_train_date = rb_date
            print(f"  [{rb_date}] 模型重訓完成 (訓練筆數: {len(y):,}) [TIMER] train_model: {_dt:.2f}s", flush=True)

        if current_model is None:
            continue

        # ── 對當日股票評分 ──
        day_feat = feat_df[feat_df["trading_date"] == rb_date].copy()
        day_feat = day_feat[day_feat["stock_id"].str.fullmatch(r"\d{4}")]

        if day_feat.empty:
            for fallback in range(1, 6):
                fb_date = rb_date - timedelta(days=fallback)
                day_feat = feat_df[feat_df["trading_date"] == fb_date].copy()
                day_feat = day_feat[day_feat["stock_id"].str.fullmatch(r"\d{4}")]
                if not day_feat.empty:
                    break

        if day_feat.empty:
            continue

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
        if momentum_penalty_cols:
            for _pc, _ps in momentum_penalty_cols.items():
                if _pc in fmat.columns:
                    fmat[_pc] = fmat[_pc] * _ps

        _t = time.time()
        scores = current_model.predict(fmat.values)
        _dt = time.time() - _t
        _timer_predict += _dt
        _count_predict += 1
        day_feat = day_feat.reset_index(drop=True)
        day_feat["score"] = scores

        # 流動性過濾
        if min_avg_turnover > 0:
            keep_ids = liquidity_eligible_map.get(rb_date, set())
            day_feat = day_feat[day_feat["stock_id"].astype(str).isin(keep_ids)]
            if day_feat.empty:
                continue

        # ── 停損冷卻過濾 ──
        if cooldown_until:
            excluded = {sid for sid, expiry in cooldown_until.items() if expiry > rb_date}
            if excluded:
                before_n = len(day_feat)
                day_feat = day_feat[~day_feat["stock_id"].astype(str).isin(excluded)]
                n_excl = before_n - len(day_feat)
                if n_excl > 0:
                    print(f"  [{rb_date}] 停損冷卻：排除 {n_excl} 檔")

        # ── 大盤環境濾網 (Market Regime Filter) ──
        effective_topn = topn
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
                        continue

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

        # ── 進場訊號過濾（entry_signal_filter）──
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
        else:
            picks = risk.pick_topn(day_feat, effective_topn)

        # ── 突破確認進場（Breakthrough Entry Filter）──
        # 月底選股後不立即進場，等每檔個股出現突破訊號（最多等 breakthrough_max_wait 個交易日）。
        # 無訊號者以後排候選補位；仍無訊號者持現金。
        # 使用向量化批次計算，避免逐股逐日掃描 price_df（效能 O(n_stocks × window)）。
        per_stock_entry_dates: Dict[str, date] = {}
        if enable_breakthrough_entry:
            # 取再平衡日後的交易日視窗（最多 breakthrough_max_wait 天）
            _bt_window = [
                d for d in bt_trading_dates if d > rb_date
            ][:breakthrough_max_wait]

            if _bt_window:
                # 擴展候補池（從 day_feat 按分數排序）
                _slots_needed = effective_topn
                _ext_n = min(_slots_needed * 3, len(day_feat)) if _slots_needed > 0 else 0

                _confirmed_rows: List[Dict] = []
                _entry_map: Dict[str, date] = {}
                _orig_topn_sids = set(picks["stock_id"].astype(str).tolist())

                if _ext_n > 0:
                    _extended = risk.pick_topn(day_feat, _ext_n)
                    _ext_sids = _extended["stock_id"].astype(str).tolist()

                    # 向量化一次算出所有新候選股的突破日（使用預計算的 rolling stats）
                    _t = time.time()
                    _bt_map = _compute_breakthrough_map(
                        _ext_sids, _bt_window, bt_stats_df, feat_df
                    )
                    _dt = time.time() - _t
                    _timer_breakthrough += _dt
                    _count_breakthrough += 1

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

        # ── 計算倉位權重（支援 position_sizing_method）──
        period_price_slice = price_df[price_df["trading_date"] <= rb_date]
        # position_sizing_method 優先（支援 mean_variance / risk_parity）
        _ps_method = position_sizing_method if position_sizing_method != "vol_inverse" else position_sizing
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
                    atr_period=atr_period,
                )
        else:
            pos_weights = risk.compute_position_weights(
                picks, method=position_sizing,
                price_df=period_price_slice if position_sizing == "vol_inverse" else None,
                atr_period=atr_period,
            )

        # ── 大盤過濾（market_filter / market_filter_tiers）：根據前期大盤報酬調整持倉 ──
        _market_filter_skip = False
        _mf_multiplier = 1.0  # 記錄實際持倉倍率供 period_results 保存
        if len(period_results) > 0 and (market_filter or market_filter_tiers):
            _prev_bm = period_results[-1].get("benchmark_return", 0.0)
            if market_filter_tiers:
                # 漸進式：按 tiers 從深到淺匹配（tiers 須由淺到深排序）
                _applied = False
                for _thr, _mult in reversed(market_filter_tiers):
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
            elif market_filter:
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
            )
            _dt = time.time() - _t
            _timer_simulate += _dt
            _count_simulate += 1
            _timer_apply_stoploss += result.get("_stoploss_time", 0.0)
            _count_apply_stoploss += 1

        # ── 大盤基準（等權，向量化計算，不設停損／無滑價，與策略套用相同流動性門檻）──
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
