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

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Feature, Label, RawPrice
from skills import risk

# ── 模型訓練（複用 train_ranker 邏輯）──
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
from sklearn.ensemble import GradientBoostingRegressor


def _parse_features(series: pd.Series) -> pd.DataFrame:
    import json
    parsed = [json.loads(v) if isinstance(v, str) else (v if isinstance(v, dict) else {}) for v in series]
    return pd.json_normalize(parsed)


def _train_model(train_X: np.ndarray, train_y: np.ndarray, fast_mode: bool = False):
    """訓練一個輕量級模型供回測使用。fast_mode=True 時減少樹數以加速。"""
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
        model.fit(train_X, train_y)
    else:
        n_est_gbr = 100 if fast_mode else 300
        model = GradientBoostingRegressor(
            n_estimators=n_est_gbr, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42,
        )
        model.fit(train_X, train_y)
    return model


def _load_all_data(
    db_session: Session,
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """從 DB 載入 features, labels, prices"""
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
    price_stmt = (
        select(
            RawPrice.stock_id, RawPrice.trading_date,
            RawPrice.open, RawPrice.high, RawPrice.low, RawPrice.close, RawPrice.volume,
        )
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.trading_date, RawPrice.stock_id)
    )
    feat_df = pd.read_sql(feat_stmt, db_session.get_bind())
    label_df = pd.read_sql(label_stmt, db_session.get_bind())
    price_df = pd.read_sql(price_stmt, db_session.get_bind())

    for col in ["open", "high", "low", "close", "volume"]:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

    return feat_df, label_df, price_df


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

    Returns:
        Dict with period results
    """
    if picks.empty:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0}

    stock_ids = picks["stock_id"].tolist()

    period_prices = price_df[
        (price_df["stock_id"].isin(stock_ids)) &
        (price_df["trading_date"] >= entry_date) &
        (price_df["trading_date"] <= exit_date)
    ].copy()

    if period_prices.empty:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0}

    # ── 1. 確定實際進場日（支援延遲 N 個交易日）──
    all_trading_dates = sorted(period_prices["trading_date"].unique())
    if entry_delay_days > 0:
        future_dates = [d for d in all_trading_dates if d > entry_date]
        if len(future_dates) < entry_delay_days:
            return {"return": 0.0, "trades": 0, "stoploss_triggered": 0}
        actual_entry_date = future_dates[entry_delay_days - 1]
    else:
        actual_entry_date = entry_date

    # ── 2. 取進場價（以實際進場日收盤價估算）──
    entry_prices = period_prices[
        period_prices["trading_date"] == actual_entry_date
    ][["stock_id", "close"]].rename(columns={"close": "entry_price"})

    if entry_prices.empty:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0}

    positions_input = entry_prices.assign(
        entry_date=actual_entry_date,
        planned_exit_date=exit_date,
    )[["stock_id", "entry_date", "planned_exit_date", "entry_price"]]

    # ── 3. ATR-based 個股動態停損 ──
    per_stock_stop: Optional[Dict[str, float]] = None
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
    if trailing_stop_pct is not None or atr_stoploss_multiplier is not None:
        stoploss_result = risk.apply_trailing_stop(
            positions_input, price_df=period_prices, 
            trailing_stop_pct=trailing_stop_pct if trailing_stop_pct else -0.15, 
            stoploss_pct=stoploss_pct, 
            per_stock_stoploss=per_stock_stop,
            atr_stoploss_multiplier=atr_stoploss_multiplier,
            atr_at_entry=atr_at_entry
        )
    else:
        stoploss_result = risk.apply_stoploss(
            positions_input, period_prices, stoploss_pct, per_stock_stop
        )

    # ── 預先計算個股滑價（ATR 的 10%，上限 0.3%，進出場各一次）──
    slippage_map: Dict[str, float] = {}
    if enable_slippage and atr_df is not None and not atr_df.empty:
        atr_for_slippage = (
            atr_df[atr_df["trading_date"] < actual_entry_date]
            .groupby("stock_id")["atr"]
            .last()
        )
        for _, ep_row in entry_prices.iterrows():
            sid = str(ep_row["stock_id"])
            ep = float(ep_row["entry_price"])
            if sid in atr_for_slippage.index and ep > 0:
                atr_pct = float(atr_for_slippage[sid]) / ep
                # 單邊滑價 = ATR 10%，上限 0.3%；來回 × 2
                slippage_one_way = min(atr_pct * 0.1, 0.003)
                slippage_map[sid] = slippage_one_way * 2  # 進出場各一次
            else:
                slippage_map[sid] = 0.0

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
                slippage_pct = slippage_map.get(sid, 0.0)
                ret = exit_px / entry_px - 1 - transaction_cost_pct - slippage_pct
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
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0, "trades_log": []}

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
    }


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
    entry_delay_days: int = 1,
    risk_free_rate: float = 0.015,
    benchmark_with_cost: bool = True,
    position_sizing: str = "vol_inverse",
    position_sizing_method: str = "risk_parity",  # vol_inverse | mean_variance | risk_parity
    trailing_stop_pct: Optional[float] = -0.15,
    atr_stoploss_multiplier: Optional[float] = 2.5,
    atr_period: int = 14,
    rebalance_freq: str = "W",
    label_horizon_buffer: int = 7,
    enable_slippage: bool = True,  # 滑價模型：ATR 的 10%，上限 0.3%，來回各一次
    fast_mode: bool = False,  # 加速模式：減少樹數（LightGBM 150 棵）
) -> Dict:
    """執行 walk-forward 回測。

    Args:
        config: AppConfig
        db_session: DB session
        backtest_months: 回測幾個月
        retrain_freq_months: 每幾個月重訓模型
        topn: 每期選幾檔
        stoploss_pct: 全局固定停損比例（如 -0.07）
        transaction_cost_pct: 來回交易成本
        min_train_days: 最低訓練天數
        entry_delay_days: 進場延遲交易日（1=決策日次一交易日執行，更符合實際）
        risk_free_rate: 無風險利率（年化，Sharpe 計算用）
        benchmark_with_cost: Benchmark 是否套用相同交易成本（公平比較）
        position_sizing: 倉位分配方式 equal|score_tiered|vol_inverse
        trailing_stop_pct: 移動停利比例（如 -0.12），None 停用
        atr_stoploss_multiplier: ATR 倍數動態停損（如 2.5），None 使用固定停損
        atr_period: ATR 計算週期（日）
        label_horizon_buffer: 訓練標籤截止日往前預留天數，避免近 rb_date 的標籤使用到未來價格（預設 7）

    Returns:
        Dict 包含完整回測結果
    """
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
    if eval_start is not None and train_lookback_days:
        # Walk-forward 測試可縮小資料抓取範圍，降低 DB I/O 與記憶體壓力
        warmup_days = max(60, atr_period * 3)
        bounded_start = eval_start - timedelta(days=train_lookback_days + warmup_days)
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

    # ── 2. 載入全部資料 ──
    feat_df, label_df, price_df = _load_all_data(db_session, data_start, data_end)
    if feat_df.empty or price_df.empty:
        raise ValueError("資料不足，無法進行回測")

    feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
    label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
    price_df["trading_date"] = pd.to_datetime(price_df["trading_date"]).dt.date

    # ── 2b. 預解析 features_json（一次性展開，避免 training loop 重複 parse）──
    print("  預解析特徵 JSON ...", flush=True)
    _parsed_cols = _parse_features(feat_df["features_json"])
    _parsed_cols = _parsed_cols.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df[["stock_id", "trading_date"]].reset_index(drop=True)
    feat_df = pd.concat([feat_df, _parsed_cols.reset_index(drop=True)], axis=1)
    del _parsed_cols  # 釋放記憶體

    # ── 2c. Schema 遷移保護：過濾掉 feature 覆蓋率不足的舊版資料 ──
    # 根因：730d 重算後 DB 存在 2016-2024 的 19-feature 舊 schema（3.14M 行）
    # 與 2024-2026 的 48-feature 新 schema（978k 行）。混合訓練時舊行 76% 欄位為
    # NaN，以 median imputation 填補會嚴重污染模型。
    # 門檻 50%：舊行 19/59≈32% < 50% → 過濾；新行 48/59≈81% > 50% → 保留。
    _feat_cols_in_df = [c for c in feat_df.columns if c not in ("stock_id", "trading_date")]
    if _feat_cols_in_df:
        _schema_threshold = max(1, int(len(_feat_cols_in_df) * 0.50))
        _valid_schema_mask = feat_df[_feat_cols_in_df].notna().sum(axis=1) >= _schema_threshold
        _n_schema_dropped = int((~_valid_schema_mask).sum())
        if _n_schema_dropped > 0:
            print(
                f"  [schema filter] 過濾 {_n_schema_dropped:,} 筆舊版特徵資料 "
                f"(threshold={_schema_threshold}/{len(_feat_cols_in_df)} features)",
                flush=True,
            )
            feat_df = feat_df.loc[_valid_schema_mask].reset_index(drop=True)

    # ── 3. 預計算 ATR（若需要）──
    atr_df: Optional[pd.DataFrame] = None
    if atr_stoploss_multiplier is not None or position_sizing == "vol_inverse":
        print("  預計算 ATR ...", flush=True)
        atr_df = risk.compute_atr(price_df, period=atr_period)
    liquidity_eligible_map = _precompute_liquidity_eligible_map(price_df, min_avg_turnover)
    market_median_ret20_map = _precompute_market_median_ret20(price_df)
    market_weekly_drop_map = _precompute_market_weekly_drop(price_df)
    market_200ma_bear_map = _precompute_market_200ma_bear(price_df)

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
            y = merged["future_ret_h"].astype(float).values
            current_model = _train_model(fmat.values, y, fast_mode=fast_mode)
            last_train_date = rb_date
            print(f"  [{rb_date}] 模型重訓完成 (訓練筆數: {len(y):,})", flush=True)

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

        scores = current_model.predict(fmat.values)
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
        median_ret = market_median_ret20_map.get(rb_date)
        weekly_drop = market_weekly_drop_map.get(rb_date)

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
        seasonal_weak = getattr(config, "seasonal_weak_months", (3, 10))
        seasonal_mult = getattr(config, "seasonal_topn_multiplier", 0.5)
        if rb_date.month in seasonal_weak:
            new_topn = max(1, int(effective_topn * seasonal_mult))
            if new_topn < effective_topn:
                print(f"  [{rb_date}] 弱勢月份 {rb_date.month}月，topN {effective_topn} → {new_topn}")
                effective_topn = new_topn

        # ── topN 絕對下限保護（避免危機+弱勢月份疊加造成過度集中）──
        # 極端空頭（週跌>10%）最低 3 支；一般情況最低 5 支
        _extreme_bear = weekly_drop is not None and weekly_drop < -0.10
        _min_topn = 3 if _extreme_bear else 5
        if effective_topn < _min_topn:
            print(f"  [{rb_date}] [topN-floor] {effective_topn}→{_min_topn} (min_topn={_min_topn})")
            effective_topn = _min_topn

        # ── 判斷是否為空頭環境（供防禦過濾使用）──
        _is_bear_env = (
            (weekly_drop is not None and weekly_drop < crisis_threshold) or
            (median_ret is not None and median_ret < -0.03)
        )

        # ── 空頭防禦①：RSI 過熱過濾（空頭時不追高 RSI>70 強勢股）──
        if _is_bear_env and "rsi_14" in day_feat.columns:
            _rsi_before = len(day_feat)
            day_feat = day_feat[day_feat["rsi_14"].fillna(0) <= 70]
            _rsi_removed = _rsi_before - len(day_feat)
            if _rsi_removed > 0:
                print(f"  [{rb_date}] 空頭RSI過濾: 移除{_rsi_removed}檔(rsi>70)")
            if day_feat.empty:
                continue

        # ── 空頭防禦②：低波動加權（atr_inv z-score 加分 0.3 倍）──
        if _is_bear_env and "atr_inv" in day_feat.columns:
            _atr_inv = day_feat["atr_inv"]
            _atr_std = float(_atr_inv.std())
            if _atr_std > 0:
                day_feat = day_feat.copy()
                day_feat["score"] = (
                    day_feat["score"] + 0.3 * (_atr_inv - _atr_inv.mean()) / _atr_std
                )

        # ── 現金保留機制（大盤跌破200日均線保留30%現金）──
        _cash_ratio = 0.30 if market_200ma_bear_map.get(rb_date, False) else 0.0

        picks = risk.pick_topn(day_feat, effective_topn)

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

        # ── 模擬持有 ──
        result = _simulate_period(
            picks, price_df, rb_date, exit_date,
            stoploss_pct, transaction_cost_pct,
            entry_delay_days=entry_delay_days,
            position_weights=pos_weights,
            trailing_stop_pct=trailing_stop_pct,
            atr_df=atr_df,
            atr_stoploss_multiplier=atr_stoploss_multiplier,
            enable_slippage=enable_slippage,
        )

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
        print(
            f"  [{rb_date} ~ {exit_date}] "
            f"組合: {sign}{period_ret:.2%}  大盤: {bm_sign}{benchmark_ret:.2%}  "
            f"持股: {result['trades']}  停損: {result['stoploss_triggered']}  "
            f"淨值: {equity:,.0f}",
            flush=True,
        )

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
