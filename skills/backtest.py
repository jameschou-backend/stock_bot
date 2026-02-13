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


def _train_model(train_X: np.ndarray, train_y: np.ndarray):
    """訓練一個輕量級模型供回測使用"""
    if _HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=500,
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
        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
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
        select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close, RawPrice.volume)
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.trading_date, RawPrice.stock_id)
    )
    feat_df = pd.read_sql(feat_stmt, db_session.get_bind())
    label_df = pd.read_sql(label_stmt, db_session.get_bind())
    price_df = pd.read_sql(price_stmt, db_session.get_bind())

    for col in ["close", "volume"]:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

    return feat_df, label_df, price_df


def _get_rebalance_dates(trading_dates: List[date]) -> List[date]:
    """找出每月第一個交易日作為再平衡日"""
    dates = sorted(trading_dates)
    rebalance = []
    prev_month = None
    for d in dates:
        ym = (d.year, d.month)
        if ym != prev_month:
            rebalance.append(d)
            prev_month = ym
    return rebalance


def _simulate_period(
    picks: pd.DataFrame,
    price_df: pd.DataFrame,
    entry_date: date,
    exit_date: date,
    stoploss_pct: float,
    transaction_cost_pct: float,
) -> Dict:
    """模擬一個持有期間的績效。

    Args:
        picks: DataFrame with 'stock_id' and 'score'
        price_df: full price data
        entry_date: 進場日（以收盤價買入）
        exit_date: 預計出場日（以收盤價賣出）
        stoploss_pct: 停損比例（如 -0.07）
        transaction_cost_pct: 來回交易成本

    Returns:
        Dict with period results
    """
    if picks.empty:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0}

    stock_ids = picks["stock_id"].tolist()
    n = len(stock_ids)

    period_prices = price_df[
        (price_df["stock_id"].isin(stock_ids)) &
        (price_df["trading_date"] >= entry_date) &
        (price_df["trading_date"] <= exit_date)
    ].copy()

    if period_prices.empty:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0}

    # 取得進場價（entry_date 的收盤價）
    entry_prices = period_prices[period_prices["trading_date"] == entry_date][["stock_id", "close"]]
    entry_prices = entry_prices.rename(columns={"close": "entry_price"})

    if entry_prices.empty:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0}

    # 逐日檢查停損
    stoploss_count = 0
    stock_returns = {}
    trading_dates = sorted(period_prices["trading_date"].unique())

    for sid in stock_ids:
        ep = entry_prices[entry_prices["stock_id"] == sid]
        if ep.empty:
            continue
        entry_px = float(ep["entry_price"].iloc[0])
        if entry_px <= 0:
            continue

        stock_prices = period_prices[period_prices["stock_id"] == sid].sort_values("trading_date")
        exit_px = entry_px  # default
        stopped = False

        for _, row in stock_prices.iterrows():
            if row["trading_date"] == entry_date:
                continue  # 進場日不檢查停損
            current_ret = float(row["close"]) / entry_px - 1
            if stoploss_pct < 0 and current_ret <= stoploss_pct:
                exit_px = float(row["close"])
                stoploss_count += 1
                stopped = True
                break
            exit_px = float(row["close"])

        ret = exit_px / entry_px - 1 - transaction_cost_pct
        stock_returns[sid] = ret

    if not stock_returns:
        return {"return": 0.0, "trades": 0, "stoploss_triggered": 0}

    # 等權重
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
) -> Dict:
    """執行 walk-forward 回測。

    Args:
        config: AppConfig
        db_session: DB session
        backtest_months: 回測幾個月
        retrain_freq_months: 每幾個月重訓模型
        topn: 每期選幾檔
        stoploss_pct: 停損比例
        transaction_cost_pct: 來回交易成本
        min_train_days: 最低訓練天數

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

    print(f"  資料範圍: {data_start} ~ {data_end}")
    print(f"  回測期間: {backtest_start} ~ {data_end}")
    print(f"  模型重訓: 每 {retrain_freq_months} 個月")
    print(f"  選股數量: {topn}")
    print(f"  成交值門檻: {min_avg_turnover} 億元")
    print(f"  停損比例: {stoploss_pct:.1%}")
    print(f"  交易成本: {transaction_cost_pct:.3%}")
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

    # ── 3. 找出回測期間的再平衡日 ──
    bt_trading_dates = sorted(price_df[price_df["trading_date"] >= backtest_start]["trading_date"].unique())
    rebalance_dates = _get_rebalance_dates(bt_trading_dates)
    print(f"  再平衡次數: {len(rebalance_dates)}")

    # ── 4. Walk-forward 執行 ──
    current_model = None
    current_feature_names = None
    last_train_date = None
    period_results: List[Dict] = []
    equity = 10000.0
    equity_curve = [{"date": rebalance_dates[0].isoformat() if rebalance_dates else data_start.isoformat(), "equity": equity}]

    # 大盤基準：等權平均
    benchmark_equity = 10000.0
    benchmark_curve = [{"date": equity_curve[0]["date"], "equity": benchmark_equity}]

    for i, rb_date in enumerate(rebalance_dates):
        # 決定退出日（下一個再平衡日前一天，或最後一天）
        if i + 1 < len(rebalance_dates):
            exit_date = rebalance_dates[i + 1] - timedelta(days=1)
            # 確保 exit_date 是交易日
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
            # 用 rb_date 之前的資料訓練
            train_feat = feat_df[feat_df["trading_date"] < rb_date]
            train_label = label_df[label_df["trading_date"] < rb_date]
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

            fmat = _parse_features(merged["features_json"])
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
            current_model = _train_model(fmat.values, y)
            last_train_date = rb_date
            print(f"  [{rb_date}] 模型重訓完成 (訓練筆數: {len(y):,})", flush=True)

        if current_model is None:
            continue

        # ── 對當日股票評分 ──
        day_feat = feat_df[feat_df["trading_date"] == rb_date].copy()
        # 只選四碼股票
        day_feat = day_feat[day_feat["stock_id"].str.fullmatch(r"\d{4}")]

        if day_feat.empty:
            # 回退找前幾天
            for fallback in range(1, 6):
                fb_date = rb_date - timedelta(days=fallback)
                day_feat = feat_df[feat_df["trading_date"] == fb_date].copy()
                day_feat = day_feat[day_feat["stock_id"].str.fullmatch(r"\d{4}")]
                if not day_feat.empty:
                    break

        if day_feat.empty:
            continue

        fmat = _parse_features(day_feat["features_json"])
        # 補齊缺失欄位
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

        # 流動性過濾：20 日平均成交值（億元）
        if min_avg_turnover > 0:
            threshold = min_avg_turnover * 1e8
            recent_prices = (
                price_df[price_df["trading_date"] <= rb_date]
                .sort_values(["stock_id", "trading_date"])
                .groupby("stock_id")
                .tail(20)
                .copy()
            )
            recent_prices["turnover"] = recent_prices["close"] * recent_prices["volume"]
            avg_turnover = recent_prices.groupby("stock_id")["turnover"].mean()
            keep_ids = set(avg_turnover[avg_turnover >= threshold].index.astype(str))
            day_feat = day_feat[day_feat["stock_id"].astype(str).isin(keep_ids)]
            if day_feat.empty:
                continue

        picks = day_feat.sort_values("score", ascending=False).head(topn)

        # ── 模擬持有 ──
        result = _simulate_period(picks, price_df, rb_date, exit_date, stoploss_pct, transaction_cost_pct)

        # 大盤基準（等權所有股票）
        all_stocks_on_date = price_df[price_df["trading_date"] == rb_date]["stock_id"].unique()
        benchmark_result = _simulate_period(
            pd.DataFrame({"stock_id": all_stocks_on_date, "score": 0}),
            price_df, rb_date, exit_date, 0, 0,  # 基準不設停損和成本
        )

        period_ret = result["return"]
        benchmark_ret = benchmark_result["return"]
        equity *= (1 + period_ret)
        benchmark_equity *= (1 + benchmark_ret)

        period_results.append({
            "rebalance_date": rb_date.isoformat(),
            "exit_date": exit_date.isoformat(),
            "return": period_ret,
            "benchmark_return": benchmark_ret,
            "excess_return": period_ret - benchmark_ret,
            "trades": result["trades"],
            "wins": result.get("wins", 0),
            "losses": result.get("losses", 0),
            "stoploss_triggered": result["stoploss_triggered"],
            "equity": equity,
            "benchmark_equity": benchmark_equity,
        })
        equity_curve.append({"date": exit_date.isoformat(), "equity": equity})
        benchmark_curve.append({"date": exit_date.isoformat(), "equity": benchmark_equity})

        sign = "+" if period_ret >= 0 else ""
        bm_sign = "+" if benchmark_ret >= 0 else ""
        print(
            f"  [{rb_date} ~ {exit_date}] "
            f"組合: {sign}{period_ret:.2%}  大盤: {bm_sign}{benchmark_ret:.2%}  "
            f"持股: {result['trades']}  停損: {result['stoploss_triggered']}  "
            f"淨值: {equity:,.0f}",
            flush=True,
        )

    # ── 5. 計算總結指標 ──
    if not period_results:
        print("\n[WARN] 無有效回測期間")
        return {"error": "no valid backtest periods"}

    returns = [p["return"] for p in period_results]
    benchmark_returns = [p["benchmark_return"] for p in period_results]
    excess_returns = [p["excess_return"] for p in period_results]

    total_return = equity / 10000 - 1
    benchmark_total = benchmark_equity / 10000 - 1
    n_periods = len(returns)
    years = n_periods / 12  # 近似（月度再平衡）

    annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if total_return > -1 else -1
    benchmark_annualized = (1 + benchmark_total) ** (1 / max(years, 0.01)) - 1 if benchmark_total > -1 else -1

    # Max Drawdown
    peak = 10000
    max_dd = 0
    for ec in equity_curve:
        v = ec["equity"]
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Sharpe (月化報酬 → 年化)
    monthly_returns = np.array(returns)
    if len(monthly_returns) > 1 and monthly_returns.std() > 0:
        sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
    else:
        sharpe = 0.0

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
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_trades": total_trades,
        "total_periods": n_periods,
        "stoploss_triggered": sum(p["stoploss_triggered"] for p in period_results),
        "backtest_start": period_results[0]["rebalance_date"],
        "backtest_end": period_results[-1]["exit_date"],
    }

    # ── 6. 輸出報告 ──
    print("\n" + "=" * 60)
    print("回測結果摘要")
    print("=" * 60)
    print(f"  回測期間: {summary['backtest_start']} ~ {summary['backtest_end']}")
    print(f"  再平衡次數: {n_periods}")
    print(f"  總交易次數: {total_trades}")
    print()
    print(f"  {'指標':<20} {'組合':>12} {'大盤':>12}")
    print(f"  {'-'*44}")
    print(f"  {'累積報酬':<18} {total_return:>11.2%} {benchmark_total:>11.2%}")
    print(f"  {'年化報酬':<18} {annualized_return:>11.2%} {benchmark_annualized:>11.2%}")
    print(f"  {'超額報酬':<18} {total_return - benchmark_total:>11.2%}")
    print(f"  {'最大回撤':<18} {max_dd:>11.2%}")
    print(f"  {'Sharpe Ratio':<20} {sharpe:>11.2f}")
    print(f"  {'勝率':<18} {win_rate:>11.2%}")
    print(f"  {'盈虧比':<18} {profit_factor:>11.2f}")
    print(f"  {'停損觸發次數':<18} {summary['stoploss_triggered']:>11}")
    print()

    # 月度報酬表
    print("  月度報酬:")
    for p in period_results:
        ret = p["return"]
        bm = p["benchmark_return"]
        excess = p["excess_return"]
        bar = "█" * max(1, int(abs(ret) * 200))
        sign = "+" if ret >= 0 else ""
        color_bar = f"{'↑' if ret >= 0 else '↓'} {bar}"
        print(f"    {p['rebalance_date'][:7]}  {sign}{ret:>7.2%}  (大盤 {bm:>+7.2%})  {color_bar}")

    print("=" * 60)

    return {
        "summary": summary,
        "periods": period_results,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
    }
