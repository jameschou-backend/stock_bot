#!/usr/bin/env python
"""月營收 PEAD（post-earnings-announcement drift）事件錨定回測引擎。

獨立於 skills/backtest.py（無 ML、無每期重訓、純因子事件研究），回測很快。
預登記：docs/prereg_pead_arm_20260711.md（判準先寫死；本檔為其唯一執行引擎）。

────────────────────────────────────────────────────────────────────────────
訊號（§1 prereg）
    主訊號 = revenue_yoy = revenue_current_month(M) / revenue_current_month(M−12mo) − 1。
    DB raw_fundamentals.revenue_yoy 全表 NULL（FinMind 未給 last-year 欄位）→ 本引擎自
    revenue_current_month 時間序列自算（同一 stock_id、trading_date 減 1 年的同月營收為分母）。
    次要 = revenue_yoy_accel = yoy(M) − yoy(M−1mo)（僅 robustness，不進裁決）。

時序（§2 prereg，deadline 口徑 = 誠實下界）
    營收月 M（DB 存 M-01）→ 法定申報截止 deadline = (M+1) 月 10 日。
    進場 = 「≥ deadline 的第一個交易日」再 +1 交易日（deadline+1，實盤最早可執行）。
    出場 = 進場 + 20 交易日（= label horizon，無 mismatch）。
    ⚠️ 此時序系統性截掉早公告（多為好消息）前段 drift → 量到的效果量是 drift 的「下界」。
       ex-ante 明文承認（偏保守方向）。

無 lookahead 鐵律（§4 prereg，測試鎖定 tests/test_backtest_pead.py）
    entry_date 嚴格 > deadline；進場價取 entry_date 當日 adj_close，不早於 deadline。
    revision 偏差（FinMind 為最終修正值非首刊）記為已知輕微 caveat。

P&L（§3 prereg）
    持有報酬 = adj_close(exit)/adj_close(entry) − 1，單筆 clip −50%。
    adj_close = raw close × 官方 adj_factor（DB price_adjust_factors，per-stock ffill→bfill→1.0，
    skills.build_features.apply_adj_factors 同源，含息）。同 cohort 全股共享 entry/exit 日。

兩臂（§5 prereg）
    Arm A（gate，gross、無成本、無門檻 universe）：top-30 等權 vs 等權零成本 universe benchmark。
        判準：超額 Sharpe paired block-bootstrap 95% CI 下界 > 0 → PASS（續 Arm B）；否則 FAIL（停 A）。
    Arm B（僅 A PASS 才跑，個人可執行口徑）：min_avg_turnover 0.0033 億、無價格上限、
        每檔 net = gross − 0.00585（整股稅費來回）− odd_lot_round_trip_cost(amt_20, entry, ×1.5)。
        判準：悲觀 Sharpe ≥ 0.70 且 MDD ≥ −50% PASS；[0.50,0.70) GRAY；否則 FAIL。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import get_session  # noqa: E402
from app.models import PriceAdjustFactor, RawFundamental, Stock  # noqa: E402
from skills.build_features import apply_adj_factors  # noqa: E402
from skills.odd_lot_costs import odd_lot_round_trip_cost  # noqa: E402
from skills.statistics import (  # noqa: E402
    deflated_sharpe_ratio,
    paired_block_bootstrap_sharpe_ci,
    returns_moments,
)
from skills.taiex_tr import compute_vs_taiex_tr  # noqa: E402
from skills.trial_registry import (  # noqa: E402
    HISTORICAL_TRIALS_BASE,
    record_backtest_trial,
    registry_trial_count,
)
from sqlalchemy import select  # noqa: E402

# ── 制度常數（ex-ante 固定）────────────────────────────────────────────────────
MONTHLY_FILING_DEADLINE_DAY = 10       # 台股次月 10 日前公告月營收
HOLD_TRADING_DAYS = 20                 # = label horizon（無 4:1 mismatch）
SINGLE_TRADE_CLIP = -0.50              # 單筆最大虧損（退市股保護，與生產同）
TOPN = 30                              # top-quantile 選股數（等權）
DECILE_FRAC = 0.10                     # long-short 診斷用十分位
RISK_FREE_RATE = 0.015                 # 年化無風險利率（與 backtest summary 同口徑）
PERIODS_PER_YEAR = 12                  # 月頻事件
ROUND_TRIP_TAX_FEE = 0.00585           # 整股稅費來回（app.config round_trip_cost，Arm B）
# 統計常數（與 scripts/run_backtest.py 同源）
BOOTSTRAP_BLOCK_SIZE = 6
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
TRIAL_SR_STD_MONTHLY = 0.5 / (12 ** 0.5)   # run_backtest：年 0.5 / sqrt(12)
STOCK_ID_RE = r"^\d{4}$"
# Arm B 個人口徑
PERSONAL_MIN_AVG_TURNOVER_YI = 0.0033       # 億元（≈333,333 元）
PERSONAL_POSITION_SIZE_TWD = 33_333.0
ODD_LOT_PESSIMISTIC_MULT = 1.5

# ── rank/winsorize 訊號 transform 常數（prereg: docs/prereg_pead_rank_arm_20260711.md §1）──
# (a) 基期效應剔除：同月去年營收下限（元）。剔除基期≈0 假 YoY（診斷定位的失敗機制）
#     + 負營收金控雜訊。10,000,000 = 前一輪 §8.6 明列候選值，非看數字後挑選、不掃描。
REVENUE_BASE_FLOOR_TWD = 10_000_000.0
# (b) winsorize 百分位（對 rank 選股為單調 no-op，僅防禦性 hygiene，見 prereg §1b）。
WINSOR_LOWER_PCT = 1.0
WINSOR_UPPER_PCT = 99.0
# 參考臂整股 tiered slippage（來回；與 skills/backtest.py 生產 --tiered-slippage 逐字相同）。
TIERED_SLIP_LARGE = 0.002    # amt_20 >= 5 億
TIERED_SLIP_MID = 0.006      # amt_20 >= 1 億（亦為 amt_20 缺失 fallback）
TIERED_SLIP_SMALL = 0.010    # amt_20 < 1 億
TIERED_SLIP_BOUND_LARGE = 5e8
TIERED_SLIP_BOUND_MID = 1e8


def winsorize_series(values: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
    """截面 winsorize：clip 到 [Plower, Pupper]（純函式，供 transform 與單元測試共用）。

    對 rank 選股為單調 no-op（clip 保序）；此步僅供防禦性 hygiene 與 value-based 報表。
    空序列或全 NaN 直接回傳原序列。
    """
    s = pd.to_numeric(values, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s
    lo = float(np.percentile(valid, lower_pct))
    hi = float(np.percentile(valid, upper_pct))
    return s.clip(lower=lo, upper=hi)


def tiered_slippage_round_trip(amt_20: float) -> float:
    """整股 tiered slippage（來回，比例）。參考臂用，隔離零股價差（prereg §4）。

    amt_20 缺失 / 非有限 → 落中型 fallback（保守，與 backtest.py 一致）。
    """
    if amt_20 is None or not math.isfinite(amt_20) or amt_20 <= 0:
        return TIERED_SLIP_MID
    if amt_20 >= TIERED_SLIP_BOUND_LARGE:
        return TIERED_SLIP_LARGE
    if amt_20 >= TIERED_SLIP_BOUND_MID:
        return TIERED_SLIP_MID
    return TIERED_SLIP_SMALL

# A 線相關性檢查用（生產 ML 主臂 v2.2 機構口徑）
PRODUCTION_ML_JSON = ROOT / "artifacts" / "backtest" / "backtest_20260710_190850.json"


# ════════════════════════════════════════════════════════════════════════════
# 1. 資料載入
# ════════════════════════════════════════════════════════════════════════════
def compute_yoy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """自 revenue_current_month 序列自算 revenue_yoy（同月去年）與 revenue_yoy_accel。

    純函式（無 DB），供 load_revenue_signals 與單元測試共用。
    輸入欄位：stock_id, revenue_month(datetime64, normalized), revenue_current_month。
    - revenue_yoy = current(M) / current(同 stock_id、M−12mo) − 1（分母 ≠ 0）。
    - revenue_yoy_accel = yoy(M) − yoy(M−1 自然月)。
    回傳僅保留 revenue_yoy 可算的列（同月去年存在且 ≠ 0）。
    """
    df = df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df["revenue_month"] = pd.to_datetime(df["revenue_month"]).dt.normalize()
    df["revenue_current_month"] = pd.to_numeric(df["revenue_current_month"], errors="coerce")
    df = df.dropna(subset=["revenue_current_month"])

    # 同月去年營收（YoY 分母）：以 (stock_id, revenue_month − 1 年) 自 merge
    prior = df[["stock_id", "revenue_month", "revenue_current_month"]].copy()
    prior["revenue_month"] = prior["revenue_month"] + pd.DateOffset(years=1)
    prior = prior.rename(columns={"revenue_current_month": "rev_prior_year"})
    df = df.merge(prior, on=["stock_id", "revenue_month"], how="left")

    denom = df["rev_prior_year"].where(df["rev_prior_year"] != 0)
    df["revenue_yoy"] = df["revenue_current_month"] / denom - 1.0

    # yoy_accel = yoy(M) − yoy(M−1 自然月)（robustness，僅供對照）
    df = df.sort_values(["stock_id", "revenue_month"])
    prev_month = df["revenue_month"] - pd.DateOffset(months=1)
    yoy_lookup = df.set_index(["stock_id", "revenue_month"])["revenue_yoy"]
    keys = list(zip(df["stock_id"], prev_month))
    df["revenue_yoy_prev"] = yoy_lookup.reindex(keys).to_numpy()
    df["revenue_yoy_accel"] = df["revenue_yoy"] - df["revenue_yoy_prev"]

    df = df.dropna(subset=["revenue_yoy"]).reset_index(drop=True)
    # rev_prior_year 一併回傳：rank/winsorize transform 的基期 floor（§1a）需要它剔除
    # 基期≈0 假 YoY 與負營收金控（rev_prior_year 可為負；revenue_yoy 只排除分母 == 0）。
    return df[[
        "stock_id", "revenue_month", "revenue_yoy", "revenue_yoy_accel", "rev_prior_year",
    ]]


def load_revenue_signals(session) -> pd.DataFrame:
    """讀 raw_fundamentals，自算 revenue_yoy（同月去年）與 revenue_yoy_accel（前月）。

    回傳欄位：stock_id, revenue_month(date, M-01), revenue_yoy, revenue_yoy_accel。
    僅保留 revenue_yoy 可算（同月去年營收存在且 ≠ 0）的列。
    """
    stmt = (
        select(
            RawFundamental.stock_id,
            RawFundamental.trading_date,
            RawFundamental.revenue_current_month,
        )
        .order_by(RawFundamental.stock_id, RawFundamental.trading_date)
    )
    df = pd.read_sql(stmt, session.get_bind())
    df = df.rename(columns={"trading_date": "revenue_month"})
    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["stock_id"].str.match(STOCK_ID_RE)].copy()
    out = compute_yoy_columns(df)
    out["revenue_month"] = pd.to_datetime(out["revenue_month"]).dt.date
    return out


def load_universe_exclusions(session) -> set:
    """回傳應排除的 stock_id 集合：EMERGING 興櫃股 + 非普通股（etf/warrant/...）。

    survivorship 中性：不用 is_listed 過濾（那會偏向現存股）——point-in-time universe
    由「該 cohort entry/exit 日價格是否齊備」決定（見事件迴圈）。
    """
    rows = session.execute(
        select(Stock.stock_id, Stock.market, Stock.security_type)
    ).fetchall()
    exclude = set()
    for sid, market, sec_type in rows:
        sid = str(sid)
        if market == "EMERGING":
            exclude.add(sid)
        elif sec_type is not None and sec_type != "stock":
            exclude.add(sid)
    return exclude


def load_adj_prices(session, start: date, end: date) -> pd.DataFrame:
    """載入 [start, end] 原始 OHLCV + 官方 adj_factor，回傳含 adj_close / turnover_20。

    - adj_close = raw close × factor（apply_adj_factors 缺日語義：per-stock ffill→bfill→1.0）。
    - turnover_20 = 近 20 交易日 raw(close×volume) 均值（min_periods=10；流動性/零股層用）。
    """
    from skills import data_store

    px = data_store.get_prices(session, start, end)
    px["stock_id"] = px["stock_id"].astype(str)
    px = px[px["stock_id"].str.match(STOCK_ID_RE)].copy()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["volume"] = pd.to_numeric(px["volume"], errors="coerce")

    factor_df = pd.read_sql(
        select(
            PriceAdjustFactor.stock_id,
            PriceAdjustFactor.trading_date,
            PriceAdjustFactor.adj_factor,
        ).where(PriceAdjustFactor.trading_date.between(start, end)),
        session.get_bind(),
    )
    # apply_adj_factors 產出 adj_close（含 factor 缺日 ffill/bfill/1.0 語義）
    adj = apply_adj_factors(px[["stock_id", "trading_date", "close"]].copy(), factor_df)
    adj["trading_date"] = pd.to_datetime(adj["trading_date"]).dt.normalize()

    px["trading_date"] = pd.to_datetime(px["trading_date"]).dt.normalize()
    out = px.merge(
        adj[["stock_id", "trading_date", "adj_close"]],
        on=["stock_id", "trading_date"], how="left",
    )
    out["turnover"] = out["close"] * out["volume"]  # raw 成交值（流動性口徑，不還原）
    out = out.sort_values(["stock_id", "trading_date"])
    out["turnover_20"] = (
        out.groupby("stock_id", sort=False)["turnover"]
        .transform(lambda s: s.rolling(20, min_periods=10).mean())
    )
    return out[["stock_id", "trading_date", "close", "adj_close", "turnover_20"]]


# ════════════════════════════════════════════════════════════════════════════
# 2. 時序（deadline 口徑）
# ════════════════════════════════════════════════════════════════════════════
def compute_deadline(revenue_month: date) -> date:
    """營收月 M（M-01）→ 法定申報截止 = (M+1) 月 10 日。"""
    nxt = revenue_month + relativedelta(months=1)
    return date(nxt.year, nxt.month, MONTHLY_FILING_DEADLINE_DAY)


def resolve_entry_exit(
    deadline: date,
    trading_days: np.ndarray,
    hold_days: int = HOLD_TRADING_DAYS,
) -> Optional[Tuple[date, date]]:
    """deadline → (entry_date, exit_date)。

    entry_idx = searchsorted(trading_days, deadline, "left") + 1
      （「≥ deadline 的第一個交易日」再 +1 交易日 = deadline+1，實盤最早可執行）。
    exit_idx  = entry_idx + hold_days。未來交易日不足回傳 None。
    """
    dl64 = np.datetime64(deadline)
    first_ge = int(np.searchsorted(trading_days, dl64, side="left"))
    entry_idx = first_ge + 1
    exit_idx = entry_idx + hold_days
    if entry_idx >= len(trading_days) or exit_idx >= len(trading_days):
        return None
    entry_date = pd.Timestamp(trading_days[entry_idx]).date()
    exit_date = pd.Timestamp(trading_days[exit_idx]).date()
    # 無 lookahead 鐵律（§4）：進場嚴格晚於申報截止
    assert entry_date > deadline, f"lookahead: entry {entry_date} <= deadline {deadline}"
    return entry_date, exit_date


# ════════════════════════════════════════════════════════════════════════════
# 3. 事件迴圈
# ════════════════════════════════════════════════════════════════════════════
def _decile_long_short(frame: pd.DataFrame) -> Optional[float]:
    """top decile − bottom decile 的 gross_ret 均值（依 revenue_yoy 排序）。"""
    n = len(frame)
    if n < 2:
        return None
    k = max(1, int(n * DECILE_FRAC))
    by_yoy = frame.sort_values("revenue_yoy", ascending=False)
    top_dec = float(by_yoy.head(k)["gross_ret"].mean())
    bot_dec = float(by_yoy.tail(k)["gross_ret"].mean())
    return top_dec - bot_dec


def run_cohorts(
    signals: pd.DataFrame,
    adj_prices: pd.DataFrame,
    exclude_ids: set,
    *,
    topn: int = TOPN,
    hold_days: int = HOLD_TRADING_DAYS,
    min_avg_turnover_yi: float = 0.0,
    odd_lot_premium_mult: float = 1.0,
    cost_mode: str = "none",
    signal_transform: bool = False,
    base_floor: float = REVENUE_BASE_FLOOR_TWD,
    winsor_lower: float = WINSOR_LOWER_PCT,
    winsor_upper: float = WINSOR_UPPER_PCT,
) -> Tuple[List[dict], dict]:
    """逐營收月 cohort 模擬，回傳 (periods, diagnostics)。

    Args:
        cost_mode: "none"（gross 無成本）/ "oddlot"（稅費 + 零股 spread ×premium_mult）/
                   "tiered"（稅費 + 整股 tiered slippage，參考臂，隔離零股價差）。
        signal_transform: True 時套 rank/winsorize transform（prereg §1）：
                   (a) 基期 floor 剔除 rev_prior_year < base_floor（含負值/NaN）；
                   (b) winsorize 存活 yoy 到 [winsor_lower, winsor_upper]（單調 no-op on rank）；
                   (c) 選股仍依 revenue_yoy 降冪 rank top-N。

    periods[i]：entry/exit_date / return(策略 net) / benchmark_return(零成本全 universe) /
                excess_return / n_universe / n_pool / n_selected。
    diagnostics：full-universe 與 post-transform-pool 兩組 long-short / rank IC 序列
                （pool 組 = sanity gate 資料源）。
    """
    if cost_mode not in ("none", "oddlot", "tiered"):
        raise ValueError(f"未知 cost_mode: {cost_mode}")
    trading_days = np.sort(adj_prices["trading_date"].unique())
    # 依 (pd.Timestamp, normalized) → DataFrame(index=stock_id) 建查詢表
    adj_by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        pd.Timestamp(d).normalize(): g.set_index("stock_id")
        for d, g in adj_prices.groupby("trading_date", sort=False)
    }

    signals = signals[~signals["stock_id"].isin(exclude_ids)].copy()
    min_turnover_twd = min_avg_turnover_yi * 1e8

    periods: List[dict] = []
    ls_returns: List[float] = []       # full universe long-short（前一輪對照）
    ic_values: List[float] = []        # full universe 逐 cohort Spearman(yoy, ret)
    pool_ls_returns: List[float] = []  # post-transform pool long-short（sanity gate）
    pool_ic_values: List[float] = []   # post-transform pool 逐 cohort rank IC（sanity gate）

    for rev_month, cohort in signals.groupby("revenue_month", sort=True):
        deadline = compute_deadline(rev_month)
        ee = resolve_entry_exit(deadline, trading_days, hold_days)
        if ee is None:
            continue
        entry_date, exit_date = ee
        entry_tbl = adj_by_date.get(pd.Timestamp(entry_date).normalize())
        exit_tbl = adj_by_date.get(pd.Timestamp(exit_date).normalize())
        if entry_tbl is None or exit_tbl is None:
            continue

        c = cohort[["stock_id", "revenue_yoy", "revenue_yoy_accel", "rev_prior_year"]].copy()
        c = c.set_index("stock_id")
        c["entry_px"] = entry_tbl["adj_close"].reindex(c.index)
        c["exit_px"] = exit_tbl["adj_close"].reindex(c.index)
        c["turnover_20"] = entry_tbl["turnover_20"].reindex(c.index)
        c = c.dropna(subset=["entry_px", "exit_px"])
        c = c[c["entry_px"] > 0]
        if len(c) < topn:
            continue

        gross = (c["exit_px"] / c["entry_px"] - 1.0).clip(lower=SINGLE_TRADE_CLIP)
        c["gross_ret"] = gross

        # ── benchmark：等權零成本全 universe（未套 floor / 門檻）──
        bench_ret = float(c["gross_ret"].mean())

        # ── 選股 pool：(a) 基期 floor（signal_transform）→ 流動性門檻 ──
        pool = c
        if signal_transform:
            # rev_prior_year < floor（含 NaN / 負值）→ 剔除（§1a）
            pool = pool[pool["rev_prior_year"].fillna(-np.inf) >= base_floor]
        if min_turnover_twd > 0:
            pool = pool[pool["turnover_20"].fillna(0.0) >= min_turnover_twd]
        if len(pool) < topn:
            continue
        if signal_transform:
            # (b) winsorize（單調 no-op on rank；hygiene / value-based 報表用）
            pool = pool.assign(
                revenue_yoy_wins=winsorize_series(pool["revenue_yoy"], winsor_lower, winsor_upper)
            )

        # ── 執行成本（cost_mode）──
        if cost_mode == "oddlot":
            cost = pool["turnover_20"].apply(
                lambda t: ROUND_TRIP_TAX_FEE + odd_lot_round_trip_cost(
                    amt_20=float(t) if pd.notna(t) else 0.0,
                    trade_date=entry_date,
                    premium_mult=odd_lot_premium_mult,
                    position_size_twd=PERSONAL_POSITION_SIZE_TWD,
                )
            )
            net = pool["gross_ret"] - cost
        elif cost_mode == "tiered":
            cost = pool["turnover_20"].apply(
                lambda t: ROUND_TRIP_TAX_FEE + tiered_slippage_round_trip(
                    float(t) if pd.notna(t) else float("nan")
                )
            )
            net = pool["gross_ret"] - cost
        else:  # "none"
            net = pool["gross_ret"].copy()
        pool = pool.assign(net_ret=net)

        # ── 選股：top-N by yoy 降冪 rank 等權（§1c）──
        ranked = pool.sort_values("revenue_yoy", ascending=False)
        selected = ranked.head(topn)
        strat_ret = float(selected["net_ret"].mean())

        periods.append({
            "revenue_month": rev_month.isoformat() if hasattr(rev_month, "isoformat") else str(rev_month),
            "deadline": deadline.isoformat(),
            "entry_date": entry_date.isoformat(),
            "exit_date": exit_date.isoformat(),
            "return": strat_ret,
            "benchmark_return": bench_ret,
            "excess_return": strat_ret - bench_ret,
            "n_universe": int(len(c)),
            "n_pool": int(len(pool)),
            "n_selected": int(len(selected)),
        })

        # ── 診斷 1：full universe（gross，不受 transform/成本影響；前一輪對照）──
        ls_full = _decile_long_short(c)
        if ls_full is not None:
            ls_returns.append(ls_full)
        ic = c["revenue_yoy"].corr(c["gross_ret"], method="spearman")
        if pd.notna(ic):
            ic_values.append(float(ic))

        # ── 診斷 2：post-transform pool（sanity gate；gross，成本無關）──
        ls_pool = _decile_long_short(pool)
        if ls_pool is not None:
            pool_ls_returns.append(ls_pool)
        ic_pool = pool["revenue_yoy"].corr(pool["gross_ret"], method="spearman")
        if pd.notna(ic_pool):
            pool_ic_values.append(float(ic_pool))

    diagnostics = {
        "long_short_returns": ls_returns,
        "ic_values": ic_values,
        "pool_long_short_returns": pool_ls_returns,
        "pool_ic_values": pool_ic_values,
    }
    return periods, diagnostics


# ════════════════════════════════════════════════════════════════════════════
# 4. 指標（與 skills/backtest.py summary 同口徑）
# ════════════════════════════════════════════════════════════════════════════
def _sharpe(returns: np.ndarray, rf_period: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std()  # ddof=0（backtest summary 口徑）
    if std <= 0:
        return 0.0
    return float((returns.mean() - rf_period) / std * math.sqrt(PERIODS_PER_YEAR))


def _max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def summarize(periods: List[dict]) -> dict:
    """由 periods 產 summary（sharpe_ratio / max_drawdown / total / excess / annualized / calmar）。"""
    rets = np.array([p["return"] for p in periods], dtype=float)
    bench = np.array([p["benchmark_return"] for p in periods], dtype=float)
    rf_month = (1 + RISK_FREE_RATE) ** (1 / 12) - 1

    total = float(np.prod(1.0 + rets) - 1.0)
    bench_total = float(np.prod(1.0 + bench) - 1.0)
    mdd = _max_drawdown(rets)
    sharpe = _sharpe(rets, rf_month)
    excess_sharpe = _sharpe(rets - bench, 0.0)

    first_entry = pd.Timestamp(periods[0]["entry_date"])
    last_exit = pd.Timestamp(periods[-1]["exit_date"])
    years = max((last_exit - first_entry).days / 365.25, 0.01)
    annualized = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    calmar = annualized / abs(mdd) if mdd < 0 else None

    wins = int((rets > 0).sum())
    return {
        "total_return": round(total, 4),
        "benchmark_total_return": round(bench_total, 4),
        "excess_return": round(total - bench_total, 4),
        "annualized_return": round(annualized, 4),
        "max_drawdown": round(mdd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "excess_sharpe": round(excess_sharpe, 4),
        "calmar_ratio": round(calmar, 4) if calmar is not None else None,
        "win_rate": round(wins / len(rets), 4) if len(rets) else 0.0,
        "n_periods": len(periods),
        "backtest_start": periods[0]["entry_date"],
        "backtest_end": periods[-1]["exit_date"],
    }


def statistical_block(periods: List[dict], n_trials: int) -> Optional[dict]:
    """paired block-bootstrap Sharpe 95% CI（含 excess）+ DSR。"""
    rets = np.array([p["return"] for p in periods], dtype=float)
    bench = np.array([p["benchmark_return"] for p in periods], dtype=float)
    if len(rets) < 2:
        return None
    boot = paired_block_bootstrap_sharpe_ci(
        rets, bench,
        block_size=BOOTSTRAP_BLOCK_SIZE, n_boot=BOOTSTRAP_N_RESAMPLES,
        seed=BOOTSTRAP_SEED, periods_per_year=PERIODS_PER_YEAR,
        risk_free_rate=RISK_FREE_RATE,
    )
    moments = returns_moments(rets)
    rf_month = (1 + RISK_FREE_RATE) ** (1 / 12) - 1
    std = rets.std()
    sr_monthly = float((rets.mean() - rf_month) / std) if std > 0 else 0.0
    dsr = deflated_sharpe_ratio(
        sr_observed=sr_monthly, n_trials=n_trials, n_observations=len(rets),
        skewness=moments["skewness"], kurtosis=moments["kurtosis"],
        sr_estimates_std=TRIAL_SR_STD_MONTHLY,
    )
    return {
        "sharpe_ci_95": {
            "sharpe_observed": round(boot.sharpe_observed, 4),
            "ci_low": round(boot.ci_low, 4),
            "ci_high": round(boot.ci_high, 4),
            "excess_sharpe_observed": round(boot.excess_sharpe_observed, 4),
            "excess_ci_low": round(boot.excess_ci_low, 4),
            "excess_ci_high": round(boot.excess_ci_high, 4),
            "method": "paired_circular_block_bootstrap",
            "block_size": boot.block_size, "n_boot": boot.n_boot,
            "seed": boot.seed, "n_observations": boot.n_observations,
        },
        "deflated_sharpe": {
            "sr_observed_monthly": round(dsr.sr_observed, 4),
            "sr_expected_max_under_null": round(dsr.sr_expected_under_null, 4),
            "p_value": round(dsr.p_value, 4),
            "is_significant_5pct": dsr.is_significant_5pct,
            "n_trials": dsr.n_trials, "n_observations": dsr.n_observations,
            "n_trials_source": f"trial_registry({n_trials - HISTORICAL_TRIALS_BASE}) + historical_base({HISTORICAL_TRIALS_BASE})",
        },
    }


def _ls_ic_block(ls_returns: List[float], ic_values: List[float]) -> dict:
    """由 long-short 月序列與逐 cohort rank IC 算 Sharpe / IC / ICIR block。"""
    ls = np.array(ls_returns, dtype=float)
    ic = np.array(ic_values, dtype=float)
    return {
        "long_short_sharpe": round(_sharpe(ls, 0.0), 4) if len(ls) else None,
        "long_short_mean_monthly": round(float(ls.mean()), 4) if len(ls) else None,
        "long_short_total": round(float(np.prod(1.0 + ls) - 1.0), 4) if len(ls) else None,
        "rank_ic_mean": round(float(ic.mean()), 4) if len(ic) else None,
        "rank_ic_std": round(float(ic.std(ddof=1)), 4) if len(ic) > 1 else None,
        "rank_icir": round(float(ic.mean() / ic.std(ddof=1)), 4) if len(ic) > 1 and ic.std(ddof=1) > 0 else None,
        "n_cohorts": int(len(ls)),
    }


def long_short_stats(diagnostics: dict) -> dict:
    """full-universe + post-transform-pool 兩組 long-short/IC；pool 組供 sanity gate。"""
    full = _ls_ic_block(diagnostics["long_short_returns"], diagnostics["ic_values"])
    pool = _ls_ic_block(
        diagnostics.get("pool_long_short_returns", []),
        diagnostics.get("pool_ic_values", []),
    )
    pool_ic = pool["rank_ic_mean"]
    return {
        # 向後相容：頂層仍是 full-universe block（前一輪 JSON 欄位不變）
        **full,
        "full_universe": full,
        "post_transform_pool": pool,
        # sanity gate（prereg §1）：transform 後 pool rank IC 須 > 0（非裁決、健全性檢查）
        "sanity_gate_pool_ic_positive": bool(pool_ic is not None and pool_ic > 0),
    }


def correlation_with_ml(periods: List[dict]) -> dict:
    """PEAD cohort 月報酬 vs 生產 ML 主臂月報酬（年-月對齊，Pearson）。<0.5 = 獨立來源。"""
    if not PRODUCTION_ML_JSON.exists():
        return {"error": f"production ML JSON not found: {PRODUCTION_ML_JSON}"}
    ml = json.loads(PRODUCTION_ML_JSON.read_text())
    ml_map = {}
    for p in ml.get("periods", []):
        rb = p.get("rebalance_date")
        r = p.get("return")
        if rb is None or r is None:
            continue
        ym = pd.Timestamp(rb).strftime("%Y-%m")
        ml_map[ym] = float(r)
    pead_map = {pd.Timestamp(p["entry_date"]).strftime("%Y-%m"): p["return"] for p in periods}
    common = sorted(set(ml_map) & set(pead_map))
    if len(common) < 3:
        return {"n_overlap": len(common), "pearson_r": None, "note": "重疊期不足"}
    a = np.array([pead_map[m] for m in common])
    b = np.array([ml_map[m] for m in common])
    r = float(np.corrcoef(a, b)[0, 1])
    return {
        "n_overlap": len(common),
        "pearson_r": round(r, 4),
        "independent_source": bool(abs(r) < 0.5),
        "ml_source": PRODUCTION_ML_JSON.name,
    }


def build_equity_curve(periods: List[dict]) -> List[dict]:
    """由 cohort net 報酬複利建 equity curve（供 compute_vs_taiex_tr；起點 10000）。

    每 cohort = entry→+20 交易日，月頻近似連續；以 exit_date 為節點複利。
    """
    eq = 10000.0
    curve = [{"date": periods[0]["entry_date"], "equity": eq}]
    for p in periods:
        eq *= (1.0 + p["return"])
        curve.append({"date": p["exit_date"], "equity": eq})
    return curve


# ════════════════════════════════════════════════════════════════════════════
# 5. 臂執行
# ════════════════════════════════════════════════════════════════════════════
def run_arm(
    arm: str,
    output: Optional[str],
    hold_days: int,
    topn: int,
    *,
    signal_transform: bool = False,
    cost_mode: Optional[str] = None,
    min_avg_turnover_yi: Optional[float] = None,
    label: Optional[str] = None,
) -> dict:
    """執行單臂 PEAD 回測。

    legacy（無新旗標）：arm A = gross/無門檻；arm B = oddlot/0.0033 門檻
        （前一輪 docs/prereg_pead_arm_20260711.md 口徑，可重現）。
    rank 版（signal_transform=True）：docs/prereg_pead_rank_arm_20260711.md——
        cost_mode ∈ {none(gross), oddlot(裁決), tiered(參考)}、門檻 0.0033。
    """
    # 由 arm 推導 legacy 預設；新旗標覆蓋
    if cost_mode is None:
        cost_mode = "oddlot" if arm == "B" else "none"
    if min_avg_turnover_yi is None:
        min_avg_turnover_yi = PERSONAL_MIN_AVG_TURNOVER_YI if arm == "B" else 0.0
    tag = label or f"{arm}{'_transform' if signal_transform else ''}_{cost_mode}"
    prereg = (
        "docs/prereg_pead_rank_arm_20260711.md" if signal_transform
        else "docs/prereg_pead_arm_20260711.md"
    )

    print(f"[pead] === {tag} 開始 ===（transform={signal_transform} cost={cost_mode} "
          f"門檻={min_avg_turnover_yi}億）", flush=True)
    with get_session() as session:
        signals = load_revenue_signals(session)
        print(f"[pead] 訊號列數（yoy 可算）: {len(signals):,}；"
              f"營收月數: {signals['revenue_month'].nunique()}", flush=True)
        exclude_ids = load_universe_exclusions(session)
        print(f"[pead] universe 排除（EMERGING+非普通股）: {len(exclude_ids)} 檔", flush=True)

        # 價格窗：最早營收月 deadline+1 ~ 最晚 deadline + 足夠交易日
        first_month = signals["revenue_month"].min()
        px_start = date(first_month.year, first_month.month, 1)
        px_end = date.today()
        adj_prices = load_adj_prices(session, px_start, px_end)
        print(f"[pead] 價格列數: {len(adj_prices):,}；"
              f"交易日: {adj_prices['trading_date'].nunique()}", flush=True)

        periods, diag = run_cohorts(
            signals, adj_prices, exclude_ids,
            topn=topn, hold_days=hold_days,
            min_avg_turnover_yi=min_avg_turnover_yi,
            odd_lot_premium_mult=ODD_LOT_PESSIMISTIC_MULT,
            cost_mode=cost_mode,
            signal_transform=signal_transform,
        )

    if not periods:
        raise RuntimeError("無有效 cohort（檢查資料與時序）")

    summary = summarize(periods)
    ls = long_short_stats(diag)
    corr = correlation_with_ml(periods)   # 所有臂都算（本輪 Arm B 需再確認獨立性）
    vs_tr = compute_vs_taiex_tr(build_equity_curve(periods), allow_fetch=False)

    # trial registry（記入後 n_trials = registry + historical_base）
    record_backtest_trial(
        {"summary": summary}, months=hold_days,
        source=f"pead_{tag}",
        params={"arm": arm, "topn": topn, "hold_days": hold_days,
                "signal": "revenue_yoy", "timing": "deadline+1",
                "signal_transform": signal_transform, "cost_mode": cost_mode,
                "min_avg_turnover_yi": min_avg_turnover_yi},
    )
    n_trials = registry_trial_count() + HISTORICAL_TRIALS_BASE
    stats = statistical_block(periods, n_trials)

    result = {
        "arm": arm,
        "label": tag,
        "engine": "backtest_pead.py",
        "prereg": prereg,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal": (
            "winsorized-rank revenue_yoy (base-floor + winsorize + rank top-N)"
            if signal_transform else
            "revenue_yoy (self-computed same-month-prior-year)"
        ),
        "timing": "deadline+1 (法定申報截止=(M+1)月10日；進場=≥deadline第一個交易日+1交易日)",
        "timing_caveat": "deadline 口徑系統性截掉早公告前段 drift → 效果量為下界（lower bound）",
        "config": {
            "topn": topn, "hold_trading_days": hold_days,
            "single_trade_clip": SINGLE_TRADE_CLIP,
            "signal_transform": signal_transform,
            "revenue_base_floor_twd": REVENUE_BASE_FLOOR_TWD if signal_transform else None,
            "winsor_pct": [WINSOR_LOWER_PCT, WINSOR_UPPER_PCT] if signal_transform else None,
            "min_avg_turnover_yi": min_avg_turnover_yi,
            "max_stock_price": None,
            "cost_mode": cost_mode,
            "odd_lot_premium_mult": ODD_LOT_PESSIMISTIC_MULT if cost_mode == "oddlot" else None,
            "round_trip_tax_fee": ROUND_TRIP_TAX_FEE if cost_mode in ("oddlot", "tiered") else None,
            "pnl_convention": "adj_from_db_official",
        },
        "summary": summary,
        "long_short_diagnostic": ls,
        "statistics": stats,
        "vs_taiex_tr": vs_tr,
        "correlation_with_ml": corr,
        "periods": periods,
    }

    # 主控台摘要
    print(f"\n[pead] === {tag} 結果 ===", flush=True)
    print(f"  期數: {summary['n_periods']}  窗口: {summary['backtest_start']} ~ {summary['backtest_end']}", flush=True)
    print(f"  策略累積: {summary['total_return']:+.2%}  大盤(零成本 universe): {summary['benchmark_total_return']:+.2%}  超額: {summary['excess_return']:+.2%}", flush=True)
    print(f"  Sharpe: {summary['sharpe_ratio']}  超額 Sharpe: {summary['excess_sharpe']}  MDD: {summary['max_drawdown']:.2%}  Calmar: {summary['calmar_ratio']}", flush=True)
    if vs_tr:
        print(f"  vs TAIEX TR: 策略 {vs_tr['strategy_total_return']:+.2%} / TR {vs_tr['taiex_tr_total_return']:+.2%} → 超額 {vs_tr['excess_total_return_vs_tr']:+.2%}", flush=True)
    if stats:
        s = stats["sharpe_ci_95"]
        d = stats["deflated_sharpe"]
        print(f"  Sharpe 95% CI: [{s['ci_low']}, {s['ci_high']}]", flush=True)
        print(f"  超額 Sharpe 95% CI: [{s['excess_ci_low']}, {s['excess_ci_high']}]（rank 臂佐證，裁決看 Sharpe/MDD）", flush=True)
        print(f"  DSR p={d['p_value']} (n_trials={d['n_trials']})", flush=True)
    print(f"  [full-universe] long-short Sharpe: {ls['long_short_sharpe']}  rank IC: {ls['rank_ic_mean']}  ICIR: {ls['rank_icir']}", flush=True)
    pool = ls["post_transform_pool"]
    print(f"  [sanity gate] pool long-short Sharpe: {pool['long_short_sharpe']}  rank IC: {pool['rank_ic_mean']}  → IC>0={ls['sanity_gate_pool_ic_positive']}", flush=True)
    if corr:
        print(f"  與 A 線 ML 主臂相關性: Pearson r={corr.get('pearson_r')} (n={corr.get('n_overlap')})  獨立來源={corr.get('independent_source')}", flush=True)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print(f"[pead] 結果寫入 {out_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="月營收 PEAD 事件臂回測（prereg: docs/prereg_pead_*_20260711.md）")
    parser.add_argument("--arm", choices=["A", "B"], required=True,
                        help="A=無門檻 gross（legacy gate）；B=個人可執行 0.0033 門檻")
    parser.add_argument("--signal-transform", action="store_true",
                        help="套 rank/winsorize transform（prereg: docs/prereg_pead_rank_arm_20260711.md）")
    parser.add_argument("--cost-mode", choices=["none", "oddlot", "tiered"], default=None,
                        help="none=gross；oddlot=零股 spread ×1.5（裁決）；tiered=整股 slippage（參考）。"
                             "省略時由 --arm 推導（A→none, B→oddlot）")
    parser.add_argument("--min-avg-turnover", type=float, default=None, dest="min_avg_turnover",
                        help="流動性門檻（億元）。省略時由 --arm 推導（A→0, B→0.0033）")
    parser.add_argument("--label", type=str, default=None, help="臂標籤（JSON/trial registry 標記）")
    parser.add_argument("--output", type=str, default=None, help="結果 JSON 路徑")
    parser.add_argument("--hold-days", type=int, default=HOLD_TRADING_DAYS, help="持有交易日數（預設 20）")
    parser.add_argument("--topn", type=int, default=TOPN, help="每 cohort 選股數（預設 30）")
    args = parser.parse_args()
    run_arm(
        args.arm, args.output, args.hold_days, args.topn,
        signal_transform=args.signal_transform,
        cost_mode=args.cost_mode,
        min_avg_turnover_yi=args.min_avg_turnover,
        label=args.label,
    )


if __name__ == "__main__":
    main()
