"""倉位配置模組：提供三種倉位計算方法，統一介面供 daily_pick / backtest 使用。

方法：
  vol_inverse   — 波動率反比加權（低波動股獲較大倉位）
  mean_variance — 均值變異數最佳化，最大化 Sharpe（PyPortfolioOpt）
  risk_parity   — 風險平價（每檔貢獻相同風險，PyPortfolioOpt）

介面統一：
  compute_weights(prices, scores, method) -> dict[stock_id, weight]

設計原則：
  - 單檔上限 15%，下限 1%（mean_variance / risk_parity）
  - vol_inverse 不設下限（等比例縮放）
  - 若最佳化失敗，fallback 至 vol_inverse
  - 不洩漏未來資訊：prices 只使用傳入的歷史資料
"""
from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd


_MAX_WEIGHT = 0.15
_MIN_WEIGHT = 0.01


def _clean_prices(prices: pd.DataFrame, stock_ids: list[str]) -> pd.DataFrame:
    """清洗價格資料：去除無效股票（全 NaN/0 或含 inf）並 forward-fill。"""
    available = [sid for sid in stock_ids if sid in prices.columns]
    if not available:
        return pd.DataFrame()
    sub = prices[available].copy()
    # 0 價格視為無效（上市前或資料異常）
    sub = sub.replace(0, np.nan)
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.ffill().bfill()
    # 只保留資料筆數 >= 30 且全部有效的股票
    valid_cols = [c for c in sub.columns if sub[c].notna().sum() >= 30]
    if not valid_cols:
        return pd.DataFrame()
    return sub[valid_cols]


def _vol_inverse_weights(prices: pd.DataFrame, stock_ids: list[str]) -> dict[str, float]:
    """以 20 日日報酬標準差的倒數計算權重（波動率反比）。"""
    weights: dict[str, float] = {}
    vols: dict[str, float] = {}

    for sid in stock_ids:
        if sid not in prices.columns:
            vols[sid] = 1.0  # 無資料者給平均
            continue
        col = prices[sid].replace([np.inf, -np.inf], np.nan).dropna()
        ret = col.pct_change(fill_method=None).dropna()
        if len(ret) < 5:
            vols[sid] = 1.0
            continue
        std = float(ret.tail(20).std())
        vols[sid] = max(std, 1e-8)

    inv = {sid: 1.0 / v for sid, v in vols.items()}
    total = sum(inv.values())
    if total <= 0:
        n = len(stock_ids)
        return {sid: 1.0 / n for sid in stock_ids}

    return {sid: w / total for sid, w in inv.items()}


def _mean_variance_weights(prices: pd.DataFrame, stock_ids: list[str]) -> dict[str, float]:
    """最大化 Sharpe Ratio，使用 PyPortfolioOpt EfficientFrontier。"""
    try:
        from pypfopt import EfficientFrontier, expected_returns, risk_models

        sub = _clean_prices(prices, stock_ids)
        if sub.shape[0] < 30 or sub.shape[1] < 2:
            raise ValueError("資料不足")

        mu = expected_returns.mean_historical_return(sub, returns_data=False)
        S = risk_models.CovarianceShrinkage(sub).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=(_MIN_WEIGHT, _MAX_WEIGHT))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef.max_sharpe(risk_free_rate=0.015)
        cleaned = ef.clean_weights()
        result = {sid: float(w) for sid, w in cleaned.items() if w > 1e-6}
        # 正規化
        total = sum(result.values())
        if total <= 0:
            raise ValueError("最佳化結果全為 0")
        return {sid: w / total for sid, w in result.items()}

    except Exception as exc:
        print(f"  [position_sizing] mean_variance 失敗（{exc}），fallback vol_inverse")
        return _vol_inverse_weights(prices, stock_ids)


def _risk_parity_weights(prices: pd.DataFrame, stock_ids: list[str]) -> dict[str, float]:
    """風險平價：每檔貢獻相同邊際風險。"""
    try:
        from pypfopt import risk_models
        from pypfopt.efficient_frontier import EfficientFrontier

        # risk_parity 透過 CLA 或 EfficientRiskParity 計算
        try:
            from pypfopt.efficient_frontier import EfficientRiskParity as _ERP
            _HAS_ERP = True
        except ImportError:
            _HAS_ERP = False

        sub = _clean_prices(prices, stock_ids)
        if sub.shape[0] < 30 or sub.shape[1] < 2:
            raise ValueError("資料不足")

        S = risk_models.CovarianceShrinkage(sub).ledoit_wolf()

        if _HAS_ERP:
            # PyPortfolioOpt >= 1.4 有 EfficientRiskParity
            erp = _ERP(S)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                erp.min_volatility()
            cleaned = erp.clean_weights()
        else:
            # 手動實作 risk parity（等風險貢獻，scipy 最佳化）
            cleaned = _manual_risk_parity(S, stock_ids)

        result = {sid: float(w) for sid, w in cleaned.items() if w > 1e-6}
        # 套用上限
        capped = {sid: min(w, _MAX_WEIGHT) for sid, w in result.items()}
        total = sum(capped.values())
        if total <= 0:
            raise ValueError("風險平價結果全為 0")
        return {sid: w / total for sid, w in capped.items()}

    except Exception as exc:
        print(f"  [position_sizing] risk_parity 失敗（{exc}），fallback vol_inverse")
        return _vol_inverse_weights(prices, stock_ids)


def _manual_risk_parity(cov_matrix: pd.DataFrame, stock_ids: list[str]) -> dict[str, float]:
    """手動 risk parity：每單位風險貢獻相等（scipy 最小化）。"""
    from scipy.optimize import minimize

    n = len(stock_ids)
    Sigma = cov_matrix.loc[stock_ids, stock_ids].values
    x0 = np.ones(n) / n

    def _obj(w):
        port_var = float(w @ Sigma @ w)
        marginal_risk = Sigma @ w
        risk_contrib = w * marginal_risk / max(port_var, 1e-12)
        target = 1.0 / n
        return float(np.sum((risk_contrib - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(_MIN_WEIGHT, _MAX_WEIGHT)] * n

    res = minimize(_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})

    if res.success:
        w = np.clip(res.x, 0, None)
        total = w.sum()
        if total > 0:
            w /= total
        return dict(zip(stock_ids, w.tolist()))
    else:
        # fallback 等權
        return {sid: 1.0 / n for sid in stock_ids}


def compute_weights(
    prices: pd.DataFrame,
    scores: Dict[str, float],
    method: str = "risk_parity",
) -> Dict[str, float]:
    """計算候選股倉位權重。

    Args:
        prices: 候選股歷史收盤價（columns=stock_id, index=date），至少 30 天
        scores: 模型分數 {stock_id: score}
        method: "vol_inverse" | "mean_variance" | "risk_parity"

    Returns:
        {stock_id: weight}，所有權重和為 1.0（或接近 1.0）
    """
    if not scores:
        return {}

    stock_ids = list(scores.keys())

    # 只保留 prices 中存在的股票
    available = [sid for sid in stock_ids if sid in prices.columns]
    if not available:
        n = len(stock_ids)
        return {sid: 1.0 / n for sid in stock_ids}

    if method == "mean_variance":
        weights = _mean_variance_weights(prices, available)
    elif method == "risk_parity":
        weights = _risk_parity_weights(prices, available)
    else:
        # vol_inverse（預設）
        weights = _vol_inverse_weights(prices, available)

    # 補上 prices 中不存在的股票（等權補齊）
    missing = [sid for sid in stock_ids if sid not in available]
    if missing:
        avg_w = np.mean(list(weights.values())) if weights else 1.0 / len(stock_ids)
        for sid in missing:
            weights[sid] = avg_w
        total = sum(weights.values())
        weights = {sid: w / total for sid, w in weights.items()}

    return weights
