"""Stage 6.2: Multi-Horizon Forecasting（多時間尺度集成）

動機：你既有 model 只預測 20 日 forward return（單 horizon），對「下一個再平衡日
是否會漲」這個目標可能不是唯一最佳信號。Multi-horizon ensemble：

  - 對 5 / 10 / 20 / 40 日各訓一個 LightGBM
  - 每個 model 看到的是同一份 features，但 label 不同（不同時間尺度的 forward return）
  - 短 horizon → 捕捉動能訊號
  - 長 horizon → 捕捉趨勢訊號
  - Ensemble: 每個 trading_date 對每個 horizon model 的預測做 rank → 取平均

預期效益：
  - **降低時間尺度單一化的風險**（model 不會只 over-fit 到 20 日 horizon）
  - **改善 IC 衰減**（CLAUDE.md memory 提過 ma_5/20/60 IC 衰減；不同 horizon model
    對不同市況的 robustness 不同）
  - **複合 stacking diversity**：跟 Stage 6.1 異質 algorithm stacking 是 orthogonal
    的軸，未來可組合 4 horizons × 3 algorithms = 12 models

文獻：López de Prado, AFML Ch 5（meta-labeling 跟 multi-task variant）、
      Marc Salin 2014 "Multi-task Learning in Finance"。

設計原則：opt-in，不接 daily pipeline；介面跟 stacking.StackingEnsemble 相容
（都有 base_models / predict / 提供 rank-averaged percentile）。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


DEFAULT_HORIZONS = (5, 10, 20, 40)


# ──────────────────────────────────────────────
# Multi-horizon label generation
# ──────────────────────────────────────────────

def compute_multi_horizon_labels(
    prices: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """從 raw prices 計算多 horizon forward returns。

    Args:
        prices: 需含 stock_id, trading_date, close
        horizons: 要算的 forward day 數 list

    Returns:
        DataFrame with stock_id, trading_date, future_ret_5, future_ret_10, ...
    """
    required = {"stock_id", "trading_date", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices 缺欄位: {sorted(missing)}")
    if not horizons:
        raise ValueError("horizons 不可為空")

    df = prices[["stock_id", "trading_date", "close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df = df.dropna(subset=["close"])
    df = df.sort_values(["stock_id", "trading_date"]).reset_index(drop=True)

    out_chunks = []
    for sid, g in df.groupby("stock_id", sort=False):
        g = g.sort_values("trading_date").reset_index(drop=True)
        out = g[["stock_id", "trading_date"]].copy()
        close = g["close"]
        for h in horizons:
            col = f"future_ret_{h}"
            # 使用 pandas shift（自動處理 n_days < h 的 stock，shift(-h) 給 NaN）
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = close.shift(-h) / close - 1.0
            # close <= 0 或 inf 都當 NaN
            ret = ret.where((close > 0) & np.isfinite(ret), np.nan)
            out[col] = ret.to_numpy()
        out_chunks.append(out)

    if not out_chunks:
        return pd.DataFrame(columns=["stock_id", "trading_date"] +
                            [f"future_ret_{h}" for h in horizons])
    return pd.concat(out_chunks, ignore_index=True)


# ──────────────────────────────────────────────
# Multi-horizon ensemble
# ──────────────────────────────────────────────

@dataclass
class MultiHorizonEnsemble:
    """4 個獨立 LightGBM model（不同 horizon labels）的 rank-averaged ensemble。"""
    horizon_models: Dict[int, object]
    feature_names: List[str]
    horizons: List[int]
    train_secs: float
    n_samples_per_horizon: Dict[int, int]

    def predict(self, X: pd.DataFrame, by_group: Optional[Sequence] = None) -> np.ndarray:
        """各 horizon model predict → rank → average。"""
        from skills.stacking import _rank_to_percentile

        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"X 缺欄位: {missing[:5]}{'...' if len(missing)>5 else ''}")

        Xf = X[self.feature_names]
        ranks = []
        for h, model in self.horizon_models.items():
            try:
                raw = model.predict(Xf)
            except Exception as exc:
                logger.warning("[multi_horizon] h=%s predict failed: %s", h, exc)
                continue
            ranks.append(_rank_to_percentile(raw, by_group=by_group))
        if not ranks:
            raise RuntimeError("所有 horizon model predict 失敗")
        return np.mean(ranks, axis=0)


def _train_lightgbm_for_horizon(train_X, train_y, val_X, val_y, seed=42):
    """獨立訓練一個 LightGBM regressor（內部 helper）。"""
    if not _HAS_LGBM:
        raise ImportError("lightgbm 必要")
    model = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def train_multi_horizon_ensemble(
    train_X: pd.DataFrame,
    label_panel: pd.DataFrame,
    val_X: pd.DataFrame,
    val_label_panel: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    seed: int = 42,
) -> MultiHorizonEnsemble:
    """對每個 horizon 訓一個 LightGBM。

    Args:
        train_X: 訓練 features（按時間排序）
        label_panel: 訓練 labels，含 future_ret_<h> 欄位（行 index 對齊 train_X）
        val_X / val_label_panel: 同上但驗證
        horizons: 要訓練的 horizons

    Note: label_panel 的 row 順序必須與 train_X 對齊（merge 前已對好）。
          每個 horizon 內部會 dropna 處理 NaN label。

    Returns:
        MultiHorizonEnsemble
    """
    if not _HAS_LGBM:
        raise ImportError("lightgbm 必要")
    if len(train_X) != len(label_panel):
        raise ValueError(f"train_X / label_panel 長度不一致: {len(train_X)} vs {len(label_panel)}")
    if len(val_X) != len(val_label_panel):
        raise ValueError(f"val_X / val_label_panel 長度不一致")

    t0 = time.monotonic()
    models: Dict[int, object] = {}
    n_samples_per_h: Dict[int, int] = {}

    for h in horizons:
        col = f"future_ret_{h}"
        if col not in label_panel.columns:
            logger.warning("[multi_horizon] label %s 不存在，跳過 h=%d", col, h)
            continue

        # Dropna for this horizon
        train_mask = label_panel[col].notna().to_numpy()
        val_mask = val_label_panel[col].notna().to_numpy()
        tX = train_X[train_mask]
        ty = label_panel.loc[train_mask, col].to_numpy()
        vX = val_X[val_mask]
        vy = val_label_panel.loc[val_mask, col].to_numpy()

        if len(tX) < 100 or len(vX) < 50:
            logger.warning("[multi_horizon] h=%d 樣本不足 (train=%d, val=%d)，跳過",
                           h, len(tX), len(vX))
            continue

        logger.info("[multi_horizon] training h=%d (%s) ...", h, col)
        models[h] = _train_lightgbm_for_horizon(tX, ty, vX, vy, seed=seed)
        n_samples_per_h[h] = len(tX)

    if not models:
        raise RuntimeError("沒有 horizon model 訓練成功")

    return MultiHorizonEnsemble(
        horizon_models=models,
        feature_names=list(train_X.columns),
        horizons=sorted(models.keys()),
        train_secs=time.monotonic() - t0,
        n_samples_per_horizon=n_samples_per_h,
    )


# ──────────────────────────────────────────────
# Diagnostic：horizon model 互相 correlation
# ──────────────────────────────────────────────

def horizon_model_correlation(
    ensemble: MultiHorizonEnsemble, X: pd.DataFrame,
) -> pd.DataFrame:
    """horizon model 之間 prediction correlation。"""
    from scipy.stats import spearmanr
    Xf = X[ensemble.feature_names]
    preds = {h: m.predict(Xf) for h, m in ensemble.horizon_models.items()}
    df = pd.DataFrame(preds)
    n = df.shape[1]
    corr = np.eye(n)
    names = list(df.columns)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            corr[i, j] = corr[j, i] = float(rho)
    return pd.DataFrame(corr, index=[f"h={h}" for h in names], columns=[f"h={h}" for h in names])
