"""Stage 6.1: 異質模型 stacking（LightGBM + CatBoost + XGBoost）。

動機：你既有的 ensemble checkpoint（Session 12 P2-2）是同一個 LightGBM 跑不同
random seed → checkpoint 間相關性高，diversity 不足。改用三種不同 GBDT 算法
（LightGBM / CatBoost / XGBoost），即使都是 gradient-boosted trees，它們的：

  - leaf split 演算法（histogram-based / oblivious trees / pre-sorted）
  - categorical 處理（直接編碼 / target encoding / 不處理）
  - NaN 處理（自動分支 / 預設方向 / 必須補值）
  - regularization strategy
  - early stopping behaviour

都有差異，產出的預測**相關性低於同 algorithm 不同 seed**。Stacking 的 alpha
增益主要來自這個 diversity。

ensemble 策略：**Rank averaging** —— 每個 model predict 後，每個 trading_date
做 cross-sectional rank → percentile（0~1）→ 三個 percentile 取平均。
這比直接平均原始分數更穩定，與 train_ranker 內既有 ensemble 邏輯一致。

設計原則：
  - Pure module，不依賴 backtest / DB
  - 訓練 / 預測介面與 train_ranker 的 _build_model 相容（傳 train_X/y, val_X/y）
  - Stacking 模型寫到 artifacts/models/stacking_*.joblib
  - opt-in：不接 daily pipeline，下游用 scripts/run_stacking_backtest.py 跑對照

文獻：López de Prado, AFML Ch 6（ensemble methods）；Wolpert 1992（stacked generalization）。
"""
from __future__ import annotations

import logging
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
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
try:
    import catboost as cb
    _HAS_CB = True
except ImportError:
    _HAS_CB = False


# ──────────────────────────────────────────────
# Single-algorithm trainers (all return predictor with `.predict(X)`)
# ──────────────────────────────────────────────

def _train_lightgbm(train_X, train_y, val_X, val_y, seed=42):
    if not _HAS_LGBM:
        raise ImportError("lightgbm required for stacking")
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


def _train_xgboost(train_X, train_y, val_X, val_y, seed=42):
    if not _HAS_XGB:
        raise ImportError("xgboost required for stacking")
    model = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, min_child_weight=10,
        tree_method="hist",  # 對應 LightGBM histogram-based
        random_state=seed, n_jobs=-1, verbosity=0,
        early_stopping_rounds=50,
    )
    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        verbose=False,
    )
    return model


def _train_catboost(train_X, train_y, val_X, val_y, seed=42):
    if not _HAS_CB:
        raise ImportError("catboost required for stacking")
    model = cb.CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, subsample=0.8,
        random_seed=seed, thread_count=-1, verbose=0,
        early_stopping_rounds=50,
    )
    model.fit(
        train_X, train_y,
        eval_set=(val_X, val_y),
        verbose=False,
    )
    return model


# ──────────────────────────────────────────────
# Stacking ensemble
# ──────────────────────────────────────────────

@dataclass
class StackingEnsemble:
    """三種異質 GBDT 的 stacking ensemble。

    `predict(X)` 對每個 base model 做 predict，每個 trading_date / 同一份 X
    取 rank percentile，三個 percentile 取平均當最終 score。
    """
    base_models: Dict[str, object]
    feature_names: List[str]
    engines_used: List[str]
    train_secs: float
    n_samples: int

    def predict(self, X: pd.DataFrame, by_group: Optional[Sequence] = None) -> np.ndarray:
        """三 model rank-average prediction.

        Args:
            X: features DataFrame，columns 須含 self.feature_names
            by_group: 若提供（與 X 同長度的 group labels，如 trading_date），
                      則做 per-group rank；否則全體 rank
        Returns:
            np.ndarray shape (len(X),)，每個 row 的 rank-averaged percentile（0~1）
        """
        if not self.base_models:
            raise ValueError("no base models trained")
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"X 缺欄位: {missing[:5]}{'...' if len(missing)>5 else ''}")

        Xf = X[self.feature_names]
        per_model_ranks = []
        for name, model in self.base_models.items():
            try:
                raw = model.predict(Xf)
            except Exception as exc:
                logger.warning("[stacking.predict] %s 失敗: %s", name, exc)
                continue
            ranks = _rank_to_percentile(raw, by_group=by_group)
            per_model_ranks.append(ranks)

        if not per_model_ranks:
            raise RuntimeError("所有 base model predict 都失敗")
        return np.mean(per_model_ranks, axis=0)


def _rank_to_percentile(scores: np.ndarray, by_group: Optional[Sequence] = None) -> np.ndarray:
    """Cross-sectional rank → percentile [0, 1]。"""
    scores = np.asarray(scores, dtype=float)
    if by_group is None:
        # 全體 rank
        from scipy.stats import rankdata
        if len(scores) <= 1:
            return scores * 0.0
        return (rankdata(scores) - 1) / (len(scores) - 1)
    # per-group rank
    by_group = pd.Series(by_group).values
    out = np.empty_like(scores)
    df = pd.DataFrame({"_score": scores, "_g": by_group})
    df["_pct"] = df.groupby("_g")["_score"].rank(method="average", pct=True)
    return df["_pct"].to_numpy()


def train_stacking_ensemble(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_X: pd.DataFrame,
    val_y: np.ndarray,
    use_lightgbm: bool = True,
    use_xgboost: bool = True,
    use_catboost: bool = True,
    seed: int = 42,
) -> StackingEnsemble:
    """訓練 3 個 base model 並組成 StackingEnsemble。

    Args:
        train_X / train_y / val_X / val_y: 訓練/驗證資料
        use_*: 個別開關（除錯用）
        seed: random seed

    Returns:
        StackingEnsemble；訓練 / 預測時 LightGBM/XGBoost/CatBoost 都有 fallback。
    """
    import time
    t0 = time.monotonic()

    base_models: Dict[str, object] = {}
    engines_used = []

    if use_lightgbm and _HAS_LGBM:
        logger.info("[stacking] training LightGBM ...")
        base_models["lightgbm"] = _train_lightgbm(train_X, train_y, val_X, val_y, seed=seed)
        engines_used.append("lightgbm")
    if use_xgboost and _HAS_XGB:
        logger.info("[stacking] training XGBoost ...")
        base_models["xgboost"] = _train_xgboost(train_X, train_y, val_X, val_y, seed=seed)
        engines_used.append("xgboost")
    if use_catboost and _HAS_CB:
        logger.info("[stacking] training CatBoost ...")
        base_models["catboost"] = _train_catboost(train_X, train_y, val_X, val_y, seed=seed)
        engines_used.append("catboost")

    if not base_models:
        raise RuntimeError("no base models trained — install lightgbm/xgboost/catboost")

    elapsed = time.monotonic() - t0
    feature_names = list(train_X.columns)
    return StackingEnsemble(
        base_models=base_models,
        feature_names=feature_names,
        engines_used=engines_used,
        train_secs=elapsed,
        n_samples=len(train_X),
    )


# ──────────────────────────────────────────────
# Diversity diagnostic
# ──────────────────────────────────────────────

def base_model_correlation(
    ensemble: StackingEnsemble, X: pd.DataFrame, by_group: Optional[Sequence] = None
) -> pd.DataFrame:
    """計算 base model 之間的 prediction correlation matrix（rank-Spearman）。

    Stacking gain 主要來自 model 間 correlation 低於 1.0。若三者都 >= 0.95，
    ensemble 效益微弱。
    """
    from scipy.stats import spearmanr
    preds = {}
    Xf = X[ensemble.feature_names]
    for name, model in ensemble.base_models.items():
        preds[name] = model.predict(Xf)
    df = pd.DataFrame(preds)
    names = list(df.columns)
    n = len(names)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            corr[i, j] = corr[j, i] = float(rho)
    return pd.DataFrame(corr, index=names, columns=names)
