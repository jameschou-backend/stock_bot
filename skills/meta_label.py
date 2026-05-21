"""Meta-Labeling（López de Prado AFML Ch 3.7）

第二層 model：給定既有 primary model（LightGBM regressor 預測 forward return）的
score，加上原始特徵，預測「**該不該執行**這次 long 建議」。

訓練資料：
- X：原始 features + primary_score（第一層 model 對該 (stock, date) 的預測）
- y：tb_label == +1（profit-take 觸發 = 該交易）；其他 (sl/time) = 0（不該交易）

執行時：
- 第一層 model 給 score → 取 score > 0 的候選
- 第二層 model 算 P(should_trade)
- 只在 P > threshold 才實際下單

預期效益（López de Prado）：
- Precision 上升（避免低勝率進場）
- Trade frequency 下降（降成本）
- Sharpe 提升（過濾雜訊交易）

Stage 4.2 設計原則：
- Opt-in：不接 daily_pick.py / train_ranker.py 預設行為
- 模組獨立：用 features parquet + TB labels parquet 訓練，產 joblib 到 artifacts/models/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

if not _HAS_LGBM:
    from sklearn.ensemble import GradientBoostingClassifier


# ──────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────

def prepare_meta_training_data(
    features: pd.DataFrame,
    primary_scores: pd.DataFrame,
    tb_labels: pd.DataFrame,
    only_positive_signal: bool = True,
    primary_score_col: str = "primary_score",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """合併 features + primary_score + tb_label。

    Args:
        features: 需含 stock_id, trading_date 和 numeric feature columns
        primary_scores: 需含 stock_id, trading_date, `primary_score_col`
        tb_labels: 需含 stock_id, trading_date, tb_label (-1/0/+1)
        only_positive_signal: 只 keep primary_score > 0 的 rows（只在 primary 認為
            該 long 時，才訓練「是否該執行」）
        primary_score_col: primary_scores 中分數欄位的名稱

    Returns:
        (X, primary_arr, y_binary)
        - X：DataFrame，最後一欄是 `__primary_score`，前面是 features
        - primary_arr：np.ndarray，與 X 對齊的 primary score
        - y_binary：np.ndarray，tb_label == +1 → 1，其他 → 0

    Raises:
        ValueError: merge 後 0 rows、缺欄位、tb_label 不在 {-1, 0, 1}
    """
    for df_name, df, required in [
        ("features", features, {"stock_id", "trading_date"}),
        ("primary_scores", primary_scores, {"stock_id", "trading_date", primary_score_col}),
        ("tb_labels", tb_labels, {"stock_id", "trading_date", "tb_label"}),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{df_name} 缺欄位 {sorted(missing)}")

    # 標籤合法性檢查
    invalid_labels = set(tb_labels["tb_label"].unique()) - {-1, 0, 1}
    if invalid_labels:
        raise ValueError(f"tb_label 必須屬於 {{-1, 0, 1}}，發現異常值: {invalid_labels}")

    df = features.merge(
        primary_scores[["stock_id", "trading_date", primary_score_col]],
        on=["stock_id", "trading_date"], how="inner",
    )
    df = df.merge(
        tb_labels[["stock_id", "trading_date", "tb_label"]],
        on=["stock_id", "trading_date"], how="inner",
    )

    if only_positive_signal:
        df = df[df[primary_score_col] > 0].copy()

    if df.empty:
        raise ValueError(
            "Meta-Label 訓練資料 merge 後為空。"
            "請檢查 features / primary_scores / tb_labels 的 (stock_id, trading_date) overlap。"
        )

    feature_cols = [
        c for c in features.columns
        if c not in ("stock_id", "trading_date") and pd.api.types.is_numeric_dtype(features[c])
    ]
    X = df[feature_cols].copy()
    X["__primary_score"] = df[primary_score_col].to_numpy()
    primary = df[primary_score_col].to_numpy()
    y = (df["tb_label"] == 1).astype(np.int8).to_numpy()

    return X, primary, y


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

@dataclass
class MetaLabelModel:
    """訓練好的 meta-label classifier。

    `feature_names` 不含 `__primary_score`（由 caller 額外傳入 primary score）。
    """
    estimator: object
    feature_names: List[str]
    engine: str
    n_train: int
    n_val: int
    train_pos_rate: float
    val_pos_rate: float
    threshold: float = 0.5
    val_metrics: dict = field(default_factory=dict)

    def predict_proba(
        self,
        features: pd.DataFrame,
        primary_scores: np.ndarray,
    ) -> np.ndarray:
        """回傳 P(該交易) ∈ [0, 1]。

        Args:
            features: 需含至少 `self.feature_names` 中的欄位
            primary_scores: shape (n,)，與 features 對齊的 primary model 預測
        """
        if len(features) != len(primary_scores):
            raise ValueError(f"len mismatch: features {len(features)} vs primary {len(primary_scores)}")
        X = self._build_X(features, primary_scores)
        return self.estimator.predict_proba(X)[:, 1]

    def predict(
        self,
        features: pd.DataFrame,
        primary_scores: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """回傳 0/1（1 = 該交易）。"""
        thr = threshold if threshold is not None else self.threshold
        return (self.predict_proba(features, primary_scores) >= thr).astype(np.int8)

    def _build_X(self, features: pd.DataFrame, primary_scores: np.ndarray) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"features 缺欄位（與訓練時不一致）: {missing[:5]}{'...' if len(missing)>5 else ''}")
        X = features[self.feature_names].copy()
        X["__primary_score"] = np.asarray(primary_scores)
        return X


def train_meta_model(
    X: pd.DataFrame,
    y: np.ndarray,
    val_frac: float = 0.2,
    seed: int = 42,
) -> MetaLabelModel:
    """訓練 meta-label classifier。

    **重要**：使用 chronological split（後 val_frac% 當 validation），**不可 shuffle**——
    避免 forward leakage（X 已含 primary_score，primary 訓練時看過更早期資料）。

    Caller 必須先確保 X 是時序排好的（依 trading_date asc）。

    Args:
        X: 含 `__primary_score` 在最後一欄；其他為 features
        y: 二元標籤 (0/1)
        val_frac: validation set 比例
        seed: 隨機種子

    Returns:
        MetaLabelModel
    """
    n = len(X)
    if n < 100:
        raise ValueError(f"訓練樣本不足 (n={n} < 100)")
    if not (0 < val_frac < 0.5):
        raise ValueError(f"val_frac in (0, 0.5)，got {val_frac}")
    if len(np.unique(y)) < 2:
        raise ValueError(f"y 標籤只有一類，無法訓練分類器：{np.unique(y)}")

    val_n = max(int(n * val_frac), 50)
    train_X, train_y = X.iloc[:-val_n].copy(), y[:-val_n]
    val_X, val_y = X.iloc[-val_n:].copy(), y[-val_n:]

    if _HAS_LGBM:
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=50,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            class_weight="balanced",  # PT 通常 < 20%，用 class_weight 平衡
        )
        model.fit(
            train_X, train_y,
            eval_set=[(val_X, val_y)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        engine = "lightgbm"
    else:
        model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=seed,
        )
        model.fit(train_X, train_y)
        engine = "sklearn_gbc"

    val_metrics = _compute_meta_metrics(model, val_X, val_y, threshold=0.5)
    feature_names = [c for c in X.columns if c != "__primary_score"]
    return MetaLabelModel(
        estimator=model,
        feature_names=feature_names,
        engine=engine,
        n_train=len(train_X),
        n_val=len(val_X),
        train_pos_rate=float(train_y.mean()),
        val_pos_rate=float(val_y.mean()),
        val_metrics=val_metrics,
    )


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def _compute_meta_metrics(estimator, X: pd.DataFrame, y: np.ndarray, threshold: float = 0.5) -> dict:
    """precision / recall / F1 / ROC-AUC + trade rate（多少 sample 通過 meta filter）。"""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    proba = estimator.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    n_pos_pred = int(pred.sum())
    out = {
        "n_total": int(len(y)),
        "n_pos_true": int(y.sum()),
        "n_pos_pred": n_pos_pred,
        "base_pos_rate": float(y.mean()),
        "trade_rate": float(pred.mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
    }
    try:
        if len(np.unique(y)) > 1:
            out["roc_auc"] = float(roc_auc_score(y, proba))
    except Exception:
        pass
    return out


def evaluate(
    model: MetaLabelModel,
    features: pd.DataFrame,
    primary_scores: np.ndarray,
    y_true: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
    """對 held-out set 計算 meta-label metrics。"""
    X = features[model.feature_names].copy()
    X["__primary_score"] = primary_scores
    thr = threshold if threshold is not None else model.threshold
    return _compute_meta_metrics(model.estimator, X, y_true, threshold=thr)
