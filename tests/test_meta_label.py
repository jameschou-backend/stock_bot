"""Meta-Labeling 單元測試（合成資料，無 DB 依賴）。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.meta_label import (
    MetaLabelModel,
    evaluate,
    prepare_meta_training_data,
    train_meta_model,
)


def _synth_dataset(n_samples: int = 800, seed: int = 0):
    """合成 features + primary_scores + tb_labels。

    設計：
    - 5 個 features，其中 feat_signal 與 tb_label 有真實相關
    - primary_score 也與 tb_label 有相關（模擬 primary 有點 skill）
    - 結果：meta model 應該能學到「該交易」的 pattern
    """
    rng = np.random.default_rng(seed)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_samples)]
    stocks = ["A"] * n_samples

    feat_signal = rng.normal(0, 1, n_samples)
    feat_noise = rng.normal(0, 1, (n_samples, 4))
    primary_score = 0.3 * feat_signal + rng.normal(0, 0.5, n_samples)

    # tb_label 受 feat_signal + primary_score 影響：高訊號 → +1，低 → -1，中間 → 0
    score_combined = 0.5 * feat_signal + 0.5 * primary_score
    tb_label = np.where(score_combined > 0.8, 1,
                np.where(score_combined < -0.8, -1, 0))

    features = pd.DataFrame({
        "stock_id": stocks,
        "trading_date": dates,
        "feat_signal": feat_signal,
        "feat_1": feat_noise[:, 0],
        "feat_2": feat_noise[:, 1],
        "feat_3": feat_noise[:, 2],
        "feat_4": feat_noise[:, 3],
    })
    primary_scores = pd.DataFrame({
        "stock_id": stocks,
        "trading_date": dates,
        "primary_score": primary_score,
    })
    tb_labels = pd.DataFrame({
        "stock_id": stocks,
        "trading_date": dates,
        "tb_label": tb_label,
        "tb_return": rng.normal(0, 0.05, n_samples),
    })
    return features, primary_scores, tb_labels


# ──────────────────────────────────────────────
# prepare_meta_training_data
# ──────────────────────────────────────────────

class TestPrepareMetaTrainingData:
    def test_basic_merge(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=200)
        X, primary, y = prepare_meta_training_data(
            features, primary_scores, tb_labels,
            only_positive_signal=False,
        )
        assert len(X) == 200
        assert "__primary_score" in X.columns
        assert "feat_signal" in X.columns
        assert len(primary) == len(y) == 200
        # y 應該是 binary（tb_label == +1 → 1）
        assert set(np.unique(y)) <= {0, 1}

    def test_only_positive_signal_filters(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=400)
        X_all, _, _ = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        X_pos, primary_pos, _ = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=True)
        assert len(X_pos) < len(X_all)
        assert (primary_pos > 0).all()

    def test_rejects_missing_columns(self):
        features = pd.DataFrame({"stock_id": ["A"], "trading_date": [date(2020, 1, 1)], "f": [0.5]})
        primary_scores = pd.DataFrame({"stock_id": ["A"], "trading_date": [date(2020, 1, 1)]})  # 缺 primary_score
        tb_labels = pd.DataFrame({"stock_id": ["A"], "trading_date": [date(2020, 1, 1)], "tb_label": [1]})
        with pytest.raises(ValueError, match="缺欄位"):
            prepare_meta_training_data(features, primary_scores, tb_labels)

    def test_rejects_invalid_tb_labels(self):
        features = pd.DataFrame({
            "stock_id": ["A"] * 5,
            "trading_date": [date(2020, 1, i+1) for i in range(5)],
            "f": [0.1] * 5,
        })
        primary_scores = pd.DataFrame({
            "stock_id": ["A"] * 5,
            "trading_date": [date(2020, 1, i+1) for i in range(5)],
            "primary_score": [0.5] * 5,
        })
        tb_labels = pd.DataFrame({
            "stock_id": ["A"] * 5,
            "trading_date": [date(2020, 1, i+1) for i in range(5)],
            "tb_label": [1, 0, 2, -1, 5],  # 2 跟 5 不合法
        })
        with pytest.raises(ValueError, match="tb_label"):
            prepare_meta_training_data(features, primary_scores, tb_labels)

    def test_empty_after_merge_raises(self):
        features = pd.DataFrame({
            "stock_id": ["A"], "trading_date": [date(2020, 1, 1)], "f": [0.1],
        })
        primary_scores = pd.DataFrame({
            "stock_id": ["B"], "trading_date": [date(2020, 1, 1)], "primary_score": [0.5],
        })  # 不同 stock → merge 為空
        tb_labels = pd.DataFrame({
            "stock_id": ["A"], "trading_date": [date(2020, 1, 1)], "tb_label": [1],
        })
        with pytest.raises(ValueError, match="merge 後為空"):
            prepare_meta_training_data(features, primary_scores, tb_labels, only_positive_signal=False)


# ──────────────────────────────────────────────
# train_meta_model + predict
# ──────────────────────────────────────────────

class TestTrainMetaModel:
    def test_trains_and_predicts(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=600, seed=42)
        X, primary, y = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        model = train_meta_model(X, y, val_frac=0.2, seed=42)

        assert isinstance(model, MetaLabelModel)
        assert model.n_train > 0
        assert model.n_val >= 50
        assert "feat_signal" in model.feature_names
        assert "__primary_score" not in model.feature_names  # 內建處理

        # predict_proba 在 [0, 1]
        proba = model.predict_proba(features.iloc[:50], primary[:50])
        assert proba.shape == (50,)
        assert (proba >= 0).all() and (proba <= 1).all()

        # predict 是 0/1
        pred = model.predict(features.iloc[:50], primary[:50])
        assert set(np.unique(pred)) <= {0, 1}

    def test_learns_signal(self):
        """有真實 signal 時，validation AUC 應該 > 0.6（明顯優於 random 0.5）。"""
        features, primary_scores, tb_labels = _synth_dataset(n_samples=1000, seed=7)
        X, _, y = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        model = train_meta_model(X, y, val_frac=0.2)
        auc = model.val_metrics.get("roc_auc")
        assert auc is not None and auc > 0.6, f"AUC 應該 > 0.6, got {auc}"

    def test_threshold_changes_pred(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=600, seed=1)
        X, primary, y = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        model = train_meta_model(X, y, val_frac=0.2)
        pred_low = model.predict(features.iloc[:100], primary[:100], threshold=0.2)
        pred_high = model.predict(features.iloc[:100], primary[:100], threshold=0.8)
        # threshold 0.2 應通過更多樣本
        assert pred_low.sum() >= pred_high.sum()

    def test_rejects_too_few_samples(self):
        X = pd.DataFrame({"a": [1, 2, 3], "__primary_score": [0.1, 0.2, 0.3]})
        y = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="樣本不足"):
            train_meta_model(X, y)

    def test_rejects_invalid_val_frac(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=300)
        X, _, y = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        with pytest.raises(ValueError, match="val_frac"):
            train_meta_model(X, y, val_frac=0.6)
        with pytest.raises(ValueError, match="val_frac"):
            train_meta_model(X, y, val_frac=0.0)

    def test_rejects_single_class(self):
        X = pd.DataFrame({
            "a": np.random.rand(200), "__primary_score": np.random.rand(200),
        })
        y = np.zeros(200, dtype=int)  # 全 0
        with pytest.raises(ValueError, match="只有一類"):
            train_meta_model(X, y)


# ──────────────────────────────────────────────
# evaluate
# ──────────────────────────────────────────────

class TestEvaluate:
    def test_metrics_present(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=600, seed=2)
        X, primary, y = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        model = train_meta_model(X, y, val_frac=0.2)

        # 用後 100 個當 held-out
        m = evaluate(model, features.iloc[-100:], primary[-100:], y[-100:])
        for k in ("n_total", "trade_rate", "base_pos_rate", "precision", "recall", "f1"):
            assert k in m
        assert 0 <= m["precision"] <= 1
        assert 0 <= m["recall"] <= 1
        assert 0 <= m["trade_rate"] <= 1

    def test_higher_threshold_lowers_trade_rate(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=600, seed=3)
        X, primary, y = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        model = train_meta_model(X, y, val_frac=0.2)

        m_low = evaluate(model, features, primary, y, threshold=0.2)
        m_high = evaluate(model, features, primary, y, threshold=0.8)
        assert m_low["trade_rate"] >= m_high["trade_rate"]
        # 通常 precision 隨 threshold 上升
        assert m_high["precision"] >= m_low["precision"] - 0.05  # 容許小波動


# ──────────────────────────────────────────────
# MetaLabelModel predict guard
# ──────────────────────────────────────────────

class TestPredictGuards:
    def test_rejects_len_mismatch(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=300)
        X, primary, y = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        model = train_meta_model(X, y, val_frac=0.2)
        with pytest.raises(ValueError, match="len mismatch"):
            model.predict_proba(features.iloc[:10], primary[:5])

    def test_rejects_missing_features(self):
        features, primary_scores, tb_labels = _synth_dataset(n_samples=300)
        X, primary, y = prepare_meta_training_data(
            features, primary_scores, tb_labels, only_positive_signal=False)
        model = train_meta_model(X, y, val_frac=0.2)
        # 砍掉 feat_signal 試試
        bad = features[["stock_id", "trading_date", "feat_1", "feat_2"]].copy()
        with pytest.raises(ValueError, match="缺欄位"):
            model.predict_proba(bad.iloc[:5], primary[:5])
