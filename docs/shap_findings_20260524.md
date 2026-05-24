# SHAP Findings 2026-05-24

對 production LightGBM ranker（`ranker_lgbm_20210227_20260226_c24e3482904135ed.pkl`,
58 features）跑 SHAP 分析（5000 samples × 24 月 features）。

## 📊 Global Importance

| Rank | Feature | mean\|SHAP\| | pct | cum% |
|------|---------|-------------|-----|------|
| 1 | **market_volatility_20** | 0.000874 | **43.50%** | 43.50% |
| 2 | **market_trend_60** | 0.000608 | **30.26%** | 73.76% |
| 3 | market_trend_20 | 0.000230 | 11.47% | 85.23% |
| 4 | bias_20 | 0.000110 | 5.49% | 90.72% |
| 5 | theme_turnover_ratio | 0.000108 | 5.35% | 96.07% |
| 6 | drawdown_60 | 0.000040 | 1.98% | 98.05% |
| 7 | ma_5 | 0.000029 | 1.46% | 99.51% |
| 8-58 | 剩 51 個 | < 0.5% | 共 0.49% | 100% |

**Top 2 features 撐起 73.76%，Top 7 撐起 99.51%！**

## 🌗 Regime Breakdown（bull vs bear，依 market_above_200ma）

| Feature | Bear weight | Bull/Bear ratio | 觀察 |
|---------|-------------|----------------|------|
| market_volatility_20 | 0.001450 | **0.46x** ⬇ bear-heavy | bear 時超重要 |
| market_trend_60 | 0.000732 | 0.77x | balanced |
| market_trend_20 | 0.000402 | **0.42x** ⬇ bear-heavy | |
| bias_20 | 0.000176 | **0.49x** ⬇ bear-heavy | |
| theme_turnover_ratio | 0.000078 | **1.52x** ⬆ bull-heavy | 牛市題材輪動 |
| drawdown_60 | 0.000068 | **0.44x** ⬇ bear-heavy | |
| volume_surge_ratio | 0.000003 | **0.54x** ⬇ bear-heavy | |

5/7 bear-heavy → **bear regime model 應該重 weight 市場波動 + 趨勢 features**

## 🔄 Redundant Pairs（|corr| > 0.85，17 對）

| Feature 1 | Feature 2 | Correlation | 弱者（可刪） |
|-----------|-----------|-------------|--------------|
| trend_persistence_inv | trend_persistence | **-1.0000** | trend_persistence |
| trust_net_5_inv | trust_net_5 | -1.0000 | trust_net_5 |
| vol_20 | vol_20_inv | -1.0000 | vol_20_inv |
| ma_5 | ma_20 | +0.9977 | ma_20 |
| vol_ratio_20 | amt_ratio_20 | +0.9948 | amt_ratio_20 |

`_inv` 系列都是顯式 -1.0 對應原版 → 訓練時 LightGBM 已用 inverse 那一邊 → 可刪原版（或反之）

## 🎯 Actionable Hypotheses（待 backfill 完 + 10y 驗證）

### H1：極簡 features ablation
- 從 48 → 7 features（只留 SHAP > 1%）
- 預期 cum/Sharpe 不退化（model 反正沒用其他）
- 風險：reduce noise 可能 marginal positive；丟掉「將來 regime 變化才需要」的 features 可能 negative

### H2：刪 _inv 冗餘
- 移除 trend_persistence_inv / trust_net_5_inv / vol_20_inv（保留原版）
- 預期 model 簡化但效能同等

### H3：刪 ma_20 / amt_ratio_20 高相關 features
- 移除 ma_20（vs ma_5 corr 0.998）
- 移除 amt_ratio_20（vs vol_ratio_20 corr 0.995）

### H4：Regime-conditional model
- bear model 重 weight：market_volatility_20 / market_trend_20 / bias_20 / drawdown_60
- bull model 重 weight：theme_turnover_ratio
- 跟 Stage 10.4/10.5 累積教訓「需要 regime-aware」一致

## ⚠️ Caveats

- mean|SHAP| = 0 不等於「無用」，可能只是 marginal redundant
- 必須 ablation backtest 驗證（10y）
- 不要根據 SHAP 直接更新 production，需要 backtest evidence

## 📁 輸出檔案

- `artifacts/shap_analysis/global_importance.csv` (58 features 完整 ranking)
- `artifacts/shap_analysis/regime_breakdown.csv` (bull/bear 對照)
- `artifacts/shap_analysis/redundant_pairs.csv` (17 對 high-corr pairs)
