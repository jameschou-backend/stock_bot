# News Features POSITIVE 發現（2026-05-25）

## TL;DR

**News features 是過去所有實驗中 ICIR 最強的訊號**，超過 SHAP 識別的所有
production features 4 倍以上。

## 結果

對 90 天 news data（340,619 篇跨 2,708 stocks）跑 cross-sectional Spearman IC：

| Feature | ICIR | IC mean | pos% | 判定 |
|---------|------|---------|------|------|
| **news_count_5d** | **+1.33** | +0.096 | **92.1%** | ✅✅ 候選 |
| **news_source_diversity_5d** | **+1.21** | +0.093 | **86.8%** | ✅✅ 候選 |
| news_count_change_5d | -0.05 | -0.003 | 34% | ⚠️ noise |
| news_count_change_1d | +0.16 | +0.010 | 57% | ⚠️ 弱 |

## 對比歷史所有實驗

| Source | Best ICIR | 結論 |
|--------|-----------|------|
| Stage 8.1a Pruned re-eval | foreign_buy_consecutive_days +0.31 | 邊際 |
| Stage 8.1b 訊號組合 | foreign_net_vol_20 +0.29 | 差 0.01 達標 |
| Production SHAP top | market_volatility_20 主導 43.5% | market-level |
| **Stage 11.1 News count** | **+1.33** | **超強 4x+** |

## 意義

1. **News 量本身就是強 alpha 訊號**（不需 LLM 情感分析）
2. **92.1% pos rate** 表示：90 天內 38 個 trading days，35 天 IC 為正（systematic）
3. **跨股 ranking**：news 多 + source 多元的股 → 預期 next 20 日 outperform

## 為什麼之前沒人發現

1. 過去 _PRUNE_SET 內無 news 相關特徵（從未 ingest）
2. SHAP 分析只看 production model 已用 features
3. FinMind 雖然有 dataset 但需要 sponsor + 特定 API quirk（每次 1 天）

## Caveats

- 只 backfill 90 天 → 樣本短
- Model train_lookback = 5 年 → 前 4.75 年 news 為 NaN → model 學不到關係
- 10y backtest 須先 backfill 5 年 news（~1 小時 + 1800 calls）

## 下一步（in progress）

1. ⏳ **5 年 news backfill** (background task `bqia0epte`, ~1h)
2. ⏸️ rebuild features for 5y → news features 進 features_json
3. ⏸️ 10y backtest 對照 baseline
4. ⏸️ 若 POSITIVE → update production + Phase 2 LLM 情感
