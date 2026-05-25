# News features 10y NEGATIVE — 又一個 IC POSITIVE → portfolio NEGATIVE

## TL;DR

90 天 news IC 評估 ICIR **+1.33**（最強訊號），但 10y backtest 整合後 Sharpe
**退化 -0.14**。

## 結果

| Metric | Baseline (no news) | with news (v4) | Δ |
|--------|-------------------|----------------|---|
| 累積 | +5115.80% | +3530.64% | **-1585pp** ❌ |
| 年化 | +49.38% | +43.99% | -5.39pp ❌ |
| MDD | -33.00% | -32.74% | +0.26pp (no change) |
| **Sharpe** | **1.33** | **1.19** | **-0.14** ❌ |
| Calmar | 1.50 | 1.34 | -0.16 ❌ |
| 勝率 | 47.23% | 46.83% | -0.40pp ❌ |

## Why（推測根因）

1. **IC vs Portfolio measure 不同**：
   - IC 衡量「有 news 的 9.4% stocks 之間排名 vs forward return」→ POSITIVE
   - Portfolio 是「全 universe 排序選 top 30」→ model over-weight 「news > 0」

2. **Selection bias**：
   - 高 news 量 ≈ 熱門股 ≈ 大型股 / 已 priced
   - LightGBM split「news > 0」→ picks 偏向熱門股
   - 但 strategy alpha 來自 mid/small-cap follow-up move

3. **Sample bias**:
   - 90 天 IC POSITIVE 但這 90 天剛好特殊期間
   - 10y averaging 內含 2017-2020 牛市 + 各類 regime
   - News alpha 可能短期有但不 stable

4. **Coverage 太低**：
   - 9.4% stocks 有 news_count > 0
   - 90.6% stocks news = 0
   - LightGBM 看「news > 0」變成「binary indicator」
   - 訊號 dilute

## Phase 2 LLM 情感方向是否有救？

**或許**：
- 量訊號（count）失敗的原因是「高 news = 熱門 ≠ outperform」
- 方向訊號（sentiment +1/-1）能區分「**利多熱門** vs **利空熱門**」
- 真實的 alpha 應該是「利多但 underpriced」

**Phase 2 設計**（待用戶 confirm cost）：
1. Claude Haiku batch 3.4M news → sentiment (-1/0/+1)
2. cost ~$80 一次性 + ~$10/月 ongoing
3. Features:
   - `sentiment_5d_avg`: 5 日平均情感
   - `sentiment_net_5d`: (pos - neg) / total
   - `sentiment_strength_5d`: 絕對值強度
   - `sentiment_pos_count_5d`: 純利多新聞數
4. Backtest 對照

## Production status

- News features 全部移回 _PRUNE_SET
- Production 維持 topn=30, no news, Sharpe 1.33
- 即將累積第 ~12 個 NEGATIVE 實驗（含今天 news）

## 累積 NEGATIVE 模式總結

| 實驗 | IC POS? | Portfolio? |
|------|--------|------------|
| 6.1 Stacking | ✅ +7% IC | ❌ -0.32 Sharpe |
| 7.3 Kelly | mixed | ❌ -0.10 Sharpe |
| 8.1 Combo features | borderline | ❌ ICIR<0.30 |
| 9.2 Optuna v2 | 60mo POS | ❌ 10y NEG |
| 10.4 D1 dd_skip | 60mo POS | ❌ 10y NEG |
| 10.5 D2 sector | - | ❌ 10y NEG |
| **11.1 News量** | **ICIR +1.33** | **❌ Sharpe -0.14** |

共同 lesson: **IC ≠ Portfolio alpha**。任何新 feature 必須 10y portfolio
backtest 驗證，IC 只能淘汰負面。
