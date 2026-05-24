### 歷史基準對照

#### 2026-03-15 停損與大盤過濾實驗系列（buffer=20，去偏後）

| 實驗 | 配置 | 累積報酬 | MDD | Sharpe | Calmar | 說明 |
|------|------|---------|-----|--------|--------|------|
| **Stage 10.1 topn=30（2026-05-23）** | **+topn=30** | **+5115%** | **-33%** | **1.33** | **1.50** | **← 現行生產** |
| 剪枝+流動性加權 topn=20（2026-03-18 快照）| +liq-weighted+pruned(48特徵) | +3351% | -27% | 1.104 | 1.583 | 舊基準，adj_close 已更新 |
| 流動性加權（56特徵）| 漸進過濾+最少2檔+liq-weighted | +2948% | -27% | 1.082 | 1.510 | |
| 無加權基準（2026-03-18）| 漸進過濾+最少2檔 | +1795% | -29% | 1.009 | 1.188 | adj_close 2026-03-18 快照 |
| 流動性加權+分級滑價 | 同上+tiered-slip | +1061% | -30% | 0.813 | 0.930 | 現實成本壓力測試 |
| 純分級滑價（無加權）| 同上 | +609% | -40% | 0.710 | 0.549 | 揭示小型股集中問題 |
| Exp D（舊快照 2026-03-15）| 同上 | +2637% | -29% | 1.042 | 1.367 | adj_close 已更新，不可重現 |
| Exp A | 漸進過濾（最少1檔） | +2687% | -29% | 1.064 | 1.376 | 有單押集中風險 |
| Exp E | 漸進過濾+最少3檔 | +2252% | -29% | 1.025 | 1.294 | 過度分散 |
| Exp C 保守 | >5% x0.5, >10% x0.5 | +2320% | -29% | 1.034 | 1.308 | |
| 無停損（基線） | 無過濾 | +1792% | -35% | 0.971 | 0.989 | |
| 原始大盤過濾 | >5%半倉, >10%全現金 | +1301% | -34% | 0.913 | 0.908 | 2018/2020 錯過反彈 |
| 固定停損 -7% | 原始設定 | +69% | -46% | 0.268 | 0.119 | 過度截斷月內波動 |
| 固定停損 -12% | | +122% | -51% | 0.353 | 0.165 | |
| ATR 動態停損 | 低波動-15%/高波動-25% | +218% | -60% | 0.451 | 0.206 | MDD 惡化 |

#### 2026-03-15 進場訊號過濾實驗系列（基於 Exp D 配置）

| 實驗 | 過濾條件 | 累積報酬 | MDD | Sharpe | Calmar | 說明 |
|------|---------|---------|-----|--------|--------|------|
| Exp D（基線） | 無過濾 | +2637% | -29% | 1.042 | 1.367 | 現行生產 |
| Exp F | foreign_buy_streak<=3 | +2703% | -29% | 1.051 | 1.379 | 微幅改善，2025 年 +3pp |
| Exp G | RSI 45-70 | +889% | -30% | 0.840 | 0.877 | ⚠️ 嚴重退化 |
| Exp H | streak<=3 + RSI 45-70 | +964% | -38% | 0.782 | 0.713 | ⚠️ 嚴重退化，MDD 惡化 |
| Exp I | streak+RSI+bias+volume | +934% | -38% | 0.774 | 0.704 | ⚠️ 條件越多越差 |

> **結論**：RSI 過濾嚴重損害策略表現（Sharpe 1.04 → 0.84），
> LightGBM 模型已隱式學到 RSI 區間的最佳權衡，人為截斷反而干擾模型判斷。
> foreign_buy_streak<=3 單獨使用有微幅改善但差異太小（+66pp / +2.5%），不足以改變生產配置。
> **生產配置維持 Exp D 不變。**

#### Stage 10.6 Beta-Hedge 後處理分析（2026-05-24，✅ POSITIVE 後處理 metric）

DD attribution 啟示「-33% MDD 含大量 systemic beta」。設計 beta-hedge 後處理：
`hedged_return_t = portfolio_return_t - hedge_ratio × benchmark_return_t`，不重跑
backtest，純後處理 metric。

**10y topn=30 baseline 後處理結果**：

| hedge | Sharpe | MDD | Calmar | Annual | Cum |
|-------|--------|------|--------|--------|-----|
| 0.00 unhedged | 1.33 | -33.00% | 1.48 | +49.0% | +5115% |
| 0.25 | 1.38 (+0.05) | -30.7% (+2.4pp) | 1.56 | +47.7% | +4673% |
| 0.50 | **1.43** (+0.10) | **-28.3%** (+4.7pp) | 1.63 | +46.1% | +4199% |
| 0.75 | 1.47 (+0.14) | -25.9% (+7.1pp) | 1.72 | +44.4% | +3713% |
| 1.00 full hedge | **1.48** (+0.15) | **-23.4%** (+9.6pp) | 1.81 | +42.4% | +3231% |
| **OLS beta (1.45) hedge** | **1.41** (+0.08) | **-19.93%** (+13.1pp) | - | - | +2398% |

**關鍵 insight**：
- 策略 OLS beta = **1.45** vs 大盤（過度 beta exposure）
- corr = 0.69（有真實 alpha 不只是 beta）
- alpha-only Sharpe / total Sharpe = **106%** — 真實 alpha 比表面 Sharpe 更強
- hedge 越多 Sharpe + MDD 都改善（trade cum）

**實作**（不動 production default）：
- `skills/backtest.compute_hedged_metrics(result, hedge_ratio)` 純函式 helper
- `scripts/run_backtest.py --hedge-ratio H` flag，計算 hedged metrics 印對照 + 寫進 JSON
- `scripts/beta_hedge_analysis.py` 對既有 result 後處理分析（multi-hedge 對照）

**Production caveat**：要真正利用此 finding 需實作 TXF 期貨對沖（margin cost、滾倉、
contract sizing 等實務問題）。目前僅為「metric-only」分析。

**第 2 個 POSITIVE 結果**（繼 [stage101 topn 20→30] 之後）。

#### Stage 10.5 D2 max_per_sector 產業集中限制（2026-05-23，⚠️ NEGATIVE）

DD attribution 觀察到 2025-03 觀光餐旅 2 檔（5301、4804）同時崩盤共貢獻 -10.92%。
設計 D2：`apply_sector_constraint` 限制同產業最多 N 檔。**這次直接 10y validation**
（不靠 60mo overfit）。

**10y 雙設定對照（vs baseline topn=30）**：

| Metric | Baseline | D2 sector=5 | D2 sector=3 |
|--------|----------|-------------|-------------|
| 累積 | +5115.80% | +4334.36% | **+2730.36%** |
| Sharpe | 1.33 | 1.31 | **1.21** ❌ |
| MDD | -33.00% | **-33.00%** | **-33.62%** ❌ |
| Calmar | 1.50 | 1.42 | 1.20 |

> **重大結論**：**-33% MDD 不是 sector concentration 造成**（sector=5 MDD 完全
> 不變 -33%）。DD attribution 找到「觀光餐旅 2 檔同崩」是事實但 actionability 為零，
> 因為 2025-03 期間大盤同跌 -6%（多產業同步下跌），sector limit 擋不住 systemic risk。
> 強迫分散在強勢期間是純 cost（model 在 topN 強勢產業被打斷）。
>
> 保留 `max_per_sector` 為 opt-in flag（CLI `--max-per-sector N`），預設 0。
> 未來若有 regime-aware 設計（大盤熊市才強制分散）可能 work，但 unconditional 不適合。

#### Stage 10.4 D1 recent_dd_skip filter（2026-05-23，⚠️ NEGATIVE）

Stage 10.3 DD attribution 發現 -33% MDD 的 28% 來自 5301（觀光餐旅）一檔，
連續 2 個月被 model 選中（4 月 -38% → 5 月還選 → 6 月再跌 -50%）。設計 D1：
`ret_20 < threshold` 排除 candidate，避免持續持有已暴雷股。

**60mo threshold sweep** (vs baseline_v2 topn=30 Sharpe 1.7551 / MDD -33.00%)：

| Threshold | Sharpe | MDD | Calmar | 勝率 | vs baseline |
|-----------|--------|------|--------|------|----|
| -15% | 1.6116 | -25.35% | 2.17 | 48.33% | Sharpe ❌ -0.14 |
| -20% | 1.7502 | -25.49% | 2.46 | 48.80% | Sharpe ≈, Calmar +0.46 ✅ |
| -25% | 1.7725 | -25.53% | 2.51 | 48.93% | Sharpe **+0.02**, Calmar **+0.51** ✅✅ |

**60mo 最佳：-25%（dominant 全面贏 baseline）**

**10y 驗證（-25% threshold）vs baseline (topn=30)**：

| Metric | Baseline | D1 -25% | Δ |
|--------|----------|---------|---|
| 累積 | +5115% | +3088% | **-2028pp** ❌ |
| MDD | -33.00% | **-36.14%** | **-3.14pp** ❌ |
| Sharpe | 1.33 | 1.30 | -0.03 |
| Calmar | 1.50 | 1.16 | -0.34 ❌ |

> **失敗根因**：60mo (2021-2026) -25% 剛好 cover 5301 那種暴雷，看似 dominant。
> 但 10y (2016-2026) 期間，-25% 過濾**把 2017-2020 中小型股牛市的「跌深反彈」
> 機會一併排掉**，alpha 大失，MDD 也沒改善（其他年份新暴雷股 emerge）。
>
> **教訓重申**：60mo POSITIVE ≠ 10y POSITIVE。Optuna v2 與 D1 都是同樣模式。
> 任何 portfolio-level filter 都必須 10y validation。
>
> **保留 `recent_dd_skip_pct` 為 opt-in flag**（CLI `--recent-dd-skip PCT`），
> 預設 0=disabled。未來若有 regime-aware 設計可能 work（例如只在大盤
> 200ma 上方啟用過濾），但 unconditional filter 不適合。

#### Stage 9.2 Optuna search v2 10y 驗證（2026-05-23，⚠️ NEGATIVE）

Optuna 30 trials × 60mo 搜尋出 best Sharpe 0.68（trial #8）+ low-MDD 候選（trial #4），
拿 top-2 對 production baseline 跑 10y 對照，兩個 candidates 全面崩盤。

| Metric | Baseline (prod) | Trial #8 best Sharpe | Trial #4 low MDD |
|--------|----------------|---------------------|------------------|
| 累積 | **+3471.73%** | +137.27% | +54.55% |
| 年化 | **+43.75%** | +9.16% | +4.52% |
| 超額 | **+3396%** | +49% | -34% ❌ |
| MDD | -39.15% | -53.32% | **-11.07%** ✅ |
| Sharpe | **1.15** | 0.38 | 0.38 |
| Calmar | **1.12** | 0.17 | 0.41 |

> **失敗根因**：
> 1. **60mo 不夠 representative**：缺 2017-2020 中小型股大牛市，TPE 學到的 best
>    僅反映 2021-2026 較弱期局部最優
> 2. **`min_avg_turnover ≥ 0.5` 強制過濾**：search space 把 turnover 起點設 0.5 億，
>    過濾掉牛市中小型股 → 17-20 期間傷害巨大
> 3. **`topn ≤ 15` 過度集中**：60mo 看不出但 10y idiosyncratic risk 累積
> 4. **缺 baseline 對應點**：search space 沒涵蓋 `min_avg_turnover=0` 與 `topn=20`
>    的 production 預設，TPE 無法回到該區域
>
> **下次重跑修正方向**：
>   - `min_avg_turnover ∈ [0, 3.0]`（含 0）
>   - `topn ∈ [15, 25]`（不要 10）
>   - search 期改 120mo（10y 直接）— 雖然每 trial 變 ~30min × 30 ≈ 15h
>
> **Stage 9 工具本身有效**（MLflow tracking、Optuna SQLite、Prefect retry），
> 此次 search space 設計失敗不影響工具價值。

#### Stage 8.1 結構化籌碼資料 IC 再評估（2026-05-22，⚠️ NEGATIVE）

兩條子路線都未取得增量：

**8.1a `_PRUNE_SET` 內 20 個特徵重新跑 36mo / 60mo IC**：
- 36mo：僅 `foreign_buy_consecutive_days` ICIR=+0.31 邊際達標，無顯著回收
- 60mo：擴大樣本後 0/20 達標（樣本擴大 ICIR 反而弱化 → 36mo 邊際是 noise）
- 多數 sponsor 特徵 coverage=0%（FinMind sponsor token 2026-05-20 過期，無法 backfill）

**8.1b 5 個訊號組合新特徵**（不需新 ingest，純用 raw_prices+institutional+margin）：
| 特徵 | ICIR | 結論 |
|------|------|------|
| inst_consensus_5d | -0.137 | ⚠️ 弱 |
| **foreign_net_vol_20** | **+0.292** | ⚠️ 差 0.01 達標 |
| intraday_strength_5d | -0.175 | ⚠️ 弱 |
| opening_gap_5d | +0.061 | ⚠️ 弱 |
| margin_share_zscore_20 | -0.065 | ⚠️ 弱 |

> **結論**：production model 已含 48 特徵，新增結構化特徵邊際 ROI 極低；LightGBM
> 從現有 ret/foreign_net/margin 等已隱式學到組合訊號。Stage 8 進一步需要：
>   - (a) 接付費 alt data（FinMind sponsor / TEJ / Yahoo Premium）
>   - (b) LLM/NLP 新聞情感（需要新 scraper + LLM API budget）
>
> 兩個 IC 評估 script 保留於 `scripts/ic_analysis_pruned_set.py` /
> `scripts/ic_analysis_combo_features.py` 作為未來再評估入口。

#### Stage 7.3 Kelly Criterion 加權實驗（2026-05-22，⚠️ NEGATIVE）

Half-Kelly fractional weighting：每月 top-N picks 不再等權，依 `f_i ∝ μ_i / σ²_i` 加權
（μ_i 從 model score percentile 線性校正至 [5%, 35%]，σ_i 用 60d daily vol 年化，
half-Kelly ×0.5，individual cap 10%）。

| 指標 | Baseline 60mo | +Kelly Half 60mo | Δ |
|------|--------------|------------------|---|
| 累積 | +772.63% | +765.91% | -6.72pp ❌ |
| Sharpe | 1.368 | 1.269 | -0.099 ❌ |
| MDD | -40.74% | -41.43% | -0.69pp ❌ |
| Calmar | 1.355 | 1.327 | -0.028 ❌ |

> **失敗原因**：
> 1. Half-Kelly 對 μ 估計誤差仍敏感，percentile → [5%, 35%] 線性外推太粗略。
> 2. 個股 60d vol 噪音大；反 vol 配重恰好偏向低 vol mean-reverting 股。
> 3. Cap 0.10 ≈ 10 個有效部位，比等權 1/20=5% 更集中，反降 diversification。
> 4. Top picks 已有產業相關性（流動性加權偏同類股），Kelly tilt 沒帶來 idiosyncratic 增益。
>
> **保留 `skills/kelly.py` + 18 tests 作為研究入口**，未整合到 backtest.py。
> 與 [[stage61-stacking-negative]]、Stage 6.2 Multi-Horizon、Stage 7.1 HRP 同列 NEGATIVE
> 結果。教訓：position-sizing 改動在乾淨 equal-weight + market filter 的基準上很難取得增量。

#### Stage 6.1 Stacking Ensemble 實驗（2026-05-22，⚠️ NEGATIVE）

異質模型 stacking：LightGBM + XGBoost + CatBoost rank-averaged。
Quick eval（60mo）顯示截面 IC lift +7.1%，但 10y WF portfolio 表現嚴重退化。

| 指標 | Baseline 10y | +Stacking 10y | Δ |
|------|-------------|---------------|---|
| 累積報酬 | +3471.73% | +716.18% | **-2755pp** ❌ |
| 年化報酬 | +43.75% | +23.75% | -20.00pp ❌ |
| 超額報酬 | +3396.05% | +640.50% | -2755pp ❌ |
| Sharpe | 1.15 | 0.83 | -0.32 ❌ |
| MDD | -39.15% | -42.87% | -3.72pp ❌ |
| Calmar | 1.12 | 0.55 | -0.57 ❌ |
| 勝率 | 46.76% | 44.92% | -1.84pp ❌ |

> **失敗原因分析**：
> 1. Quick eval IC lift（+7.1%）量測「截面排名相關」，不直接對應 portfolio Sharpe。
> 2. Stacking 切 20% 樣本做 early-stopping val，實際訓練集縮減 20%，反而傷害 LightGBM 在百萬筆資料下的優勢。
> 3. XGBoost / CatBoost 在 48 個樹狀特徵 + 數百萬樣本上沒比 LightGBM 強，rank-averaging 反而稀釋 LGBM 的單獨贏家訊號。
> 4. 三 base model 各自 800 顆樹 + early stopping 對 portfolio top-K 排名一致性貢獻有限。
>
> **保留 code 但 production flag 預設關閉**：
> `--use-stacking` 仍可用做未來進一步研究（如改用 Sharpe-weighted ensemble 而非等權 rank-average），但生產配置維持 LightGBM 單模型。
> 與 Stage 6.2 Multi-Horizon、Stage 7.1 HRP 同列 NEGATIVE 結果保留。

#### Strategy B 日頻策略實驗（2026-03-15）

獨立的日頻策略（`scripts/backtest_daily.py`），每日掃描進場訊號，個別部位管理。

- **進場**：模型分數前 20% + 外資買超 + 成交量 > 20 日均量 × 1.2 + RSI 45-70
- **出場**：持有 > 20 天 / 外資連賣 3 天 / RSI > 80 / 虧損 > -10%
- **倉位**：最多 6 檔，每檔 15%，10% 現金緩衝
- **大盤過濾**：沿用 Exp D 漸進式設定

| 指標 | Strategy B（日頻） | Strategy A Exp D（月頻） |
|------|-------------------|----------------------|
| 總報酬 | +198% | +2637% |
| 年化 | +12.12% | +39.92% |
| MDD | -53.95% | -29.20% |
| Sharpe | 0.482 | 1.042 |
| Calmar | 0.225 | 1.367 |
| 勝率 | 43.27% | 45.45% |
| 交易次數 | 1553 | 2009 |
| 平均持有 | 8.3 天 | ~20 天（月頻） |

出場原因分佈：外資連賣 57.0%、RSI 超買 20.3%、停損 12.8%、時間到期 9.5%

> **結論**：日頻策略表現遠不如月頻（Sharpe 0.48 vs 1.04，MDD -54% vs -29%）。
> 原因：(1) 20 天 label horizon 的模型預測月度趨勢，日頻進出場與預測尺度不匹配；
> (2) RSI 45-70 進場條件過度限縮候選（同前期 Exp G 結論）；
> (3) 外資連賣出場觸發率 57%，導致多數持倉提前出場（平均僅持 8 天）錯過月度漲幅。
> **結論：維持月頻 Strategy A（Exp D）為生產策略。**

#### 2026-03-13 去偏基準（Experiment F，固定停損 -7%）

- **Experiment F（去偏，buffer=20，含 EMERGING 過濾，2026-03-13）**：
  累積 +205.17%、超額 +149.29%、MDD -32.62%、Sharpe +0.4893、Calmar +0.3674

#### 2026-03-11 之前的實驗（含 label 洩漏，⚠️ 僅供參考）

> ⚠️ **重要說明**：Experiment A~E（舊編號）均使用 `label_horizon_buffer=0`，存在訓練標籤前向洩漏，
> 回測績效虛高。

- **Experiment E（含bias，buffer=0，56 特徵，2026-03-11，⚠️ 標籤洩漏）**：
  累積 +10004.80%、超額 +9956.59%、MDD -27.57%、Sharpe +1.3028、Calmar +2.166
- **Experiment D（含bias，buffer=0，seasonal_filter，53 特徵，commit b5974be，2026-03-11，⚠️ 標籤洩漏）**：
  累積 +9552.75%、超額 +9504.54%、MDD -27.57%、Sharpe +1.2958、Calmar +2.1392
- **Experiment C（含bias，buffer=0，無 seasonal_filter，commit 4440fc5，2026-03-10，⚠️ 標籤洩漏）**：
  累積 +1216.35%、超額 +1167.99%、MDD -33.35%、Sharpe +0.9085、Calmar +0.8963

### 突破確認進場實驗（2026-03-13，`--breakthrough-entry`）

月底模型選股後，不立即進場，等每檔股票出現突破訊號才進場（最多等 10 個交易日），
無突破者由後排候選補位，補位仍無突破則持現金。

**突破條件（任一成立）**：
- 條件一（價格）：收盤 > 前 20 日最高收盤 AND 當日量 > 20 日均量 × 1.5
- 條件二（籌碼）：`foreign_buy_consecutive_days ≥ 3` AND 收盤 > 20 日均線

**效果說明（隱式市場擇時）**：
- 多頭市場：大多數股票快速突破 → 近全倉進場，不影響收益
- 空頭市場：少數股票突破 → 大量持現金 → 自動避開下跌股

**10y Walk-Forward 結果（Experiment F+，2016-05-03 ~ 2026-01-30，117 期）**：

| 指標 | 基準 F | 實驗 F+（突破進場）| 變化 |
|------|--------|-------------------|------|
| 累積報酬 | +205.17% | **+827.15%** | +622pp ✅ |
| 年化報酬 | +11.99% | **+25.35%** | +13pp ✅ |
| 超額報酬 | +149.29% | **+771.26%** | +622pp ✅ |
| MDD | -32.62% | **-22.47%** | -10pp ✅ |
| Sharpe | 0.4893 | **0.8581** | +0.369 ✅ |
| Calmar | 0.3674 | **1.1280** | +0.761 ✅ |
| 勝率 | 38.14% | **41.06%** | +2.9pp ✅ |
| 總交易 | 2,019 | 1,924 | -95 |

**逐年報酬（F+ vs F基準 vs 大盤）**：
| 年份 | F+（突破）| F（基準）| 大盤 | 超額（F+）|
|------|---------|---------|------|---------|
| 2016 | +2.42% | -11.23% | +1.87% | +0.55% |
| 2017 | +12.64% | +32.89% | +11.57% | +1.07% |
| 2018 | +2.88% | -7.38% | -15.35% | +18.22% |
| 2019 | +25.43% | +13.91% | +12.01% | +13.42% |
| 2020 | +56.08% | +42.90% | +18.13% | +37.95% |
| 2021 | +40.32% | +25.92% | +19.71% | +20.61% |
| 2022 | +16.30% | -3.68% | -15.83% | +32.13% |
| 2023 | +30.66% | +14.43% | +22.37% | +8.29% |
| 2024 | +7.87% | -5.05% | +2.99% | +4.88% |
| 2025 | +50.19% | +13.97% | -5.69% | +55.87% |
| 2026 | +15.52% | +14.26% | +2.23% | +13.29% |

**實作位置**：`skills/backtest.py`
- `_compute_breakthrough_map()`: 向量化批次計算所有候選股突破日（避免逐股逐日掃描）
- `run_backtest(..., enable_breakthrough_entry=True, breakthrough_max_wait=10)`
- `_simulate_period(..., per_stock_entry_dates)`: 支援各股獨立進場日

**CLI**：`python scripts/run_backtest.py --months 120 --seasonal-filter --breakthrough-entry`

**注意**：`backtest.py` 預設 `enable_breakthrough_entry=False`，不影響現有流程。
生產端 `daily_pick.py` 尚未整合突破過濾，需另外規劃。
2017 是唯一略遜年份（+12.64% vs +32.89%），強多頭市場可能延遲進場導致追高成本。

### 時間加權訓練（2026-03-08 新增，目前已停用）

`backtest.py` 訓練循環可選時間加權（`time_weighting=True`）：
- 近 1 年樣本（≤365 天）：`sample_weight = 2.0`
- 1~2 年樣本（365~730 天）：`sample_weight = 1.0`
- >2 年樣本（>730 天）：`sample_weight = 0.5`

> 結論：10y walk-forward 驗證（2026-03-10）顯示時間加權在 equal-weight + 月頻設定下
> 未帶來額外改善，故現行生產設定 `time_weighting=False`（等同原始基準）。

### 優化歷史記錄（2026-03 完整系列）

#### 24m 窗口實驗（2026-03-08/09，已知有 overfitting 問題）

| 版本 | 主要改動 | 超額報酬 | MDD | Sharpe |
|------|---------|---------|-----|--------|
| v1 | 基準線 | -13.04% | - | - |
| v2 | topN floor + 200MA 現金水位 + RSI 過濾 | -14.54% | - | - |
| v3 | 動態現金水位 | -14.64% | - | - |
| v4 | 時間加權訓練 + 市場特徵（全 0）| +7.02% | -46.79% | - |
| v5 | 全量市場特徵（非零）| -1.71% | -41.88% | -0.68 |
| v6 | Market Regime Switching | +24.71% | -34.79% | -0.18 |
| v7 | 移除 Regime Switching，週頻 + 時間加權 | +0.38%（10y） | -87.74% | -0.36 |

> **根本問題（2026-03-10 發現）**：v2-v7 全在 24m 窗口驗證，且 v7 使用**週頻再平衡（W）**
> 與 20 日 label horizon 造成 4:1 mismatch → 2016-2019 超額 -88.83%。

#### 10y walk-forward 正式驗證（2026-03-10）

| 版本 | 累積 | 超額 | MDD | Sharpe | 說明 |
|------|------|------|-----|--------|------|
| v7（週頻，錯誤）| -84.96% | +0.38% | -87.74% | -0.36 | 月頻修正前 |
| 月頻修正 | +67.37% | - | - | +0.26 | rebalance_freq W→M |
| 還原原始設定 | +338.31% | - | - | +0.56 | 移除 8 個「優化」|
| 實驗 B（退市過濾）| 已 revert | - | - | - | 2021/2023 誤殺強勢股 |
| **實驗 C（clip -50%）** | **+1216.35%** | **+1167.99%** | **-33.35%** | **+0.9085** | **✅ 現行生產** |

### Market Regime Switching 研究紀錄（2026-03-09 實驗，已移除）

> **⚠️ 已刪除（2026-05-18）**：`skills/market_regime.py` 已從 source tree 移除。
> 此處保留 24m 實驗紀錄供未來研究比較。如需重建模組，可參考此處規則描述與 git 歷史
> （`git log --all -- skills/market_regime.py`）。

**三態規則（向量化）**：
- bull: 等權指數 > 200MA AND 60日報酬 > 0 AND 20日報酬 > -5%
- bear: 等權指數 < 200MA AND 60日報酬 < -10%
- sideways: 其餘

**24m 實驗結果（2024-02 ~ 2026-01，大盤 -36.39%）**：超額 +24.71%，MDD -34.79%，Sharpe -0.18

**10y Walk-Forward 驗證（2026-03-09，週頻 v7，已過時）**：此版本使用週頻再平衡，有根本性 label-horizon mismatch，結果已無效。

**移除原因**：Sideways（35%）topN×75% 縮減造成長期損耗（超額 -0.21%/期），10y MDD 惡化。

### 10y 正式優化實驗（2026-03-10，月頻正確基準）

**背景**：2026-03-10 發現 v7 使用週頻再平衡（W）與 20 日 label horizon 造成根本性 4:1 mismatch，
修正為月頻（M）後重新在乾淨 10y walk-forward 上逐步驗證。

**實驗結果（期間 2016-03-16 ~ 2026-01-23，119 月頻再平衡期）**：

| 實驗 | 累積 | MDD | Sharpe | 說明 |
|------|------|-----|--------|------|
| v7（週頻，錯誤基準）| -84.96% | -87.74% | -0.36 | 根本錯誤，已廢棄 |
| 月頻修正（M）| +67.37% | - | +0.26 | rebalance_freq W→M |
| 還原 8 個原始設定 | +338.31% | - | +0.56 | equal/no-trailing/no-slippage 等 |
| 實驗 B：退市過濾 B1 | **reverted** | - | - | 2021: -28pp，2023: -46pp（誤殺強勢股）|
| 實驗 B2：精準退市過濾 | **reverted** | - | - | 2021: -19pp，2023: -34pp（仍有誤殺）|
| **實驗 C：clip -50%** | **+1216.35%** | **-33.35%** | **+0.9085** | **✅ 現行生產（commit 4440fc5）** |

**實驗 B 失敗原因**：零成交量過濾誤排除牛市反彈強勢股（如 2021/2023 多頭），任何「零成交量 →
退市徵兆」的邏輯在小型股高速輪動時均會造成重大誤殺。

**實驗 C 成功原因**：`max(ret, -0.50)` 一行修正，將退市股單月虧損從 -100% 截斷至 -50%。
等權 20 股組合下，一檔退市股原本造成 -5pp 額外損耗（-100% vs 真實 -50%），
10 年累計 39 個 clip 事件 × 2.5pp/事件 = ~97.5pp 累積損耗被消除 → +878pp 累積改善。

**關鍵修正碼（`skills/backtest.py`）**：
```python
ret = exit_px / entry_px - 1 - transaction_cost_pct - slippage_pct
ret = max(ret, -0.50)  # 單筆最大損失 clip -50%，防止退市股拖垮整月組合
stock_returns[sid] = ret
```

**ATR Bug Fix（`skills/backtest.py` L555，隨月頻修正同步提交）**：
```python
# 修正前：slippage 在 equal-weight + trailing stop 模式下靜默失效（atr_df=None）
if atr_stoploss_multiplier is not None or position_sizing == "vol_inverse":
# 修正後：
if atr_stoploss_multiplier is not None or position_sizing == "vol_inverse" or enable_slippage:
```

## 進出場（回測）規則

在 `skills/risk.py`：

- 固定停損：`stoploss_pct`
- 移動停利：`trailing_stop_pct`
- ATR 動態停損/停利容忍：`atr_stoploss_multiplier`
- 階段保護：
  - 峰值獲利 >= 10% 時收緊保護
  - 峰值獲利 >= 20% 時進一步收緊
- 時間汰弱：長時間不創高且報酬偏弱時提前出場
- 若均未觸發，於期末再平衡日平倉

## Multi-Agent 選股補充

`skills/multi_agent_selector.py` 的各 agent 分數合併邏輯：

- **`model_alignment_weight`**：當主模型分數可用時，加入 z-score 正規化後的模型分數作為
  額外 agent，權重由 `multi_agent_weights["model_alignment"]` 控制（預設 0）。
  由 `daily_pick.py` 傳入 `model_score_map`。

## 風控補充（`skills/risk.py`）

- **EMERGING 興櫃過濾（2026-03-13 新增）**：`get_universe()` 加入 `.where(Stock.market != "EMERGING")`，
  從選股 universe 排除興櫃股（2340 → 1965 股）。興櫃為議價交易，外資無法參與，
  `foreign_buy_*` 特徵永遠為 0，且流動性、漲跌機制與上市上櫃不同，不應納入。
  注意：`backtest.py` 不呼叫 `get_universe()`，此過濾僅影響生產選股，不影響回測結果。
- **ATR `min_periods`**：`compute_atr()` 的 ewm 加入 `min_periods=period`，新上市股票
  資料不足時 ATR 為 NaN，避免使用不可靠的早期估計。
- **流動性過濾新股保護**：`apply_liquidity_filter()` 在門檻 > 0 時，排除資料筆數 < 10
  的股票，避免新上市股票因樣本不足造成流動性誤判。

## Promotion Tracker（`scripts/update_promotion_tracker.py`）

A 級穩定性（`_evaluate_a_stability`）門檻說明：

- Shadow monitor 目前使用 `[1m, 3m, 6m]` windows。
- 以每月再平衡計算，1m 最多 2 個再平衡日、3m 最多 4 個，均無法達到 `rb_n >= 6`。
- 故 `qualified_windows >= 1`（實際上等同於 6m window 必須通過）。
- 若未來擴展為 `[6m, 12m, 18m]` windows，可恢復 `qualified_windows >= 3` 嚴格標準。

## 效能基準（2026-03-08 優化後）

### DB Index 現況
所有重要 index 已存在（`make check-index` 通過 13/13）：
- `raw_prices (stock_id, trading_date)` — 主鍵（PRIMARY）
- `raw_prices (trading_date)` — 次要
- `raw_prices (trading_date, stock_id)` — 覆蓋索引（GROUP BY COUNT(DISTINCT) 用）
- `raw_institutional / raw_margin_short` — 同上
- `picks (stock_id)` — 依股票查詢
- `jobs (started_at), (status)` — 工作查詢

### build_features.py 效能（10 年全量重建）
| 步驟 | 優化前 | 優化後 | 改善 |
|------|--------|--------|------|
| fetch prices | ~228s | ~228s | — |
| calc features | ~95s | ~95s | — |
| save to DB | ~1051s | ~220s（估）| **5x** |
| **總計** | **~23.3min** | **~9min（估）** | **~60%** |

優化方式：
- `iterrows()` → `numpy to_numpy() + zip`（25x，15k→380k rows/s）
- `BATCH_SIZE` 1000 → 5000（5x 減少 commit 次數）

### Dashboard 覆蓋率查詢
`fetch_recent_coverage()` 加 WHERE 日期範圍限制：
- raw_prices: 2.2s → 0.04s（53x）
- raw_institutional/margin: 2.0s → 0.013s（154x）

### DB 連線池
`app/db.py`: pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=1800

## 重要限制與已知風險

- `ingest_trading_calendar.py` 目前使用 weekday heuristic，尚未串官方 TWSE 行事曆。
- `ingest_corporate_actions.py` 外部來源尚未接妥，常見為 `adj_factor=1.0` 保底。
- 回測交易成本口徑需留意（單邊 vs 來回）並統一設定。
- 週頻回測（`rebalance_freq="W"`）與 20 日 label horizon 存在 4:1 mismatch，**不可使用**。
  現行預設 `rebalance_freq="M"`，與 label horizon 匹配。
- `pd.merge_asof(by="stock_id")` 需要 left key 全局單調，跨股資料不可直接使用，
