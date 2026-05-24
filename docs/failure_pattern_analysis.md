# Failure Pattern Meta-Analysis（2026-05-24）

跨 8 個 NEGATIVE 實驗的系統性分析，目的：guide 未來方向選擇。

## 失敗實驗清單

| Stage | 改動類型 | 改動範圍 | 60mo Result | 10y Result |
|-------|---------|---------|-------------|-----------|
| 6.1 Stacking | Model ensemble（LGBM+XGB+CatBoost） | Model 架構 | IC +7.1% | Sharpe -0.32 ❌ |
| 6.2 Multi-Horizon | Label 平均（多 horizon） | Label 設計 | IC -2.2% | （未跑 10y） |
| 7.1 HRP | Position weighting（階層聚類） | Portfolio 加權 | ΔSharpe -0.13 | （未跑 10y） |
| 7.3 Kelly | Position weighting（μ/σ²） | Portfolio 加權 | ΔSharpe -0.10 | （未跑 10y） |
| 8.1a Pruned re-eval | Feature 回收 | 特徵集 | 0/20 候選 | - |
| 8.1b Combo features | 新訊號組合 | 特徵設計 | ICIR < 0.30 | - |
| 9.2 Optuna v2 | 超參數搜尋 | 配置 | Sharpe 0.68 | Sharpe -0.77 ❌ |
| 10.4 D1 dd_skip | Portfolio filter | 選股過濾 | Sharpe +0.02 | Sharpe -0.03, MDD -3.14pp ❌ |
| 10.5 D2 sector cap | Portfolio filter | 選股過濾 | - | Sharpe -0.02, cum -781pp ❌ |

## 成功實驗清單（對照）

| Stage | 改動類型 | 改動範圍 | 結果 |
|-------|---------|---------|------|
| 10.1 topn 20→30 | Portfolio size | 配置 | Sharpe **+0.18** / MDD **+6.15pp** / cum **+1644pp** ✅ |
| 10.6 Beta-hedge（analysis-only） | 後處理（外部對沖） | metric | Sharpe **+0.15** / MDD **+9.6pp** ✅ |
| 7.2 Vol Targeting | Cash share 控制 | 配置（已整合 opt-in） | 10y Sharpe Δ +0.078 / MDD +4.29pp ✅ |

## 🔍 共通失敗 Pattern

### Pattern 1：複雜模型架構 ≠ 更高 portfolio Sharpe
- **案例**：6.1 Stacking, 6.2 Multi-Horizon, 7.1 HRP, 7.3 Kelly
- **共同點**：理論上有 alpha source，IC / 樣本層級驗證 marginal POSITIVE
- **失敗根因**：portfolio-level 與 cross-sectional IC 不同單位
  - IC 衡量「跨股 ranking 相關性」
  - portfolio Sharpe 衡量「top-K 等權報酬」
  - 兩者不總是 monotonic
- **教訓**：**任何 alpha source 必須 portfolio-level backtest 驗證**，IC 只能淘汰負面

### Pattern 2：60mo POSITIVE ≠ 10y POSITIVE
- **案例**：9.2 Optuna v2（60mo Sharpe 0.68 → 10y 0.38）、10.4 D1（60mo +0.02 → 10y -0.03）
- **共同點**：60mo 期間特定 regime（2021-2026）的 over-fitting
- **失敗根因**：60mo 缺 2017-2020 大牛市，找到的「優化」在缺失期間是 noise
- **教訓**：**60mo quick eval 只能淘汰 NEGATIVE，不能 confirm POSITIVE**

### Pattern 3：DD attribution 的「症狀」常不可治
- **案例**：10.3 → 10.4 / 10.5（找到 5301 暴雷 + 觀光餐旅集中，但 D1/D2 都失敗）
- **共同點**：個股暴雷 / 產業集中是真實事實，但 actionability 為零
- **失敗根因**：systemic regime 才是根本，特定症狀只是表徵
- **教訓**：**找到症狀不等於可治症狀**。Unconditional filter 救症狀通常會 hurt alpha

### Pattern 4：Unconditional filter 都 hurt 多頭策略
- **案例**：10.4 D1（unconditional dd skip）、10.5 D2（unconditional sector cap）
- **共同點**：「永遠啟用」的 filter
- **失敗根因**：strong regime 期間 filter 是 cost，weak regime 期間救不了 systemic
- **教訓**：**filter 應該 regime-aware**（強勢期 disabled、弱勢期 enabled）

### Pattern 5：Search space 的「未訪區」是 missing alpha
- **案例**：9.2 Optuna v2（search topn ∈ {10,15,20,25}，從未試 30 → 真正 winner 在 30）
- **共同點**：人為定義的 search space 排除了關鍵區域
- **失敗根因**：boundary assumption 太緊
- **教訓**：**search space 必須涵蓋 production baseline + 適度外推**

## 🎯 成功 Pattern

### Pattern S1：簡單 portfolio size 改動最有效
- **案例**：10.1 topn 20→30
- **共同點**：1-line config change，不改 model 不改 features
- **成功根因**：production 在 ridge 上，水平移動小步測試
- **教訓**：**先試最簡單的維度（topn, freq, weight）再試複雜的**

### Pattern S2：後處理 metric 有效（不需 backtest）
- **案例**：10.6 Beta-hedge
- **共同點**：不重跑 backtest，純對 result 後處理
- **成功根因**：揭露隱藏 alpha（106% alpha/total ratio）
- **教訓**：**後處理可低成本探索 portfolio 變數**（hedge / vol target / scaling）

### Pattern S3：行為對齊 production 一致
- **案例**：7.2 Vol Targeting（在現有 cash_ratio 之上 max）
- **共同點**：「在現有邏輯之上 add」而非「取代」
- **成功根因**：不破壞既有 alpha source
- **教訓**：**新 component 設計成「ridge 上的補強」**而非「取代既有 component」

## 📋 對未來實驗的指引

### ✅ 應該嘗試
1. **Regime-aware** 版本的 D1/D2/D3（之前 unconditional 失敗的）
2. **後處理 metric** 探索（不需 backtest 的低成本實驗）
3. **新資料源**（FinMind sponsor 籌碼、新聞 NLP、期貨指標）
4. **單維度小步調整**（topn 30→35/25、retrain freq、market filter thresholds）

### ⚠️ 應該謹慎
1. **Position weighting 改動**（HRP/Kelly 都失敗）→ equal weight 是 robust 基準
2. **Multi-model ensemble**（stacking 失敗）→ single LightGBM + checkpoint 已足
3. **Multi-horizon label**（6.2 失敗）→ 20d horizon 已是 sweet spot

### ❌ 不應該再試
1. **Unconditional filter**（D1/D2 都失敗）→ 必須 regime-aware
2. **複雜 ensemble combined**（multi-strategy ensemble combined 失敗）→ 拿單一最佳 sub 即可
3. **60mo only validation**（Optuna v2 + D1 都失敗）→ 必須 10y 直接驗證

## 🚀 真正可能突破 ceiling 的方向

### 短期（1-2 週可實驗）
1. **Regime-conditional D1**：只在大盤 < 200ma 啟用 dd_skip
2. **Regime-conditional D2**：只在 bear regime 啟用 sector cap
3. **Sponsor features 解鎖**（backfill 跑完後）
4. **期貨 leading features**（Week 2 規劃中）

### 中期（1-2 月可實驗）
5. **Regime-conditional Model**（bull/bear/sideways 各訓 LightGBM）
6. **Long-short market-neutral**（top-K vs bottom-K）
7. **動態 hedge ratio**（regime-aware）

### 長期（需新資料源或架構大改）
8. **DL Transformer**（替代 LightGBM）
9. **付費 alt data**（broker / sentiment / 衛星）
10. **Multi-asset 擴展**（ETF / 期貨 / 海外股）
