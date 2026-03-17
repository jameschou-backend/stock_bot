# 專案現況

> 最後更新：2026-03-17（session 6）

---

## 目前最佳回測結果

### Strategy A（月頻，現行生產）— Exp D

| 指標 | 數值 |
|------|------|
| 期間 | 2016-05-03 ~ 2026-02-03 |
| 累積報酬 | **+2637.11%** |
| 年化報酬 | **+39.92%** |
| MDD | **-29.20%** |
| Sharpe | **1.042** |
| Calmar | **1.367** |
| 超額報酬 | +2583.43% |
| 交易次數 | 2009 筆 |

配置：月頻 + 無停損 + 漸進大盤過濾（-5%:×0.5, -10%:×0.25, -15%:×0.10）+ 最少 2 檔 + seasonal filter + label_horizon_buffer=20

### Strategy C（日頻輪動）— **Label-10 最佳配置**（2026-03-16 session 4 更新）

| 指標 | 0.3% 成本 | 0.585% 真實成本 |
|------|-----------|----------------|
| 期間 | 2016-03-28 ~ 2026-02-03 | 同左 |
| 累積報酬 | **+1,038,061%** | **+487,108%** |
| 年化報酬 | +163.21% | +143.18% |
| MDD | **-27.23%** | **-28.08%** |
| Sharpe | **1.987** | **1.847** |
| Calmar | 5.995 | 5.100 |
| 交易筆數 | 1,602 | 1,602 |
| 平均持倉 | 9.0 天 | 9.0 天 |
| 年化成本 | 50.3% | 98.1% |

配置：`--train-label-horizon 10`，rank_threshold=0.20，max_hold=30d

**C2 基準（舊）**：+18,395%，Sharpe 1.622，MDD -30.25%（20d label horizon，1,259 筆）

#### 流動性過濾後更保守的估計（2026-03-17 更新）

| 配置 | 累積報酬 | Sharpe | MDD | 說明 |
|------|---------|--------|-----|------|
| ~~buggy（buffer=10, 0.585%）~~ | ~~+487,108%~~ | ~~1.847~~ | ~~-28.1%~~ | ❌ 含前瞻偏差（已廢棄）|
| **fixed（buffer=20, 0.585%）** | **+267,029%** | **2.314** | **-25.6%** | **✅ 現行最佳，含微型股** |
| Liq-5千萬（0.585%）| +62,321% | 1.754 | -28.3% | 較保守 |
| Liq-1億（0.585%）| +26,197% | 1.541 | -35.2% | 中型股以上 |
| **Liq-1億+分級滑價** | **+12,523%** | **1.367** | **-37.2%** | **最接近真實可執行** |

alpha 高度集中在日均量 <1億 小型股；加過濾後報酬驟降 97%，但 Sharpe 仍優於 Strategy A（1.042）

---

## 今日完成的實驗（2026-03-16，sessions 3-4）

| 實驗 | 結論 |
|------|------|
| C2 最小持倉（Hold5/10/15）| 無效，Force Exit 替代 Rank Drop |
| C2 風控出場 Risk v1/v2 | MA Break 太靈敏，成本反升 |
| Oracle 分析 | 每筆平均少吃 23pp；最佳出場：RSI=68、外資剛中斷 |
| C2-Oracle v1~v3 | 皆遜於基準（Sharpe 1.156 max）|
| C2 雙門檻 | entry_threshold 無效；exit=30% Sharpe+0.026 但報酬-5600pp |
| **Quality Gate** | Sharpe 1.667（+0.045），+8,343pp ✅ |
| **Label-10**（核心突破）| Sharpe 1.987（+0.365），×56 總報酬 ✅✅ |
| **Label-10 真實成本** | Sharpe 1.847，+487,108% ✅ 仍遠優於 C2 |
| **Chip Exit 補充** | **0 次觸發**，與 Label-10 完全相同，放棄 |
| **流動性+滑價（session 5）** | Liq-1億+分級滑價：Sharpe 1.367，+12,523%；alpha 97% 來自微型股 |
| **嚴格 Walk-Forward v1（session 6）** | 🟠 存疑：5/8 Folds Sharpe>1.0；平均 Sharpe 1.518 |
| **Walk-Forward v2（session 6）** | 🟠 存疑：9/14 Folds（64.3%）；Sharpe 均值 1.810；大盤過濾無效（Rank Drop 已隱性替代）|
| **波動率過濾（session 6）** | 效果極小（-1 筆交易，Sharpe +0.016）；2020 +15pp；2024H2 無效；不列生產預設 |
| **分數穩定過濾（session 7）** | ❌ 全面劣於基準：Sharpe 1.83-1.90 vs 2.31；Stop Loss 21%；MDD -33% vs -26%；不採用 |

**核心發現**：Label-10（10日 label horizon 對齊 9日平均持倉）是最重要的單一改進。Rank Drop 比任何絕對指標更快捕捉弱化訊號。阻止 Rank Drop 出場的所有嘗試（risk/oracle/vol filter/score stability）均失敗。

---

## 目前已知問題

| 問題 | 影響範圍 | 說明 |
|------|----------|------|
| `backtest.py` 不呼叫 `get_universe()` | 回測 | 回測不過濾興櫃股，與生產行為不一致 |
| `ingest_trading_calendar.py` 用 weekday heuristic | pipeline | 尚未串接官方 TWSE 行事曆 |
| `ingest_corporate_actions.py` 外部來源未接妥 | 除權除息 | 常見 `adj_factor=1.0` 保底 |

---

## 待優化項目

### 最高優先（下次 session 優先跑）

| 項目 | 說明 | 預期效果 |
|------|------|----------|
| 突破進場（F+）去偏驗證 | 需在 label_horizon_buffer=20 正確基準重跑 | 在去偏後看是否仍有效 |
| Fold 12/14 失敗原因深挖 | 2023H2 風格轉換 + 2024H2 高波動成本 | 找出應對方案（如震盪市降頻）|

### 中優先

| 項目 | 說明 |
|------|------|
| 突破進場（F+）去偏驗證 | 需在 label_horizon_buffer=20 正確基準重跑 |
| TWSE 行事曆接入 | 取代 weekday heuristic |
| 特徵 SHAP 分析 | 56 個特徵是否有負貢獻 |

---

## 新增工具（2026-03-16）

| 工具 | 路徑 | 說明 |
|------|------|------|
| Oracle 分析 | `scripts/oracle_analysis.py` | 找出每筆交易最佳出場時機，分析特徵分佈 |
| 輪動回測 | `scripts/backtest_rotation.py` | Strategy C 日頻輪動，支援 rank/risk/oracle 三種出場模式 |

---

## 近期 Commit 摘要

| Hash | 說明 |
|------|------|
| c0d956b | docs: add mandatory memory update rules to CLAUDE.md |
| b008880 | feat: add minimum holding period to reduce trading costs |
| 6f82aa0 | docs: add persistent memory system |
| be96c90 | feat: add daily frequency Strategy B |
| d53c92f | experiment: entry signal refinement |

---

## 主要 scripts 清單

| 腳本 | 用途 |
|------|------|
| `scripts/run_backtest.py` | Strategy A 主回測（月頻 walk-forward）|
| `scripts/backtest_rotation.py` | Strategy C 日頻輪動（rank/risk/oracle 出場）|
| `scripts/backtest_daily.py` | Strategy B 日頻回測 |
| `scripts/oracle_analysis.py` | Oracle 最佳出場分析（新）|
| `scripts/generate_review_pack.py` | 產生復盤報告 |
| `scripts/run_walkforward.py` | Walk-forward 驗證 |
| `scripts/diagnose_stock.py` | 個股診斷工具 |
| `scripts/run_shadow_monitor.py` | Shadow monitor 評估 |
