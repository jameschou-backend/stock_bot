# 專案現況

> 最後更新：2026-03-16（session 2）

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

### Strategy C（日頻輪動）— C2 基準

| 指標 | 數值 |
|------|------|
| 期間 | 2016-03-28 ~ 2026-02-03 |
| 累積報酬 | **+18,395%** |
| 年化報酬 | **+72.68%** |
| MDD | -30.25% |
| Sharpe | **1.622** |
| Calmar | 2.403 |
| 交易筆數 | 1,259 |
| 年化成本 | **39.53%**（問題所在）|

配置：daily rotation，rank_threshold=0.20，max_hold=30d，top_entry=10

---

## 今日完成的實驗（2026-03-16）

| 實驗 | 結論 |
|------|------|
| C2 最小持倉（Hold5/10/15）| 無效，Force Exit 替代 Rank Drop，成本幾乎不降 |
| C2 風控出場 Risk v1 | MA Break 太靈敏，成本反升至 51.6% |
| C2 風控出場 Risk v2 | MA Break 更嚴重（81%出場），成本 80.8% |
| Oracle 分析 | 每筆平均少吃 23pp；最佳出場特徵：RSI=68、Bias=9%、外資剛中斷 |
| C2-Oracle v1 | Bug：_feat_map 未建立，訊號不觸發 |
| C2-Oracle v2（修 bug）| Boll 太靈敏，平均持倉縮至 8.2 天，成本 55% |
| C2-Oracle v3 | RSI>78 OR (RSI>72 AND ret5>10%)，Sharpe 1.156，仍遜基準 |

**核心結論**：Rank-based exit 是有效的隱含相對強度風控，所有風控出場版本均無法在 Sharpe 上超越 C2 基準（1.622）。

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
| **C2 雙門檻** | entry top10% / exit top30% | 成本降 30-40%，Sharpe 維持 1.5+ |

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
