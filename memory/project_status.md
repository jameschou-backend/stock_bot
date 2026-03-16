# 專案現況

> 最後更新：2026-03-16

---

## 目前最佳回測結果

**Exp D（現行生產）**，2026-03-15 驗證：

| 指標 | 數值 |
|------|------|
| 期間 | 2016-05-03 ~ 2026-02-03 |
| 累積報酬 | **+2637.11%** |
| 年化報酬 | **+39.92%** |
| MDD | **-29.20%** |
| Sharpe | **1.042** |
| Calmar | **1.367** |
| 超額報酬 | +2583.43% |
| 勝率 | 45.45% |
| 交易次數 | 2009 筆 |

配置：月頻 + 無停損 + 漸進大盤過濾（-5%:×0.5, -10%:×0.25, -15%:×0.10）+ 最少 2 檔 + seasonal filter + label_horizon_buffer=20

---

## 目前已知問題

### 已確認但尚未修復

| 問題 | 影響範圍 | 說明 |
|------|----------|------|
| `backtest.py` 不呼叫 `get_universe()` | 回測 | 回測不過濾興櫃股，與生產行為不一致 |
| `ingest_trading_calendar.py` 用 weekday heuristic | pipeline | 尚未串接官方 TWSE 行事曆 |
| `ingest_corporate_actions.py` 外部來源未接妥 | 除權除息 | 常見 `adj_factor=1.0` 保底 |

### 已知架構注意事項

- `market_context_features`（5 個市場環境特徵）**不可**在 ProcessPoolExecutor worker 內計算
- `pd.merge_asof(by="stock_id")` 跨股資料需 per-stock groupby loop，不可用全域 merge
- `_get_rebalance_dates()` 預設參數是 `"W"`，但 `run_backtest()` 預設是 `"M"`（命名不一致但不影響結果）

---

## 待優化項目

### 近期可驗證

| 項目 | 說明 | 優先級 |
|------|------|--------|
| 突破進場（F+）在去偏版本驗證 | F+ 在含 label 洩漏版本 Sharpe 0.86，需在 buffer=20 正確基準重跑 | 高 |
| 特徵重要性分析 | 56 個特徵是否有負貢獻或冗餘，可考慮 SHAP 分析 | 中 |
| TWSE 行事曆接入 | 取代 weekday heuristic，提高假日判斷準確性 | 中 |

### 觀察中

| 項目 | 說明 |
|------|------|
| 2025 年表現 | Exp D 全年 +8.70%，策略整體轉弱；需觀察是否為市場環境變化或模型退化 |
| foreign_buy_streak<=3 | 微幅改善但差異不顯著，暫不採用 |

---

## 近期 Commit 摘要（最近 10 個）

| Hash | 說明 |
|------|------|
| be96c90 | feat: add daily frequency Strategy B（日頻策略實驗，結論劣於月頻）|
| d53c92f | experiment: entry signal refinement（Exp F/G/H/I，結論：RSI 過濾嚴重損害）|
| 164db20 | docs: update CLAUDE.md with current production config and experiment rule |
| e56efcb | docs: add strategy documentation |
| 5305365 | experiment: minimum position count protection（最少 2 檔保護）|
| 7aeb002 | experiment: gradual market filter（漸進式大盤過濾）|
| 632ae96 | experiment: ATR dynamic stop-loss and market filter |
| 8ab0427 | experiment: stop_loss tuning and momentum penalty |
| 3a7433a | feat: add generate_review_pack.py for backtest review |
| 42954ec | fix: strengthen breakthrough.py cond2 guard |

---

## 主要 scripts 清單

| 腳本 | 用途 |
|------|------|
| `scripts/run_backtest.py` | 主回測入口（walk-forward）|
| `scripts/backtest_daily.py` | 日頻策略回測（Strategy B）|
| `scripts/run_experiment_matrix.py` | 矩陣式多實驗批次回測 |
| `scripts/generate_review_pack.py` | 產生復盤報告 Markdown |
| `scripts/run_walkforward.py` | Walk-forward 驗證 |
| `scripts/ic_analysis_new_features.py` | IC/ICIR 特徵分析 |
| `scripts/compare_periods.py` | 不同期間績效比較 |
| `scripts/update_promotion_tracker.py` | 模型升級追蹤 |
| `scripts/run_shadow_monitor.py` | Shadow monitor 評估 |
| `scripts/diagnose_stock.py` | 個股診斷工具 |
