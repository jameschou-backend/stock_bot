# Risk / Regime / Data Quality Refactor Plan

## 1. 現況問題清單

- Data Quality 可觀測性不足：目前僅執行檢核，缺少可回溯的日報表落地資料表。
- `raw_fundamentals` 目前為 0 rows，無法判斷是資料來源問題、排程問題或映射問題。
- `raw_theme_flow` 目前為 0 rows，且 dashboard 缺少長期可視化訊號。
- `labels.max_date` 落後於 `raw_prices.max_date`（`labels: 2026-01-14` vs `raw_prices: 2026-02-11`），可能影響訓練與選股時效性。
- Risk / Regime 規則散落於 `daily_pick.py` 與 `backtest.py`，可維護性與一致性風險高。
- model 評估指標偏單一，`metrics_json` 缺乏排序任務常見的 Top-K 與 hitrate 指標。

## 2. 改造目標（可量化）

- 可觀測性：新增 `data_quality_reports`，可查最近 30 天每張關鍵資料表的缺漏率、最新交易日、異常註記。
- 可擴充性：抽離 `skills/risk.py` 與 `skills/regime.py`，讓規則可重用與可插拔（保留預設行為）。
- 維護性：將共用邏輯集中於模組化介面，降低重複實作與行為分歧風險。
- 相容性：預設參數下，不應無故改變 daily picks 與 backtest 主結果；若變動需在 commit notes 註明原因。
- 回滾性：每個里程碑獨立 commit，可用 `git revert <commit>` 單點回退。

## 3. 里程碑規劃與修改檔案清單

### M1: Data Quality 落地（DB + Dashboard）

**目標**
- 新增可查詢的資料品質日報表，讓「今日資料異常」可追蹤、可視覺化。

**預計修改檔案**
- `storage/migrations/005_data_quality_reports.sql`（新增）
- `skills/data_quality.py`（修改）
- `app/dashboard.py`（修改）
- `tests/test_data_quality_reports.py`（新增）

**關鍵行為**
- `data_quality.run()` 結尾寫入 `data_quality_reports`（upsert）。
- `report_date` 以最新交易日為主，避免非交易日誤判。
- Dashboard 新增 Data Quality 區塊，提供 30 天紅黃綠灰燈號。

### M2: Risk Layer 抽離（daily_pick/backtest 共用）

**目標**
- 將流動性過濾、TopN、停損規則集中，維持既有預設行為不變。

**預計修改檔案**
- `skills/risk.py`（新增）
- `skills/daily_pick.py`（修改）
- `skills/backtest.py`（修改）
- `tests/test_risk.py`（新增）

**關鍵行為**
- `daily_pick` 與 `backtest` 共用 risk layer。
- `bear_topn` 邏輯保留，僅改為統一呼叫 `pick_topn`。

### M3: Regime 模組化（預設 MA Detector）

**目標**
- 讓市場狀態偵測可插拔，先保持 MA 判斷規則一致。

**預計修改檔案**
- `skills/regime.py`（新增）
- `skills/daily_pick.py`（修改）
- `app/config.py` 或 `config.yaml`（依現況最小修改）
- `app/dashboard.py`（修改）
- `tests/test_regime.py`（新增）

**關鍵行為**
- 新增 `BaseRegimeDetector` 與 `MovingAverageRegimeDetector`。
- 將 `daily_pick` 既有 MA 空頭判斷搬遷至 detector，並保留 `MARKET_FILTER_*` 行為。

### M4: Ranker 評估指標標準化（model_versions + Dashboard）

**目標**
- 擴充排序評估指標並落地於 `metrics_json`，便於模型比對與監控。

**預計修改檔案**
- `skills/train_ranker.py`（修改）
- `app/dashboard.py`（修改）
- `tests/test_train_metrics.py`（新增）

**關鍵行為**
- 保留 `ic_spearman`，新增 `topk_mean_future_ret`、`hitrate_at_k`、`pred_score_distribution`。
- `metrics_json` 使用版本化結構（`v: 1`），Dashboard 支援 graceful fallback。

## 4. 風險與回滾策略

### 主要風險

- DB migration 導致 pipeline 例外：新增表結構若與 ORM/SQL 使用不一致，可能中斷 daily pipeline。
- 行為漂移風險：抽離 risk/regime 過程可能引入排序、過濾或停損細節差異。
- Dashboard 相容性風險：舊資料缺欄位時，可能發生渲染錯誤。
- 訓練指標格式變更風險：舊版 `metrics_json` 解析邏輯可能失效。

### 控制策略

- 採最小修改原則：先抽介面、再接線，不重寫核心計算。
- 每個里程碑獨立 commit，完成一個里程碑就執行測試與驗收。
- 對於舊欄位/舊格式採顯性相容（graceful fallback），不使用 silent fallback 改變行為。

### 回滾方式

- 單里程碑回滾：`git revert <該里程碑commit>`
- 多里程碑回滾：依時間序由新到舊逐一 `git revert`
- 嚴禁 `reset --hard` 與 force push，確保主幹歷史可追溯。

## 5. 驗收清單（確認 daily pipeline 未破壞）

每個里程碑完成後執行：

1. 測試驗證
   - `pytest`（至少涵蓋當次新增測試）
2. Pipeline 最小可用驗證
   - `scripts/run_daily.py` 可執行至 `build_labels`/`daily_pick`（或等效 dry-run/mock）
3. 專案規範驗收（涉及 DB/pipeline/ingest 時）
   - `make test`
   - `make pipeline`
   - `make api`
   - `curl -s http://127.0.0.1:8000/health`
   - `curl -s "http://127.0.0.1:8000/picks"`
   - `curl -s "http://127.0.0.1:8000/models"`
   - `curl -s "http://127.0.0.1:8000/jobs?limit=10"`

## 6. Commit 策略

- M0: `docs: add refactor plan for risk/regime/data-quality`
- M1: `feat: persist data quality reports + show in dashboard`
- M2: `refactor: introduce risk layer and reuse in daily_pick/backtest`
- M3: `refactor: modularize market regime detector (default MA)`
- M4: `feat: add ranker eval metrics (topk/hitrate) and show in dashboard`

