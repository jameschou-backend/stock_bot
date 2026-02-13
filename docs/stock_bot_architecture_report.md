# Stock Bot 專案架構報告

> 更新時間：2026-02-12  
> 專案路徑：`/Users/james.chou/JamesProject/stock_bot`

## 1) 專案目錄結構（Tree）

```text
stock_bot/
├── .env.example
├── .gitignore
├── AGENTS.md
├── Makefile
├── README.md
├── config.yaml
├── requirements.txt
├── .github/
│   └── workflows/nightly.yml
├── app/
│   ├── api.py
│   ├── dashboard.py
│   ├── config.py
│   ├── db.py
│   ├── models.py
│   ├── schemas.py
│   ├── strategy_doc.py
│   ├── finmind.py
│   ├── market_calendar.py
│   └── (ai_client/job_utils/rate_limiter...)
├── pipelines/
│   └── daily_pipeline.py
├── skills/
│   ├── ingest_stock_master.py
│   ├── ingest_prices.py
│   ├── ingest_institutional.py
│   ├── ingest_margin_short.py
│   ├── ingest_fundamental.py
│   ├── ingest_theme_flow.py
│   ├── data_quality.py
│   ├── build_features.py
│   ├── build_labels.py
│   ├── train_ranker.py
│   ├── daily_pick.py
│   ├── backtest.py
│   └── export_report.py
├── scripts/
│   ├── run_daily.py
│   ├── run_api.py
│   ├── run_backtest.py
│   ├── run_grid_backtest.py
│   ├── run_walkforward.py
│   ├── research_factors.py
│   ├── backfill_history.py
│   └── migrate.py
├── storage/
│   └── migrations/
│       ├── 001_init.sql
│       ├── 002_stock_master.sql
│       ├── 003_margin_short.sql
│       └── 004_research_data.sql
├── tests/
│   └── test_*.py
├── docs/
│   ├── api.md
│   ├── schema.md
│   └── stock_bot_architecture_report.md
└── artifacts/   # 產出目錄（ai_answers, ai_prompts, backtest, models, reports）
```

---

## 2) 主要模組說明（pipeline / feature / model / backtest / api / dashboard）

- `pipeline`：`pipelines/daily_pipeline.py`，串接 ingest → 檢核 → 特徵/標籤 → 訓練 → 選股 → 報表。
- `feature`：`skills/build_features.py`，將價格/法人/融資券/基本面/題材資料合併成特徵，寫入 `features`。
- `model`：`skills/train_ranker.py`，訓練排序模型並寫 `model_versions` 與模型檔（`artifacts/models`）。
- `backtest`：`skills/backtest.py`，做 walk-forward 月度再平衡回測，含停損與交易成本。
- `api`：`app/api.py`，提供 `/health` `/picks` `/models` `/jobs` `/stock/{id}` `/strategy`。
- `dashboard`：`app/dashboard.py`，Streamlit 視覺化看 picks、模型、回測摘要、風險燈號。

---

## 3) DB schema（所有 table 與欄位）

### `stocks`

- `stock_id` (PK)
- `name`
- `market`
- `is_listed`
- `listed_date`
- `delisted_date`
- `industry_category`
- `security_type`
- `updated_at`

### `stock_status_history`

- `id` (PK)
- `stock_id`
- `effective_date`
- `status_type`
- `payload_json`
- `created_at`

### `raw_prices`

- `stock_id` (PK)
- `trading_date` (PK)
- `open`
- `high`
- `low`
- `close`
- `volume`

### `raw_institutional`

- `stock_id` (PK)
- `trading_date` (PK)
- `foreign_buy`
- `foreign_sell`
- `foreign_net`
- `trust_buy`
- `trust_sell`
- `trust_net`
- `dealer_buy`
- `dealer_sell`
- `dealer_net`

### `raw_margin_short`

- `stock_id` (PK)
- `trading_date` (PK)
- `margin_purchase_buy`
- `margin_purchase_sell`
- `margin_purchase_cash_repay`
- `margin_purchase_limit`
- `margin_purchase_balance`
- `short_sale_buy`
- `short_sale_sell`
- `short_sale_cash_repay`
- `short_sale_limit`
- `short_sale_balance`
- `offset_loan_and_short`
- `note`

### `raw_fundamentals`

- `stock_id` (PK)
- `trading_date` (PK)
- `revenue_current_month`
- `revenue_last_month`
- `revenue_last_year`
- `revenue_mom`
- `revenue_yoy`

### `raw_theme_flow`

- `theme_id` (PK)
- `trading_date` (PK)
- `turnover_amount`
- `turnover_ratio`
- `theme_return_5`
- `theme_return_20`
- `hot_score`

### `features`

- `stock_id` (PK)
- `trading_date` (PK)
- `features_json`

### `labels`

- `stock_id` (PK)
- `trading_date` (PK)
- `future_ret_h`

### `model_versions`

- `model_id` (PK)
- `train_start`
- `train_end`
- `feature_set_hash`
- `params_json`
- `metrics_json`
- `artifact_path`
- `created_at`

### `picks`

- `pick_date` (PK)
- `stock_id` (PK)
- `score`
- `model_id`
- `reason_json`

### `jobs`

- `job_id` (PK)
- `job_name`
- `status`
- `started_at`
- `ended_at`
- `error_text`
- `logs_json`

> 註：目前主要為邏輯關聯，資料庫層未強制 FK。

---

## 4) 每日 pipeline 流程（資料流向）

- 入口：`scripts/run_daily.py` → `pipelines/daily_pipeline.py`
- Ingest：
  - `ingest_stock_master`
  - `ingest_prices`
  - `ingest_institutional`
  - `ingest_margin_short`
  - `ingest_fundamental`
  - `ingest_theme_flow`
- 檢核：`skills/data_quality.py`
- 建模資料：
  - `skills/build_features.py`（寫入 `features`）
  - `skills/build_labels.py`（寫入 `labels`）
- 訓練：`skills/train_ranker.py`
- 產出：
  - `skills/daily_pick.py`（寫入 `picks`）
  - `skills/export_report.py`（寫入 `artifacts/reports`）

---

## 5) 訊號產生邏輯

- 特徵訊號：`build_features` 產生動能、均線、波動、法人流、融資券、基本面、題材熱度等特徵。
- 標籤訊號：`build_labels` 以未來報酬 `future_ret_h` 建立監督訊號。
- 選股訊號：`daily_pick` 載入最新模型做 `predict score`，依分數排序取 Top-N。
- Universe 過濾：`security_type=stock` 且 `is_listed=True`。
- 流動性過濾：使用 `min_avg_turnover`（平均成交值門檻）。

---

## 6) 目前模型類型與訓練方式

- 模型類型：
  - 優先 `LightGBM Regressor`
  - fallback `sklearn GradientBoostingRegressor`
- 任務形式：回歸排序（目標值為 `future_ret_h`）。
- 訓練方式：時間序列切分（train + 近幾個月 validation）。
- 重訓節奏：由 pipeline 規則觸發（含週期重訓）。
- 輸出：
  - 模型檔：`artifacts/models`
  - 版本資訊：`model_versions`

---

## 7) 回測流程

- 核心：`skills/backtest.py` 的 `run_backtest()`
- 方法：walk-forward、月度再平衡、可設定重訓頻率。
- 交易規則：打分 → Top-N 等權持有 → 停損/期末出場。
- 成本假設：含交易成本（預設約 0.585%）。
- 輸出：
  - 單次回測：`artifacts/backtest/backtest_*.json`
  - 網格結果：`artifacts/ai_answers/grid_backtest_results.csv`
  - Walk-forward 匯總：`artifacts/ai_answers/walkforward_summary.csv`

---

## 8) 是否有 regime 判斷邏輯

- 有。
- `skills/daily_pick.py` 透過市場均價相對均線判斷空頭/多頭（`_is_bear_market()`）。
- 空頭時可套用 `bear_topn`，降低持股數量或風險曝險。
- `app/dashboard.py` 也有市場狀態（BULL/BEAR）顯示。

---

## 9) 是否有風控層（停損、倉位管理）

- 有停損：`stoploss_pct`。
- 有流動性風控：`min_avg_turnover`。
- 有持股分散控制：Top-N 等權配置。
- 目前未看到進階動態倉位（如 volatility targeting 或風險平價）。

---

## 10) 是否有模型評估指標（AUC、IC、Sharpe）

- `IC`：有（`train_ranker` 使用 Spearman；`research_factors.py` 也做月度 IC 分析）。
- `Sharpe`：有（`backtest.py` 計算策略 Sharpe）。
- `AUC`：目前無（專案主體為回歸排序，不是分類）。
- 其他指標：`annualized_return`、`max_drawdown`、`win_rate`、`profit_factor`、`excess_return`、`stoploss_triggered`。

---

## 11) 改造所需關鍵細節（A）主入口、主要 function、呼叫鏈

### `pipelines/daily_pipeline.py`

- 主入口：
  - `run_daily_pipeline(skip_ingest: bool = False)`
  - `_should_train(config)`：決定是否訓練（`force_train` / 無模型 / 週一）
- 主要 function：
  - `run_skill(skill_name, runner)`：統一包裝 skill 執行、DB session、錯誤 AI assist
- 呼叫鏈：
  - `run_daily_pipeline`
  - （可選）`bootstrap_history.run` → `ingest_stock_master.run` → `ingest_prices.run` → `ingest_institutional.run` → `ingest_margin_short.run`（可失敗不中斷）→ `ingest_fundamental.run`（可失敗）→ `ingest_theme_flow.run`（可失敗）
  - `data_quality.run` → `build_features.run` → `build_labels.run`
  - 若 `_should_train=True`：`train_ranker.run`
  - `daily_pick.run` → `export_report.run`

### `skills/build_features.py`

- 主入口：
  - `run(config, db_session, **kwargs)`
- 主要 function：
  - `_fetch_data(session, start_date, end_date)`：拉 `raw_prices`/`raw_institutional`/`raw_margin_short`/`stocks`/`raw_fundamentals`/`raw_theme_flow`
  - `_compute_features(df)`：逐股計算技術、法人、籌碼、基本面、題材特徵
  - 常數：`CORE_FEATURE_COLUMNS`、`EXTENDED_FEATURE_COLUMNS`、`FEATURE_COLUMNS`
- 呼叫鏈：
  - `run` → 區間推算（增量 + 回看 120 天）→ `_fetch_data` → `_compute_features`
  - 核心欄位完整性過濾 + 擴充欄位填補
  - 批次 upsert 寫入 `features`

### `skills/train_ranker.py`

- 主入口：
  - `run(config, db_session, **kwargs)`
- 主要 function：
  - `_parse_features(series)`：JSON 特徵轉矩陣
  - `_build_model(train_X, train_y, val_X, val_y)`：LightGBM 優先，fallback GBR
- 呼叫鏈：
  - `run` → 載入 `Feature`/`Label` 訓練窗 → `_parse_features` → 缺值處理
  - 時間切分 train/val（近 6 個月驗證）→ `_build_model`
  - 計算 `ic_spearman` / `topn_mean_future_ret` / importance
  - 序列化模型至 `artifacts/models`，寫 `model_versions`

### `skills/daily_pick.py`

- 主入口：
  - `run(config, db_session, **kwargs)`
- 主要 function：
  - `_is_bear_market`：市場均價 vs MA 判空頭
  - `_get_valid_stock_universe`：過濾 `security_type=stock` 且 `is_listed=True`
  - `_load_latest_model`、`_parse_features`、`_impute_features`
  - `_load_price_universe`、`_choose_pick_date`
- 呼叫鏈：
  - `run` → 取候選日與模型 → universe/filter → regime 決定 `effective_topn`
  - `_choose_pick_date`（含流動性門檻）→ `_impute_features` → `model.predict`
  - score 排序取 TopN → 覆寫當日 `picks`

### `skills/backtest.py`

- 主入口：
  - `run_backtest(...)`
- 主要 function：
  - `_load_all_data`：讀 `features`、`labels`、`raw_prices`
  - `_get_rebalance_dates`：每月首交易日
  - `_train_model`：回測內重訓
  - `_simulate_period`：單期報酬、停損、交易成本
- 呼叫鏈：
  - `run_backtest` → 決定回測窗與再平衡日
  - 每期：條件重訓 → 打分取 TopN → `_simulate_period`
  - 匯總輸出：`annualized_return`、`max_drawdown`、`sharpe_ratio`、`win_rate` 等

### `app/api.py`

- 主入口：
  - `app = FastAPI(...)`
- 主要 function：
  - `_to_float`、`_price_out`、`_inst_out`、`_feature_out`、`_pick_out`
- 呼叫鏈（endpoint）：
  - `GET /health`
  - `GET /strategy`（`load_config` + `get_selection_logic`）
  - `GET /picks`
  - `GET /stock/{stock_id}`
  - `GET /models`
  - `GET /jobs`
  - 各 endpoint：`get_session()` 查詢資料表後輸出 schema

### `app/dashboard.py`

- 主入口：
  - Streamlit 腳本頂層執行（非 class）
- 主要 function（查詢/監控）：
  - 新鮮度：`fetch_data_freshness`、`fetch_raw_price_freshness`
  - 覆蓋率：`fetch_recent_coverage`
  - 策略監控：`fetch_market_regime`、`load_latest_backtest_summary`、`fetch_hot_themes`
  - 選股展示：`fetch_picks`、`fetch_model`、`fetch_stock_detail`、`fetch_price_history`
- 呼叫鏈：
  - `load_config` + `get_engine` → 分區塊查詢 → 視覺化顯示資料層、pipeline 狀態、策略風險與 picks

---

## 12) 改造所需關鍵細節（B）`config.yaml` 所有 key 與用途

| Key | 目前值 | 用途 |
|---|---:|---|
| `FINMIND_TOKEN` | `""` | FinMind API token |
| `DB_DIALECT` | `mysql` | DB 方言 |
| `DB_HOST` | `127.0.0.1` | DB host |
| `DB_PORT` | `3307` | DB port |
| `DB_NAME` | `stock_bot` | DB 名稱 |
| `DB_USER` | `""` | DB 使用者 |
| `DB_PASSWORD` | `""` | DB 密碼 |
| `TOPN` | `20` | 每期選股數 |
| `LABEL_HORIZON_DAYS` | `20` | 標籤 horizon |
| `TRAIN_LOOKBACK_YEARS` | `5` | 訓練回看年數 |
| `SCHEDULE_TIME` | `16:50` | 排程執行時間 |
| `TZ` | `Asia/Taipei` | 時區 |
| `API_HOST` | `0.0.0.0` | API host |
| `API_PORT` | `8000` | API port |
| `BOOTSTRAP_DAYS` | `365` | bootstrap 檢查視窗 |
| `MIN_AVG_TURNOVER` | `0.5` | 20日平均成交值門檻（億元） |
| `FALLBACK_DAYS` | `10` | 選股日期回退天數 |
| `MARKET_FILTER_ENABLED` | `true` | regime 過濾開關 |
| `MARKET_FILTER_MA_DAYS` | `60` | 判斷多空 MA 天數 |
| `MARKET_FILTER_BEAR_TOPN` | `10` | 空頭時 TopN |
| `STOPLOSS_PCT` | `-0.07` | 停損比例 |
| `TRANSACTION_COST_PCT` | `0.00585` | 交易成本（來回） |
| `FINMIND_REQUESTS_PER_HOUR` | `6000` | FinMind 每小時請求上限 |
| `CHUNK_DAYS` | `180` | 抓取 chunk 天數 |
| `INSTITUTIONAL_BULK_CHUNK_DAYS` | `90` | 法人全市場 chunk |
| `MARGIN_BULK_CHUNK_DAYS` | `90` | 融資券全市場 chunk |
| `FINMIND_RETRY_MAX` | `3` | API 最大重試次數 |
| `FINMIND_RETRY_BACKOFF` | `1.0` | API 重試退避秒數 |

> 註：實際執行還會受 `AppConfig` 內額外 key 影響（例如 DQ 門檻、`BACKFILL_YEARS`），但未全數寫在 `config.yaml`。

---

## 13) 改造所需關鍵細節（C）`.env.example` 所有環境變數與用途

| Env Var | 範例值 | 用途 |
|---|---|---|
| `FINMIND_TOKEN` | `__在這裡填token__` | FinMind 金鑰 |
| `DB_DIALECT` | `mysql` | DB 方言 |
| `DB_HOST` | `127.0.0.1` | DB host |
| `DB_PORT` | `3307` | DB port |
| `DB_NAME` | `stock_bot` | DB 名稱 |
| `DB_USER` | `__在這裡填使用者__` | DB 帳號 |
| `DB_PASSWORD` | `__在這裡填密碼__` | DB 密碼 |
| `TOPN` | `20` | 每期選股數 |
| `LABEL_HORIZON_DAYS` | `20` | 預測 horizon |
| `TRAIN_LOOKBACK_YEARS` | `5` | 訓練回看年數 |
| `SCHEDULE_TIME` | `16:50` | 排程時間 |
| `TZ` | `Asia/Taipei` | 時區 |
| `OPENAI_API_KEY` | `__填你的key__` | AI assist 金鑰 |
| `OPENAI_MODEL` | `gpt-4.1-mini` | AI assist 模型 |
| `AI_ASSIST_ENABLED` | `0` | 是否啟用 AI assist |
| `AI_ASSIST_MAX_CODE_LINES` | `200` | AI assist 程式碼上限 |
| `AI_ASSIST_MAX_LOG_LINES` | `200` | AI assist log 上限 |
| `BOOTSTRAP_DAYS` | `365` | bootstrap 天數 |
| `MIN_AVG_TURNOVER` | `0.5` | 流動性門檻（億元） |
| `FALLBACK_DAYS` | `10` | 選股 fallback 天數 |
| `MARKET_FILTER_ENABLED` | `true` | regime 開關 |
| `MARKET_FILTER_MA_DAYS` | `60` | regime MA 天數 |
| `MARKET_FILTER_BEAR_TOPN` | `10` | 空頭 TopN |
| `STOPLOSS_PCT` | `-0.07` | 停損比例 |
| `TRANSACTION_COST_PCT` | `0.00585` | 交易成本 |
| `FINMIND_REQUESTS_PER_HOUR` | `6000` | API 限額 |
| `CHUNK_DAYS` | `180` | ingest chunk |
| `INSTITUTIONAL_BULK_CHUNK_DAYS` | `90` | 法人 chunk |
| `MARGIN_BULK_CHUNK_DAYS` | `90` | 融資券 chunk |
| `FINMIND_RETRY_MAX` | `3` | 最大重試 |
| `FINMIND_RETRY_BACKOFF` | `1.0` | 退避秒數 |

---

## 14) 改造所需關鍵細節（D）資料表規模與時間範圍、pipeline 耗時

### D-1. 每張表 row count / 時間範圍（目前實際值）

| Table | Rows | 範圍欄位 | 最小 | 最大 |
|---|---:|---|---|---|
| `stocks` | 3,033 | - | - | - |
| `stock_status_history` | 4,034 | `created_at` | 2026-02-05 16:24:19 | 2026-02-09 10:55:14 |
| `raw_prices` | 4,742,032 | `trading_date` | 2016-02-15 | 2026-02-11 |
| `raw_institutional` | 3,536,000 | `trading_date` | 2016-02-15 | 2026-02-11 |
| `raw_margin_short` | 3,805,075 | `trading_date` | 2016-02-15 | 2026-02-11 |
| `raw_fundamentals` | 0 | `trading_date` | - | - |
| `raw_theme_flow` | 0 | `trading_date` | - | - |
| `features` | 4,099,273 | `trading_date` | 2016-05-13 | 2026-02-11 |
| `labels` | 4,591,542 | `trading_date` | 2016-02-15 | 2026-01-14 |
| `model_versions` | 1 | `created_at` | 2026-02-12 03:20:32 | 2026-02-12 03:20:32 |
| `picks` | 20 | `pick_date` | 2026-02-11 | 2026-02-11 |
| `jobs` | 162 | `started_at` | 2026-02-04 03:22:37 | 2026-02-12 04:24:51 |

### D-2. daily pipeline 跑一次需要多久（目前估計）

- 近 30 天 `jobs` 平均耗時（秒）：
  - `ingest_prices`: 93.7
  - `ingest_institutional`: 135.4
  - `build_features`: 264.6
  - `build_labels`: 43.1
  - `train_ranker`: 10.4（僅重訓時）
  - `daily_pick`: 0.9
  - `ingest_margin_short`: 667.5（波動較大）
- 綜合估計：
  - 一般日（不含重型 margin）：約 **9~12 分鐘**
  - 含 margin 且較重：約 **20~35 分鐘**
  - 歷史補抓/資料延遲時：可能 **> 1 小時**

