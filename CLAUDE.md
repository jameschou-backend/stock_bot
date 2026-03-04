# CLAUDE.md

本文件提供給 Claude / AI coding assistant 在本專案工作的上下文與操作規範。

## 專案定位

- 專案：台股波段 ML 選股系統（Python）
- 資料源：FinMind
- DB：MySQL（本機 `127.0.0.1:3307`，DB=`stock_bot`）
- 主流程：每日資料更新 -> 特徵/標籤 -> 模型訓練 -> 當日選股 -> API/報表

## 快速指令

- 建表：`make migrate`
- 首次完整回補（10 年）：`make backfill-10y`
- 每日流程：`make pipeline`
- 測試：`make test`
- 啟動 API：`make api`
- 啟動 Dashboard：`make dashboard`
- 產出報表：`make report`

## 核心流程與檔案

- Pipeline 入口：`pipelines/daily_pipeline.py`
  - 依序執行：
    1) `bootstrap_history`
    2) `ingest_stock_master`
    3) `ingest_trading_calendar`
    4) `ingest_prices`
    5) `ingest_institutional`
    6) `ingest_corporate_actions`
    7) `ingest_margin_short`（選用）
    8) `ingest_fundamental`（研究用，失敗不中斷）
    9) `ingest_theme_flow`（研究用，失敗不中斷）
    10) `data_quality`
    11) `build_features`
    12) `build_labels`
    13) `train_ranker`（依條件觸發）
    14) `daily_pick`
    15) `export_report`

- FinMind 封裝：`app/finmind.py`
- 特徵工程：`skills/build_features.py`
- 標籤建置：`skills/build_labels.py`
- 訓練：`skills/train_ranker.py`
- 每日選股：`skills/daily_pick.py`
- 回測：`skills/backtest.py`
- 風控與停損：`skills/risk.py`
- API：`app/api.py`

## 目前選股邏輯（daily pick）

預設 `selection_mode=model`：

1. 讀取最近 `fallback_days + 1` 個特徵日期。
2. 建立 universe（上市普通股）並套用 tradability filter。
3. 套用 20 日平均成交值流動性過濾（`min_amt_20` / `min_avg_turnover`）。
4. 若啟用 market regime filter，空頭時下修有效 `topn`。
5. 若最新日候選不足，往前 fallback。
6. 使用最新模型對候選股打分（`model.predict`）。
7. 可選擇過熱過濾（預設關閉）。
8. 依分數排序取 TopN，寫入 `picks`。

備註：
- 若 `selection_mode=multi_agent`，改走 `skills/multi_agent_selector.py`。
- 若 data quality 在 research mode degraded 且法人資料缺失，會使用研究用啟發式分數 fallback。

## 回測機制摘要

主回測在 `skills/backtest.py`，採 walk-forward：

- 以再平衡日（`W` 或 `M`）滾動評估。
- 每期僅使用 `trading_date < rb_date` 訓練資料，避免直接看未來。
- 每 `retrain_freq_months` 重訓模型。
- 進場可設定 `entry_delay_days`（預設 1）。
- 報酬扣除 `transaction_cost_pct`（主回測未含滑價模型）。
- 輸出：累積/年化報酬、MDD、Sharpe、Calmar、勝率、profit factor、交易紀錄與淨值曲線。

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

## 重要限制與已知風險

- `ingest_trading_calendar.py` 目前使用 weekday heuristic，尚未串官方 TWSE 行事曆。
- `ingest_corporate_actions.py` 外部來源尚未接妥，常見為 `adj_factor=1.0` 保底。
- 回測交易成本口徑需留意（單邊 vs 來回）並統一設定。
- 週頻回測時，年化/Sharpe 的期頻假設需檢查是否與實際頻率一致。

## 開發規範（務必遵守）

- 不可把任何 secret（`FINMIND_TOKEN`、DB 密碼、API key）寫入 repo。
- 不可修改 `.env`（只可改 `.env.example`）。
- 禁止使用會造成環境行為不一致的 silent fallback。
- `stock_id` 預設只允許四碼台股（`^\\d{4}$`），例外需註解清楚。
- features/labels 嚴禁資料洩漏（只能使用當日可得資訊）。

## 驗收規範

每次改動後至少跑：

1. `make test`

若涉及 pipeline / DB / ingest，需完整驗收：

1. `make test`
2. `make pipeline`（需可重跑，idempotent）
3. `make api`
4. `curl -s http://127.0.0.1:8000/health`
5. `curl -s "http://127.0.0.1:8000/picks"`
6. `curl -s "http://127.0.0.1:8000/models"`
7. `curl -s "http://127.0.0.1:8000/jobs?limit=10"`

## Commit / Push 規範

- 使用 Conventional Commits：`feat` / `fix` / `chore` / `docs` / `test`
- 每個明確問題小步提交，避免 WIP commit
- push 前至少確保 `make test` 通過
- 不可 force push；若回歸，優先用 `git revert`

## AI Assist 規範

- 僅在 `make pipeline` / `make test` / `make api` 失敗時啟用 AI assist 回問。
- 輸出位置：`artifacts/ai_prompts/`、`artifacts/ai_answers/`。
- prompt / answer 必須遮罩 secrets，不可洩漏敏感資訊。
- AI 回覆僅作建議，需轉為實際 patch 或 TODO 後再提交。
