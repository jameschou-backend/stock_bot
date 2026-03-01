# Research P0 Gap Report

## 已補齊的 4 個 P0 缺口

## 1) 公司行為 / 還原價
- 新增 `storage/migrations/008_corporate_actions.sql`
  - `corporate_actions`（事件表）
  - `price_adjust_factors`（每日還原因子）
- 新增 `skills/ingest_corporate_actions.py`
  - 提供 `run_date_range(date_range)` 與 `run(...)`
  - 當外部來源尚未接上時，會明確記錄 warning，並為既有 `raw_prices` 日期寫入 `adj_factor=1.0`
- `skills/build_features.py` 改為支援還原價流程
  - 在 `ret_* / ma_* / breakout_* / drawdown_* / vol_20` 前使用 `adj_close`（可由 `USE_ADJUSTED_PRICE` 開關控制）
  - 若因子缺失或未啟用，會在 job logs 記錄 `price_adjustment_mode` 與 warning
- 新增測試
  - `tests/test_adjusted_price_features.py`：驗證除權假跳空不污染 `ret_20/ma_20`

## 2) 可交易性狀態標準化 + 選股硬過濾
- 新增 `app/constants.py`
  - `StockStatusType` enum：`HALT, SUSPEND, DISPOSITION, FULL_DELIVERY, DELISTED, NOT_LISTED, OTHER`
  - `map_external_status(...)` 統一映射器
- 新增 `skills/tradability_filter.py`
  - `is_tradable(stock_id, asof_date) -> bool + reasons`
  - `filter_universe(df, asof_date) -> filtered_df`（可選回傳統計）
  - 缺狀態資料採「允許通過 + warning + 缺失率」策略
- 整合 `skills/daily_pick.py`
  - universe 決定點加入 tradability filter
  - `reason_json["_selection_meta"]` 與 job logs 記錄剔除比例、原因統計、缺失率
- 新增測試
  - `tests/test_tradability_filter.py`

## 3) 流動性（成交值）指標
- `skills/build_features.py` 新增：
  - `amt = close * volume`
  - `amt_20 = rolling_mean(amt, 20)`
  - `amt_ratio_20 = amt / amt_20`
- `risk.apply_liquidity_filter(...)` 與 `daily_pick` 整合
  - 新配置 `MIN_AMT_20`（元）優先
  - 向後相容舊配置 `MIN_AVG_TURNOVER`（億元）
  - 選股 metadata 記錄流動性剔除比例與剔除數
- 新增測試
  - `tests/test_liquidity_features.py`

## 4) 交易日曆
- 新增 `storage/migrations/009_trading_calendar.sql`
  - `trading_calendar(date, is_open, session_type, note, created_at)`
- 新增 `skills/ingest_trading_calendar.py`
  - 可 seed 過去 N 年日曆資料
  - 提供 `next_trading_day(...)`、`prev_trading_day(...)`
- `app/market_calendar.py` 與 `skills/data_quality.py` 調整
  - 優先使用 `trading_calendar` 推導交易日與 lag
  - 若無資料則回退舊邏輯（raw_prices/估算）
- pipeline 整合
  - `pipelines/daily_pipeline.py` 增加 `ingest_trading_calendar`

## 設定檔變更
- `config.yaml` 新增：
  - `USE_ADJUSTED_PRICE`
  - `MIN_AMT_20`
  - `ENABLE_TRADABILITY_FILTER`
- `app/config.py` 已加入對應欄位與讀取邏輯

## 仍需外部資料源的項目（已留 TODO）
- 公司行為（影響還原因子品質）
  - 候選來源：TWSE 除權除息公告、公開資訊觀測站（MOPS）、FinMind（若提供對應 dataset）
- 交易日曆（半日市/臨時休市）
  - 候選來源：TWSE 官方交易日曆（含颱風休市與半日市）

目前系統已可「跑通、回測、記錄與稽核」，但若要達到研究等級嚴謹，下一步應接上官方公司行為與交易日曆來源，讓 `adj_factor` 與 `session_type` 由真實資料驅動。
