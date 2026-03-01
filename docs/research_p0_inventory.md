# Research P0 Inventory

## 1) Migration / Model / Pipeline 盤點

### `storage/migrations/*.sql`
- `storage/migrations/001_init.sql`
  - `raw_prices(stock_id, trading_date, open, high, low, close, volume)`
  - `raw_institutional(stock_id, trading_date, foreign_buy, foreign_sell, foreign_net, trust_buy, trust_sell, trust_net, dealer_buy, dealer_sell, dealer_net)`
  - `features(stock_id, trading_date, features_json)`
  - `labels(stock_id, trading_date, future_ret_h)`
  - `model_versions(model_id, train_start, train_end, feature_set_hash, params_json, metrics_json, artifact_path, created_at)`
  - `picks(pick_date, stock_id, score, model_id, reason_json)`
  - `jobs(job_id, job_name, status, started_at, ended_at, error_text, logs_json)`
- `storage/migrations/002_stock_master.sql`
  - `stocks(stock_id, name, market, is_listed, listed_date, delisted_date, industry_category, security_type, updated_at)`
  - `stock_status_history(id, stock_id, effective_date, status_type, payload_json, created_at)`
- `storage/migrations/003_margin_short.sql`
  - `raw_margin_short(stock_id, trading_date, margin_purchase_buy, margin_purchase_sell, margin_purchase_cash_repay, margin_purchase_limit, margin_purchase_balance, short_sale_buy, short_sale_sell, short_sale_cash_repay, short_sale_limit, short_sale_balance, offset_loan_and_short, note)`
- `storage/migrations/004_research_data.sql`
  - `raw_fundamentals(stock_id, trading_date, revenue_current_month, revenue_last_month, revenue_last_year, revenue_mom, revenue_yoy)`
  - `raw_theme_flow(theme_id, trading_date, turnover_amount, turnover_ratio, theme_return_5, theme_return_20, hot_score)`
- `storage/migrations/005_data_quality_reports.sql`
  - `data_quality_reports(report_date, table_name, expected_rows, actual_rows, missing_ratio, max_trading_date, notes, created_at)`
- `storage/migrations/005_strategy_factory.sql`
  - `strategy_configs(config_id, name, config_json, created_at)`
  - `strategy_runs(run_id, config_id, start_date, end_date, initial_capital, transaction_cost_pct, slippage_pct, metrics_json, created_at)`
  - `strategy_trades(run_id, trade_id, trading_date, stock_id, action, qty, price, fee, reason_json)`
  - `strategy_positions(run_id, trading_date, stock_id, qty, avg_cost, market_value, unrealized_pnl)`
- `storage/migrations/006_strategy_factory_strategy_name.sql`
  - `strategy_trades.strategy_name`（欄位補丁）
  - `strategy_positions.strategy_name`（欄位補丁）
- `storage/migrations/008_corporate_actions.sql`
  - `corporate_actions(id, stock_id, action_date, action_type, adj_factor, payload_json, created_at)`
  - `price_adjust_factors(stock_id, trading_date, adj_factor, created_at)`
- `storage/migrations/009_trading_calendar.sql`
  - `trading_calendar(trading_date, is_open, session_type, note, created_at)`

### `app/models.py`（主表模型）
- `Stock`, `StockStatusHistory`, `RawPrice`, `RawInstitutional`, `RawMarginShort`, `RawFundamental`, `RawThemeFlow`
- `Feature`, `Label`, `ModelVersion`, `Pick`, `Job`
- `StrategyConfig`, `StrategyRun`, `StrategyTrade`, `StrategyPosition`
- `CorporateAction`, `PriceAdjustFactor`, `TradingCalendar`

### `skills/build_features.py`
- 來源資料：`raw_prices + raw_institutional + raw_margin_short + raw_fundamentals + raw_theme_flow + price_adjust_factors`
- 主要流程：拉取資料 -> 計算技術/籌碼/基本面/題材特徵 -> 寫入 `features.features_json`

### `skills/daily_pick*.py / pipelines/* / scripts/*`
- `skills/daily_pick.py`: 模型打分、fallback 選股、流動性過濾、tradability 過濾、寫入 `picks`
- `pipelines/daily_pipeline.py`: ingest -> data quality -> feature/label -> train -> pick -> report
- `scripts/run_daily.py`: 每日 pipeline 入口
- `scripts/backfill_history.py`: 歷史回補
- `scripts/run_backtest.py`, `scripts/run_strategy_backtest.py`, `scripts/run_grid_backtest.py`, `scripts/run_walkforward.py`: 回測研究
- `scripts/research_factors.py`: 因子研究
- `scripts/migrate.py`: migration runner

## 2) 目前資料表與頻率

- **日頻**
  - `raw_prices`, `raw_institutional`, `raw_margin_short`, `features`, `labels`, `picks`, `data_quality_reports`, `price_adjust_factors`, `trading_calendar`
- **月頻 / 低頻**
  - `raw_fundamentals`（月營收資料，以日期對齊）
- **事件頻**
  - `stock_status_history`（上市/下市/狀態變更）
  - `corporate_actions`（除權息/分割/合併等）
- **設定/結果管理**
  - `model_versions`, `jobs`, `strategy_*`, `stocks`

## 3) 目前 feature columns（來自 `skills/build_features.py`）

- 核心欄位：
  - `ret_5`, `ret_10`, `ret_20`, `ret_60`
  - `ma_5`, `ma_20`, `ma_60`
  - `bias_20`, `vol_20`, `vol_ratio_20`, `amt_20`
  - `foreign_net_5`, `foreign_net_20`, `trust_net_5`, `trust_net_20`, `dealer_net_5`, `dealer_net_20`
- 擴充欄位：
  - `rsi_14`, `macd_hist`, `kd_k`, `kd_d`
  - `margin_balance_chg_5`, `margin_balance_chg_20`, `short_balance_chg_5`, `short_balance_chg_20`, `margin_short_ratio`
  - `market_rel_ret_20`, `breakout_20`, `drawdown_60`
  - `amt`, `amt_ratio_20`
  - `foreign_buy_streak_5`, `chip_flow_intensity_20`
  - `fund_revenue_mom`, `fund_revenue_yoy`, `fund_revenue_trend_3m`
  - `theme_turnover_ratio`, `theme_return_20`, `theme_hot_score`

## 4) 目前 universe 過濾規則

- 來源 universe（`risk.get_universe`）：
  - `stocks.security_type == "stock"`
  - `stocks.is_listed == True`
- 代碼基本校驗（`daily_pick`）：
  - `stock_id` 必須符合 `^\d{4}$`
- Tradability hard filter（`tradability_filter` + `daily_pick`）：
  - 排除 `DELISTED`, `NOT_LISTED`, `HALT`, `SUSPEND`, `FULL_DELIVERY`, `DISPOSITION`
  - 缺狀態資料：允許通過，但記錄 warning 與缺失率
- 流動性過濾（`risk.apply_liquidity_filter`）：
  - 近 20 日平均成交值門檻（優先使用 `MIN_AMT_20`，回溯相容 `MIN_AVG_TURNOVER`）
