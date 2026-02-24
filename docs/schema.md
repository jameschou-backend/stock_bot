# Schema 說明

本專案使用 MySQL 8.0，資料庫名稱 `stock_bot`。以下為 MVP 所需的最小表結構與用途說明。

## raw_prices
- 目的：台股日 K 原始價量資料。
- 主鍵：`(stock_id, trading_date)`
- 索引：`trading_date`
- 欄位：open/high/low/close/volume。

## raw_institutional
- 目的：三大法人買賣超原始資料（外資/投信/自營商）。
- 主鍵：`(stock_id, trading_date)`
- 索引：`trading_date`
- 欄位：buy/sell/net。

## features
- 目的：每日特徵向量（MVP 以 JSON 儲存）。
- 主鍵：`(stock_id, trading_date)`
- 索引：`trading_date`

## labels
- 目的：訓練標籤，`future_ret_h = close(t+H)/close(t) - 1`。
- 主鍵：`(stock_id, trading_date)`

## model_versions
- 目的：模型訓練版本資訊與指標。
- 主鍵：`model_id`
- 欄位：訓練區間、特徵雜湊、參數/指標 JSON、模型檔路徑、建立時間。

## picks
- 目的：每日 TopN 選股結果。
- 主鍵：`(pick_date, stock_id)`
- 索引：`pick_date`、`score`
- 欄位：分數、模型版本、理由 JSON。

## jobs
- 目的：技能與 pipeline 執行紀錄。
- 主鍵：`job_id`
- 欄位：名稱、狀態、起訖時間、錯誤訊息、logs JSON。

## strategy_configs
- 目的：組合式策略設定（規則、參數、權重）。
- 主鍵：`config_id`
- 欄位：名稱、設定 JSON、建立時間。

## strategy_runs
- 目的：策略回測/模擬執行記錄。
- 主鍵：`run_id`
- 索引：`config_id`、`(start_date, end_date)`
- 欄位：日期區間、初始資金、交易成本/滑價、績效 JSON、建立時間。

## strategy_trades
- 目的：交易明細。
- 主鍵：`(run_id, trade_id)`
- 索引：`trading_date`、`stock_id`
- 欄位：策略名稱、買賣動作、數量、成交價、手續費、原因 JSON。

## strategy_positions
- 目的：每日持倉快照（選用）。
- 主鍵：`(run_id, trading_date, stock_id)`
- 索引：`trading_date`
- 欄位：策略名稱、持倉數量、均價、市值、未實現損益。
