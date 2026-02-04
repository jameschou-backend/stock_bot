# 台股波段 ML 選股系統 MVP

## 快速開始

1. 設定環境變數：
   - `cp .env.example .env`
   - 填入 `FINMIND_TOKEN` 與 DB 帳密（連線固定 `127.0.0.1:3307`, DB=`stock_bot`）。

2. 建表：
   - `make migrate`

3. 執行每日 pipeline：
   - `make pipeline`

4. 啟動 API：
   - `make api`

5. 啟動 Dashboard：
   - `make dashboard`

## 測試
- `make test`

## API
- 詳細說明：`docs/api.md`
