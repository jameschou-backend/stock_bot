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

6. 產出報表：
   - `make report`（輸出到 `artifacts/reports/<YYYY-MM-DD>/`）

## 測試
- `make test`

## L2 自動化（GitHub Actions）
### 需要設定的 Secrets
- `FINMIND_TOKEN`
- `MYSQL_ROOT_PASSWORD`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### 本機手動跑 L2
- `make migrate`
- `make pipeline`
- 報表輸出在 `artifacts/reports/<YYYY-MM-DD>/`

### 注意事項
- Actions 使用臨時 MySQL 容器，每次 run 都是 fresh 資料。
- 若需要資料累積，請改成外部 DB（TODO）。

## API
- 詳細說明：`docs/api.md`
