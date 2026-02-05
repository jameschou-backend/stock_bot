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

## 日頻必備資料說明

### FinMind Dataset
本系統使用以下 FinMind 日頻 Dataset：

| 資料表 | FinMind Dataset | 頻率 | 用途 |
|--------|-----------------|------|------|
| `raw_prices` | `TaiwanStockPrice` | 日頻 | 股價 OHLCV |
| `raw_institutional` | `TaiwanStockInstitutionalInvestorsBuySell` | 日頻 (21:00 更新) | 三大法人買賣超 |

### Data Quality Check
Pipeline 會在 ingest 後執行資料品質檢查：

1. **raw_prices**:
   - 最新 trading_date 不可落後超過 5 天
   - 最近 10 個交易日，每日 distinct stock_id >= 1200

2. **raw_institutional**:
   - 最新 trading_date 不可落後 raw_prices 超過 1 天
   - 最近 10 個交易日，每日 distinct stock_id >= 800
   - 0050 最近 30 交易日筆數 >= 20（驗證日頻）

### 常見資料不齊原因與解法

| 錯誤類型 | 原因 | 解法 |
|---------|------|------|
| `api_issue` | FINMIND_TOKEN 無效或過期 | 重新申請 token: https://finmind.github.io/ |
| `prices_insufficient` | Token 無權存取 TaiwanStockPrice | 升級 FinMind 會員等級 |
| `inst_insufficient` | Token 無權存取法人資料 | 升級 FinMind 會員等級 |
| `wrong_frequency` | 法人資料非日頻（月/週頻） | 確認 dataset 為 `TaiwanStockInstitutionalInvestorsBuySell` |
| `universe_too_small` | 每日股票數過少 | 檢查 token 權限或 API 限流 |
| `prices_stale` | 資料過舊 | 確認排程正常執行，或手動 `make pipeline` |

### Backfill 機制
- Fresh DB 或資料不足時，自動回補近 365 日曆天
- 可透過 `BOOTSTRAP_DAYS` 環境變數調整
- Backfill 結果記錄於 `jobs.logs_json`

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

### 本機 crontab 設定
- 週一到週五 17:30：`30 17 * * 1-5 /bin/bash -lc "bash <repo>/scripts/cron_daily.sh"`
- 測試方式：先改成每分鐘跑一次，例如 `* * * * * /bin/bash -lc "bash <repo>/scripts/cron_daily.sh"`
- 常見坑：cron PATH、venv、dotenv（需確保 `.env` 可被讀取）

### 注意事項
- Actions 使用臨時 MySQL 容器，每次 run 都是 fresh 資料。
- 若需要資料累積，請改成外部 DB（TODO）。

## API
- 詳細說明：`docs/api.md`
