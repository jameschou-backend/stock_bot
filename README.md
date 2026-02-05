# 台股波段 ML 選股系統 MVP

## 快速開始

1. 設定環境變數：
   - `cp .env.example .env`
   - 填入 `FINMIND_TOKEN` 與 DB 帳密（連線固定 `127.0.0.1:3307`, DB=`stock_bot`）。

2. 建表：
   - `make migrate`

3. **初始化歷史資料（首次使用必跑）**：
   - `make backfill-10y`（10 年完整回補，包含價格/法人/融資融券）
   - 支援中斷續傳，可隨時 Ctrl+C 再重新執行

4. 執行每日 pipeline：
   - `make pipeline`（會自動更新當日資料並產生選股）

5. 啟動 API：
   - `make api`

6. 啟動 Dashboard：
   - `make dashboard`

7. 產出報表：
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
| `raw_margin_short` | `TaiwanStockMarginPurchaseShortSale` | 日頻 | 融資融券（選用） |
| `stocks` | `TaiwanStockInfo` + `TaiwanStockDelisting` | 日更新 | 股票基本資料 |

### 股票 Universe 過濾
選股時會自動過濾：
- `security_type = 'stock'`（排除 ETF、權證等）
- `is_listed = 1`（排除已下市股票）

### Data Quality Check（L2 Hardening）
Pipeline 會在 ingest 後執行資料品質檢查，採用**比例 + 最小值雙門檻**：

1. **raw_prices**（必要）:
   - 最新 trading_date 落後 <= 1 交易日
   - 覆蓋率 >= 70%，每日 distinct stock_id >= 1200

2. **raw_institutional**（必要）:
   - 最新 trading_date 落後 <= 1 交易日
   - 覆蓋率 >= 50%，每日 distinct stock_id >= 800
   - 0050 最近 30 交易日筆數 >= 20（驗證日頻）

3. **raw_margin_short**（選用，不影響核心流程）:
   - 覆蓋率 >= 50%，每日 distinct stock_id >= 800

**門檻可在 `.env` 中配置：**
- `DQ_MIN_STOCKS_PRICES`: prices 每日最小股票數（預設 1200）
- `DQ_MIN_STOCKS_INSTITUTIONAL`: institutional 每日最小股票數（預設 800）
- `DQ_COVERAGE_RATIO_PRICES`: prices 覆蓋率門檻（預設 0.7）
- `DQ_MAX_LAG_TRADING_DAYS`: 允許最大落後交易日（預設 1）

### 常見資料不齊原因與解法

| 錯誤類型 | 原因 | 解法 |
|---------|------|------|
| `api_issue` | FINMIND_TOKEN 無效或過期 | 重新申請 token: https://finmind.github.io/ |
| `prices_insufficient` | Token 無權存取 TaiwanStockPrice | 升級 FinMind 會員等級 |
| `inst_insufficient` | Token 無權存取法人資料 | 升級 FinMind 會員等級 |
| `wrong_frequency` | 法人資料非日頻（月/週頻） | 確認 dataset 為 `TaiwanStockInstitutionalInvestorsBuySell` |
| `universe_too_small` | 每日股票數過少 | 檢查 token 權限或 API 限流 |
| `prices_stale` | 資料過舊 | 確認排程正常執行，或手動 `make pipeline` |
| `raw_prices empty` | 資料庫為空 | 執行 `make backfill-10y` |

### 歷史資料回補

**首次使用 - 10 年完整回補：**
```bash
# 估算 API 用量
make backfill-estimate

# 執行 10 年回補（prices + institutional + margin）
make backfill-10y

# 查看回補進度
make backfill-status
```

**日常增量更新：**
- `make pipeline` 會自動更新當日增量資料
- 若 DB 為空，會提示先執行 `make backfill-10y`

**其他回補指令：**
```bash
# 只回補 5 年（較快）
make backfill

# 只回補融資融券
make backfill-margin

# 指定日期範圍
python scripts/backfill_history.py --start 2016-01-01 --end 2026-02-05

# 重新開始（清除進度）
python scripts/backfill_history.py --reset --years 10
```

**API 用量優化說明：**

| 模式 | 10 年資料 API 次數 | 預估時間 |
|------|------------------|----------|
| 全市場模式（chunk=180天） | ~46 次 | ~2 分鐘 |
| 逐檔批次模式（chunk=180天） | ~800 次 | ~10 分鐘 |

**相關設定（`.env` 或環境變數）：**
- `FINMIND_REQUESTS_PER_HOUR`: 每小時 API 限制（付費 6000，免費 600）
- `CHUNK_DAYS`: 每個 chunk 天數（預設 180 天）
- `BACKFILL_YEARS`: 預設回補年數（預設 10）

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
