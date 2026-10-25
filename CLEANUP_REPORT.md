# CLEANUP_REPORT.md

## 2026-10-26：本機 MySQL 資料表壓縮與封存

範圍為 `poyue.mysql` 的 `127.0.0.1:3307/stock_bot`（MySQL 5.7.44）。
保留全部 39 張表的結構；本次沒有更改應用程式、排程、`.env` 或其他資料庫。

### 實際處理結果

以下大小使用主機檔案的 `st_blocks × 512`，單位為十進位 GB，並非資料庫估計列數或容量上限。

| 項目 | 清理前 | 清理後 | 處理方式 |
|------|-------:|-------:|----------|
| `features.ibd` | 15.230890 GB | 3.766485 GB | 無損壓縮，保留 4,653,980 筆全部資料與原有索引 |
| `news_sentiment.ibd` | 0.006361 GB | 0.000115 GB | 62,550 筆已停用研究資料完整封存後清空，再重建表檔 |
| `jobs.ibd` | 0.016777 GB | 0.011534 GB | 封存、刪除 2,023 筆舊成功紀錄，再重建表檔 |

三個目標表檔合計減少 **11,475,894,272 bytes（11.48 GB）**。
保留的壓縮資料備份共 **2,704,750,674 bytes（2.70 GB）**，因此目標檔案加備份的合計占用淨減 **8.77 GB**。
這是本次目標檔案的占用差額，不代表整台 Mac 的 `df` 可用空間增加同樣數量；全機同時有其他容器、程序及交換檔使用空間。

`features` 仍有 API、資料品質檢查及歷史腳本直接讀取，不能因為已有 Parquet 就刪除。
先取 2016–2026 各年 6,000 筆，共 66,000 筆，在獨立暫存資料庫比較壓縮設定，並驗證讀取、500 筆新增及修改。
4 KB 設定的樣本從 281.02 MB 降至 58.72 MB，原始資料 SHA-256 相同，才執行：

```sql
ALTER TABLE stock_bot.features
  ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=4,
  ALGORITHM=INPLACE, LOCK=NONE;
```

這是一次性的實體儲存格式調整，沒有加入會在每次 migrate 時重建大表的 migration。
壓縮增加部分 CPU 成本；正式表單次讀取當日 1,864 筆從 0.319 秒變成 0.458 秒，
讀取 2330 的 2,583 筆歷史資料從 0.157 秒變成 0.162 秒，兩組結果雜湊均相同。這些是當次觀測，並非通用效能保證。

### 刪除條件與保留資料

- `news_sentiment` 僅由 `scripts/archive/` 的已結案腳本引用，沒有現行 runtime 使用者。資料封存後保留空表結構。
- `jobs` 僅刪除 `status='success'`、非 `ai_assist:%`，且開始與結束時間均早於 `2026-07-28` 的紀錄。
  刪除前在交易內鎖定候選資料並比對備份列數與 SHA-256；當時保留 7,419 筆成功與全部 343 筆失敗紀錄。
- 持倉、成交、原始行情、新聞、調整因子、labels、模型與研究成果均保留；重疊索引亦保留。
- `.env`、11 年現行特徵 Parquet 及其餘受保護檔案，共 13 份檔案，在清理前後 SHA-256 一致。

### 備份、還原與驗收

備份位於專案外的 `~/Backups/stock_bot/db-cleanup-20261026/`，包含原始 CREATE TABLE、
JSONL gzip 資料、欄位順序、列數、SHA-256、還原程式與操作紀錄。這是受影響資料的備份，並非整個資料庫備份。
完整解壓檢查全部備份；另實際還原全部情緒與舊工作紀錄，以及 500 筆特徵資料至獨立暫存表。
還原程式保留 MySQL JSON 的負零數值表示，並只允許建立新的 `stock_bot_restore_*` 資料庫，不會覆寫現行表。

壓縮後逐筆讀取全部 4,653,980 筆特徵，以主鍵排序後的完整序列校驗，SHA-256 與壓縮前完全一致：

```text
ac69930eca139e97159284972d678a0988b04f3ba28a01ded8ad8e0f895bb2b7
```

- `make test`：**7,582 passed**，37 warnings，256.59 秒。
- `make pipeline`：完整執行成功，exit 0；保留現有 `SPONSOR_INGEST=off` 設定，未修改 `.env`。
- `make api` 後 `/health`、`/picks`、`/models`、`/jobs?limit=10` 均 HTTP 200。
- 額外查詢 `/stock/2330?date=2026-10-22`：HTTP 200，成功讀取壓縮表中的 87 個特徵欄位。
- 暫存資料庫已移除；表結構仍為 39 張，失敗工作紀錄仍為 343 筆。

本機完整證據另存於 `.cache/db-cleanup-20261026/`。Docker 官方空閒區塊回收亦已執行，
未觀察到額外明顯縮減；沒有執行全域 container/image/volume prune。復原資料請依備份目錄內 `README.txt` 與 `restore.py` 操作。

參考：[MySQL 5.7 線上 DDL](https://docs.oracle.com/cd/E17952_01/mysql-5.7-en/innodb-online-ddl-operations.html)、
[MySQL 5.7 資料表壓縮](https://docs.oracle.com/cd/E17952_01/mysql-5.7-en/innodb-compression-usage.html)、
[Docker Desktop 磁碟回收](https://docs.docker.com/desktop/troubleshoot-and-support/faqs/macfaqs/)。

---

## 2026-10-19：本機檔案清理與目錄說明

刪除 1,431 個未受 Git 追蹤的檔案，合計 2,944,116,158 bytes（約 2.94 GB／2.74 GiB）。
這是刪除當時的檔案大小；Python 與 pytest 快取在後續執行時會重新產生。

| 刪除項目 | 檔案數 | 依據 |
|----------|-------:|------|
| `artifacts/features.bak_preadj/`、`artifacts/features.bak_preohlc/` | 22 | 6／7 月修復前的特徵備份；專案紀錄已標示穩定後可刪。現行 FeatureStore 指向 `artifacts/features/`，2016–2026 共 11 份 Parquet metadata 均可讀；目前程式未讀取舊備份 |
| `artifacts/reports/all_signals_2019_20261002.xlsx.inspect.ndjson` | 1 | Excel 檢查用中間檔；同名原始 `.xlsx` 保留，程式與文件未引用中間檔 |
| 主要 Python 目錄內的 `*.pyc` | 1,401 | 可重新產生的位元碼 |
| 根目錄 `.pytest_cache/`、`.DS_Store` | 7 | 測試與 Finder 快取 |

刪除前記錄明細與大型刪除檔的 SHA-256 至本機 `.cache/project-cleanup-20261019/manifest.json`。
刪除後核對 19 份保留檔的 SHA-256 未變，包括 `.env`、持倉、11 年現行特徵、還原股價及相關帳本。
未刪除研究輸入、回測結果、續跑紀錄、trial registry、模型、MCP 環境或 MySQL 資料。
本次亦補上 README 目錄指南，並將 7 個本機資料產出目錄加入 `.gitignore`；忽略不代表可刪除。

---

## 2026-03-04：歷史清理與強化紀錄

> 以下為當時紀錄，非目前程式與驗收狀態。

---

## 一、專案健診發現

### 1.1 Bug 修復

| 檔案 | 問題 | 修復方式 |
|------|------|---------|
| `skills/train_ranker.py` line 157 | `train_end` 在賦值前被使用（`NameError` 風險）。原始寫法 `effective_train_end = train_end - timedelta(...)` 中 `train_end` 尚未定義 | 改為 `train_end = max_label_date - timedelta(days=LABEL_HORIZON_BUFFER_DAYS)`，直接從 `max_label_date` 計算 |
| `app/api.py` line 106 | 路由函式 `def get_strategy():` 命名與 line 43 的 `from skills.strategy_factory.registry import get as get_strategy` import 衝突，導致 line 243 的 `get_strategy(name)` 呼叫的是 FastAPI 路由函式而非 registry 函式（`TypeError`） | 路由函式重命名為 `strategy_doc()` |

### 1.2 未使用 Import / 死碼

經逐一檢查，核心模組（`app/`、`skills/`、`pipelines/`）中：
- 所有 import 均有被使用
- 私有函式（`_xxx`）均有被呼叫
- 無孤兒函式

`_LiquidityConfig` dataclass（`skills/risk.py` line 16）目前未被外部呼叫，僅作型別提示存在。標注如下（保留，非安全刪除項目）：

```python
# DEAD CODE? _LiquidityConfig 目前未被使用，原設計為 apply_liquidity_filter 的型別安全包裝
```

### 1.3 TODO / FIXME 位置（仍未修復的已知問題）

| 檔案 | 位置 | 內容 | CLAUDE.md 對應章節 |
|------|------|------|-------------------|
| `skills/ingest_trading_calendar.py` | line 25 | `"note": "TODO: replace with TWSE official calendar (holiday/half-day aware)"` | ✅ 已知，週末 heuristic 仍在使用 |
| `skills/ingest_corporate_actions.py` | line 35 | `# TODO: 接上正式來源（TWSE / MOPS / FinMind）` | ✅ 已知，`adj_factor=1.0` 保底 |

以上兩個 TODO 屬於資料來源整合問題，需接上外部 API，**非本次清理範圍**。

### 1.4 CLAUDE.md 已知問題狀態

| 問題 | 狀態 |
|------|------|
| `ingest_trading_calendar.py` 使用 weekday heuristic | ⚠️ 仍未修復（TODO 存在） |
| `ingest_corporate_actions.py` 外部來源未接妥 | ⚠️ 仍未修復（TODO 存在） |
| `pd.merge_asof(by="stock_id")` 全局單調限制 | ✅ 已正確使用 per-stock groupby loop |
| 回測交易成本口徑（單邊 vs 來回） | ✅ `backtest.py` 已標注「來回合計」 |

---

## 二、清理作業

### 2.1 Bug 修復
- `skills/train_ranker.py`: 修復 `train_end` 使用前未定義的 bug
- `app/api.py`: 修復 `get_strategy` 命名衝突

### 2.2 新增模組頂部說明 Docstring

以下檔案原本缺少模組級說明，已補充：

| 檔案 | 說明 |
|------|------|
| `skills/build_features.py` | 特徵工程模組用途說明 |
| `skills/daily_pick.py` | 每日選股模組用途說明 |
| `skills/risk.py` | 風控模組用途說明 |
| `skills/train_ranker.py` | 模型訓練模組用途說明 |
| `app/api.py` | FastAPI 應用端點說明 |

### 2.3 標注死碼（待人工確認）

| 位置 | 標注 | 說明 |
|------|------|------|
| `skills/risk.py` `_LiquidityConfig` | `# DEAD CODE?` | dataclass 目前無外部使用，可能是重構中間態 |

---

## 三、新增特徵（`skills/build_features.py`）

所有特徵均嚴禁資料洩漏（只使用當日可得資訊）。

### 3.1 外資連續買超天數 `foreign_buy_consecutive_days`

```python
is_buy = (group["foreign_net"] > 0).astype(int)
run_id = (is_buy != is_buy.shift()).cumsum()
group["foreign_buy_consecutive_days"] = is_buy * (is_buy.groupby(run_id).cumcount() + 1)
```

與既有 `foreign_buy_streak_5`（5 日窗格加總）不同，此特徵計算**真實連續天數**（例如連買 10 日為 10，中斷後重新計算）。無資料洩漏風險：只使用截至當日的歷史序列。

### 3.2 月營收 YoY 加速度 `fund_revenue_yoy_accel`

```python
fund_df["fund_revenue_yoy_accel"] = fund_df.groupby("stock_id")["fund_revenue_yoy"].diff(1)
```

計算位置：`_fetch_data()` 的 fund_df 處理階段，在 45 天公告延遲調整**之前**計算（`available_date = trading_date + 45 days`），透過 `merge_asof` 以 `available_date` 與 `trading_date` 對齊，確保不使用未公開資料。

正值代表 YoY 成長率加速（相較上月更強），負值代表放緩。

### 3.3 布林帶位置百分位 `boll_pct`

```python
boll_mid = close.rolling(20).mean()
boll_std = close.rolling(20).std()
boll_upper = boll_mid + 2 * boll_std
boll_lower = boll_mid - 2 * boll_std
boll_range = (boll_upper - boll_lower).replace(0, np.nan)
group["boll_pct"] = ((close - boll_lower) / boll_range).clip(0, 1)
```

0 = 價格在下軌，1 = 價格在上軌。使用 20 日標準差（2σ），並 clip 至 [0, 1] 處理突破情況。

### 3.4 近 60 日報酬偏態與峰態 `ret_60_skew` / `ret_60_kurt`

```python
group["ret_60_skew"] = daily_ret.rolling(60, min_periods=30).skew()
group["ret_60_kurt"] = daily_ret.rolling(60, min_periods=30).kurt()
```

使用 pandas rolling `.skew()` 和 `.kurt()`，`min_periods=30` 避免資料不足時的不可靠估計。正偏態代表右尾厚（偶爾大漲），正超額峰態（>0）代表尾部風險高。

### 3.5 價量背離信號 `price_volume_divergence`

```python
vol_ma_10 = volume.rolling(10).mean()
price_near_high = close >= close.rolling(10).max() * 0.98
price_near_low = close <= close.rolling(10).min() * 1.02
vol_shrink = volume < vol_ma_10 * 0.8
group["price_volume_divergence"] = (price_near_low & vol_shrink).astype(int) - (price_near_high & vol_shrink).astype(int)
```

- `+1`：正背離（股價接近 10 日低點但量縮，賣壓衰竭）
- `-1`：負背離（股價接近 10 日高點但量縮，上攻乏力）
- `0`：無明顯背離

---

## 四、Sharpe 提升措施

### 4.1 LOW_IC 特徵標注

以下特徵已在 `FEATURE_COLUMNS` 中加上 `# LOW_IC?` 註解，建議以實際 IC 數據確認後決定是否刪除或調整：

| 特徵 | 位置 | 標注理由 |
|------|------|---------|
| `vol_20` | `CORE_FEATURE_COLUMNS` | 波動率本身非方向性 alpha，IC 方向不確定 |
| `dealer_net_5`, `dealer_net_20` | `CORE_FEATURE_COLUMNS` | 自營商操作含避險性質，IC 不穩 |
| `drawdown_60` | `EXTENDED_FEATURE_COLUMNS` | 多為風險衡量指標，未必有預測力 |
| `fund_revenue_trend_3m` | `EXTENDED_FEATURE_COLUMNS` | 60 日滾動均值可能過度平滑 |
| `theme_hot_score` | `EXTENDED_FEATURE_COLUMNS` | 題材資料品質不穩（見 CLAUDE.md） |

**注意：這些特徵尚未被刪除，需實際計算 IC 後再決定。**

### 4.2 簡易滑價模型（`skills/backtest.py`）

在 `_simulate_period()` 加入 `enable_slippage: bool = True` 參數，計算邏輯：

```python
# 單邊滑價 = min(ATR/close * 0.1, 0.003)；來回（進 + 出）共 × 2
atr_pct = atr_value / entry_price
slippage_one_way = min(atr_pct * 0.1, 0.003)  # 上限 0.3%
slippage_total = slippage_one_way * 2  # 進出場各一次
```

報酬計算：`ret = exit_px / entry_px - 1 - transaction_cost_pct - slippage_pct`

大盤 benchmark 不套用滑價（`enable_slippage=False`）以保持指數特性。

### 4.3 Market Regime Filter 狀態確認

`AppConfig.market_filter_enabled: bool = True` 已為預設開啟。`load_config()` 中默認值為 `"true"`，**無需修改**。

---

## 五、仍需人工確認的項目

| 項目 | 類型 | 說明 |
|------|------|------|
| `skills/risk.py` `_LiquidityConfig` | DEAD CODE? | 目前未被使用，需確認是否可刪除 |
| `vol_20` | LOW_IC? | 計算實際 IC 後決定是否保留在 CORE（保留則無法訓練時被 dropna） |
| `dealer_net_5`, `dealer_net_20` | LOW_IC? | 考慮移至 EXTENDED（允許 NaN 填補），降低因缺資料被 dropna 的影響 |
| `drawdown_60` | LOW_IC? | 若 IC < 0.01 可考慮移除 |
| `fund_revenue_trend_3m` | LOW_IC? | 60 日窗口的平滑 YoY 均值，需驗證是否優於直接使用 `fund_revenue_yoy` |
| `theme_hot_score` | LOW_IC? | 題材資料品質不穩，建議在充足資料下計算 IC 後再決定 |
| `ingest_trading_calendar.py` TODO | 未修復 | 仍使用 weekday heuristic，假日不準確 |
| `ingest_corporate_actions.py` TODO | 未修復 | `adj_factor=1.0` 保底，除權息調整不準確 |
| `backtest.py` 無 `run()` 函式 | 設計確認 | 只有 `run_backtest()`，與其他 skill 的 `run()` 介面不一致。若未來需由 pipeline 呼叫，需補 wrapper |

---

## 六、測試結果

```
93 passed, 10 warnings in 1.29s
```

所有原有測試全部通過，無新增測試失敗。
