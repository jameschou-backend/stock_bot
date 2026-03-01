# Dashboard Picks Debug Report

## 結論摘要（Root Cause）

本次問題主要有兩個根因：

1. **Dashboard picks 區塊被 `DASHBOARD_SHOW_ML` 條件包住**  
   當 `DASHBOARD_SHOW_ML=false`（目前 `config.yaml` 預設），前台不會顯示 picks 區塊，即使 `picks` 表有資料。

2. **Data Quality 報表日期被未來交易日污染**  
   `data_quality` 報表日期先前取自 `trading_calendar` 的 `MAX(is_open=1)`，而該表有未來日期（到 2027），導致 `report_date=2027-02-26`，進而讓多數表出現 `missing_ratio=1` 與 `empty,stale_or_empty` 的失真紅燈。

---

## A) Root Cause 檢查

### A1. Dashboard picks 查詢邏輯

- 位置：`app/dashboard.py`
- picks 查詢：
  - `fetch_latest_pick_date()`：`SELECT MAX(pick_date) FROM picks`
  - `fetch_picks()`：`SELECT ... FROM picks WHERE pick_date = :pick_date ORDER BY score DESC`
- 原本問題：
  - picks 區塊放在 `if show_ml:` 裡（`show_ml` 由 `DASHBOARD_SHOW_ML` 控制）
  - 使用者若不是 ML 顯示模式，會看不到 picks 區塊
  - 若選到無資料日期，原本只顯示「當日無 picks 資料」並停止，不會自動回退到最新可用日期
- Data Quality RED 是否會隱藏 picks：
  - 程式上不直接用 RED 隱藏 picks
  - 但因 show_ml gate + 日期選擇邏輯，使用者體感會是「看不到 picks」

### A2. DB 實查結果（同一套 repo 連線設定）

查詢結果（本機現況）：

- `picks`: COUNT=80, MAX(pick_date)=2026-02-26
- `features`: COUNT=4,106,613, MAX(trading_date)=2026-02-26
- `raw_prices`: COUNT=4,750,064, MAX(trading_date)=2026-02-26
- `raw_institutional`: COUNT=3,543,368, MAX(trading_date)=2026-02-26
- `raw_margin_short`: COUNT=3,812,366, MAX(trading_date)=2026-02-26
- `raw_fundamentals`: COUNT=214,372, MAX(trading_date)=2026-02-01
- `data_quality_reports`: COUNT=35, MAX(report_date)=2027-02-26

補充檢查：

- `trading_calendar` 的 `MAX(is_open=1)` = 2027-02-26
- `trading_calendar` 未來 open days 數量 = 260
- `data_quality_reports` 未來日期列數 = 7

### A3. Dashboard 與 Pipeline 是否同 DB

- Dashboard 使用：`app/dashboard.py -> app.db.get_engine() -> app.config.load_config()`
- Pipeline 使用：`scripts/run_daily.py -> pipelines/daily_pipeline.py`（同樣透過 `app.config` / `app.db`）
- 目前實查連線配置：`mysql@127.0.0.1:3307/stock_bot`（帳密已設定）

結論：**dashboard 與 pipeline 使用同一個 DB 設定來源，非「查錯 DB」問題。**

### A4. 日期異常（為何出現 2027-02-26）

根因不是系統時區，也不是 `today` 直接寫錯；主因是：

- `data_quality` 的 report_date 先前優先拿 `trading_calendar` 最新開市日
- `trading_calendar` 已預先鋪到未來（2027）
- 因此報表以未來日期計算，對多數資料表在該日自然是 0 筆，導致 `missing_ratio=1` / `empty,stale_or_empty`

---

## B) Dashboard picks 顯示修正

已修改 `app/dashboard.py`：

1. **picks 區塊改為永遠顯示，不受 `DASHBOARD_SHOW_ML` 限制**
2. **日期 fallback 機制**
   - 若使用者選的日期無 picks，自動回退 `MAX(pick_date)`，並顯示提示
   - 顯示文字：`目前顯示最新可用選股日期：YYYY-MM-DD`（若非今日）
3. **picks 表為空時給明確原因**
   - 顯示 `picks table is empty`
   - 同時顯示 `picks/features` 計數與最新日期
   - 顯示 `daily_pick` / `build_features` 最近 job 狀態
4. **Data Quality RED 不隱藏 picks**
   - 若 RED 存在但 picks 有資料，顯示 warning badge，仍繼續顯示 picks
5. **Data Quality 查詢忽略未來報表日期**
   - `fetch_data_quality_reports()` 增加 `report_date <= CURDATE()`（SQLite 同步處理）

---

## C) Pipeline 缺口檢查（若 picks 真的為空時）

本次 DB 實查顯示 `picks` 非空（80 筆），所以「資料真的沒有」不是主因。  
但若未來遇到 picks 空，最短追查順序如下：

1. `features` 是否有產出（`features` count / max date）
2. `daily_pick` job 是否成功（jobs 表）
3. `labels` 缺失是否影響 picks：
   - 一般而言 picks 主要依賴 feature + model 推論，labels 多用於訓練/評估，不是 picks 寫入的直接硬依賴
4. `raw_fundamentals` 缺失是否中止：
   - 在 `research` 模式可 degraded，不一定中止整條流程（視 data_quality_mode 與當次 issue）

最短修復命令（依情況）：

- `make pipeline-build`（跳過 ingest，重跑 dq + features/labels/train/pick）
- `make pipeline`（完整流程）
- 只補報表：`make dq-report`

---

## D) 直接修復內容

已完成程式修正：

- `app/dashboard.py`
  - picks 區塊改為 always-on
  - 無資料日期自動 fallback 到最新可用 pick_date
  - picks 空表時顯示明確原因
  - Data Quality RED 不再阻斷 picks 顯示
  - Data Quality 查詢排除未來 `report_date`
- `skills/data_quality.py`
  - `_resolve_report_date()` 優先使用 `raw_prices` 最新有資料日期（並 cap 到 today）
  - 避免 future calendar date 造成報表失真
- `app/market_calendar.py`
  - `get_latest_trading_day()` 查 `trading_calendar` 時限制 `<= today`

---

## E) 驗收

1. 主要根因（1~2 個）：
   - `DASHBOARD_SHOW_ML=false` 導致 picks 區塊直接不顯示
   - `data_quality report_date` 取到未來交易日，導致 RED 失真

2. 修改檔案：
   - `app/dashboard.py`
   - `skills/data_quality.py`
   - `app/market_calendar.py`
   - `docs/dashboard_picks_debug_report.md`

3. 修正後 dashboard picks 顯示狀態：
   - 可顯示最新可用 picks（非今日也可）
   - 若當日無 picks 會自動 fallback 並提示
   - 若 Data Quality RED 仍顯示 picks + warning

4. 若仍看不到 picks，剩餘必要步驟：
   - 確認有重新啟動 dashboard（`make dashboard`）
   - 若 DB 實際為空，執行：`make pipeline-build` 或 `make pipeline`
