# CLAUDE.md

本文件提供給 Claude / AI coding assistant 在本專案工作的上下文與操作規範。

## 強制規則

> ⚠️ 以下規則為最高優先級，任何情況下都必須遵守。

1. **每次 session 開始**：必須先讀取 `memory/` 下所有檔案（decisions.md、preferences.md、project_status.md）
2. **每次有重要發現或跑完回測**：立刻更新對應的 memory 檔案，不等到 session 結束
3. **每次 session 結束前**：更新 `memory/project_status.md` 和 `memory/decisions.md`，commit 並 push 到 main
4. **如果使用者說「結束」或「今天到這」**：自動觸發記憶更新流程，依序執行：
   - 更新 `memory/decisions.md`（新增實驗結果與結論）
   - 更新 `memory/project_status.md`（更新現況、待辦、下一步）
   - `git add memory/ && git commit -m "docs: update memory after session" && git push origin main`

---

## 專案簡介

台股波段 ML 選股系統。使用 LightGBM 以 20 日 forward return 為 label，每月再平衡選股（topN 等權），
配合漸進式大盤過濾、季節性降倉，實現超越大盤的長期報酬。
現行 10y walk-forward 結果（buffer 交易日制去偏後，2026-06-17 快照）：累積 **+1141%**、Sharpe **0.99**、MDD **-34%**（topn 30 + 流動性加權 + SHAP 剪枝 + buffer 交易日制）。

> ⚠️ **績效數字不可逐位重現**：adj_close 隨除權息回溯調整持續漂移，memory 記錄區間 +5115%（2026-05-23）~ +1444%（2026-04-23），依時間點與特徵集而異，重跑以實際輸出為準。當前 **FEATURE_COLUMNS=87、PRUNED_FEATURE_COLS=58**（`tests/test_production_invariants.py` 鎖定）。
> ⚠️ **已知回測偏差（2026-06 審計）**：raw_prices 缺 2016–2021 下市股（survivorship bias，日月光/矽品/樂陞等 0 rows）；回測以 T 日收盤價成交 T 盤後才公布的籌碼特徵（point-in-time 違反）；`label_horizon_buffer` 用日曆天未完全覆蓋 20 交易日 horizon（殘餘 ~6 交易日洩漏）。**絕對績效偏高估，作實盤資金配置前需先處理。**

## 持久記憶系統

**每次 session 開始時，請先讀取以下檔案（若與當前任務相關）：**

```
memory/decisions.md     — 策略決策、實驗結果、已試過的方向
memory/preferences.md   — 工作偏好、Git 規範、輸出路徑慣例
memory/project_status.md — 目前最佳結果、已知問題、待優化項目
```

**每次 session 結束時（有重要進展時），請更新上述記憶檔案：**
- 新增實驗結果 → `memory/decisions.md`
- 新的工作偏好或規範 → `memory/preferences.md`
- 進度或問題變化 → `memory/project_status.md`

---

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

## Stage 9 工程化工具（2026-05-22）

### 9.1 MLflow 實驗追蹤
回測自動寫入 `mlruns/`，跨 run 比較、artifact 保留：
```bash
python scripts/run_backtest.py --months 120 ... --mlflow --mlflow-experiment my_exp
mlflow ui --port 5000   # http://localhost:5000 dashboard
```
`--mlflow` 啟用，否則 byte-identical 跳過所有追蹤邏輯（`skills/mlflow_tracking.py`）。

### 9.2 Optuna 超參數搜尋
5 維 TPE 搜尋（topn / min_avg_turnover / vol_target / ensemble_n / liquidity_weighting），
SQLite 可續跑，自動進 MLflow tracking：
```bash
python scripts/optuna_search.py --n-trials 30 --months 60   # ~2-3h
python scripts/optuna_search.py --resume                     # 中斷續跑
```
搜尋完印 top-5，手動跑 `--months 120` 完整驗證 top-3 候選。

### 9.3 Prefect 工作流編排
`pipelines/prefect_flow.py` 將 daily pipeline 拆 5 個 task 群組，加 retry/checkpoint/Web UI：
```bash
python -m pipelines.prefect_flow              # 本地直接跑
prefect server start                          # Web UI :4200
prefect deploy ...                            # scheduled flow
```
原 `make pipeline` 走 `pipelines/daily_pipeline.py` 保留向後相容。

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
2. 建立 universe（上市/上櫃普通股，**排除興櫃 EMERGING**）並套用 tradability filter。
3. 套用 20 日平均成交值流動性過濾（`min_amt_20` / `min_avg_turnover`）。
4. 若啟用 market regime filter，空頭時下修有效 `topn`。
5. 若最新日候選不足，往前 fallback。
6. 使用最新模型對候選股打分（`model.predict`）。
7. 可選擇過熱過濾（預設關閉）。
8. 依分數排序取 TopN，寫入 `picks`。

備註：
- 若 `selection_mode=multi_agent`，改走 `skills/multi_agent_selector.py`。
- 若 data quality 在 research mode degraded 且法人資料缺失，會使用研究用啟發式分數 fallback。

## 特徵 Schema 管理

`skills/build_features.py` 採增量建置，每次只補算新日期。若 `FEATURE_COLUMNS` 新增欄位，
舊日期不會自動重算，需觸發補算機制：

- **自動偵測**：`_detect_schema_outdated()` 檢查 DB 最新一筆 `features_json` 的欄位數
  是否低於預期的 80%，若是則自動觸發往前 180 天重算。
- **手動強制**：設定 `force_recompute_days=N`（config 或 env），強制刪除並重算最近 N 天特徵。
- **fund revenue 45 天 publication delay**：月營收資料有約 45 天發布延遲，
  使用 `available_date = trading_date + 45 days` 進行 `merge_asof`，以 per-stock groupby
  方式執行（不可用全域 `merge_asof(by=stock_id)`，因為 `trading_date` 在跨股時非全局單調）。
- **新增特徵（2026-03-04）**：`foreign_buy_consecutive_days`、`fund_revenue_yoy_accel`（YoY 加速度，含 45 天延遲）、
  `boll_pct`（布林帶位置 0~1）、`price_volume_divergence`（價量背離 ±1/0）、
  `ret_60_skew`、`ret_60_kurt`（近 60 日報酬偏態/峰態）。
  觸發 schema_outdated 自動補算（或設 `force_recompute_days=180`）。
- **新增強勢訊號特徵（2026-03-11）**：`foreign_buy_streak`（外資連續高於20日均買量天數，嚴格版）、
  `volume_surge_ratio`（近5日均量/近20日均量，週成交量放大比例）、
  `foreign_buy_intensity`（近5日外資淨買超/近20日均量，負IC因子 ICIR=-0.578）。
  FEATURE_COLUMNS: 53→56。觸發 schema_outdated 自動補算（threshold 0.80→0.95）。
  10y 驗證（Experiment E）：累積 +10004.80%，Sharpe +1.3028，MDD -27.57%（vs D: +9552.75%, Sharpe +1.2958）。
- **新增市場環境特徵（2026-03-08）**：`market_trend_20`、`market_trend_60`（等權市場指數近 20/60 日報酬）、
  `market_above_200ma`（市場是否在 200 日均線以上，0/1，min_periods=40）、
  `market_volatility_20`（市場日報酬近 20 日波動率）、`sector_momentum`（產業近 20 日報酬 - 市場近 20 日報酬）。
  **架構注意**：這 5 個特徵須在 `_compute_features()` 的 `pd.concat(result_parts)` 後，
  以 `_compute_market_context_features(df)` + `_compute_sector_momentum(df, mkt_ctx_df)` 計算並 merge，
  **不可在 ProcessPoolExecutor worker 內計算**（無法存取全市場資料）。
  `calc_start` 從 120 天延長至 250 天以支援 200 日均線計算。
  觸發 schema_outdated 自動補算（或設 `force_recompute_days=180`）。
- **全量重建注意事項（2026-03-08）**：新增市場環境特徵後須執行全量 10 年重建，否則訓練資料存在 train-test 分佈偏移。
  正確做法：用腳本設 `FORCE_RECOMPUTE_DAYS=3650` env var（在 `load_config()` 之前），並加 `if __name__ == '__main__':` guard
  防止 macOS spawn multiprocessing workers 重複執行主邏輯。結果：4,198,531 rows，53 columns，2016-03-15 ~ 2026-03-04。

## 回測機制摘要

主回測在 `skills/backtest.py`，採 walk-forward：

- 以再平衡日（`M`，月頻）滾動評估。
- 每期僅使用 `trading_date < rb_date` 訓練資料，避免直接看未來。
- 每 `retrain_freq_months` 重訓模型。
- 進場：`entry_delay_days=0`，即再平衡日當日收盤進場（原始基準設定）。
- 交易成本：`transaction_cost_pct × 4.1`（含稅費），無滑價模型（`enable_slippage=False`）。
- 單筆最大虧損 clip -50%（`max(ret, -0.50)`），防止退市股拖垮整月組合。
- **benchmark 一致性**：大盤基準套用與策略相同的流動性門檻（`min_avg_turnover`），
  確保 benchmark universe 不含低流動性股票。
- 輸出：累積/年化報酬、MDD、Sharpe、Calmar、勝率、profit factor、交易紀錄與淨值曲線。

### 預設回測參數（現行生產，2026-03-15 更新，Exp D 配置）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `rebalance_freq` | `"M"` | 月頻再平衡（匹配 20 日 label horizon） |
| `entry_delay_days` | `0` | 再平衡日當日收盤進場（原始基準） |
| `position_sizing` | `"equal"` | 等權重倉位 |
| `stoploss_pct` | `0.0`（`--no-stoploss`） | 無固定停損，月底換股即出場 |
| `trailing_stop_pct` | `None` | 不啟用移動停利 |
| `atr_stoploss_multiplier` | `None` | 不啟用 ATR 動態停損 |
| `label_horizon_buffer` | `20` | 消除訓練標籤前向洩漏（label horizon = 20 交易日）|
| `enable_slippage` | `False` | 不啟用滑價模型 |
| `time_weighting` | `False` | 等權樣本 |
| `liquidity_weighting` | `True`（`--liq-weighted`） | 流動性加權訓練：sample_weight ∝ log(1+amt_20）|
| `feature_columns` | `PRUNED_FEATURE_COLS`（`--pruned-features`）| SHAP 剪枝：FEATURE_COLUMNS 87 → PRUNED 58 特徵（含後續新增 PER/fracdiff/news 等）|
| `enable_complex_filter` | `False` | 不啟用 RSI/熊市/200MA 等複雜過濾 |
| `enable_seasonal_filter` | `True` | 啟用季節性降倉（3/10月 topN×0.5，floor=5）|
| `market_filter_tiers` | `[(-0.05,0.5),(-0.10,0.25),(-0.15,0.10)]` | 漸進式大盤過濾 |
| `market_filter_min_positions` | `2` | 大盤過濾後最低持股數（防止單押集中風險） |
| 單筆 clip | `-0.50` | `max(ret, -0.50)`：退市股最大虧損 -50% |

> **生產 CLI 指令（2026-05-23 更新，topn 20→30）**：
> ```bash
> python scripts/run_backtest.py --months 120 --topn 30 --seasonal-filter --no-stoploss \
>   --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2 \
>   --liq-weighted --pruned-features
> ```
> 注意：`topn` 預設值已從 20 改為 30（app/config.py + skills/backtest.py）。
> 若 .env 顯式設了 `TOPN=20` 需手動移除或改為 30。

> **`label_horizon_buffer` 說明（2026-03-13 修正）**：
> 標籤定義為 `future_ret_h = close_{T+20} / close_T - 1`（20 交易日 forward return）。
> 當 `buffer=0` 時，訓練截止 `rb_date` 前 20 個交易日的樣本，其標籤涉及測試期收盤價 → **訓練標籤前向洩漏**。
> **2026-06-17 修正為「交易日制」**：cutoff 取 rb_date 前第 20 個「交易日」。先前用 20 日曆天
> （≈14 交易日）蓋不住 20 交易日 horizon，殘餘 ~6 交易日洩漏。`backtest.py` 用全交易日序列
> searchsorted、`train_ranker.py` 用 `Label.trading_date` distinct 取第 20 個交易日。buffer 值仍 20，語義由日曆天→交易日。

> **`enable_seasonal_filter` 說明**：2026-03-11 新增（commit b5974be），
> 對應 `daily_pick.py` 在 3/10 月永遠啟用季節性降倉的行為，解決 production ≠ backtest 不一致。
> `run_backtest.py` CLI 加 `--seasonal-filter` 旗標啟用。

> **現行 10y walk-forward 結果（buffer 交易日制去偏後，2026-06-17，topn=30）**：
> 累積 **+1140.68%**、大盤 **+67.56%**、超額 **+1073.12%**、MDD **-34.46%**、
> Sharpe **0.994**、Calmar **0.845**、年化 **+29.11%**
> 勝率 45.38%、交易次數 3028、期間 2016-07-06 ~ 2026-05-15
> 配置：topn=30 + 無停損 + 漸進大盤過濾 + 最少 2 檔 + 流動性加權 + SHAP剪枝 + buffer 交易日制
>
> ⚠️ 與舊 headline +5115%（2026-05-23, buffer 日曆天）**不可直接比較**：落差混雜
> (1) buffer 交易日制消除殘餘洩漏 (2) adj_close 一個月回溯漂移 (3) 資料延伸至 2026-05
> （含 2025-03~05 連跌 -9.7/-14.8/-11.3%）。Sharpe 0.99 與 2026-04-23 快照 0.949 同級。
> **survivorship bias（缺下市股）尚未回補，絕對績效仍偏高估。**
>
> **舊基準（Stage 10.1，2026-05-23, buffer 日曆天，含殘餘洩漏，已過時）**：
> 累積 +5115.80%、Sharpe +1.33、MDD -33.00%、Calmar +1.50
>
> 逐年報酬（現行基準，2026-03-18）：
> | 年份 | 策略 | 大盤 | 超額 |
> |------|------|------|------|
> | 2016 | +13.43% | +1.87% | +11.56% |
> | 2017 | +44.09% | +11.57% | +32.52% |
> | 2018 | +19.34% | -15.35% | +34.69% |
> | 2019 | +36.20% | +12.01% | +24.19% |
> | 2020 | +135.55% | +18.13% | +117.42% |
> | 2021 | +83.80% | +19.71% | +64.09% |
> | 2022 | -9.08% | -15.83% | +6.75% |
> | 2023 | +95.79% | +22.37% | +73.42% |
> | 2024 | +10.61% | +2.99% | +7.62% |
> | 2025 | +26.66% | -5.69% | +32.35% |
> | 2026 | +20.33% | +1.37% | +18.96% |
>
> **前期基準對照**（Experiment F，去偏 buffer=20，固定停損 -7%，2026-03-13）：
> 累積 +205.17%、Sharpe +0.4893、MDD -32.62%、年化 +11.99%

## 歷史實驗對照

完整實驗紀錄已遷至 [docs/experiments_history.md](docs/experiments_history.md)：
- Stage 6.1 / 6.2 / 7.1 / 7.3 / 8.1 / 9.2 / 10.4 / 10.5 NEGATIVE 對照表
- Stage 7.2 / 10.1 / 10.6 POSITIVE 對照表
- 突破確認進場實驗、Strategy B 日頻、優化歷史記錄
- 訓練標籤前向洩漏修正歷史

失敗 pattern meta-analysis：[docs/failure_pattern_analysis.md](docs/failure_pattern_analysis.md)

## 重要限制與已知風險

- `ingest_trading_calendar.py` 目前使用 weekday heuristic，尚未串官方 TWSE 行事曆。
- `ingest_corporate_actions.py` 外部來源尚未接妥，常見為 `adj_factor=1.0` 保底。
- 回測交易成本口徑需留意（單邊 vs 來回）並統一設定。
- 週頻回測（`rebalance_freq="W"`）與 20 日 label horizon 存在 4:1 mismatch，**不可使用**。
  現行預設 `rebalance_freq="M"`，與 label horizon 匹配。
- `pd.merge_asof(by="stock_id")` 需要 left key 全局單調，跨股資料不可直接使用，
  須改為 per-stock groupby loop。
- **訓練標籤前向洩漏歷史（已修正）**：Experiment A~E 使用 `label_horizon_buffer=0`，
  訓練截止前 20 個交易日的標籤洩漏測試期收盤價，導致回測績效虛高（+10004% → 去偏後 +205%）。
  2026-03-13 改為 `label_horizon_buffer=20`，Experiment F 為去偏後真實基準。

## 開發規範（務必遵守）

- 不可把任何 secret（`FINMIND_TOKEN`、DB 密碼、API key）寫入 repo。
- 不可修改 `.env`（只可改 `.env.example`）。
- 禁止使用會造成環境行為不一致的 silent fallback。
- `stock_id` 預設只允許四碼台股（`^\\d{4}$`），例外需註解清楚。
- features/labels 嚴禁資料洩漏（只能使用當日可得資訊）。
- **回測實驗自動更新規則**：每次回測實驗完成後，須同步更新以下兩處：
  1. **CLAUDE.md「預設回測參數」區塊**：若實驗結果優於現行生產配置且決定採用，
     更新參數表、生產 CLI 指令、現行結果數據和逐年報酬表。
  2. **CLAUDE.md「歷史基準對照」區塊**：將新實驗結果加入對照表，
     包含實驗名稱、配置摘要、累積報酬、MDD、Sharpe、Calmar。
  3. **docs/strategy.md**：若生產配置變更，同步更新策略文件的績效數據和優化歷程。

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
