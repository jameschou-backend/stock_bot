# CLAUDE.md

本文件提供給 Claude / AI coding assistant 在本專案工作的上下文與操作規範。

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
2. 建立 universe（上市普通股）並套用 tradability filter。
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

- 以再平衡日（`W` 或 `M`）滾動評估。
- 每期僅使用 `trading_date < rb_date` 訓練資料，避免直接看未來。
- 每 `retrain_freq_months` 重訓模型。
- 進場可設定 `entry_delay_days`（預設 1）。
- 報酬扣除 `transaction_cost_pct` + 滑價成本（`enable_slippage=True`，ATR 的 10%，上限 0.3%，來回各一次）。
- **`label_horizon_buffer = 7`**：訓練標籤截止日往前預留 7 天，避免近 `rb_date`
  的標籤使用到尚未公開的未來價格。`train_ranker.py` 採相同邏輯。
- **benchmark 一致性**：大盤基準套用與策略相同的流動性門檻（`min_avg_turnover`），
  確保 benchmark universe 不含低流動性股票。
- 輸出：累積/年化報酬、MDD、Sharpe、Calmar、勝率、profit factor、交易紀錄與淨值曲線。

### 預設回測參數（最佳化後）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `backtest_position_sizing` | `vol_inverse` | 反波動率加權，低波動股獲更大倉位 |
| `backtest_trailing_stop_pct` | `-0.12` | 移動停利，從峰值回落 12% 出場 |
| `stoploss_pct` | `-0.07` | 固定停損 |
| `label_horizon_buffer` | `7` | 標籤截止預留天數 |
| `enable_slippage` | `True` | 滑價模型開關（ATR 的 10%，上限 0.3%，來回合計）|

> 參考回測結果（walk-forward 10 年，2026-03-09 已修正 label_horizon_buffer + benchmark 流動性過濾後）：
> 累積 **-84.96%**、大盤 **-85.35%**、超額 **+0.38%**、MDD **-87.74%**、Sharpe **-0.36**
> 注意：早期 "+105%" 基準為修正前版本（無 label_horizon_buffer、benchmark 未套流動性過濾），已過時。

### 時間加權訓練（2026-03-08 新增）

`backtest.py` 訓練循環在每次 `_train_model` 呼叫前計算時間加權：
- 近 1 年樣本（≤365 天）：`sample_weight = 2.0`（強調近期市場規律）
- 1~2 年樣本（365~730 天）：`sample_weight = 1.0`（正常權重）
- >2 年樣本（>730 天）：`sample_weight = 0.5`（降低陳舊規律影響）

效果（walk-forward 24 月回測，市場環境特徵為 0）：超額報酬從 **-14.64%** 改善至 **+7.02%**，MDD 從 -55%+ 改善至 **-46.79%**。

### 風控層面優化記錄（2026-03 系列）

| 版本 | 主要改動 | 超額報酬 | MDD | Sharpe |
|------|---------|---------|-----|--------|
| v1 | 基準線 | -13.04% | - | - |
| v2 | topN floor (min 5/3) + 200MA 現金水位 30% + RSI 空頭過濾 | -14.54% | - | - |
| v3 | 動態現金水位 (0/10/30%) + 放寬 RSI 過濾 | -14.64% | - | - |
| **v4** | **時間加權訓練 + 市場環境特徵架構（特徵值=0）** | **+7.02%** | **-46.79%** | - |
| v5 | 全量 10 年重建市場環境特徵（非零值啟用） | -1.71% | -41.88% | -0.68 |
| **v6** | **Market Regime Switching（bull/sideways/bear）** | **+24.71%** | **-34.79%** | **-0.18** |
| **v7（現行）** | **移除 Regime Switching，回歸 v4 時間加權訓練** | **+0.38%（10y）** | **-87.74%** | **-0.36** |

結論：
- 風控層（RSI 過濾/現金水位）已優化至極限；真正的改善來自模型訓練品質（時間加權）。
- 市場環境特徵在 2024-2026 熊市測試窗口（大盤 -36.39%）中無法提升超額報酬。
  v4 的 +7.02% 實際上是「模型忽略 market_context=0 → 專注個股特徵」的結果。
- **v6 Regime Switching** 24m 超額 **+24.71%**（對 2024-2026 極端熊市有效），
  但 10y walk-forward 驗證顯示 sideways（35%）topN×75% 造成持續損耗（超額 -0.21%/期），整體有害。
- **v7（現行）**：移除 regime switching，回歸時間加權訓練。10y 超額 +0.38%，
  等同於跑贏同等條件大盤基準，但 Sharpe 仍負（測試窗口本身大盤極端走弱）。

### Market Regime Switching 研究紀錄（2026-03-09 實驗，已移除）

> **⚠️ 已移除**：`skills/market_regime.py` 標記為 `# NOT IN USE`，`backtest.py` / `daily_pick.py` 不再呼叫。
> 保留供未來研究參考。

**三態規則（向量化）**：
- bull: 等權指數 > 200MA AND 60日報酬 > 0 AND 20日報酬 > -5%
- bear: 等權指數 < 200MA AND 60日報酬 < -10%
- sideways: 其餘

**24m 實驗結果（2024-02 ~ 2026-01，大盤 -36.39%）**：超額 +24.71%，MDD -34.79%，Sharpe -0.18

**10y Walk-Forward 驗證（2026-03-09）**：

```
回測期間: 2016-03-28 ~ 2026-01-23  再平衡: 508期
累積: -82.16%  大盤: -85.35%  超額: +3.18%
MDD: -93.53%  Sharpe: -0.33
```

| Regime | 期數 | 佔比 | 平均超額 |
|--------|------|------|---------|
| bull | 288 | 57% | +0.40% |
| sideways | 180 | 35% | **-0.21%** ❌ |
| bear | 40 | 8% | -0.76% |

**移除原因**：Sideways（35%）topN×75% 縮減造成長期損耗（超額 -0.21%/期），10y MDD 惡化至 -93.53%。

**10y（無 Regime Switching，v7 現行，2026-03-09）**：

```
回測期間: 2016-03-28 ~ 2026-01-23  再平衡: 508期  交易: 8030次
累積: -84.96%  大盤: -85.35%  超額: +0.38%
MDD: -87.74%  Sharpe: -0.36  Calmar: -0.20
勝率: 39.56%  盈虧比: 1.40  停損觸發: 1082次
```

### 10y 增量優化實驗（2026-03-09，從乾淨 Baseline 逐步驗證）

**目的**：先前 v2-v7 全部在 24m 窗口驗證，導致系統性 overfitting。改以 10y walk-forward 為主要驗證窗口，每次只改一個變數。

**Baseline 設定**（`--baseline` flag）：
- equal sizing、trailing=-15%、no time_weighting、no complex_filter、no slippage
- 32 個 `BASELINE_FEATURE_COLS`（動量/均線/技術/法人/月營收）
- benchmark: 大盤 -85.35%（所有實驗共用）

**實驗結果（期間 2016-03-28 ~ 2026-01-23，508 再平衡期）**：

| 組別 | 累積 | 超額 | MDD | Sharpe | 勝率 | 停損 | 備注 |
|------|------|------|-----|--------|------|------|------|
| Baseline | -86.80% | -1.45% | -93.18% | -0.356 | 41.20% | 1534 | 乾淨基準 |
| **Change A (+IC 特徵)** | **-83.59%** | **+1.76%** | **-88.29%** | **-0.329** | 41.20% | 1538 | ✅ 採用 |
| Change B (topN floor=5) | -86.80% | -1.45% | -93.18% | -0.356 | 41.20% | 1534 | 🔄 等同 Baseline（no-op） |
| Change C (+slippage) | -99.26% | -13.92% | -99.32% | -0.820 | 36.70% | 1534 | ❌ 有害 |
| **v7（現行生產）** | -84.96% | +0.38% | -87.74% | -0.36 | 39.56% | 1082 | 含時間加權/vol_inverse |

**Change A 內容**（`CHANGE_A_FEATURE_COLS` = BASELINE + 3 IC 特徵）：
- `trust_net_5_inv`（信託淨買力 5 日反轉）
- `theme_turnover_ratio`（題材輪動比率）
- `fund_revenue_mom`（月營收環比動能）

**Change B 無效原因**：topN floor=5 從未觸發。Baseline 的熊市過濾 `topN → topN//3 = 20//3 = 6 ≥ 5`，floor 永遠不是下限。

**Change C 有害原因**：ATR 滑價模型（10%，上限 0.3%）在 equal-weight 模式下成本過高。
9326 筆交易，來回滑價累積造成 ~12pp 累積損耗。建議：滑價模型與 vol_inverse sizing 搭配使用，
不適合純 equal-weight 模式。

**ATR Bug Fix（本 session，`skills/backtest.py` L555）**：
```python
# 修正前：slippage 在 equal-weight + trailing stop 模式下靜默失效（atr_df=None）
if atr_stoploss_multiplier is not None or position_sizing == "vol_inverse":
# 修正後：
if atr_stoploss_multiplier is not None or position_sizing == "vol_inverse" or enable_slippage:
```

## 進出場（回測）規則

在 `skills/risk.py`：

- 固定停損：`stoploss_pct`
- 移動停利：`trailing_stop_pct`
- ATR 動態停損/停利容忍：`atr_stoploss_multiplier`
- 階段保護：
  - 峰值獲利 >= 10% 時收緊保護
  - 峰值獲利 >= 20% 時進一步收緊
- 時間汰弱：長時間不創高且報酬偏弱時提前出場
- 若均未觸發，於期末再平衡日平倉

## Multi-Agent 選股補充

`skills/multi_agent_selector.py` 的各 agent 分數合併邏輯：

- **`model_alignment_weight`**：當主模型分數可用時，加入 z-score 正規化後的模型分數作為
  額外 agent，權重由 `multi_agent_weights["model_alignment"]` 控制（預設 0）。
  由 `daily_pick.py` 傳入 `model_score_map`。

## 風控補充（`skills/risk.py`）

- **ATR `min_periods`**：`compute_atr()` 的 ewm 加入 `min_periods=period`，新上市股票
  資料不足時 ATR 為 NaN，避免使用不可靠的早期估計。
- **流動性過濾新股保護**：`apply_liquidity_filter()` 在門檻 > 0 時，排除資料筆數 < 10
  的股票，避免新上市股票因樣本不足造成流動性誤判。

## Promotion Tracker（`scripts/update_promotion_tracker.py`）

A 級穩定性（`_evaluate_a_stability`）門檻說明：

- Shadow monitor 目前使用 `[1m, 3m, 6m]` windows。
- 以每月再平衡計算，1m 最多 2 個再平衡日、3m 最多 4 個，均無法達到 `rb_n >= 6`。
- 故 `qualified_windows >= 1`（實際上等同於 6m window 必須通過）。
- 若未來擴展為 `[6m, 12m, 18m]` windows，可恢復 `qualified_windows >= 3` 嚴格標準。

## 效能基準（2026-03-08 優化後）

### DB Index 現況
所有重要 index 已存在（`make check-index` 通過 13/13）：
- `raw_prices (stock_id, trading_date)` — 主鍵（PRIMARY）
- `raw_prices (trading_date)` — 次要
- `raw_prices (trading_date, stock_id)` — 覆蓋索引（GROUP BY COUNT(DISTINCT) 用）
- `raw_institutional / raw_margin_short` — 同上
- `picks (stock_id)` — 依股票查詢
- `jobs (started_at), (status)` — 工作查詢

### build_features.py 效能（10 年全量重建）
| 步驟 | 優化前 | 優化後 | 改善 |
|------|--------|--------|------|
| fetch prices | ~228s | ~228s | — |
| calc features | ~95s | ~95s | — |
| save to DB | ~1051s | ~220s（估）| **5x** |
| **總計** | **~23.3min** | **~9min（估）** | **~60%** |

優化方式：
- `iterrows()` → `numpy to_numpy() + zip`（25x，15k→380k rows/s）
- `BATCH_SIZE` 1000 → 5000（5x 減少 commit 次數）

### Dashboard 覆蓋率查詢
`fetch_recent_coverage()` 加 WHERE 日期範圍限制：
- raw_prices: 2.2s → 0.04s（53x）
- raw_institutional/margin: 2.0s → 0.013s（154x）

### DB 連線池
`app/db.py`: pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=1800

## 重要限制與已知風險

- `ingest_trading_calendar.py` 目前使用 weekday heuristic，尚未串官方 TWSE 行事曆。
- `ingest_corporate_actions.py` 外部來源尚未接妥，常見為 `adj_factor=1.0` 保底。
- 回測交易成本口徑需留意（單邊 vs 來回）並統一設定。
- 週頻回測時，年化/Sharpe 的期頻假設需檢查是否與實際頻率一致。
- `pd.merge_asof(by="stock_id")` 需要 left key 全局單調，跨股資料不可直接使用，
  須改為 per-stock groupby loop。

## 開發規範（務必遵守）

- 不可把任何 secret（`FINMIND_TOKEN`、DB 密碼、API key）寫入 repo。
- 不可修改 `.env`（只可改 `.env.example`）。
- 禁止使用會造成環境行為不一致的 silent fallback。
- `stock_id` 預設只允許四碼台股（`^\\d{4}$`），例外需註解清楚。
- features/labels 嚴禁資料洩漏（只能使用當日可得資訊）。

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
