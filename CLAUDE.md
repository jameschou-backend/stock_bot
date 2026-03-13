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

### 預設回測參數（現行生產，2026-03-13 更新）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `rebalance_freq` | `"M"` | 月頻再平衡（匹配 20 日 label horizon） |
| `entry_delay_days` | `0` | 再平衡日當日收盤進場（原始基準） |
| `position_sizing` | `"equal"` | 等權重倉位 |
| `stoploss_pct` | `-0.07` | 固定停損 7% |
| `trailing_stop_pct` | `None` | 不啟用移動停利 |
| `atr_stoploss_multiplier` | `None` | 不啟用 ATR 動態停損 |
| `label_horizon_buffer` | `20` | 消除訓練標籤前向洩漏（label horizon = 20 交易日）|
| `enable_slippage` | `False` | 不啟用滑價模型 |
| `time_weighting` | `False` | 等權樣本（原始基準） |
| `enable_complex_filter` | `False` | 不啟用 RSI/熊市/200MA 等複雜過濾 |
| `enable_seasonal_filter` | `True` | 啟用季節性降倉（3/10月 topN×0.5，floor=5）|
| 單筆 clip | `-0.50` | `max(ret, -0.50)`：退市股最大虧損 -50% |

> **`label_horizon_buffer` 說明（2026-03-13 修正）**：
> 標籤定義為 `future_ret_h = close_{T+20} / close_T - 1`（20 交易日 forward return）。
> 當 `buffer=0` 時，訓練截止 `rb_date` 前 20 個交易日的樣本，其標籤涉及測試期收盤價 → **訓練標籤前向洩漏**。
> 改為 `buffer=20`（日曆天，≈14 個交易日）後，訓練截止日往前移，消除最嚴重的洩漏區間。
> `train_ranker.py` 的 `LABEL_HORIZON_BUFFER_DAYS` 同步改為 20（從 7）。

> **`enable_seasonal_filter` 說明**：2026-03-11 新增（commit b5974be），
> 對應 `daily_pick.py` 在 3/10 月永遠啟用季節性降倉的行為，解決 production ≠ backtest 不一致。
> `run_backtest.py` CLI 加 `--seasonal-filter` 旗標啟用。

> **現行 10y walk-forward 結果（Experiment F，去偏後真實基準，2026-03-13）**：
> 累積 **+205.17%**、大盤 **+55.88%**、超額 **+149.29%**、MDD **-32.62%**、
> Sharpe **+0.4893**、Calmar **+0.3674**、年化 **+11.99%**
> 期間：2016-05-03 ~ 2026-01-30，117 再平衡期，2019 筆交易，停損觸發 888 次
> ⚠️ 注意：前期 Experiment E 的 +10004% 含訓練標籤前向洩漏（buffer=0），本結果為去偏後真實績效。
>
> 逐年報酬（Experiment F，去偏基準）：
> | 年份 | 策略（F）| 大盤 | 超額 |
> |------|---------|------|------|
> | 2016 | -11.23% | +1.87% | -13.11% |
> | 2017 | +32.89% | +11.57% | +21.32% |
> | 2018 | -7.38% | -15.35% | +7.97% |
> | 2019 | +13.91% | +12.01% | +1.89% |
> | 2020 | +42.90% | +18.13% | +24.77% |
> | 2021 | +25.92% | +19.71% | +6.20% |
> | 2022 | -3.68% | -15.83% | +12.14% |
> | 2023 | +14.43% | +22.37% | -7.94% |
> | 2024 | -5.05% | +2.99% | -8.04% |
> | 2025 | +13.97% | -5.69% | +19.66% |
> | 2026 | +14.26% | +2.23% | +12.03% |

### 歷史基準對照

> ⚠️ **重要說明**：Experiment A~E 均使用 `label_horizon_buffer=0`，存在訓練標籤前向洩漏，
> 回測績效虛高。Experiment F 為 2026-03-13 去偏後的真實基準。

- **Experiment F（去偏，buffer=20，含 EMERGING 過濾，2026-03-13）**：
  累積 **+205.17%**、超額 +149.29%、MDD -32.62%、Sharpe +0.4893、Calmar +0.3674 ← **現行真實基準**
- **Experiment E（含bias，buffer=0，56 特徵，2026-03-11，⚠️ 標籤洩漏）**：
  累積 +10004.80%、超額 +9956.59%、MDD -27.57%、Sharpe +1.3028、Calmar +2.166
- **Experiment D（含bias，buffer=0，seasonal_filter，53 特徵，commit b5974be，2026-03-11，⚠️ 標籤洩漏）**：
  累積 +9552.75%、超額 +9504.54%、MDD -27.57%、Sharpe +1.2958、Calmar +2.1392
- **Experiment C（含bias，buffer=0，無 seasonal_filter，commit 4440fc5，2026-03-10，⚠️ 標籤洩漏）**：
  累積 +1216.35%、超額 +1167.99%、MDD -33.35%、Sharpe +0.9085、Calmar +0.8963

### 突破確認進場實驗（2026-03-13，`--breakthrough-entry`）

月底模型選股後，不立即進場，等每檔股票出現突破訊號才進場（最多等 10 個交易日），
無突破者由後排候選補位，補位仍無突破則持現金。

**突破條件（任一成立）**：
- 條件一（價格）：收盤 > 前 20 日最高收盤 AND 當日量 > 20 日均量 × 1.5
- 條件二（籌碼）：`foreign_buy_consecutive_days ≥ 3` AND 收盤 > 20 日均線

**效果說明（隱式市場擇時）**：
- 多頭市場：大多數股票快速突破 → 近全倉進場，不影響收益
- 空頭市場：少數股票突破 → 大量持現金 → 自動避開下跌股

**10y Walk-Forward 結果（Experiment F+，2016-05-03 ~ 2026-01-30，117 期）**：

| 指標 | 基準 F | 實驗 F+（突破進場）| 變化 |
|------|--------|-------------------|------|
| 累積報酬 | +205.17% | **+827.15%** | +622pp ✅ |
| 年化報酬 | +11.99% | **+25.35%** | +13pp ✅ |
| 超額報酬 | +149.29% | **+771.26%** | +622pp ✅ |
| MDD | -32.62% | **-22.47%** | -10pp ✅ |
| Sharpe | 0.4893 | **0.8581** | +0.369 ✅ |
| Calmar | 0.3674 | **1.1280** | +0.761 ✅ |
| 勝率 | 38.14% | **41.06%** | +2.9pp ✅ |
| 總交易 | 2,019 | 1,924 | -95 |

**逐年報酬（F+ vs F基準 vs 大盤）**：
| 年份 | F+（突破）| F（基準）| 大盤 | 超額（F+）|
|------|---------|---------|------|---------|
| 2016 | +2.42% | -11.23% | +1.87% | +0.55% |
| 2017 | +12.64% | +32.89% | +11.57% | +1.07% |
| 2018 | +2.88% | -7.38% | -15.35% | +18.22% |
| 2019 | +25.43% | +13.91% | +12.01% | +13.42% |
| 2020 | +56.08% | +42.90% | +18.13% | +37.95% |
| 2021 | +40.32% | +25.92% | +19.71% | +20.61% |
| 2022 | +16.30% | -3.68% | -15.83% | +32.13% |
| 2023 | +30.66% | +14.43% | +22.37% | +8.29% |
| 2024 | +7.87% | -5.05% | +2.99% | +4.88% |
| 2025 | +50.19% | +13.97% | -5.69% | +55.87% |
| 2026 | +15.52% | +14.26% | +2.23% | +13.29% |

**實作位置**：`skills/backtest.py`
- `_compute_breakthrough_map()`: 向量化批次計算所有候選股突破日（避免逐股逐日掃描）
- `run_backtest(..., enable_breakthrough_entry=True, breakthrough_max_wait=10)`
- `_simulate_period(..., per_stock_entry_dates)`: 支援各股獨立進場日

**CLI**：`python scripts/run_backtest.py --months 120 --seasonal-filter --breakthrough-entry`

**注意**：`backtest.py` 預設 `enable_breakthrough_entry=False`，不影響現有流程。
生產端 `daily_pick.py` 尚未整合突破過濾，需另外規劃。
2017 是唯一略遜年份（+12.64% vs +32.89%），強多頭市場可能延遲進場導致追高成本。

### 時間加權訓練（2026-03-08 新增，目前已停用）

`backtest.py` 訓練循環可選時間加權（`time_weighting=True`）：
- 近 1 年樣本（≤365 天）：`sample_weight = 2.0`
- 1~2 年樣本（365~730 天）：`sample_weight = 1.0`
- >2 年樣本（>730 天）：`sample_weight = 0.5`

> 結論：10y walk-forward 驗證（2026-03-10）顯示時間加權在 equal-weight + 月頻設定下
> 未帶來額外改善，故現行生產設定 `time_weighting=False`（等同原始基準）。

### 優化歷史記錄（2026-03 完整系列）

#### 24m 窗口實驗（2026-03-08/09，已知有 overfitting 問題）

| 版本 | 主要改動 | 超額報酬 | MDD | Sharpe |
|------|---------|---------|-----|--------|
| v1 | 基準線 | -13.04% | - | - |
| v2 | topN floor + 200MA 現金水位 + RSI 過濾 | -14.54% | - | - |
| v3 | 動態現金水位 | -14.64% | - | - |
| v4 | 時間加權訓練 + 市場特徵（全 0）| +7.02% | -46.79% | - |
| v5 | 全量市場特徵（非零）| -1.71% | -41.88% | -0.68 |
| v6 | Market Regime Switching | +24.71% | -34.79% | -0.18 |
| v7 | 移除 Regime Switching，週頻 + 時間加權 | +0.38%（10y） | -87.74% | -0.36 |

> **根本問題（2026-03-10 發現）**：v2-v7 全在 24m 窗口驗證，且 v7 使用**週頻再平衡（W）**
> 與 20 日 label horizon 造成 4:1 mismatch → 2016-2019 超額 -88.83%。

#### 10y walk-forward 正式驗證（2026-03-10）

| 版本 | 累積 | 超額 | MDD | Sharpe | 說明 |
|------|------|------|-----|--------|------|
| v7（週頻，錯誤）| -84.96% | +0.38% | -87.74% | -0.36 | 月頻修正前 |
| 月頻修正 | +67.37% | - | - | +0.26 | rebalance_freq W→M |
| 還原原始設定 | +338.31% | - | - | +0.56 | 移除 8 個「優化」|
| 實驗 B（退市過濾）| 已 revert | - | - | - | 2021/2023 誤殺強勢股 |
| **實驗 C（clip -50%）** | **+1216.35%** | **+1167.99%** | **-33.35%** | **+0.9085** | **✅ 現行生產** |

### Market Regime Switching 研究紀錄（2026-03-09 實驗，已移除）

> **⚠️ 已移除**：`skills/market_regime.py` 標記為 `# NOT IN USE`，`backtest.py` / `daily_pick.py` 不再呼叫。
> 保留供未來研究參考。

**三態規則（向量化）**：
- bull: 等權指數 > 200MA AND 60日報酬 > 0 AND 20日報酬 > -5%
- bear: 等權指數 < 200MA AND 60日報酬 < -10%
- sideways: 其餘

**24m 實驗結果（2024-02 ~ 2026-01，大盤 -36.39%）**：超額 +24.71%，MDD -34.79%，Sharpe -0.18

**10y Walk-Forward 驗證（2026-03-09，週頻 v7，已過時）**：此版本使用週頻再平衡，有根本性 label-horizon mismatch，結果已無效。

**移除原因**：Sideways（35%）topN×75% 縮減造成長期損耗（超額 -0.21%/期），10y MDD 惡化。

### 10y 正式優化實驗（2026-03-10，月頻正確基準）

**背景**：2026-03-10 發現 v7 使用週頻再平衡（W）與 20 日 label horizon 造成根本性 4:1 mismatch，
修正為月頻（M）後重新在乾淨 10y walk-forward 上逐步驗證。

**實驗結果（期間 2016-03-16 ~ 2026-01-23，119 月頻再平衡期）**：

| 實驗 | 累積 | MDD | Sharpe | 說明 |
|------|------|-----|--------|------|
| v7（週頻，錯誤基準）| -84.96% | -87.74% | -0.36 | 根本錯誤，已廢棄 |
| 月頻修正（M）| +67.37% | - | +0.26 | rebalance_freq W→M |
| 還原 8 個原始設定 | +338.31% | - | +0.56 | equal/no-trailing/no-slippage 等 |
| 實驗 B：退市過濾 B1 | **reverted** | - | - | 2021: -28pp，2023: -46pp（誤殺強勢股）|
| 實驗 B2：精準退市過濾 | **reverted** | - | - | 2021: -19pp，2023: -34pp（仍有誤殺）|
| **實驗 C：clip -50%** | **+1216.35%** | **-33.35%** | **+0.9085** | **✅ 現行生產（commit 4440fc5）** |

**實驗 B 失敗原因**：零成交量過濾誤排除牛市反彈強勢股（如 2021/2023 多頭），任何「零成交量 →
退市徵兆」的邏輯在小型股高速輪動時均會造成重大誤殺。

**實驗 C 成功原因**：`max(ret, -0.50)` 一行修正，將退市股單月虧損從 -100% 截斷至 -50%。
等權 20 股組合下，一檔退市股原本造成 -5pp 額外損耗（-100% vs 真實 -50%），
10 年累計 39 個 clip 事件 × 2.5pp/事件 = ~97.5pp 累積損耗被消除 → +878pp 累積改善。

**關鍵修正碼（`skills/backtest.py`）**：
```python
ret = exit_px / entry_px - 1 - transaction_cost_pct - slippage_pct
ret = max(ret, -0.50)  # 單筆最大損失 clip -50%，防止退市股拖垮整月組合
stock_returns[sid] = ret
```

**ATR Bug Fix（`skills/backtest.py` L555，隨月頻修正同步提交）**：
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

- **EMERGING 興櫃過濾（2026-03-13 新增）**：`get_universe()` 加入 `.where(Stock.market != "EMERGING")`，
  從選股 universe 排除興櫃股（2340 → 1965 股）。興櫃為議價交易，外資無法參與，
  `foreign_buy_*` 特徵永遠為 0，且流動性、漲跌機制與上市上櫃不同，不應納入。
  注意：`backtest.py` 不呼叫 `get_universe()`，此過濾僅影響生產選股，不影響回測結果。
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
- 週頻回測（`rebalance_freq="W"`）與 20 日 label horizon 存在 4:1 mismatch，**不可使用**。
  現行預設 `rebalance_freq="M"`，與 label horizon 匹配。
- `pd.merge_asof(by="stock_id")` 需要 left key 全局單調，跨股資料不可直接使用，
  須改為 per-stock groupby loop。
- **`backtest.py` 不呼叫 `get_universe()`**：回測的候選股過濾只用 `stock_id.str.fullmatch(r"\d{4}")`，
  `risk.py` 的 EMERGING 過濾對回測無效。若需在回測中排除興櫃，須在 `backtest.py` 的 candidates 建立處另行過濾。
- **訓練標籤前向洩漏歷史（已修正）**：Experiment A~E 使用 `label_horizon_buffer=0`，
  訓練截止前 20 個交易日的標籤洩漏測試期收盤價，導致回測績效虛高（+10004% → 去偏後 +205%）。
  2026-03-13 改為 `label_horizon_buffer=20`，Experiment F 為去偏後真實基準。

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
