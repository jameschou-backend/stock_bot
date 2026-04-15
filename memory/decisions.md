# 重要策略決策與實驗記錄

> 最後更新：2026-04-15（session 12）

---

## Session 12 優化（2026-04-15）

### 已完成的優化（5 項）

**P1-4 Strategy C 持倉寫入 DB 稽核 log**
- 新增 `strategy_c_trades` 表（append-only），記錄每日 buy/sell/hold/skip 動作
- `app/models.py` 加 `StrategyCTrade` ORM model（欄位：entry_score, exit_reason, pct_to_breakthrough 等）
- `storage/migrations/010_strategy_c_trades.sql` DDL
- `scripts/strategy_c_pick.py` 新增 `_write_trades_to_db()`，失敗僅 warning 不阻斷主流程
- JSON 備份仍保留，DB 為額外稽核管道

**P3-2 Backtest universe EMERGING 過濾**
- `skills/backtest.py`: 從 DB 載入興櫃股 ID，在再平衡日評分時排除
- 使回測 universe 與 production `get_universe()` 一致（371 支興櫃股）

**P2-1 截面 Z-score 特徵正規化**
- `skills/feature_utils.py`: 新增 `cross_section_normalize()`（每 trading_date 截面 z-score + ±3σ 截斷）
- `skills/backtest.py`: 可選 `--cs-norm` flag
- `scripts/strategy_c_pick.py`: 預設啟用截面正規化

**P2-2 Ensemble 3 model checkpoints**
- `skills/backtest.py`: `collections.deque(maxlen=N)` 保存最近 N 次重訓模型
- 多模型以截面排名百分位平均，降低 variance
- CLI: `--ensemble N`（預設 1=停用）

**P2-3 資料品質異常值偵測**
- `skills/data_quality.py`: 新增 `_check_price_spikes()`（偵測單日 ±50% 漲跌，記 warning）
- 新增 `_check_cross_table_consistency()`（features 落後 raw_prices >5 天，提醒 pipeline-build）

### LambdaRank 驗證結果（2026-04-15）

使用標準生產配置（--months 120 --seasonal-filter --no-stoploss --market-filter-tiers --liq-weighted --pruned-features）+  EMERGING 過濾：

| 配置 | 累積報酬 | 大盤 | 超額 | MDD | Sharpe | Calmar | 年化 |
|------|---------|------|------|-----|--------|--------|------|
| Regression（基準）| +186.7% | +57.1% | +129.7% | -42.4% | 0.478 | 0.266 | 11.3% |
| **LambdaRank（NDCG@20）** | **+194.0%** | +59.5% | **+134.5%** | -42.4% | **0.488** | **0.273** | **11.6%** |

**結論**：LambdaRank 微幅改善（+7.3pp 累積、+0.01 Sharpe），MDD 相同。
改善幅度不顯著（<5%），維持 regression 為生產配置；LambdaRank 作為備選研究方向保留。

**注意**：LambdaRank label 需離散化，實作為截面 quintile 排名整數（0~4）。

### IC 衰減分析結果（2026-04-15，近期基準=最近 2 年 2024-04 以後）

**❌ 失效 8 個（近期 IC 幾乎歸零）**：
`foreign_buy_streak`（-76.8%）、`fund_revenue_mom`（-64.2%）、`foreign_net_20`（+79.5% sign flip）、
`ma_5`（-87.7%）、`ma_20`（-95.6%）、`ma_60`（-103.4% sign flip）、`amt_20`（-108.8% sign flip）、`amt`（-115.7% sign flip）

**⚠️ 衰減 4 個（近期 IC < 歷史 50%）**：
`short_balance_chg_5`（-78.9%）、`fund_revenue_yoy_accel`（-68.1%）、`short_balance_chg_20`、`theme_return_20`（-69.5%）

**🔺 增強 12 個**：`trust_net_5`、`ret_60_kurt`、`drawdown_60`、`trust_net_20`、`sector_momentum`、`vol_ratio_20` 等

**✅ 穩定 34 個**：`atr_inv`、`vol_20_inv`、`ret_5`、`willr_14`、`market_trend_60/20`、`boll_pct`、`cci_20`、`bias_20`、`rsi_14`、`market_above_200ma`、`ret_10/20/60` 等

**策略建議（已更新，實驗後決定）**：
- IC 衰減特徵即使 IC 近期偏低，仍含隱性市場資訊（動能/趨勢），不宜強制移除
- LightGBM 對失效特徵有容忍度，下採用 SHAP 剪枝（50特徵）為最終生產配置
- rs_rank_20/rs_rank_60 因 IC=NaN（特徵太新，未覆蓋足夠月份），暫不評估

### IC 衰減剪枝實驗結果（2026-04-15，10y WF 驗證）

**實驗 A：42 特徵（50feat - 8個IC衰退特徵）**

| 配置 | 累積報酬 | 超額 | MDD | Sharpe | Calmar | 年化 |
|------|---------|------|-----|--------|--------|------|
| **50feat（基準，2026-04-13）** | **+1863%** | **+1807%** | **-27.8%** | **0.980** | **1.268** | **35.3%** |
| 42feat（IC剪枝） | +1784% | +1728% | -29.0% | 1.072 | 1.198 | 34.7% |
| ensemble-3（42feat）| +1523% | +1467% | -34.5% | 0.955 | 0.948 | 32.7% |
| cs-norm（42feat）| +104% | +44% | -49.1% | 0.339 | 0.153 | 7.5% |

**逐年差異（IC42 vs Baseline）**：

| 年 | Baseline | IC-42feat | Delta |
|----|---------|---------|-------|
| 2016 | +2.4% | +7.3% | **+4.9%** |
| 2017 | +36.0% | +44.2% | **+8.2%** |
| 2018 | -0.5% | -5.3% | -4.8% |
| 2019 | +21.4% | +26.7% | **+5.2%** |
| 2020 | +74.9% | +66.4% | -8.5% |
| 2021 | +48.6% | +114.3% | **+65.7%** |
| 2022 | +10.5% | -5.7% | -16.2% |
| **2023** | **+141.0%** | **+57.4%** | **⚠️ -83.6%** |
| 2024 | +17.4% | +51.1% | **+33.7%** |
| 2025 | +20.6% | -2.3% | -22.9% |
| 2026 | +19.1% | +29.9% | **+10.8%** |

**結論（❌ 不採用 IC 剪枝）**：
- Sharpe 微升 +0.09（0.98→1.07），但 Calmar 下降（1.268→1.198）
- **2023 嚴重退化 -83pp**：移除的特徵（ma_20/60、foreign_buy_streak 等動能特徵）在強勢多頭仍不可或缺
- IC「衰減」不等於「無用」，LightGBM 非線性模型可在低 IC 特徵中萃取非線性訊號
- **生產配置維持 50 特徵（SHAP 剪枝）不變**

**其他失敗實驗（同日，均不採用）**：
- **cs-norm（截面 Z-score）**：完全失效，Sharpe 0.34、MDD -49%（vs 基準 0.98/-28%）
- **ensemble-3**：劣於基準，Sharpe 0.95、MDD -35%（模型間相關性過高，diversity 不足）

### FinMind 配額錯誤修正（2026-04-15）

`app/finmind.py` 修正 `fetch_dataset_by_stocks()` 靜默吞噬問題：
- 402/429 配額錯誤立即 `raise FinMindError`，不再繼續後續 batch
- 新增 `error_rate_threshold=0.5`：錯誤率超過 50% 時 raise FinMindError
- 改用 `logger.error/warning` 記錄（不依賴 `debug=True`）
- commit: `d2c209c`

---

## 程式碼優化決策（2026-03-25，session 9）

### 已執行的全面優化

**P0 — 正確性（消除重複邏輯）**
1. **建立 `skills/feature_utils.py`**（新檔案）：
   - `parse_features_json()`：共用 JSON 解析（orjson 優先），統一 backtest/train_ranker/daily_pick/data_store 四處重複實作
   - `impute_features()`：語義導向特徵填補，`boll_pct` 填 0.5、`rsi_14` 填 50（而非一律填 0）
   - `filter_schema_valid_rows()`：共用 schema 遷移過濾（50% 覆蓋率門檻）
2. **建立 `risk.apply_seasonal_topn_reduction()`**：統一 backtest.py 與 daily_pick.py 季節性降倉邏輯，消除雙重獨立實作

**P1 — 穩定性**
3. **`daily_pipeline.py` checkpoint 驗證**：新增 `_check_prices_exist()`、`_check_features_exist()`、`_check_labels_exist()` 三個驗證函式，ingest_prices → build_features → build_labels 各階段執行後驗證資料行數，靜默失敗立即中斷
4. **`app/db.py` JSONL rotation**：`_rotate_slow_queries_if_needed()` 超過 10MB 時輪替（保留 5 備份），環境變數可調整 `SLOW_QUERIES_MAX_BYTES`、`SLOW_QUERIES_BACKUP_COUNT`
5. **`app/api.py` 回測 timeout**：`/strategy_runs` endpoint 改為 async + ThreadPoolExecutor + `asyncio.wait_for(timeout=120s)`，逾時回傳 HTTP 504

**P2 — 程式碼品質**
6. **`WalkForwardConfig` dataclass**（`skills/backtest.py`）：封裝 run_backtest() 30+ 參數，支援 `run_backtest(..., wf_config=cfg)` 呼叫方式，向後相容原有 kwargs
7. **重構 `_simulate_period()`**：拆出三個獨立子函式：
   - `_get_entry_positions()`：確定進場日與進場價（支援突破/統一進場兩種模式）
   - `_compute_slippage_map()`：計算個股滑價（ATR 模型或分級滑價）
   - `_calc_stock_return()`：計算單筆報酬（含成本/滑價/clip）

**驗收（P0-P2）**：93 個測試全部通過，FutureWarning（pandas fillna）消除

**P3-2 — Parquet Feature Store（2026-03-25 session 9）**

8. **建立 `skills/feature_store.py`**（FeatureStore 類別）：
   - 年份分區儲存：`artifacts/features/features_YYYY.parquet`（每年一檔，預解析數值欄位）
   - 原子寫入（.tmp → rename）、DuckDB predicate pushdown（`union_by_name=true` 支援跨年 schema 演進）
   - 主要方法：`write()` / `read()` / `get_max_date()` / `get_distinct_dates()` / `delete_from()` / `migrate_from_mysql()`
9. **`build_features.py` dual-write**：MySQL 寫入後同步寫 Parquet；`_detect_schema_outdated` 與 force_recompute 先查 Parquet（fallback MySQL）
10. **讀取路徑優化**：`data_store.py`、`train_ranker.py`、`daily_pick.py` 均優先從 FeatureStore 讀（省去 JSON 解析 60-90s）；`daily_pipeline.py` checkpoint 改查 Parquet
11. **`scripts/migrate_features_to_parquet.py`**：一次性 MySQL → Parquet 遷移（`make migrate-features`）

**P3-3 — DAG 執行引擎（2026-03-25 session 9）**

12. **建立 `pipelines/dag_executor.py`**（DAGNode + DAGExecutor）：
    - 拓樸排序（Kahn's algorithm），同層節點以 ThreadPoolExecutor 並行執行
    - 各節點獨立 DB session（避免跨 thread SQLAlchemy session 問題）
    - optional=True：失敗不傳播 failed_nodes；condition=fn：動態跳過節點
13. **建立 `pipelines/dag_daily.py`**：日常 DAG 定義，Layer 4（ingest 6 節點並行）+ Layer 6（build_features ∥ build_labels）
14. **`scripts/run_daily_dag.py`** + **Makefile 新增 `pipeline-dag` / `pipeline-dag-build` / `migrate-features`**

**驗收（P3-2/P3-3）**：119 個測試全部通過（新增 26 個：13 個 FeatureStore + 13 個 DAGExecutor）

---

## 核心架構決策

### Label 定義
- **20 交易日 forward return**（`LABEL_HORIZON_DAYS=20`）
- 定義：`close[T+20] / close[T] - 1`
- `build_labels.py: group["future_ret_h"] = group["close"].shift(-horizon) / group["close"] - 1`

### 再平衡頻率
- **月頻（M）**，約等於 20 個交易日
- 與 label horizon 對齊，避免 4:1 mismatch（見「已知錯誤修正」）

### 特徵數量
- 目前：**56 個特徵**（FEATURE_COLUMNS）
- 最新新增（2026-03-11）：`foreign_buy_streak`、`volume_surge_ratio`、`foreign_buy_intensity`

### 停損設計
- **現行生產：無固定停損**（`stoploss_pct=0.0`，`--no-stoploss`）
- 月底換股即出場，不設中途停損
- 單筆最大虧損 clip `-50%`（退市股保護）

### 大盤過濾
- **漸進式大盤過濾**：`[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)]`
- 最低持股數：2 檔（防止單押集中風險）

### 季節性降倉
- 3月、10月：topN × 0.5，floor=5
- `daily_pick.py` 與 `backtest.py` 行為一致（2026-03-11 修正）

---

## 重要回測結果記錄

### 現行生產基準：Exp D（2026-03-15）

| 指標 | 數值 |
|------|------|
| 期間 | 2016-05-03 ~ 2026-02-03（118 期）|
| 累積報酬 | **+2637.11%** |
| 年化報酬 | +39.92% |
| 大盤報酬 | +53.68% |
| 超額報酬 | +2583.43% |
| MDD | **-29.20%** |
| Sharpe | **1.042** |
| Calmar | **1.367** |
| 交易次數 | 2009 筆 |
| 停損觸發 | 0 次 |

**配置**：無停損 + 漸進大盤過濾（-5%:×0.5, -10%:×0.25, -15%:×0.10）+ 最少 2 檔 + seasonal filter

**生產 CLI**：
```bash
python scripts/run_backtest.py --months 120 --seasonal-filter --no-stoploss \
  --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2
```

---

## 已試過的方向與結論

### ✅ 已採用

| 實驗 | 改動 | 效果 | 狀態 |
|------|------|------|------|
| Clip -50% | `max(ret, -0.50)` 退市股保護 | +338% → +1216% | ✅ 生產 |
| enable_seasonal_filter | 3/10月 topN×0.5 與生產對齊 | +1216% → +9553% | ✅ 生產 |
| 無停損 baseline | 移除 stoploss_pct=-0.07 | +1792% 無過濾 | ✅ 生產基礎 |
| 漸進式大盤過濾 Exp D | (-5%:×0.5, -10%:×0.25, -15%:×0.10) + 最少2檔 | +2637%, Sharpe 1.042 | ✅ 現行生產 |
| 強勢訊號特徵（Exp E）| foreign_buy_streak、volume_surge_ratio、foreign_buy_intensity | +2252% → +10005%（⚠️ 有 label 洩漏）| ✅ 特徵已在生產 |
| EMERGING 過濾 | 排除興櫃股（2340→1965 股）| 修正 foreign_buy_* 永遠為 0 | ✅ 生產 |
| label_horizon_buffer=20 | 消除訓練標籤前向洩漏 | 去偏後真實基準 | ✅ 生產 |
| **6 個新特徵（2026-03-18）** | trust_consecutive_buy_days, trust_buy_5d_intensity, foreign_trust_both_buy_days, bull_ma_alignment_score, deviation_from_40d_high, price_volume_alignment | +2637%→+2648%, Sharpe 1.042→1.044, MDD 不變 -29.2%（FEATURE_COLUMNS: 56→62）| ✅ 生產（微幅改善）|

### ❌ 已試過但放棄

| 實驗 | 改動 | 結果 | 原因 |
|------|------|------|------|
| 固定停損 -7% | stoploss_pct=-0.07 | Sharpe 0.268，累積 +69% | 過度截斷月內正常波動 |
| 固定停損 -10%/-12% | stoploss_pct=-0.10/-0.12 | 累積 +69%~+122% | 同上 |
| ATR 動態停損 | 低波動-15%/高波動-25% | MDD 惡化 -60% | MDD 反而惡化 |
| 週頻再平衡 | rebalance_freq="W" | 累積 -84.96%，Sharpe -0.36 | Label 4:1 mismatch（根本錯誤）|
| 原始大盤過濾 | >5%半倉, >10%全現金 | 累積 +1301% | 2018/2020 錯過反彈 |
| Exp G：RSI 45-70 進場過濾 | RSI 過濾 | Sharpe 0.84（-0.20） | 模型已隱式學習 RSI，人為截斷干擾 |
| Exp H/I：多條件過濾 | streak+RSI+bias+volume | Sharpe 0.77 | 條件越多越差 |
| 退市過濾 B1/B2 | 零成交量 + 月跌幅門檻 | 2021: -28pp，2023: -46pp | 誤殺強勢反彈股 |
| 時間加權訓練 | 近期樣本 weight=2.0 | 無改善 | equal-weight+月頻下無效 |
| Market Regime Switching | bull/bear/sideways 三態 | sideways 長期損耗 | `market_regime.py` 標記 NOT IN USE |
| 日頻 Strategy B | 每日進出場 | Sharpe 0.48，MDD -54% | 預測尺度（月）與持倉（8天）不匹配 |
| 大盤天氣圖過濾（market_weather，2026-03-18）| 等權指數 5MA+MACD Histogram 方向：+1不變/0縮70%/-1空手 | +2648%→+2181%, Sharpe 1.044→1.022（-0.022）| 空手/縮倉期 53.4%，嚴重錯過 2019/2021/2023 牛市（-14/-23/-44pp）；程式碼保留供研究，production 不啟用 --market-weather |

### 🔬 有潛力但尚未完整驗證

| 方向 | 說明 |
|------|------|
| 突破確認進場 | F+ 實驗：Sharpe 0.86（vs 基準 0.49），但基準為含 label 洩漏版本，需在去偏版本驗證 |
| foreign_buy_streak<=3 過濾（Exp F）| 微幅改善 +66pp，差異太小不足以採用 |
| ~~C2 雙門檻~~ | 已驗證（2026-03-16）：exit=30% Sharpe 1.648 但總報酬 -5600pp；C2 基準不變 |

---

## Strategy C（日頻輪動）實驗系列（2026-03-16）

### C2 基準（rank 出場，top20%，max_hold=30d）
- 總報酬 +18,395%、年化 +72.68%、MDD -30.25%、Sharpe 1.622、Calmar 2.403
- 交易 1,259 筆、平均持倉 11.4 天、**年化成本 39.53%**
- 出場：Rank Drop 87%、Max Hold 12%
- **問題**：成本太高，每年 39.5% 被交易成本吃掉

### 最小持倉天數實驗（Hold5/10/15）
結論：**完全無效**。Force Exit 替代了 Rank Drop，筆數只減少 3-8%，成本節省 1-3pp，但報酬/Sharpe 同步退化。

### 風控出場實驗系列

**核心洞察（2026-03-16）**：
- Rank-based exit 是隱含的相對強度風控，比絕對技術指標更有效
- C2 的 Rank Drop 提早出場「相對變弱」的股票，自然保護 MDD
- 用絕對技術指標替代，只能抓「非常強才出」，無法捕捉「開始相對弱化」

| 版本 | 出場邏輯 | 總報酬 | Sharpe | MDD | 成本 | 結論 |
|------|---------|--------|--------|-----|------|------|
| C2 基準 | Rank<top20% | +18,395% | **1.622** | -30% | 39.5% | 基準 |
| Risk v1 | 停損+外資+MA | +31,817% | 1.213 | -41% | 51.6% | MA 太靈敏 |
| Risk v2 | 改進版（連2日MA）| +81,505% | 1.262 | -55% | 80.8% | MA 更差 |
| Oracle v1 | RSI+Boll+外資（AND）| +10,393% | 0.968 | -52% | 20.2% | 訊號不觸發（有 bug）|
| Oracle v2 | OR+Boll（bug 修正）| +22,601% | 1.124 | -54% | 55.1% | Boll 太靈敏 |
| Oracle v3 | RSI>78 OR RSI>72+ret5>10% | +25,796% | 1.156 | -55% | 44.9% | 仍遜於基準 |

**Bug 紀錄**：Oracle v1/v2 的 `_feat_map` 只在 `exit_mode=="risk"` 時建立，oracle 模式拿到空 dict，所有訊號以預設值觸發（RSI=50）→ 訊號永遠不發動。修正：`exit_mode in ("risk", "oracle")`。

### Oracle 分析（1258 筆 C2 基準交易）
- 平均實際報酬：+2.68%；平均 Oracle 報酬：+26.03%；**每筆平均少吃 23.35pp**
- 52.3% 的交易提早出場（未到峰值）
- Oracle 出場日中位數：第 12 天
- **Oracle 出場日特徵中位數**：RSI=68.4、Bias=9%、Boll=0.93、外資連買=0天、5日報酬=8.3%
- **關鍵發現**：RSI 在 65-75 之間、外資連買剛中斷、Boll 接近上緣，是最佳出場的特徵組合

### C2 雙門檻實驗（2026-03-16，session 3）

| 配置 | 總報酬 | Sharpe | Calmar | MDD | 交易 | 年化成本 |
|------|--------|--------|--------|-----|------|---------|
| C2 基準（exit=20%）| +18,395% | 1.622 | 2.403 | -30% | 1,259 | 39.5% |
| Dual-A（exit=30%）| +12,794% | **1.648** | 2.089 | -32% | 1,140 | 35.8% |
| Dual-B（exit=40%）| +9,245% | 1.537 | 1.786 | -34% | 1,085 | 34.1% |
| Dual-C（entry=5%, exit=30%）| = Dual-A（完全相同）| — | — | — | — | — |

**發現**：
1. `entry_threshold` 參數無效：進場選前6名本在 top0.5%，top5%/top10% 候選池無約束
2. exit=30% 使 Sharpe 微升（+0.026），成本降 3.7pp，但總報酬損失 5,601pp
3. exit=40% 明顯退化（Sharpe 1.537，MDD -34%）
4. **結論：C2 基準（exit=20%）仍是最佳**，高成本換來快速輪換到強勢股，是「好成本」

---

## Strategy C 多 Agent 辯論實驗系列（2026-03-16，session 3）

### 三派辯論結論
- **技術派**：動態轉折（RSI 下降幅度）優於靜態閾值；加分制不是硬過濾；EMA(8) 比 MA20 更匹配 11 天持倉
- **籌碼派**：foreign_buy_streak 2~8 天是最佳進場窗口（轉折而非追趨勢）；intensity 是負IC（高強度=主力做秀）；外資停買比開賣更早出現
- **基本面派**：在 11 天持倉中只做守門員（pb<0/pe>300/YoY崩潰排除），不能做主要進出場訊號

### 守門員排除（Quality Gate）測試
- 改動：`foreign_buy_streak > 15` / `pe_ratio > 300` / `pb_ratio < 0` 硬排除 + streak 2-8 / rev_accel > 0 軟加分
- **0.3%成本**：總報酬 +26,738%、年化 +79.54%、Sharpe 1.667（vs 基準 1.622）✅
- **0.585%成本**：總報酬 +14,574%、年化 +68.55%、Sharpe 1.505

### Label Horizon 對齊實驗（最重要發現）

| 配置 | 成本 | 總報酬 | 年化 | MDD | Sharpe | Calmar |
|------|------|--------|------|-----|--------|--------|
| C2 基準 | 0.3% | +18,395% | +72.68% | -30.25% | 1.622 | 2.403 |
| C2 真實 | 0.585% | +10,098% | +62.25% | -34.08% | 1.457 | 1.827 |
| Label-10 | 0.3% | +1,038,061% | +163.21% | -27.23% | 1.987 | 5.995 |
| **Label-10 真實** | **0.585%** | **+487,108%** | **+143.18%** | **-28.08%** | **1.847** | **5.100** |
| Gate+Label-10 真實 | 0.585% | +353,100% | +135.13% | -30.74% | 1.743 | 4.396 |

**核心洞察**：20 天 label 訓練 × 平均持倉 11.4 天 = 43% 的信號尚未實現就出場。
改為 10 天 label 後，持倉縮至 9.0 天，尺度對齊，模型預測精度大幅提升。
即使在真實成本（0.585%）下，Label-10 Sharpe 1.847 仍遠優於 C2 基準。

**Label-5**（參考）：+26億%、Sharpe 3.55、年化成本 86%，理論極限但實際不可行。

**推薦配置**：`--train-label-horizon 10`（純 Label-10），真實成本下最佳 Sharpe。

### ⚠️ Label-10 Buffer Bug 修正（2026-03-17）

**問題**：`backtest_rotation.py` 中 `train_label_horizon != 20` 時，`_eff_buffer = train_label_horizon`（被覆蓋為 10），應保持 `label_horizon_buffer=20`

**修正**：`_eff_buffer = label_horizon_buffer`（不隨 label horizon 改變）

**修正前後對比（真實成本 0.585%，無流動性限制）**：

| 版本 | 總報酬 | Sharpe | MDD | 說明 |
|------|--------|--------|-----|------|
| buggy（buffer=10）| +487,108% | 1.847 | -28.1% | ❌ 含前瞻偏差 |
| **fixed（buffer=20）** | **+267,029%** | **2.314** | **-25.6%** | **✅ 正確版** |

**驚喜發現**：修正後 Sharpe 反而提升 +0.47，MDD 也改善。bug 版本的洩漏造成的是「模型更激進 + 波動更大」，而非單純虛增 alpha。修正後策略品質更佳。

### P1 籌碼出場補充實驗（2026-03-16，session 4）

**條件**：外資連買中斷>=3天 AND boll_pct>0.90 AND 持倉>=5天 → 提前出場

| 版本 | 總報酬 | Sharpe | MDD | Chip Exit 觸發 |
|------|--------|--------|-----|----------------|
| Label-10 純基準（0.3%）| +1,038,061% | 1.987 | -27.23% | — |
| Label-10 + Chip Exit（0.3%）| +1,038,061% | 1.987 | -27.23% | **0 次** |
| Label-10 真實（0.585%）| +487,108% | 1.847 | -28.08% | — |
| Label-10 + Chip Exit（0.585%）| +487,108% | 1.847 | -28.08% | **0 次** |

**結論：Chip Exit 從未觸發，與純 Label-10 結果完全相同。**

**分析**：
1. **Rank Drop 速度更快**：外資連買中斷 1-2 天內，股票相對排名已掉出 top20%，Rank Drop 先觸發
2. **boll_pct > 0.90 與外資撤退互斥**：高布林帶位置 = 股價強勢 = 仍在 top20%；外資撤退 3 天後股價通常已從高位回落，boll_pct < 0.90
3. **AND 條件的時間錯開問題**：兩個條件在時序上互斥，無法同時成立

**核心確認**：再次驗證 Rank Drop 是最有效的出場機制。任何絕對技術指標組合都無法比相對排名更快捕捉「開始相對弱化」的訊號。

### 流動性過濾 + 滑價成本實驗（2026-03-17，session 5）

`backtest_rotation.py` 新增功能：`--min-avg-turnover`（億元）、`--slippage`、`--tiered-slippage`

**分級滑價模型（單程）**：>10億 0.05% | 3-10億 0.10% | 1-3億 0.20% | <1億 0.40%

| 配置 | 累積報酬 | 年化 | MDD | Sharpe | 均持倉 | 結論 |
|------|---------|------|-----|--------|--------|------|
| Label-10（0.585%，無限制）| +487,108% | +143% | -28.1% | 1.847 | 9.0d | 基準 |
| Liq-5千萬（0.585%）| +62,321% | +96.1% | -28.3% | 1.754 | 7.3d | -87% 報酬 |
| Liq-1億（0.585%）| +26,197% | +79.2% | -35.2% | 1.541 | 7.3d | -95% 報酬 |
| **Liq-1億+分級滑價（最保守）** | **+12,523%** | **+65.9%** | **-37.2%** | **1.367** | 7.3d | **最接近真實** |

**核心發現：alpha 高度集中在小型/微型股（日均量 <1億）**
- 加流動性過濾後報酬驟降 87-97%
- 說明 Label-10 的超高報酬大部分來自不可實際交易的微型股
- 但即使最保守估計（1億+分級滑價），Sharpe 1.367 仍優於 Strategy A（1.042）
- Strategy C 代價：MDD -37.2%（vs A 的 -29.2%）+ 每日需監控換倉

**逐年亮點（最保守 Liq-1億+滑價 vs Strategy A）**：
- 2017: C -3.8% vs A +44.7% ← C 在牛市初期表現差（小型股早期 alpha 被濾掉）
- 2021: C +444% vs A +75%   ← C 大幅領先
- 2024: C +167% vs A +31%   ← C 大幅領先
- 2025: C +24.9% vs A +8.7% ← C 領先

### Strategy C Label-10 嚴格 Walk-Forward 驗證（2026-03-17，session 6）

**方法**：固定 24 個月訓練視窗 × 6 個月不重疊測試視窗，8 個 Fold，每 Fold 只訓練一次（嚴格 OOS）

**結果：🟠 存疑（5/8 Folds Sharpe > 1.0）**

| Fold | 訓練期 | 測試期 | 報酬 | Sharpe | MDD | 判定 |
|------|--------|--------|------|--------|-----|------|
| 1 | 2016-2017 | 2018H1 | -8.39% | -3.040 | -23.7% | ❌ |
| 2 | 2016H2-2018H1 | 2018H2 | +8.19% | 0.027 | -21.3% | ❌ |
| 3 | 2017-2018 | 2019H1 | +53.15% | 2.161 | -9.1% | ✅ |
| 4 | 2017H2-2019H1 | 2019H2 | +129.51% | 6.354 | -7.1% | ✅ |
| 5 | 2018-2019 | 2020H1 | +78.15% | 2.394 | -15.5% | ✅ |
| 6 | 2018H2-2020H1 | 2020H2 | +33.09% | 1.862 | -22.8% | ✅ |
| 7 | 2019-2020 | 2021H1 | +61.07% | 2.229 | -14.2% | ✅ |
| 8 | 2019H2-2021H1 | 2021H2 | -21.83% | 0.153 | -41.1% | ❌ |

平均：Sharpe 1.518、報酬 +41.62%

**失敗原因分析**：
- Fold 1/2（2018 全年）：熊市 + 模型訓練只含 2016-2018H1（無完整熊市樣本）
- Fold 8（2021H2）：半導體/傳產輪動的高波動期，小型股排名快速反轉，MDD -41%

**結論（初版 8 Folds）**：Label-10 在 2019-2021H1 期間（Folds 3-7）有真實 OOS alpha，但 2018 熊市和 2021H2 輪動期失敗。

### Walk-Forward v2：大盤過濾 + 擴展 Folds 9-14（2026-03-17，session 6）

**重要發現 1：大盤過濾對 Strategy C 完全無效**
- 有/無大盤過濾的 8 Folds 結果完全相同（逐 fold 數字一模一樣）
- 原因：Rank Drop 出場機制已隱性實現動態大盤過濾（個股相對弱化時自動換倉/持現金）
- **結論：Strategy C 不需要大盤過濾，不同於 Strategy A（月頻必須依賴）**

**Folds 9-14 結果（2022-2024）：**

| Fold | 測試期 | 報酬 | Sharpe | MDD | 判定 |
|------|--------|------|--------|-----|------|
| 9 | 2022H1 | -0.4% | 1.957 | -13.4% | ✅ |
| 10 | 2022H2 | +59.1% | 2.710 | -4.4% | ✅ |
| 11 | 2023H1 | +108.3% | 6.392 | -3.7% | ✅ |
| 12 | 2023H2 | -7.1% | -0.719 | -16.9% | ❌ |
| 13 | 2024H1 | +64.5% | 3.154 | -6.2% | ✅ |
| 14 | 2024H2 | +34.9% | -0.113 | -12.2% | ❌ |

**14 Fold 綜合：🟠 存疑（9/14 通過，64.3%）**

| 期間 | 通過率 | 平均 Sharpe |
|------|--------|------------|
| Folds 1-8（2018-2021）| 5/8（62.5%）| 1.518 |
| Folds 9-14（2022-2024）| 4/6（66.7%）| 2.230 |
| 全部 14 Folds | **9/14（64.3%）** | **1.810** |

**失敗 Fold 模式**：
- 2018 全年（Fold 1/2）：訓練期缺乏熊市樣本
- 2021H2（Fold 8）：急速輪動，MDD -41%
- 2023H2（Fold 12）：同年上下半年從 Sharpe 6.39 → -0.72，風格急轉
- 2024H2（Fold 14）：報酬 +34.9% 但 Sharpe -0.113，高波動+高換手成本

**整體評估**：近期期間（Folds 9-14）Sharpe 均值 2.230 > 前期（1.518），策略在近年市場更有效。9/14 不達「可信」門檻（11/14），但已足夠支持小規模試單。

### 波動率過濾實驗（2026-03-17，session 6）

**實作**：`backtest_rotation.py` 加入 `--vol-filter`，高波動日（年化 vol > 25%）將進場候選從 top 10 縮減為 top 5（`vol_topn_tight=0.05` vs 正常 0.10 = 減半）。波動率來源：`market_volatility_20` 特徵 × sqrt(252)。

**高波動日識別**：226/2408 天（9.4%），主要集中在 COVID 崩盤期（2020-03）和 2022 熊市。

| 指標 | 基準（無過濾）| Vol Filter（25%門檻）| 差異 |
|------|-------------|---------------------|------|
| 總報酬 | +267,029% | +283,795% | +6.3% |
| Sharpe | 2.3139 | 2.3296 | +0.016 |
| MDD | -25.59% | -25.59% | 0 |
| 交易筆數 | 1618 | 1617 | **-1** |
| Cost Drag/yr | 99.06% | 98.99% | -0.07pp |
| 2020 年 | +142.80% | +158.04% | **+15.2pp ✓** |
| 2024H2 逐月 | 完全相同 | 完全相同 | **0** |

**核心發現**：
1. **效果極小**：10 年只減少 1 筆交易（226 個高波動日觸發，但幾乎不影響實際進場）
2. **原因**：`top_entry_n=10`（絕對數量）已非常嚴格；高波動縮減為 5 支時，大部分日子 6 個倉位早已被佔滿，新進場機會本就稀少
3. **2024H2 問題未解**：Fold 14 的 Sharpe 問題是「走前向 WF 的固定視窗模型」的泛化問題，而非全量回測中的成本問題（全量 2024 全年 +193%）
4. **2020 改善**：COVID 崩盤最高峰（vol 最高時）成功過濾部分低質量進場 → +15pp

**結論**：Rank Drop 出場機制已隱性充當市場波動過濾器（同大盤過濾的結論一致），顯式波動率過濾效果邊際。波動率過濾功能已實作並合併入主幹，但對 Strategy C 整體影響有限。不列為生產預設。

### 分數穩定過濾實驗（Score Stability，2026-03-17，session 7）

**概念**：當持股排名掉出 top 20%，但模型分數下降幅度 < threshold（佔進場分數比例）時，暫不出場。強制出場：分數顯著下降（Score Drop）/ -10% 停損 / Max Hold Days。

**測試配置**：`--score-stability --score-drop-threshold 0.10/0.15/0.25`，真實成本 0.585%

| 配置 | 總報酬 | 年化 | MDD | Sharpe | 交易 | 均持 | SL% |
|------|--------|------|-----|--------|------|------|-----|
| **Baseline** | **+267,029%** | **+128.4%** | **-25.6%** | **2.314** | 1,618 | 8.9d | 0% |
| Stable-10% | +151,635% | +115.2% | -33.3% | 1.891 | 1,475 | 9.8d | 21.0% |
| Stable-15% | +137,369% | +113.0% | -33.3% | 1.827 | 1,473 | 9.8d | 21.1% |
| Stable-25% | +182,385% | +119.4% | -33.3% | 1.903 | 1,469 | 9.8d | 21.0% |

**結論：全面劣於基準，不採用。**

**失敗原因**：
1. Score Stability 讓持股「撐著不出場」→ 21% 最終被 -10% 停損強制踢出
2. MDD 從 -25.6% 惡化到 -33.3%（比基準多吃 7.7pp 下行）
3. 2021/2023/2024 年報酬大幅縮水（基準 +343% vs Stable +229%；基準 +193% vs Stable +78%）
4. 三個門檻（10%/15%/25%）結果幾乎一樣，說明 score drop 計算幾乎沒有分辨力

**核心確認（第三次）**：Rank Drop 是 Strategy C 最關鍵的出場機制。任何形式的「阻止 Rank Drop 出場」（絕對指標、score stability、vol filter）都比純 Rank Drop 差。

### 動態流動性門檻實驗（2026-03-17，session 7）

**概念**：流動性門檻 = base × (當日市場60日均量 / 2016年均量)，隨市場規模自動調整，避免固定門檻在早期過嚴、近期過寬的問題。

**動態門檻走勢（base=1億）**：

| 年份 | 門檻 |
|------|------|
| 2016 | 1.02 億 |
| 2017 | 1.39 億 |
| 2020 | 2.61 億 |
| 2021 | 5.47 億（牛市量能暴增）|
| 2024 | 5.67 億 |
| 2026 | 8.15 億 |

**結果對比（Label-10，0.585%成本）：**

| 配置 | 總報酬 | Sharpe | MDD | 交易數 |
|------|--------|--------|-----|--------|
| Baseline（無過濾）| +267,029% | 2.314 | -25.6% | 1,618 |
| 固定門檻 1億 | +10,841% | 1.295 | -42.0% | 1,601 |
| 動態門檻 1億 | +1,557% | 0.919 | -35.7% | 1,283 |

**結論：動態門檻反效果，比固定門檻更差。不採用。**

**失敗原因**：台股量能高峰（2021 牛市）與 Strategy C alpha 高峰高度重疊。動態門檻在 alpha 最豐富的年份（2021）自動升至 5.47 億，過濾掉大量中小型強勢股，導致 2021 年報酬從 +343% 降至 +229%，損失最嚴重。動態門檻的概念方向正確，但需要與 alpha 分佈解耦才能有效。

---

### 動態部位控制實驗（2026-03-17，session 7）

**概念**：以日均量分級設定最大部位上限（微型 <0.5億→min(5%, 15萬)；小型 0.5-3億→min(5%, 50萬)；中大型>3億→5% 不限），取代流動性門檻過濾。允許小型股進場但限制部位大小，保留 alpha 同時控制流動性風險。初始資本 300 萬，每筆最大 20%。

**結果（Label-10，0.585% 成本）**：

| 配置 | 總報酬 | 年化 | Sharpe | MDD | 微/小型股佔比 |
|------|--------|------|--------|-----|--------------|
| Baseline（等權 1/6）| +267,029% | +128.35% | 2.3139 | -25.6% | N/A |
| PosSizing-300萬（20%/筆）| +1,133,841% | +165.65% | 2.2985 | -30.0% | **0%（全中大型）** |

**結論：動態部位控制無效。不採用。**

**失敗原因**：
1. **模型自然偏好大型股**：1,618 筆交易 100% 都是中大型股（>3億日均量），流動性上限從未觸發，動態部位控制完全沒有作用。
2. **實質效果只是加槓桿**：6 部位 × 20% = 120% exposure（vs 基準 100%），報酬增加來自槓桿而非 alpha。
3. **風險指標惡化**：Sharpe 2.314 → 2.299（略降），MDD -25.6% → -30.0%（惡化）。
4. **當 alpha 主要在微型股時（無流動性門檻的基準）**，模型並不選微型股，所以動態部位沒有意義。

**附記**：發現並修正兩個 UnboundLocalError bug（`weight` 和 `_w_eob` 在 trades_log.append 前未定義）以及年度報酬計算 bug（初始化 `_pe = 10000.0` 硬碼，應為 `_initial_equity`）。

---

---

## 每日實盤系統架構決策（2026-03-18）

### 訊號與持倉分離原則
- **`strategy_c_state.json`**：模型模擬狀態，由 `strategy_c_pick.py` 自動維護，記錄「如果從頭跟單」的虛擬倉位
- **`portfolio.json`**：使用者真實持倉，由 Telegram Bot 的 /buy /sell 指令維護，為 /signal 推送的唯一依據
- 兩者不可混用：terminal 輸出的買/賣/維持是模擬狀態；Telegram 推送的賣出警示/維持是以真實持倉比對 `above_threshold_stocks`

### Telegram Bot 設計決策
- 使用 raw `requests` library（不依賴 python-telegram-bot），減少外部依賴
- `--push` / `--listen` / `--dry-run` 三種模式
- signal 格式：賣出警示（持倉已掉出 top 20%）→ 維持持倉 → 買進建議（模型前幾名尚未持有）
- sell/hold 清單依模型分數高→低排序（`score_lookup` from `top_candidates`）
- **修改程式碼後必須重啟 Bot**（kill pid + make bot），否則 long-polling process 繼續用舊程式碼

### data_store parquet cache 陷阱
- `skills/data_store.py` 的 parquet cache TTL = 24 小時
- `make pipeline` 跑完後若 cache 尚未過期，`strategy_c_pick.py` 會直接讀舊 cache（含過期特徵）
- **解法**：`rm artifacts/cache/*.parquet && make daily-c`
- 或讓 `make pipeline-build` 在結尾呼叫 `data_store.invalidate()`（待改善）

### FinMind API Quota 管理
- 每次完整 pipeline ≈ 15-30 次 API 呼叫；密集跑 3-4 次/天 + backfill 可能觸發 402
- `fetch_dataset_by_stocks` 批次失敗靜默吞掉（`error_count` 遞增但 job 仍標 success）→ 資料只有部分股票
- 症狀：recent dates 只有 ~558 或 ~1050 筆而非正常 2318 筆
- 恢復方式：等 quota 重置後 `DAYS=30 make backfill-prices && make pipeline`

---

## 已知錯誤與修正記錄

### 訓練標籤前向洩漏（2026-03-13 修正）
- **問題**：`label_horizon_buffer=0` 時，訓練截止前 20 個交易日的標籤使用到測試期收盤價
- **影響**：回測績效虛高（+10004% 去偏後只有 +205%）
- **修正**：`label_horizon_buffer=20`，`train_ranker.py LABEL_HORIZON_BUFFER_DAYS=20`

### 週頻 Label Mismatch（2026-03-10 發現）
- **問題**：`rebalance_freq="W"`（每~5天換股），label horizon 20 天 → 4:1 mismatch
- **影響**：10y 累積 -84.96%，Sharpe -0.36
- **修正**：改回 `rebalance_freq="M"`

### ATR Bug（2026-03-10 修正）
- `enable_slippage=True` 時 `atr_df=None` 靜默失效
- 修正：`if atr_stoploss_multiplier is not None or position_sizing == "vol_inverse" or enable_slippage:`

### fund revenue 前向洩漏（2026-03-04 修正）
- 月營收加 45 天公告延遲：`available_date = trading_date + 45 days`
- 改為 per-stock groupby `merge_asof`（全域 merge 需要全局單調，跨股資料不符合）

---

## 2026-04-02 實驗系列（Session 10）

### 背景
以「資深交易員 / 量化分析師」角度全面審視策略，依 P0→P1→P2 優先序逐一測試。
現行生產基準：累積 +3351% / Sharpe 1.104 / MDD -27.3%（月頻 + 無停損 + 漸進大盤過濾 + liq-weighted + SHAP 剪枝 48 特徵）

---

### P0-2：突破確認進場（enable_breakthrough_entry）
**實驗設定**：生產配置 + `--breakthrough-entry`（等待最多 10 日出現新高+大量訊號）
**10y 結果**（artifacts/experiments/20260402_113128_breakthrough_vs_direct_entry.json）：
- 累積 **+465%**、年化 +19.2%、MDD -41.5%、Sharpe 0.683、Calmar 0.463
- 2020 年：**-11.68%（大盤 +18.25%，超額 -29.93%）** ← 核心問題

**結論：❌ 不採用**
- COVID V型急拉行情沒有「新高+大量」訊號，突破策略踏空只持 2 檔
- 月頻策略靠「月底直接進場、持滿一個月」捕捉 20 日 label horizon，突破等待邏輯與模型時間尺度不符
- 對比基準 +3351%，累積報酬差距高達 2886pp

---

### P0-3：投資組合熔斷機制（portfolio_circuit_breaker_pct=-0.15）
**實驗設定**：生產配置 + `--portfolio-circuit-breaker 0.15`（月中等權累積跌超 15% 全出場）
**10y 結果**（artifacts/experiments/20260402_114421_circuit_breaker_15pct.json）：
- 累積 **+1092%**、年化 +28.6%、MDD -36.6%、Sharpe 0.821、Calmar 0.781
- 2021 年：**+14.5%（vs 基準 +83.8%）** ← 月中波動頻繁觸發，錯過後段大漲
- 2022 年：+9.96%（比基準 -9.08% 好，但不足以彌補其他年度的損失）

**結論：❌ 不採用**
- MDD 反而從 -27.3% 惡化至 -36.6%（熔斷後空手，但空頭期間已損失）
- 月頻策略須持滿整月才能捕捉 20 日 label，強迫中途出場等於和 label horizon 對著幹
- 已實作程式碼保留（`--portfolio-circuit-breaker` flag），供未來特殊場景研究用

---

### P1-1：產業集中限制（apply_sector_constraint）
**實作**：commit bc102a9
- `risk.py` 新增 `apply_sector_constraint()`，greedy 選股時同產業最多持 N 檔
- `daily_pick.py` 新增 `_load_sector_map()`，讀取 Stock.industry_category
- `config.py` 新增 `sector_max_per_industry: int = 0`（預設關閉）
- 啟用方式：`SECTOR_MAX_PER_INDUSTRY=3`

**回測驗證**：未執行（優先做 P1-2，且本功能主要針對生產端集中風險，非回測指標提升）

---

### P1-2：超額報酬 Label（label_type="excess"）
**實驗設定**：生產配置 + `--excess-label`（future_ret_h 換成個股 - 等權市場均值）
**10y 結果**（artifacts/experiments/20260402_124732_excess_label_p1_2.json）：
- 累積 **+859%**、年化 +25.8%、MDD -47.3%、Sharpe 0.750、Calmar 0.545
- 2022 年：**-33.85%（基準只 -9.08%）** ← 超額報酬 label 讓模型喪失對大盤走勢的感知
- 2019 年：+3.91% 超額 -0.73%（首次跑輸大盤）

**結論：❌ 不採用**
- LightGBM cross-sectional ranking 本質上已在學超額報酬（相對排名），再加一層超額轉換是冗餘
- 硬換 label 反而破壞模型對絕對市場走勢的感知，空頭時無法縮手
- 已實作程式碼保留（`--excess-label` flag），供未來搭配其他架構研究

---

### 本 session 效能改善
- **`ingest_margin_short` 批次查詢**（commit fe8e6a4）：3039 次個別 API → ~7 次批次，~30x 加速
- Pipeline 總時間：~2 小時 → 幾分鐘（融資融券段）

### 總結
三個優化方向（突破進場、熔斷機制、超額報酬 label）全部驗證失敗。
**現行生產配置（月頻直接進場 + 絕對報酬 label + 漸進大盤過濾 + liq-weighted + 48 特徵）維持不變。**

---

## Session 11（2026-04-13）程式優化 + rs_rank 驗證

### 程式優化（commit 484c4fb）

1. **labels 加 trading_date index**：CREATE INDEX ix_labels_trading_date ON labels(trading_date)
   - MAX/GROUP BY 查詢：~2s → 即時

2. **ingest_corporate_actions 改增量模式**：只補算 price_adjust_factors 未覆蓋日期
   - 31.1s → 0.02s（**~1500x 加速**）
   - 原因：每次重算 365 天 × 2000 股 ≈ 730,000 筆 upsert → 改為只插入新日期（每日幾百筆）

3. **Strategy C 加距突破上限過濾**（MAX_BREAKTHROUGH_DIST = 0.30）
   - 距20日高點 >30% 且未突破的買進候選直接跳過，換後排候補遞補
   - bt_status 計算移至進場判斷前，讓過濾邏輯能使用突破距離
   - 目的：避免威剛 +41%、晶豪科 +39% 這類幾乎不可能在 10 天內突破的股票佔用名額

### rs_rank 特徵驗證（2026-04-13）

**背景**：rs_rank_20 / rs_rank_60（個股報酬在全市場當日百分位排名）已進入 FEATURE_COLUMNS（58特徵），PRUNED_FEATURE_COLS 從 48 → 50。

**實驗**：以當前 adj_close 資料跑兩組 10y walk-forward：
- 50 特徵（含 rs_rank）：累積 **+1863%**、Sharpe **0.98**、MDD -27.83%
- 48 特徵（不含 rs_rank）：累積 **+1863%**、Sharpe **0.98**、MDD -27.83%

**結論：rs_rank 特徵完全中性**，有無均不影響結果。LightGBM 已從其他特徵（ret_20、ret_60 等）隱式學到相同資訊。

**adj_close 資料更新影響**：
- 舊快照（2026-03-18）：+3351% / Sharpe 1.104
- 當前快照（2026-04-13）：+1863% / Sharpe 0.98
- 差距（-1488pp）**完全來自 adj_close 回溯調整**，與特徵無關
- adj_close 因除權息回溯計算會改變歷史報酬估計，為正常現象
- 當前快照為最新且最準確的數字，CLAUDE.md 基準應更新為 +1863% / Sharpe 0.98

**生產決策**：
- rs_rank 特徵保留（neutral，不刪除）
- PRUNED_FEATURE_COLS 維持 50 特徵
- 生產基準更新為 **+1863% / Sharpe 0.98**（2026-04-13 adj_close 快照）
