# 專案現況

> 最後更新：2026-04-23（session 15）

---

## 目前最佳回測結果

### Strategy A（月頻，現行生產）— 2026-04-23 最新快照

| 指標 | 數值 |
|------|------|
| 期間 | 2016-05-16 ~ 2026-03-23 |
| 累積報酬 | **+1444%** |
| MDD | **-32.0%** |
| Sharpe | **0.9493** |
| Calmar | **1.00** |
| 超額報酬 | +1381% |

配置：月頻 + 無停損 + 漸進大盤過濾（-5%:×0.5, -10%:×0.25, -15%:×0.10）+ 最少 2 檔 + seasonal filter + label_horizon_buffer=20 + liq-weighted + SHAP 剪枝 50 特徵（含 rs_rank）

> ⚠️ adj_close 隨除權息回溯調整持續變化，各期快照均不可重現：
> - 2026-03-18 快照：+3351%/Sharpe 1.104（已廢棄）
> - 2026-04-13 快照：+1863%/Sharpe 0.98（已廢棄）
> - **2026-04-23 快照（最新）：+1444%/Sharpe 0.9493**（68 特徵 DB，回測用 50 特徵）

CLI：
```
python scripts/run_backtest.py --months 120 --seasonal-filter --no-stoploss \
  --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2 \
  --liq-weighted --pruned-features
```

#### 2026-04-02 Session 10 已驗證排除方向（見 decisions.md 詳細結果）

| 實驗 | 累積報酬 | Sharpe | MDD | 結論 |
|------|---------|--------|-----|------|
| 突破進場（P0-2）| +465% | 0.683 | -41.5% | ❌ 2020 踏空 -30% 超額 |
| 熔斷 -15%（P0-3）| +1092% | 0.821 | -36.6% | ❌ MDD 惡化、年化腰斬 |
| 超額報酬 label（P1-2）| +859% | 0.750 | -47.3% | ❌ 2022 -33.85% 大虧 |

特徵欄位：56 → 62（新增 trust_consecutive_buy_days, trust_buy_5d_intensity, foreign_trust_both_buy_days, bull_ma_alignment_score, deviation_from_40d_high, price_volume_alignment）

### Strategy C（日頻輪動）— **Excess Label 最佳配置**（2026-04-16 session 14 更新）

| 指標 | 數值（新生產配置）|
|------|----------------|
| 期間 | 10y Walk-Forward（~2016-2026）|
| 累積報酬 | **+31,043%** |
| 年化報酬 | **82.6%** |
| MDD | **-37.0%** |
| Sharpe | **1.438** |
| Calmar | **2.233** |

配置：`--months 120 --train-label-horizon 10 --min-avg-turnover 1.0 --tiered-slippage --transaction-cost 0.00585 --excess-label --max-positions 4`

Strategy C 生產 CLI：
```bash
python scripts/backtest_rotation.py --months 120 --train-label-horizon 10 \
  --min-avg-turnover 1.0 --tiered-slippage --transaction-cost 0.00585 --excess-label --max-positions 4
```

#### Session 14 實驗總結

**Label 實驗（三組對照）**

| 配置 | 累積報酬 | Sharpe | MDD | Calmar | 採用？ |
|------|---------|--------|-----|--------|--------|
| A Baseline（原始 label）| +17,877% | 1.433 | -36.1% | 2.005 | 基準 |
| B Liq-Weighted | +20,785% | 1.266 | **-51.1%** | 1.472 | ❌ MDD 惡化 |
| **C Excess Label** | **+31,043%** | **1.438** | -37.0% | **2.233** | **✅ 採用** |

**P1 持倉數實驗（excess label 基礎上）**

| 配置 | 累積報酬 | Sharpe | MDD | Calmar | 採用？ |
|------|---------|--------|-----|--------|--------|
| excess (pos=6, 基準) | +31,043% | 1.438 | -37.0% | 2.233 | 前基準 |
| P1-A excess+liq-weighted | +34,561% | **1.630** | -36.8% | 2.302 | 待評估 |
| P1-B excess+adaptive rank | +31,043% | 1.438 | -37.0% | 2.233 | ❌ 無效 |
| **P1-C excess+pos=4** | **+102,240%** | 1.480 | -40.4% | **2.649** | **✅ 採用** |
| P1-D excess+pos=8 | +24,124% | 1.596 | -34.0% | 2.290 | ❌ 報酬下降 |

#### 歷史配置對照

| 配置 | 累積報酬 | Sharpe | MDD | 說明 |
|------|---------|--------|-----|------|
| ~~buggy（buffer=10, 0.585%）~~ | ~~+487,108%~~ | ~~1.847~~ | ~~-28.1%~~ | ❌ 前瞻偏差（已廢棄）|
| fixed（buffer=20, 含微型股）| +267,029% | 2.314 | -25.6% | 含微型股，不可執行 |
| Liq-1億+分級滑價（baseline）| +17,877% | 1.433 | -37.0% | 現行基準 |
| Liq-1億+分級滑價+Excess Label (pos=6) | +31,043% | 1.438 | -37.0% | 前生產 |
| **Liq-1億+分級滑價+Excess Label+pos=4** | **+102,240%** | **1.480** | **-40.4%** | **✅ 新生產** |

alpha 高度集中在日均量 <1億 小型股；加流動性過濾後報酬驟降，但 Sharpe 仍優於 Strategy A（0.98）

---

## 今日完成的實驗（2026-03-16，sessions 3-4）

| 實驗 | 結論 |
|------|------|
| C2 最小持倉（Hold5/10/15）| 無效，Force Exit 替代 Rank Drop |
| C2 風控出場 Risk v1/v2 | MA Break 太靈敏，成本反升 |
| Oracle 分析 | 每筆平均少吃 23pp；最佳出場：RSI=68、外資剛中斷 |
| C2-Oracle v1~v3 | 皆遜於基準（Sharpe 1.156 max）|
| C2 雙門檻 | entry_threshold 無效；exit=30% Sharpe+0.026 但報酬-5600pp |
| **Quality Gate** | Sharpe 1.667（+0.045），+8,343pp ✅ |
| **Label-10**（核心突破）| Sharpe 1.987（+0.365），×56 總報酬 ✅✅ |
| **Label-10 真實成本** | Sharpe 1.847，+487,108% ✅ 仍遠優於 C2 |
| **Chip Exit 補充** | **0 次觸發**，與 Label-10 完全相同，放棄 |
| **流動性+滑價（session 5）** | Liq-1億+分級滑價：Sharpe 1.367，+12,523%；alpha 97% 來自微型股 |
| **嚴格 Walk-Forward v1（session 6）** | 🟠 存疑：5/8 Folds Sharpe>1.0；平均 Sharpe 1.518 |
| **Walk-Forward v2（session 6）** | 🟠 存疑：9/14 Folds（64.3%）；Sharpe 均值 1.810；大盤過濾無效（Rank Drop 已隱性替代）|
| **波動率過濾（session 6）** | 效果極小（-1 筆交易，Sharpe +0.016）；2020 +15pp；2024H2 無效；不列生產預設 |
| **分數穩定過濾（session 7）** | ❌ 全面劣於基準：Sharpe 1.83-1.90 vs 2.31；Stop Loss 21%；MDD -33% vs -26%；不採用 |
| **動態流動性門檻（session 7）** | ❌ 反效果：Sharpe 0.919 vs 固定1億 1.295；2021 量能暴增→門檻5.47億→過濾 alpha 最豐富年份；不採用 |
| **動態部位控制（session 7）** | ❌ 無效：100% 交易皆為中大型股，流動性上限從未觸發；效果等同加槓桿（120% exposure）；Sharpe 2.299 vs 2.314；MDD -30% vs -26%；不採用 |

**核心發現**：Label-10（10日 label horizon 對齊 9日平均持倉）是最重要的單一改進。Rank Drop 比任何絕對指標更快捕捉弱化訊號。阻止 Rank Drop 出場的所有嘗試（risk/oracle/vol filter/score stability/dynamic sizing）均失敗。模型自然偏好大型股，不需要流動性過濾也不選微型股。

---

## 目前已知問題

| 問題 | 影響範圍 | 說明 |
|------|----------|------|
| `backtest.py` 不呼叫 `get_universe()` | 回測 | 回測不過濾興櫃股，與生產行為不一致 |
| `ingest_trading_calendar.py` 用 weekday heuristic | pipeline | 尚未串接官方 TWSE 行事曆 |
| `ingest_corporate_actions.py` 外部來源未接妥 | 除權除息 | 常見 `adj_factor=1.0` 保底 |
| **parquet cache 24h TTL** | strategy_c_pick.py | pipeline 跑完後若 cache 未過期，daily-c 仍用舊資料。解法：`rm artifacts/cache/*.parquet` 再跑 |
| ~~**FinMind API Quota（402）**~~ | ~~ingest~~ | ~~**已修正（2026-04-15）**~~：402/429 配額錯誤現在立即拋出 FinMindError，不再靜默繼續 |

---

## Session 15 完成事項（2026-04-23）

### Sponsor Dataset NaN 修正 + 全量重建

| 項目 | 說明 |
|------|------|
| 0-vs-NaN bug 修正 | `build_features.py` 新增 `_SPONSOR_FEATURES` 集合，fillna(0) 排除此集合；JSON 構建時 NaN→None |
| `broker_net_5d` / `broker_buy_days_5` 修正 | `.where(_has_broker_data, other=np.nan)` 遮罩，無原始資料日期正確保持 NaN |
| `gov_bank_net_5d` 修正 | 同上，notna() 判斷有無原始資料 |
| 全量重建（4,271,120 rows）| FORCE_RECOMPUTE_DAYS=3650 + 乾淨刪除 Parquet + MySQL，68 features 2016-02-15 ~ 2026-04-22 |
| FinMind API 確認 | `TaiwanStockTradingDailyReport` 不支援日期區間（1 stock × 1 day 每次），歷史回補 500 萬 call 不可行 |
| Sponsor 特徵評估 | 10y WF 結果 +1444%/Sharpe 0.949，與去除前基線相同，零改善。保留 FEATURE_COLUMNS，不加入 PRUNED_FEATURE_COLS |
| commit | `ed6194c` |

### 特徵 DB 現況

- MySQL: 4,271,120 rows，68 features，2016-02-15 ~ 2026-04-22
- Parquet: 11 年份檔案（2016-2026），含 NaN 修正
- Sponsor 每日增量：broker_trades + kbar_features（最近 2 天）繼續寫入
- large_holder：週報，持股分級資料稀疏（99% 股票無資料）
- gov_bank / fear_greed：日更

---

## Session 13 完成事項（2026-04-15）

### 三項優化驗證 + FinMind 修正

**驗證結果（10y WF，均不採用為生產配置）**：

| 實驗 | 累積報酬 | MDD | Sharpe | Calmar | 結論 |
|------|---------|-----|--------|--------|------|
| **Baseline（50feat）** | **+1863%** | **-27.8%** | **0.980** | **1.268** | ← 生產配置 |
| IC衰減剪枝（42feat）| +1784% | -29.0% | 1.072 | 1.198 | ❌ 2023 -83pp 退化 |
| ensemble-3（42feat）| +1523% | -34.5% | 0.955 | 0.948 | ❌ 全面劣於基準 |
| cs-norm（截面Z-score）| +104% | -49.1% | 0.339 | 0.153 | ❌ 完全失效 |

**關鍵發現**：
- IC「衰減」不等於「無用」：ma_20/60、foreign_buy_streak 等動能特徵在 2023 強勢多頭至關重要
- 截面 Z-score 正規化在月頻設定下破壞跨期排名信號，不適合此策略
- Ensemble 3 checkpoints：模型間相關性過高，diversity 不足以提升 Sharpe

**FinMind 配額錯誤修正（commit d2c209c）**：
- `app/finmind.py`：402/429 配額錯誤現立即 raise，不靜默繼續
- 新增 `error_rate_threshold=0.5`
- 改用 logger.error/warning 記錄

---

## Session 9 完成事項（2026-03-25）

### 全面程式碼優化（7+9 項改動，119 tests passed）

**P0-P2（先前完成）**

| 項目 | 說明 |
|------|------|
| `skills/feature_utils.py`（新） | 共用 parse_features_json / impute_features / filter_schema_valid_rows |
| `risk.apply_seasonal_topn_reduction()` | 統一季節性降倉，消除 backtest/daily_pick 雙重實作 |
| `daily_pipeline.py` checkpoint | 三階段資料驗證（prices/features/labels），靜默失敗立即中斷 |
| `app/db.py` JSONL rotation | 超過 10MB 自動輪替，保留 5 備份 |
| `app/api.py` 回測 timeout | asyncio 120s 逾時保護，防止 CPU 密集型卡死伺服器 |
| `WalkForwardConfig` dataclass | 封裝 run_backtest() 30+ 參數，向後相容 |
| `_simulate_period()` 拆分 | 3 個子函式：_get_entry_positions / _compute_slippage_map / _calc_stock_return |

**P3-2（Parquet Feature Store）**

| 項目 | 說明 |
|------|------|
| `skills/feature_store.py`（新） | FeatureStore：年份 Parquet 分區（artifacts/features/features_YYYY.parquet）|
| `build_features.py` dual-write | MySQL + Parquet 同步寫入；_detect_schema_outdated 先查 Parquet |
| `data_store.py` 讀取優化 | _build_features 從 FeatureStore 讀（省去 JSON 解析 60-90s）|
| `train_ranker.py` 讀取優化 | 直接從 FeatureStore 讀特徵，跳過 JSON parse 路徑 |
| `daily_pick.py` 讀取優化 | 候選日期與特徵均從 FeatureStore 讀 |
| `scripts/migrate_features_to_parquet.py`（新） | 一次性 MySQL → Parquet 遷移 CLI（`make migrate-features`）|

**P3-3（DAG 執行引擎）**

| 項目 | 說明 |
|------|------|
| `pipelines/dag_executor.py`（新） | DAGNode + DAGExecutor：拓樸排序 + ThreadPoolExecutor 分層並行 |
| `pipelines/dag_daily.py`（新） | 每日 DAG 定義：Layer 4（ingest 6 節點並行）+ Layer 6（features ∥ labels）|
| `scripts/run_daily_dag.py`（新） | CLI 入口（`make pipeline-dag`），支援 --skip-ingest / --dry-run |

---

## 新增指令（Session 9 P3-2/P3-3）

```makefile
make pipeline-dag         # DAG 版 pipeline（並行 ingest，預估快 30-40%）
make pipeline-dag-build   # DAG 版（跳過 ingest）
make migrate-features     # 一次性遷移 MySQL features → 年份 Parquet
```

### FeatureStore 遷移注意事項

1. 首次使用前須先執行 `make migrate-features`（否則 build_features 首次跑時會 dual-write，之後才走 Parquet 快路徑）
2. Parquet 存放路徑：`artifacts/features/features_YYYY.parquet`（每年一檔）
3. 遷移後 train_ranker / daily_pick / data_store 自動走 Parquet，不需任何手動設定
4. 回滾方式：刪除 `artifacts/features/` 資料夾，系統自動 fallback 至 MySQL

---

## Session 12 完成事項（2026-04-15）

| 項目 | 說明 |
|------|------|
| P1-4 Strategy C DB 稽核 log | `strategy_c_trades` 表 + ORM + _write_trades_to_db() |
| P3-2 Backtest EMERGING 過濾 | 回測與生產 universe 一致（371 股）|
| P2-1 截面 Z-score 正規化 | `cross_section_normalize()` + --cs-norm flag |
| P2-2 Ensemble checkpoints | deque buffer + --ensemble N flag |
| P2-3 資料品質異常偵測 | price spike ±50% + features/prices 一致性 |
| LambdaRank 10y 驗證 | 微幅改善（+7.3pp, +0.01 Sharpe），不採用為生產配置 |
| IC 衰減分析 | 8 失效特徵（ma_5/20/60 等）+ 4 衰減特徵，34 穩定 |

### 已知問題更新

| 問題 | 狀態 |
|------|------|
| `backtest.py` 不過濾興櫃股 | ✅ P3-2 已修正（session 12）|
| `ingest_trading_calendar.py` weekday heuristic | ❌ 待修（P3-1）|
| `ingest_corporate_actions.py` 外部來源未接妥 | ❌ 待修 |
| parquet cache 24h TTL | ❌ 待修 |
| FinMind API Quota 靜默失敗 | ❌ 待修 |

---

## 待優化項目

### 最高優先（下次 session 優先跑）

| 項目 | 說明 | 預期效果 |
|------|------|----------|
| 失效特徵剪枝（IC 衰減）| 移除 ma_5/ma_20/ma_60/amt/amt_20 等 8 個近期失效特徵 | 模型穩定性提升 |
| ~~FinMind backfill 靜默失敗修正~~ | ✅ **已修正（2026-04-15，commit d2c209c）**：402/429 立即拋出 FinMindError | — |
| **Sponsor 特徵重新評估**（60 天後）| broker_trades + kbar_features 累積 60+ 天後，跑 SHAP 分析。若 SHAP 排名進入 top 30，加入 PRUNED_FEATURE_COLS | 潛在 alpha 提升 |

### 中優先

| 項目 | 說明 |
|------|------|
| TWSE 行事曆接入（P3-1）| 取代 weekday heuristic |
| --cs-norm backtest 驗證 | 截面 Z-score 是否改善 10y Sharpe |
| --ensemble 3 backtest 驗證 | 3 checkpoint ensemble 是否穩定改善 |
| Strategy C 每日流程自動化 | 目前需手動 `make daily`；可設 cron |

---

## 新增工具（2026-03-16 ~ 2026-03-18）

| 工具 | 路徑 | 說明 |
|------|------|------|
| Oracle 分析 | `scripts/oracle_analysis.py` | 找出每筆交易最佳出場時機，分析特徵分佈 |
| 輪動回測 | `scripts/backtest_rotation.py` | Strategy C 日頻輪動，支援 rank/risk/oracle 三種出場模式 |
| 每日訊號輸出 | `scripts/daily_signal.py` | 讀 picks DB，比較前日，輸出 buy/sell/hold + 建議金額 JSON |
| Strategy C 每日選股 | `scripts/strategy_c_pick.py` | 訓練 LightGBM（Label-10），打分今日股票，輸出 strategy_c_YYYY-MM-DD.json |
| Telegram Bot | `scripts/telegram_bot.py` | 推送每日訊號、/signal /portfolio /buy /sell /help 指令管理持倉 |

## Telegram Bot 架構說明（2026-03-18）

- **`portfolio.json`**：`artifacts/daily_signal/portfolio.json`，使用者**真實持倉**，由 /buy /sell 指令維護
- **`strategy_c_state.json`**：`artifacts/daily_signal/strategy_c_state.json`，模型**模擬狀態**，由 strategy_c_pick.py 自動更新
- 兩者完全分離：`_format_push_message` 以 portfolio.json 為主，不使用模型模擬狀態
- signal 內 sell/hold 清單依模型分數高→低排序（2026-03-18 修正）
- Bot 使用 raw requests（不依賴 python-telegram-bot），--push / --listen / --dry-run 三種模式
- **注意**：修改 telegram_bot.py 後需重啟 Bot（`kill <pid> && make bot`），否則舊 process 繼續用舊程式碼

## make 指令更新（2026-03-18）

```makefile
make daily      # run_daily.py + strategy_c_pick.py + daily_signal.py + telegram_bot --push
make daily-c    # 只跑 strategy_c_pick.py（資料已最新時使用）
make bot        # 啟動 Telegram Bot 監聽模式
```

---

## 近期 Commit 摘要

| Hash | 說明 |
|------|------|
| 6a74d52 | fix: base signal on real portfolio instead of model simulated state |
| 381518b | feat: add Telegram bot for trading signals and portfolio management |
| dfc8bcc | feat: add Strategy C daily pick script and make daily target |
| fbfcf7d | feat: add daily signal generation script |
| e00111a | experiment: dynamic position sizing - not adopted; fix backtest_rotation bugs |

---

## 主要 scripts 清單

| 腳本 | 用途 |
|------|------|
| `scripts/run_backtest.py` | Strategy A 主回測（月頻 walk-forward）|
| `scripts/backtest_rotation.py` | Strategy C 日頻輪動（rank/risk/oracle 出場）|
| `scripts/backtest_daily.py` | Strategy B 日頻回測 |
| `scripts/oracle_analysis.py` | Oracle 最佳出場分析 |
| `scripts/strategy_c_pick.py` | Strategy C 每日實盤選股（Label-10，輸出 JSON + 模擬狀態）|
| `scripts/daily_signal.py` | Strategy A 每日訊號（讀 picks DB，輸出 buy/sell/hold）|
| `scripts/telegram_bot.py` | Telegram Bot（推送訊號 + 持倉管理）|
| `scripts/generate_review_pack.py` | 產生復盤報告 |
| `scripts/run_walkforward.py` | Walk-forward 驗證 |
