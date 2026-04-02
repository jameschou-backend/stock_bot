# 專案現況

> 最後更新：2026-04-02（session 10）

---

## 目前最佳回測結果

### Strategy A（月頻，現行生產）— 2026-04-02 確認基準

| 指標 | 數值 |
|------|------|
| 期間 | 2016-04-25 ~ 2026-03-04 |
| 累積報酬 | **+3351%** |
| 年化報酬 | **+43.2%** |
| MDD | **-27.3%** |
| Sharpe | **1.104** |
| Calmar | **1.583** |
| 超額報酬 | +3296.9% |

配置：月頻 + 無停損 + 漸進大盤過濾（-5%:×0.5, -10%:×0.25, -15%:×0.10）+ 最少 2 檔 + seasonal filter + label_horizon_buffer=20 + liq-weighted + SHAP 剪枝 48 特徵

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

### Strategy C（日頻輪動）— **Label-10 最佳配置**（2026-03-16 session 4 更新）

| 指標 | 0.3% 成本 | 0.585% 真實成本 |
|------|-----------|----------------|
| 期間 | 2016-03-28 ~ 2026-02-03 | 同左 |
| 累積報酬 | **+1,038,061%** | **+487,108%** |
| 年化報酬 | +163.21% | +143.18% |
| MDD | **-27.23%** | **-28.08%** |
| Sharpe | **1.987** | **1.847** |
| Calmar | 5.995 | 5.100 |
| 交易筆數 | 1,602 | 1,602 |
| 平均持倉 | 9.0 天 | 9.0 天 |
| 年化成本 | 50.3% | 98.1% |

配置：`--train-label-horizon 10`，rank_threshold=0.20，max_hold=30d

**C2 基準（舊）**：+18,395%，Sharpe 1.622，MDD -30.25%（20d label horizon，1,259 筆）

#### 流動性過濾後更保守的估計（2026-03-17 更新）

| 配置 | 累積報酬 | Sharpe | MDD | 說明 |
|------|---------|--------|-----|------|
| ~~buggy（buffer=10, 0.585%）~~ | ~~+487,108%~~ | ~~1.847~~ | ~~-28.1%~~ | ❌ 含前瞻偏差（已廢棄）|
| **fixed（buffer=20, 0.585%）** | **+267,029%** | **2.314** | **-25.6%** | **✅ 現行最佳，含微型股** |
| Liq-5千萬（0.585%）| +62,321% | 1.754 | -28.3% | 較保守 |
| Liq-1億（0.585%）| +26,197% | 1.541 | -35.2% | 中型股以上 |
| **Liq-1億+分級滑價** | **+12,523%** | **1.367** | **-37.2%** | **最接近真實可執行** |

alpha 高度集中在日均量 <1億 小型股；加過濾後報酬驟降 97%，但 Sharpe 仍優於 Strategy A（1.042）

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
| **FinMind API Quota（402）** | ingest | 密集跑 pipeline 會觸發 402 上限；backfill 靜默失敗（error_count 遞增但不拋例外）→ job 顯示 success 但資料不完整 |

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

## 待優化項目

### 最高優先（下次 session 優先跑）

| 項目 | 說明 | 預期效果 |
|------|------|----------|
| 突破進場（F+）去偏驗證 | 需在 label_horizon_buffer=20 正確基準重跑 | 在去偏後看是否仍有效 |
| Fold 12/14 失敗原因深挖 | 2023H2 風格轉換 + 2024H2 高波動成本 | 找出應對方案（如震盪市降頻）|
| FinMind backfill 靜默失敗修正 | `fetch_dataset_by_stocks` 批次失敗被吞掉，job 顯示 success 但資料不完整 | 加 exception 或 row count 驗證 |

### 中優先

| 項目 | 說明 |
|------|------|
| TWSE 行事曆接入 | 取代 weekday heuristic |
| 特徵 SHAP 分析 | 56 個特徵是否有負貢獻 |
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
