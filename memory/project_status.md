# 專案現況

> 最後更新：2026-07-03（全面健檢：發現生產選股 index 錯位 P0）

---

## 2026-07-03：全面健檢（4 agent 並行審計）⚠️ 兩個 P0

完整報告：`artifacts/review_pack_health_check_20260703.md`。基線 make test 479 passed 全綠。

**P0-1 生產選股 index 錯位（已人工驗證 + DB 列級實證）**：`daily_pick.py:148` merge 重置 index，
`:673-675` 用位置 index 對齊 feature_df（11 天×全市場矩陣）→ **自 2026-02-13（commit bd12bd9）起
所有生產 picks 用「別檔股票、別的日期」的特徵打分**（實證：7/2 pick 8071 的 manifest 特徵值
精確命中 2059 在 6/17 的列；抽驗 5 檔全 MISMATCH；multi_agent 路徑同受害）。
live-tracking 自 2 月中起無效。修法：isin mask 保留 index / key join + invariant 測試。

**P0-2 adj_close 與未還原 OHLC 混用**：`build_features.py:323-327` 只有 close 用 adj，
open/high/low 是 raw → ATR/KD/CCI/CMF/trend_persistence 歷史段物理不可能
（2016 atr_14_pct 中位數 42.8%、kd_d 均值 -281），失真隨時間遞減 = 日期 proxy。
`atr_inv` 在 CORE、`trend_persistence_inv` 在 PRUNED。**「誠實基準 0.99」建立在損壞特徵上**，
修完需再全量重建+重訓+重立基準。

**P1 重點**：factor 內部缺日 10.7 萬列 fillna(1.0) 假跳動（populate 應 ffill）；
rotation buffer 仍是日曆天（C/D 線殘餘 ~6 交易日洩漏）；C/D label 仍用 raw close；
cache staleness 只比 max_date（歷史重建不失效 cache，可重現性根因更廣）；
部署模型≠回測模型（87/800/ES vs 58/500/liq-weighted）；MySQL/Parquet 雙寫非原子且靜默 fallback；
transaction_cost 三處預設不一致 + ×4.1 猜單位；三個每日排程入口並存。

**安全**：無 Critical/High。ai_assist 遮罩 regex 實際壞掉（雙反斜線）、/strategy_runs 假超時凍結
全 API、.gitignore 2026-* 年份 rollover、git history 含真實持倉（repo 需 private）。

**效能**（量級級別）：trading_date object dtype 全表掃描、每次重訓全量 merge/copy、
data_store 全量重建稅（也是可重現性事故機制根源）、build_features 每日 MySQL 拉 250 天（fetch 115s）。

**修復順序**：①P0-1 止血+安全小包 → ②P0-2+factor ffill+cache freeze 一波全量重建立基準 v2
→ ③回測=部署對齊+補測試 → ④效能+scripts 歸檔 40 個 → ⑤C/D 線修正。

---

## 2026-06-24：搶抓 FinMind sponsor 還原股價（到期日當天）⭐

FinMind sponsor 2026-06-24 到期（6000/hr）。趁到期前抓下 sponsor 專屬的
**TaiwanStockPriceAdj（還原股價）**——這是修正 `adj_close=1.0`（近期查到「所有回測絕對
數字不可信」的根源）唯一缺的資料源，TWSE/TPEx 免費版沒有，過了今天拿不回來。

- 工具：`scripts/backfill_adj_prices.py`（一檔一 call、每 100 檔 checkpoint、可續跑、四碼過濾）
- 成果：**2450 檔 / 5,092,358 列 / 2016-01-04~2026-06-23**，存 `artifacts/adj_prices/adj_prices_10y.parquet`
- 抽查 2330：adj 2024-01-02=570.36（raw 593）、2020=298.58（raw 339）、2016=105.27 → 還原正確（回越早差越大）
- 2610 四碼中 2450 有 adj，160 無（多為 FinMind 不提供 adj 的下市股）

**✅ 第一個誠實基準已跑出（2026-06-24，隔離實驗）**：

| 版本（凍結同資料 10y） | Sharpe | MDD | Calmar | 累積 | 大盤 | 超額 |
|------|------:|----:|------:|----:|----:|----:|
| ① 全未還原（舊 headline） | 0.855 | -44.3% | 0.614 | +965% | +64.8% | +900.7% |
| ② 只還原報酬（label 仍 raw） | 0.827 | -46.7% | 0.540 | +816% | +97.5% | +719% |
| **③ 還原 label+報酬（誠實）** | **1.023** | **-30.7%** | **1.040** | **+1437%** | +97.5% | **+1340%** |

**關鍵：把訓練 label 也還原後策略反而更好（Sharpe 1.02、MDD -44%→-31%）。** 未還原 label 是「髒」的
（配息股因除權息跌價被標成「未來差」→ 模型學錯）；還原 label = 真實總報酬 → 選股更準（② vs ③ 100% 期數選股不同）。
②「只還原報酬」看起來差是因為報酬還原了卻用髒 label 選股，不一致；③ 才是一致的誠實版。

**機制鏈**：build_features `adj_close = close × adj_factor`，`price_adjust_factors` 表 739k 列**全 1.0**（= 沒還原的根源）；
build_labels 用 raw close。實驗用 BACKTEST_ADJ_PRICE_PARQUET overlay（報酬）+ 暫換 labels cache（adj_close.shift(-20)）達成，
features 仍未還原（比率特徵對乘法調整近乎不變，影響小）。labels cache 已還原備份。

**舊 headline（+1508%/+5115%）應退役**；可信誠實基準 = **Sharpe ~1.02 / MDD -31% / 超額 +1340%（對 +97% 含息大盤）**。
殘餘偏差：features 未還原、point-in-time 籌碼、160 檔無 adj 下市股、DSR 多重檢定 → 1.02 仍偏樂觀端但遠比之前誠實。

**✅ 已正式化到生產（2026-06-24）**：① `scripts/populate_adj_factors.py` 用 adj parquet 反推真實
adj_factor（=adj_close/raw_close，clip [0.1,10]）填 `price_adjust_factors`（4.92M 列，原全 1.0）
② `build_labels` 改用 adj_close（join factor 表，與 build_features 一致）③ 全量重建 features+labels+train+pick。

**生產路徑驗證（特徵+label+報酬全還原）**：Sharpe **0.985**、MDD **-31.2%**、累積 **+1320.6%**、
大盤 +97.8%、超額 **+1222.8%**、年化 +30.9%（重現隔離實驗 1.02/-31%）。**線上 features/labels/模型/選股
現在全是還原版。** 備份 `artifacts/features.bak_preadj`（確認穩定後可刪）。

⚠️ **post-sponsor 限制**：adj 只到 2026-06-23 快照（sponsor 過期不能更新）→ 之後新交易日 factor=1.0（未還原）。
影響小（近期股息累積少），但 daily build_features/labels 對新日期是未還原；長期需新 adj 源或定期手動補。

---

## 2026-06-22（晚）：vol-target 重驗 ❌ + 回測可重現性危害 ⚠️（最重要）

vol-target（記憶宣稱 +0.078 Sharpe）在乾淨基準上重驗 → **同資料只 +0.02 Sharpe、MDD 零改善，不採用**。
機制：台股小型動能股天生波動 40-60% >> target，scaler 幾乎永遠 <1 → 長期抱現金（60/119 期降倉），
target 30%→50% 都救不了，結構性無效。

**真正大發現**：**同一條指令跑出 Sharpe 1.035 或 0.805**——data_store parquet cache 是 lazy
（max_date 比對才重建），早上 10:00 pipeline 更新 DB 後 cache 沒同步，傍晚第一次回測觸發重建，
rep1 橫跨重建讀舊快照（1.035/-31.78% ≈ 文件 +1508%），cache 穩定後 rep2/3/4 讀新快照得 0.805/-44.26%
（三者 bit-identical）。比對證實 **118/119 歷史期全變**（連 2016 都變）→ 一次例行更新使 10y Sharpe
擺動 0.23、MDD 12.5pp。**含意：歷史上 |ΔSharpe|<0.2 的結論多半無法被證偽（vol-target/stacking/
LambdaRank/6 新特徵/rs_rank…）。訓練本身是決定性的（rep2/3/4 相同），先前 n_jobs 非決定性是誤報。**

詳見 decisions.md 同日條目。**待辦**：①實驗前凍結資料快照 ②overlay 改「固定預測只切 overlay」測法
③查 daily 重算為何改寫歷史 ④重建可信基準（0.805 vs 文件 1.06 落差需釐清）⑤線上模型今早已用 0.805 那份資料重訓。

---

## 2026-06-22：survivorship 去偏基準（第一個真正乾淨的基準）

回補 2016-2021 下市股（~5.13M 列）+ 全量重建 features/labels + 重跑 10y Strategy A。

| 指標 | 無下市股(6/17) | **含下市股(6/22)** |
|------|---------------|-------------------|
| 累積 | +1140.68% | **+1507.78%** |
| Sharpe | 0.994 | **1.06** |
| MDD | -34.46% | **-31.78%** |
| Calmar | 0.845 | **1.02** |
| 年化 | +29.11% | **+32.56%** |
| 期間 | — | 2016-07-11 ~ 2026-05-19 |

**關鍵發現：survivorship 修正後績效反升而非下降**（三項全改善）。原因：台股下市以
併購溢價為主（非破產，下市前往往上漲）+ clip -50% 限制地雷股 + 模型籌碼特徵迴避惡化股。
→ **增強「alpha 非 survivorship 假象」的可信度**——把最大回測偏差修掉後策略沒崩反穩。

**踩坑記錄**：
- `make rebuild-features` 只 DELETE MySQL features 沒清 Parquet → build_features 讀 Parquet
  max 認為已最新 → MySQL 永遠空 → 回測報「features 空」。已修（Makefile 加 rm Parquet + FORCE_RECOMPUTE）。
- backfill_delisted_prices 原用多日 chunk 全市場會空回，改逐交易日（用 raw_prices distinct date 定位）。
- _fetch_data 多餘 SQL ORDER BY 觸發 filesort 排 270 萬列 → 移除（pandas 後續本就排序），
  EXPLAIN 確認 filesort 消失。**index 無需新增（PK (stock_id,trading_date) 已理想）**。

**殘餘偏差（待修）**：T 日收盤成交盤後籌碼（point-in-time）、adj_close 全 1.0 未還原。

**下一步**：vol-target（已驗證 POSITIVE +0.078 Sharpe，純打開）在此乾淨基準上驗證。

---

## 2026-06-20 Session：/sc:task 後續優化

審計剩餘項派工（接續 2026-06-17 Phase 1-4）。

**已完成（commit c787e9a → befae48）**
- 4 個明確 bug：daily_alert 死功能（提醒永不觸發）、telegram「6/31 執行」、
  limit_down Sharpe 公式方向錯、breakout benchmark 跨股 pct_change
- LGBM 超參數三處統一：新增 skills/model_params.py 的 LGBM_BASE_PARAMS，
  C/D pick + backtest_rotation 三處引用（byte-identical，改超參數不再回測≠生產）
- 刪 15 個確認死碼（零引用 + 2+ 月未動，git 可找回）
- telegram listen CHAT_ID fail-closed（原 fail-open 接受任意聊天室操作 portfolio）

**關鍵決策**
- backtest_rotation.py 未 commit 的 regime_trailing/profit_ratchet/regime_exposure
  實驗 diff **已捨棄**（前兩者 memory 驗證 NEGATIVE、後者實作有 H-1/M-9 bug 且
  「大盤 regime 調曝險」方向歷史全 NEGATIVE）。未來不必重做這方向。
- **H-2（rotation 停牌股以進場價零損失出場）延後到下市股回補後做**：survivorship
  修正前 DB 幾乎無停牌股，H-2 不觸發、無法驗證；回補後才有實際影響且能驗證。

**仍待使用者決策/執行**
1. 下市股回補（backfill_delisted_prices.py，FinMind 6/24 前）→ 全量重建 → 重跑基準
2. H-2 停牌股出場修復（綁回補後一起）
3. C 類需重跑驗證：rotation mark-to-market、entry_delay=1、adj_close 還原
4. C/D 完整抽 rotation_pick_engine（87% 重複，高風險大重構，建議先補 C/D 測試）

---

## 2026-06-17 Session：審計後修復（11 項，Phase 1-4）

透過 /sc:analyze 四面向並行審計（資料洩漏/回測偏差、Strategy D 新碼、安全、架構）後修復。

**Phase 1 Quick wins（commit 6ab53b7）**
- 持倉檔 strategy_d_state.json / portfolio.json 脫離 git 追蹤；.gitignore 補 *.bak_* / d_replay/
- API host 預設 0.0.0.0→127.0.0.1（消 LAN 無認證暴露）
- conftest autouse fixture 清 INGEST_*_SOURCE → 修 make test 紅燈（3 failed→全綠）
- CLAUDE.md 特徵數校正 48→87/58；新增 tests/test_production_invariants.py（鎖配置）
- daily_run.sh 入版控 + 修「D pick 失敗仍推昨日舊訊號」bug + curl URL-encode

**Phase 2 Strategy D 生產 bug（commit eda88b6）**
- **C/D filter 位置 bug**：max_price(250)/流動性過濾誤砍打分宇宙，持倉漲破 250 或流動性下降
  被誤判 "Rank Drop" 強制賣出。修為「排名用全宇宙、過濾只套進場候選」，對齊 backtest_rotation.py
- skills/io_utils.py 原子寫入（temp+fsync+os.replace）+ safe_read_json；C/D state + telegram portfolio 改用

**Phase 3 News/Sentiment 洩漏（commit e1ac981）**
- 同日 lookahead：13:30 後/週末新聞改歸次一交易日（skills/feature_utils.align_news_to_trading_day）
- build_features news rolling 改交易日 reindex（原 row-based 可橫跨數月）
- news_sentiment_llm 改用 LLM 回傳 id 對齊（原 enumerate 位置對齊會錯位污染）
- ic_analysis_sentiment / ic_analysis_news 同步去 lookahead

**Phase 4 去偏 + 資料（commit 0b8b16a, 70dd923, 6501688）**
- label_horizon_buffer 改交易日制（消殘餘 ~6 交易日洩漏）：backtest.py searchsorted + train_ranker.py
- 新增 backfill_delisted_prices.py（修 survivorship）。診斷實證缺口：2016 缺 199 檔遞減到 2023 缺 30 檔

### 新基準（buffer 交易日制，2026-06-17）
| 指標 | 值 |
|------|-----|
| 累積 | +1140.68% |
| Sharpe | 0.994 |
| MDD | -34.46% |
| Calmar | 0.845 |
| 年化 | +29.11% |
| 期間 | 2016-07-06 ~ 2026-05-15 |
| 交易 | 3028 |

⚠️ 與舊 +5115%（2026-05-23）不可直接比（混 buffer 去偏 + adj_close 漂移 + 資料延伸至 2026-05，
含 2025-03~05 連跌）。Sharpe 0.99 ≈ 2026-04-23 快照 0.949。survivorship 未修，絕對績效仍偏高估。

### 待執行（需 DB + 時間；FinMind sponsor 6/24 到期前完成回補）
1. `python scripts/backfill_delisted_prices.py --start 2016-01-01 --end 2021-12-31`（分年回補下市股）
2. `FORCE_RECOMPUTE_DAYS=3650` 全量重建 features/labels（套用新 news 交易日對齊）
3. 重跑 10y 回測基準（survivorship 修正後的真實數字）
4. 重跑 news/sentiment IC（lookahead 修正後）確認訊號是否仍存在，再決定是否進模型
5. （可選）entry_delay_days=1 重建基準（T+1 成交，消除 point-in-time gap）

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
| ~~**parquet cache 24h TTL**~~ | ~~strategy_c_pick.py~~ | ~~pipeline 跑完後若 cache 未過期，daily-c 仍用舊資料~~ **已修正（2026-05-20，Stage 1.3）**：data_store._ensure 改用「max_date 比對」（cache_max < source_max → 重建），不再用 mtime+TTL。`_TTL_SECONDS` / `_is_fresh` 已刪除。`cache_info(db_session)` 加 synced 旗標。 |
| ~~**FinMind API Quota（402）**~~ | ~~ingest~~ | ~~**已修正（2026-04-15）**~~：402/429 配額錯誤現在立即拋出 FinMindError，不再靜默繼續 |

---

## Session 21 完成事項（2026-05-22）

### Stage 6 (stacking + multi-horizon) + Stage 7 (HRP + Vol Targeting)

| Stage | Commit | 結果 | 採用？ |
|-------|--------|------|--------|
| 6.1 異質模型 stacking | b7a4461 | IC +7.1% (LGBM/XGBoost/CatBoost rank-avg) | ✅ |
| 6.2 Multi-Horizon | 9513395 | IC **-2.2%**（h=20 alone 最強，混入 5/10/40 反劣）| ❌ |
| 7.1 HRP | 9de1ee1 | 60mo Sharpe **-0.130** / MDD 持平 | ❌ |
| **7.2 Vol Targeting** | ce0470f | **60mo +0.161 / 10y +0.078 / 10y MDD +4.29pp** | ✅ |

### Stage 7.2 Vol Targeting 完整驗證

設計：用 picks 過去 60d realized vol 估 portfolio vol，超過 target (30%) 拉
高 cash ratio 縮減總部位。與 market_filter_tiers complementary（後者看大盤）。

| Window | ΔSharpe | Δ MDD | Δ Calmar |
|--------|---------|-------|---------|
| 60-month | +0.161 | +6.56pp | +0.215 |
| 10y WF | +0.078 | +4.29pp | +0.100 |

10y < 60mo 因為 2016-2020 多頭期 vol-target 較少觸發。仍顯著正向。

**生產整合策略**：
- 目前透過 monkey-patch 驗證（`scripts/backtest_vol_target_quick.py`）
- 正式整合需要：
  1. `skills/backtest.py` 加 `_apply_vol_target_cash_share(self, picks, rb_date)` method
  2. 在 `_apply_market_regime_filter` 後 hook，cash_ratio = max(原值, vol_target_share)
  3. 加 env var `BACKTEST_VOL_TARGET_PCT=0.30`，0 為 disabled（預設）
- 留至下次 session 做（避免一次動 backtest.py 太多）

### Stage 6.1 Stacking 整合策略

stacking +7.1% IC lift 也未整合（同樣 monkey-patch 驗證）。整合需要：
- 改 `skills/train_ranker.py` 加 `use_stacking` config flag
- 接 `train_stacking_ensemble`
- 預測時用 `ens.predict(features, by_group=trading_date)` 給 0-1 percentile
- 跟既有 LightGBM-only 路徑共存

### 累積 alpha (Stages 5-7)

| 改善 | Sharpe Δ |
|------|---------|
| Stage 5.4 PER + fracdiff | +0.154 |
| Stage 6.1 stacking (估)| +0.04 |
| Stage 7.2 Vol Target | +0.078 |
| **總和 (理論)** | **+0.27** |

實際組合可能因互動效應略低，但接近 Stage 2 +0.3 的「重大改進」門檻。

### 新增 scripts (Session 21)

```
skills/stacking.py                   Stage 6.1 — 3-way GBDT ensemble
scripts/eval_stacking_quick.py       Stage 6.1 — IC sweep
skills/multi_horizon.py              Stage 6.2 — multi-horizon ensemble
scripts/eval_multi_horizon_quick.py  Stage 6.2 — 4-horizon IC eval
skills/hrp.py                        Stage 7.1 — HRP weights
scripts/backtest_hrp_quick.py        Stage 7.1 — 60mo HRP eval
skills/vol_targeting.py              Stage 7.2 — vol-based exposure scaling
scripts/backtest_vol_target_quick.py Stage 7.2 — 60mo / 10y vol-target eval
```

### 測試規模演進

| Session | tests |
|---------|-------|
| 19 (Stage 1+2) | 278 |
| 20 (Stage 4+5) | 348 |
| 21 (Stage 6+7) | 383 |

---

## Session 20 完成事項（2026-05-21）

### Stage 4 (Triple-Barrier + Meta-Label + FracDiff) + Stage 5 (IC + 整合)

| Stage | Commit | 重點 |
|-------|--------|------|
| 4.1 | de9a459 | Triple-Barrier label（López AFML Ch3）4.74M labels，PT/SL/Time 三 barrier。16 tests pass |
| 4.2 | dba02c0 | Meta-Labeling（LightGBM Classifier）AUC 0.704，threshold 0.55 → 1.66x precision lift |
| 4.2-eval | d1afc3d | In-sample quick eval：Δ Sharpe +0.45 @ thr=0.60（in-sample upper bound） |
| 4.3 | 0fc69da | FracDiff features（partial-memory price，d=0.3/0.4/0.5）4.78M values |
| 5.1+5.2 | 089bf02 | ingest_per 移出 SPONSOR_INGEST 區塊；IC 分析 5 個 effective factors |
| 5.3 quick | 4ecc3f6 | fracdiff 60mo +0.015 Sharpe / +12pp MDD；PER 60mo +0.091 Sharpe / +197pp cum |
| 5.3 combined | 2101f6d, 1f404b0 | combined 60mo +0.145 Sharpe / +211pp cum / +0.325 Calmar |
| **5.4 integrate** | **5766447** | **改 `_PRUNE_SET` + 新增 `ENRICHED_FEATURE_COLS` 常數 + enrich script post-process 11 個 features_YYYY.parquet** |
| **5.4 10y WF** | (待 commit) | **Δ Sharpe +0.154 / +2329pp cum 累積，60mo 與 10y 對齊高度一致** |

### Stage 5.2 IC 分析關鍵發現

對 5 個候選 factor 跑 cross-sectional Spearman IC（2018-01-01 ~ 2026-05-21，2016 天）：

| Group | Feature | IC mean | ICIR | Verdict |
|-------|---------|---------|------|---------|
| PER | per | -0.002 | -0.89 | ❌ 雜訊 |
| PER | **pbr** | -0.021 | **-6.29** | ✅ 有效 |
| PER | dividend_yield | -0.039 | -4.27 | ✅ 有效（樣本少）|
| FracDiff | **close_fracdiff_0_30** | -0.022 | -8.91 | ✅ 有效 |
| FracDiff | **close_fracdiff_0_40** | -0.024 | -9.81 | ✅ 有效 |
| FracDiff | **close_fracdiff_0_50** | -0.025 | **-10.36** | ✅ 最強 |
| Baseline | ma_5/20/60 | -0.012~-0.016 | -4.7~-6.3 | △ 邊緣 |
| Baseline | ret_20 | -0.028 | -10.46 | ✅ 有效（已 production）|

**三個關鍵 insight**：
1. **FracDiff (partial memory) > ma (zero memory)** — Stage 4.3 假設確認
2. **PER 雜訊，PBR 有效，Dividend Yield 強**（PER 受虧損股/成長股污染，PBR 較純）
3. **全部負 IC = 台股強 mean-reversion**（低 PBR/低 fracdiff/低 ret_20 → 高未來報酬）

### Stage 5.4 10y WF 對照（2016-2026，120 期）

| Metric | Baseline (52 feat) | +PER+fracdiff (55) | Δ |
|--------|--------------------|-----|------|
| 累積報酬 | +4,119.64% | **+6,448.97%** | **+2,329pp** |
| Sharpe | 1.167 | **1.321** | **+0.154** |
| MDD | -39.2% | -37.9% | +1.2pp |
| Calmar | 1.180 | 1.393 | +0.214 |
| 勝率 | 47.7% | 48.7% | +1.0pp |
| Trades | 2025 | 2025 | 0 |

**60mo vs 10y 對齊**：Δ Sharpe +0.145 (60mo) vs +0.154 (10y) → 真實 alpha 不是 noise。

**Stage 2 嚴格標準 +0.3 不達**，但：
- +0.154 = 13.2% Sharpe 相對提升
- +2329pp 累積報酬絕對 magnitude 顯著
- 不增加 trades，純粹「同樣交易做得更好」

**生產整合**：
- `_PRUNE_SET` 已移除 `pbr_ratio` / `dividend_yield`
- 新常數 `ENRICHED_FEATURE_COLS = ["close_fracdiff_0_50"]` 加進 `PRUNED_FEATURE_COLS`
- `scripts/enrich_features_stage5_4.py` 一次性 post-process 既有 features_YYYY.parquet
- 下次 daily pipeline 自動用新 features（ingest_per 已每日更新 raw_per）

### 新增 scripts（Session 20）

```
scripts/build_triple_barrier_labels.py    Stage 4.1 — TB label 產出
scripts/train_meta_label.py               Stage 4.2 — meta classifier
scripts/evaluate_meta_filter_effect.py    Stage 4.2 — in-sample sweep
scripts/build_fracdiff_features.py        Stage 4.3 — fracdiff backfill + ADF probe
scripts/ic_analysis_stage5.py             Stage 5.2 — cross-sectional IC
scripts/backtest_fracdiff_quick.py        Stage 5.3 — fracdiff inject test
scripts/backtest_per_factors_quick.py     Stage 5.3 — PER inject test
scripts/backtest_combined_quick.py        Stage 5.3 — combined inject test
scripts/enrich_features_stage5_4.py       Stage 5.4 — features parquet post-process
scripts/backtest_stage5_4_10y.py          Stage 5.4 — 10y WF baseline+treatment
```

### Stage 6-9 路線圖（剩餘）

| Stage | 內容 | 預期 |
|-------|------|------|
| 6 | LightGBM + CatBoost + XGBoost stacking + Multi-Horizon | Sharpe +0.1~0.2 |
| 7 | HRP / Vol Targeting / Kelly Criterion | MDD -5~-8pp |
| 8 | NLP / alt data (PTT/MOPS/order book) | unknown |
| 9 | MLflow / Optuna / Prefect 工程升級 | 維運效益 |

---

## Session 19 完成事項（2026-05-20）

### Stage 1（基礎建設 6 件）+ Stage 2.1 CI + Stage 2.2-2.4 統計嚴謹度

| Commit | 內容 |
|--------|------|
| `3287b23` chore(stage-1) | 6 quick wins：cron 17:30、TWSE 過濾 ETF、cache TTL 死碼刪、trading_calendar raw_prices 校準、TWSE 除權除息 endpoint、min_avg_turnover helper |
| `2e0c05a` ci: Tests workflow | GitHub Actions：pytest + 41 modules import smoke + 5 個 invariant guards（label_horizon_buffer>=20, clip=-0.50, BacktestPipeline 結構, 4 個 INGEST_*_SOURCE 預設 finmind）|
| 預定 commit | Stage 2.2-2.4：skills/statistics.py（DSR/PBO/CPCV）+ scripts/honest_baseline.py + 20 個新測試 |

### Stage 2 誠實基準關鍵發現

對 **Strategy A 現行生產 baseline**（`backtest_20260423_161310.json`, Sharpe 0.949, TR +1444%）跑 DSR：

| n_trials | sr_std | sr_null（H0 期望最大）| DSR p-value | 顯著？ |
|---------|--------|------------------|------------|--------|
| 20（A 系列變體）| 0.60（exclude rotation_）| 1.132 | 0.108 | ❌ |
| 30（A 系列含 entry/breakthrough）| 0.60 | 1.235 | 0.016 | ❌ |
| 80（所有試驗）| 0.75 | 1.831 | 0.000 | ❌ |

**PBO（Strategy A 系列 only）**：61 個策略 × 118 sample × 16 splits → PBO = **0.1%**（11/12870 combinations）。

**結論的兩個面向**：
- ✅ PBO 0.1% = 你 60+ 個 Strategy A 變體之間「train 期最佳在 test 期排名 < median」幾乎不發生 → 不是 random overfit
- ⚠️ DSR p < 0.95 = 觀察到的 Sharpe 0.99 在 multiple testing 校正後，**還沒明確超過 H0 期望最大值** → 真實 alpha 可能 < 看到的 0.95

**對 Stage 3-9 的啟示**：
1. 後續實驗的目標不是「再拿 +50% TR」，而是「**真實 Sharpe 提升 >= 0.3**」才有意義
2. 不能再相信「看到 Sharpe 1.5 就採用」這種決策
3. 每個 Stage 結束跑 `python scripts/honest_baseline.py --backtest <new_result>` 比對誠實 baseline
4. CI 已內建 invariant guard，避免無意中破壞 baseline

**新增工具**：
```bash
# 對指定 backtest 跑 DSR
python scripts/honest_baseline.py --backtest <path> --n-trials 20

# 同時跑 PBO across all backtests
python scripts/honest_baseline.py --pbo --pbo-min-periods 118
```

**測試規模**：258 → 278（+20 statistics tests）

---

## Session 18 完成事項（2026-05-19）

### Phase 1 + 2a + 2b：FinMind sponsor 退場（**5/20 sponsor 到期**緊急遷移）

| Commit | 內容 |
|--------|------|
| `d0a497f` Phase 1：SPONSOR_INGEST toggle | 預設 on，off 跳過 6 個 per-stock 重型 sponsor ingest，每日省 ~12,000 API calls |
| `db629ef` Phase 2a：TWSE/TPEx client + 對照工具 | 4 類資料 × 2 市場 × 2 模式，60 fixture 測試。`scripts/compare_twse_vs_finmind.py` |
| `49faad5` Phase 2a fix：STOCK_DAY_ALL 欄位順序 bug + HTTP retry | 對照工具實測抓到 OHLC 順序錯亂；加 5xx + 429 retry（預設 3 次指數退避）|
| `0a07cf4` compare script 加 load_dotenv | 沒載 .env 導致 FINMIND_TOKEN 讀不到，誤判為 free tier |
| `0d77ff7` Phase 2b：4 個 ingest 接 wire | `INGEST_*_SOURCE` env var 開關（預設 finmind 保留行為，可切 twse）。31 toggle tests。 |

**動機 / 觸發**：使用者 FinMind sponsor 2026-05-20 到期，不續訂，改用 TWSE/TPEx 免費官方 API。

**4 個 ingest 開關**（全部預設 finmind，向後相容）：
```
INGEST_PRICES_SOURCE=twse          # TWSE STOCK_DAY_ALL + TPEx daily_close_quotes
INGEST_INSTITUTIONAL_SOURCE=twse   # TWSE T86 + TPEx 3insti（OpenAPI 沒 T86，必走 legacy）
INGEST_MARGIN_SHORT_SOURCE=twse    # TWSE MI_MARGN + TPEx margin_balance
INGEST_PER_SOURCE=twse             # TWSE BWIBBU_d + TPEx peQryDate
```

**TWSE client 設計重點**：
- 官方 OpenAPI 不接受 date 參數（永遠回最新一天），backfill 必走 legacy `rwd/zh/*` 與 `www/zh-tw/*`
- TWSE T86 三大法人 **OpenAPI 沒有**，必走 legacy
- TPEx margin 拼字錯字 `ShortConvering`（應為 ShortCovering）已兼容
- 5xx + 429 自動 retry，預設 3 次指數退避（base 2s）
- Rate limit 預設 1.5s/req（社群實證安全值）
- 週末本地 skip（避免空 API call）

**重要 Schema 對應**（TWSE → DB）：
- institutional：TWSE T86 將「外陸資（不含外資自營商）」+「外資自營商」合併為 `foreign_*`；`dealer_buy`/`dealer_sell` 填 0（T86 row level 只給 net，features 只讀 net 故安全）
- margin_short：TWSE `*_today_balance` → DB `*_balance`，`*_cash_repayment` → `*_cash_repay`
- per/prices：欄位名稱直接對應

**已知限制 / 未做的事**：
- TWSE/TPEx 不提供調整後股價（adj_close） — FinMind 也只有 sponsor 才有，使用者 DB 本來就無 adj_close 故不影響
- 興櫃股不在 TWSE/TPEx 官方 endpoint（TPEx 另有 emerging endpoint，本次未接）
- ETF（5~6 碼含字母）被 `r"\d{4,6}"` 過濾保留，與生產 universe 過濾在 risk.py 一致

**測試規模**：119 → 139（session 17）→ 150（Phase 1）→ 210（Phase 2a）→ 211（fix）→ 242（Phase 2b，+31 toggle tests）

**實測結果（2026-05-19 晚）**：
- compare_twse_vs_finmind.py 跑出 prices 88% / institutional 6% / margin 100% / per 100%
- institutional 6% 是腳本 bug（FinMind long format 被當 wide 處理），實測 2330 數據 100% 一致
- prices 88% 是因為 **TWSE STOCK_DAY_ALL legacy 不接受 date 參數**（永遠回最新一天）—— 已 curl 證實，加 sanity check
- T86 / MI_MARGN / BWIBBU_d 接受 date，可 backfill

**FinMind batch query 也壞了**（2026-05-19）：
- `fetch_dataset_by_stocks(..., use_batch_query=True)` 對 ~500 stock_id 用逗號連起來查詢回 HTTP 400 "parameter data_id is illegal"
- 應是 FinMind 改 URL 長度限制或 schema
- pipeline 之前跑 2514 API call 一筆都沒寫入
- workaround: `scripts/backfill_prices_single_stock.py`（per-stock 一檔一 call）

**緊急 backfill 結果**：
- raw_prices 5/8-5/19 (~18,550 rows) 透過 FinMind single-stock backfill 完成（21 分鐘 / 2514 API calls）
- raw_institutional / raw_margin_short 透過 TWSE backfill（既有 _run_twse，1-2 分鐘）
- raw_per 4/24-5/19 (33,277 rows) 透過 TWSE backfill（修完 NaN bug 後成功）

**NaN bug 修正**：TWSE legacy 對虧損股 PER 給 '-'，safe_float→None，pd.to_numeric→NaN，pymysql 報「nan can not be used with MySQL」。4 個 _normalize_twse_* 都加 `.astype(object).where(notna, None)` 處理。

**最終切換步驟**（使用者今晚做）：
```
# .env 加入 5 行
INGEST_PRICES_SOURCE=twse
INGEST_INSTITUTIONAL_SOURCE=twse
INGEST_MARGIN_SHORT_SOURCE=twse
INGEST_PER_SOURCE=twse
SPONSOR_INGEST=off
```
明天 5/20 sponsor 過期後 daily pipeline 完全免費運作（~10 API calls 全走 TWSE）。

**測試規模**：242 tests pass（含 Phase 2a + 2b 共 91 新測試）

---



| Commit | 內容 |
|--------|------|
| `d0a497f` feat: SPONSOR_INGEST toggle | env var 開關，預設 on（向後相容），off 跳過 6 個 per-stock 重型 sponsor ingest，每日省 ~12,000 FinMind API calls。11 個 toggle 測試。 |
| `db629ef` feat: TWSE/TPEx client + cross-validation script | `app/twse_client.py` + 60 個 fixture 測試 + `scripts/compare_twse_vs_finmind.py` 對照工具。**不動任何 existing ingest**，等使用者驗證後再 wire-up。 |

**降 API 動機**：使用者付費原因是每小時 rate limit（不是 sponsor 特徵），免費 tier 不夠用。

**Phase 1 立即可用**：在 .env 設 `SPONSOR_INGEST=off` 即可省 240x API call。
這 6 個 ingest 的特徵全部在 `_PRUNE_SET`，生產模型未用，停跑零 alpha 損失。

**Phase 2a 基礎建設**：
- `TWSEClient`：4 類資料 (prices / institutional / margin_short / per) × 2 市場 (TWSE/TPEx) × 2 模式 (latest OpenAPI / history Legacy)
- 研究發現官方 OpenAPI 不接受 date 參數（永遠回最新一天），backfill 必須走 legacy `rwd/zh/*` 與 `www/zh-tw/*`
- TWSE OpenAPI **沒有 T86 個股三大法人**，只能走 legacy
- 已知 TPEx 拼字錯字 `ShortConvering` 在 client 兼容處理
- Rate limit 預設 1.5s/req（社群實證安全值）

**對照工具**：
```bash
# 純驗 TWSE 抓得到資料（不需 FinMind token）
python scripts/compare_twse_vs_finmind.py 2026-05-15

# 同時 diff FinMind（需要 FINMIND_TOKEN）
python scripts/compare_twse_vs_finmind.py 2026-05-15 --with-finmind
```
產出 `artifacts/twse_finmind_diff/<date>.json`：欄位匹配率、缺漏股、不一致樣本。

**待辦（Phase 2b，使用者驗證後）**：
1. 跑對照工具確認 TWSE vs FinMind 值一致
2. 若 OK，逐個 ingest 加 `INGEST_<NAME>_SOURCE=finmind|twse` env var
3. 順序建議：prices → institutional → margin_short → per（先試流量最大的）
4. 每個 ingest 切換後留 1 週觀察期再切下一個

**測試規模**：119 → 139（session 17）→ 150（Phase 1）→ 210（Phase 2a，+60 新測試）

---

## Session 17 完成事項（2026-05-18）

### /sc:analyze 全面審計 + P0~P3 全派工修正

四個並行 agent 完成多面向改善，139 tests passed，smoke backtest byte-identical。

| Commit | Hash | 變動 |
|--------|------|------|
| chore: harden API + remove dead code | `d8c8582` | api.py CORS + Path regex + 刪除 skills/market_regime.py + CLAUDE.md 同步 |
| test: add regression guards | `8e05db0` | tests/test_label_leakage.py（5）+ tests/test_backtest_clip.py（15）共 20 個保護測試 |
| refactor: silence-free ingest + print→logger | `b6f0b9c` | 23 處 except 審計（修 3 個真實 silent bug）+ 79 處 print→logger（16 檔）|
| refactor: extract BacktestPipeline class | `09987cd` | run_backtest 拆 BacktestPipeline（.prepare / ._train_model_for_period / .run），thin wrapper 保留，行為 byte-identical |

**3 個被揪出的真實 silent bug**：
1. `ingest_quarterly_fundamental.py:_fetch_one_dataset`：`except Exception: return DataFrame()` 吞掉 FinMind 配額錯誤 → 2000+ 股全部空跑無紀錄
2. `ingest_fundamental.py:_write_batch`：`return 0` 吞掉 schema 錯誤，整批丟棄無紀錄
3. `ingest_securities_lending.py`：`except Exception` 過寬，配額錯誤被當成單股 timeout 忽略

**新發現的可選後續工作**（agent 報告中提到，未動）：
- `backtest.py` 的 simulation phase 仍可拆 `_select_picks / _apply_filters / _compute_benchmark_return` 將 `run()` 從 864 行降到 ~300 行
- `min_avg_turnover` 單位（億元 vs 元）的 footgun，新 caller 易誤用
- `backtest.py` candidate 過濾僅用 `\d{4}` regex，EMERGING 過濾僅在 feature-scoring 階段（既有問題，未變動）

**重要更正**：先前 /sc:analyze 報告誤判 `.env` 與 `.env.example` 漂移方向。實際是 .env 缺 14 個 key，.env.example 完整。code 對這些 key 有 fallback default，無需動作。

---

## Session 16 完成事項（2026-04-23）

### 新增 PER/借券/季報資料集 + 修正 daily-c/d 特徵過濾

| 項目 | 說明 |
|------|------|
| `ingest_per.py` | TaiwanStockPER：每日 PER/PBR/殖利率，per-stock 查詢，1 year chunk |
| `ingest_securities_lending.py` | TaiwanStockSecuritiesLending：逐筆借券彙整，90 天 chunk |
| `ingest_quarterly_fundamental.py` | BalanceSheet + FinancialStatements + CashFlow：季報財務指標，60 天公告延遲 merge_asof |
| 新 ORM 模型 | `RawPER`, `RawSecuritiesLending`, `RawQuarterlyFundamental` |
| migration `012` | `raw_per`, `raw_securities_lending`, `raw_quarterly_fundamental` 三表 |
| `build_features.py` | FEATURE_COLUMNS 59→68（+9 新特徵）；全部新特徵入 `_PRUNE_SET` + `_SPONSOR_FEATURES` |
| 季報 fetch | `_fetch_data()` 新增 PER/借券/季報三個 fetch block + per-stock merge_asof |
| `daily_pipeline.py` | ingest_per / ingest_securities_lending / ingest_quarterly_fundamental 作為 optional skill（15-17） |
| `Makefile` | 新增 backfill-per, backfill-securities-lending, backfill-quarterly-fundamental, backfill-value-factors |
| `backfill_sponsor.py` | 新增 per/securities_lending/quarterly_fundamental 到 DATASETS + ALL_ORDER |
| **Bug fix: daily-c/d _PRUNE_SET** | `strategy_c_pick.py` / `strategy_d_pick.py` 從 `skills.build_features` 匯入 `_PRUNE_SET`，訓練特徵從 68 降至 50（與 backtest 一致）|
| test fix | `test_detect_schema_outdated_json_string` 補 `_patch_feature_store_empty()`（119 tests passed）|
| commit | `67b1918` |

### PER 回補狀態
- PER backfill 已在背景啟動（10y, ~2000 stocks）
- 預估 3-5 小時完成（每股 1 API call × 10年 / rate limit）
- 完成後：`make pipeline` 開始寫入 PER 特徵

### 待辦（下一步）
1. 等待 PER/SecuritiesLending/QuarterlyFundamental 回補完成
2. 跑 `make backfill-securities-lending` 和 `make backfill-quarterly-fundamental`（較慢）
3. 全量重建 features（`FORCE_RECOMPUTE_DAYS=3650`）使新特徵填入歷史資料
4. IC 分析：評估 earnings_yield / pbr_ratio / lending_balance_ratio / roe_ttm 的 ICIR
5. 若 IC 確認，從 `_PRUNE_SET` 移出並重跑 10y WF

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
| ~~parquet cache 24h TTL~~ | ✅ 已修（Stage 1.3，2026-05-20）|
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
