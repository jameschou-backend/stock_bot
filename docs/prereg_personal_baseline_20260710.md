# 預登記：personal-baseline 第三臂（個人可執行口徑）

> **預登記時間戳：2026-07-10 21:19 CST（Asia/Taipei）**
> 本文件在跑任何回測數字**之前**寫完並固定。判準、配置、裁決規則以本文件為準，
> 跑完後不得回頭修改判準（結果段落於跑完後追加，判準段落凍結）。

---

## 1. 背景與動機

- 基準 v2.2（2026-07-10 官方 factor 切換後，`backtest_20260710_190850/192027.json`）：
  主臂（0.5 億門檻，機構口徑）超額 **-41pp（轉負）**；無門檻歸因臂 Sharpe 0.881 / 超額 +623pp。
  → **機構口徑在正確資料下無超額；系統全部價值在微型股臂。**
- 容量分析（`scripts/capacity_analysis.py`）：使用者資金 1,000,000 TWD、topn 30、
  參與率 10% → 個人等效門檻 ≈ 333,333 TWD/日，遠低於 0.5 億。0.5 億門檻對個人資金**過嚴**。
- 本臂回答唯一問題：**「使用者實際可執行」的口徑下，這個系統值不值得實盤。**

## 2. 配置（機械推導，ex-ante 固定，不掃描）

生產 preset（`--production-baseline`）為基底，僅改門檻相關參數。所有推導只依賴資金結構，
**不看任何回測結果**：

| 參數 | 值 | 推導 |
|------|-----|------|
| 資金 C | 1,000,000 TWD | 使用者實際資金 |
| 持股數 N | 30 | 生產 preset topn（不改） |
| 每檔部位 | 33,333 TWD | C / N |
| 單日參與率上限 p | 10% | 部位 ≤ 日成交金額 10%（不砸盤） |
| **min_avg_turnover** | **333,333 TWD ≈ 0.0033 億** | (C/N) / p；CLI `--min-avg-turnover 0.0033`（固定，不掃描） |
| **max_stock_price** | **33 元（原始收盤價）** | 一張 = 1,000 股；33,333 / 1,000 ≈ 33 元，>33 元買不起一張；CLI `--max-price 33` |
| 滑價（基準臂） | tiered slippage | amt_20 <1 億 1.0% / 1~5 億 0.6% / >5 億 0.2%（來回）；CLI `--tiered-slippage` |
| 滑價（悲觀臂） | tiered × **1.5** | CLI `--slippage-mult 1.5`（本次新實作精確倍率，非近似） |
| 其餘 | 生產 preset | seasonal filter、no-stoploss、market-filter-tiers、min-pos 2、liq-weighted、pruned 58 特徵、entry_delay=0、retrain 3 個月、月頻、clip -50% |

### max-price 機制（本次新實作，`skills/backtest.py`）

比照 `backtest_rotation.py --max-price` 語義：**排名用全宇宙（打分先完成）、過濾只套進場
候選**；口徑為**原始收盤價**（`close_raw` = 實際下單價，不可用還原價）；排除 px ≤ 0。
benchmark **不套** max-price（理由：價格上限是「本策略 30 檔等權結構」的後果，不是市場
不可投資性；個人的機會成本臂另有 vs_taiex_tr 與零成本等權大盤）。benchmark 仍套用與
策略相同的 min_avg_turnover（引擎既有行為）。

## 3. P&L 口徑（本次新實作：`BACKTEST_ADJ_FROM_DB=1`）

- P&L 直接用 DB `price_adjust_factors`（TWSE/TPEx 官方自算 factor，2026-07-10 切換）還原
  OHLC（raw × factor，per-stock ffill/bfill 對齊 `apply_adj_factors` 語義），
  **與 label/特徵同源**，消除 v2.2 的「label=官方 factor、P&L=凍結 FinMind parquet」混口徑。
- 優先序：`BACKTEST_ADJ_FROM_DB=1` > `BACKTEST_ADJ_PRICE_PARQUET` > raw close；
  summary 記 `pnl_convention: "adj_from_db_official"` + `adj_db_snapshot`（factor 表快照）。
- **ex-ante factor 表快照（2026-07-10 21:18）**：4,512,766 列 / 1,997 檔 /
  2016-02-15 ~ 2026-07-09。三臂結果 JSON 的 `adj_db_snapshot` 必須一致，否則不可比。
- 三臂一律 `DATA_STORE_FREEZE=1`（prices/features/labels 凍結同一 cache 快照；
  結果 JSON `data_snapshot.cache_rebuilt_during_run` 必須為 false，否則該臂作廢重跑一次並記錄）。

## 4. 三臂（各跑一次，不因結果重跑調參）

```bash
# 臂 1：基準臂（個人門檻 + max-price 33 + tiered slippage）
BACKTEST_ADJ_FROM_DB=1 DATA_STORE_FREEZE=1 python scripts/run_backtest.py \
  --months 120 --production-baseline --min-avg-turnover 0.0033 --max-price 33 \
  --tiered-slippage --output artifacts/backtest/personal_baseline_20260710_arm1_base.json

# 臂 2：悲觀臂（同臂 1 + 滑價 ×1.5）——判準只看這一臂
BACKTEST_ADJ_FROM_DB=1 DATA_STORE_FREEZE=1 python scripts/run_backtest.py \
  --months 120 --production-baseline --min-avg-turnover 0.0033 --max-price 33 \
  --tiered-slippage --slippage-mult 1.5 \
  --output artifacts/backtest/personal_baseline_20260710_arm2_pessimistic.json

# 臂 3：連續性對照（生產 preset 原樣 = 0.5 億門檻、無 max-price、無滑價；
#        僅 P&L 口徑由 parquet 換成 DB 官方 factor → 與 v2.2 主臂唯一差異 = P&L 來源）
BACKTEST_ADJ_FROM_DB=1 DATA_STORE_FREEZE=1 python scripts/run_backtest.py \
  --months 120 --production-baseline \
  --output artifacts/backtest/personal_baseline_20260710_arm3_continuity.json
```

臂 3 配置說明：任務語義「同配置但門檻 0.5 億」的目的是**驗證與 v2.2 主臂銜接**——
v2.2 主臂 = `--production-baseline` + parquet overlay（無 max-price、無滑價）。
故臂 3 取「生產 preset 原樣 + DB 口徑」，使臂 3 vs v2.2 主臂（0.520/-38.8%/+252%）的差
**只歸因於 P&L 來源**（凍結 FinMind parquet → DB 官方 factor）。若臂 3 加上 max-price
與滑價則同時混三個變因，銜接失效。

## 5. 判準（單一、先寫死，只適用**悲觀臂（臂 2）**）

| 裁決 | 條件 |
|------|------|
| **PASS**（值得個人口徑實盤） | 悲觀臂 Sharpe ≥ 0.70 **且** MDD ≥ -50%（即 max_drawdown > -0.50） |
| **GRAY**（paper 半年再議） | 悲觀臂 Sharpe 在 [0.50, 0.70) **且** MDD ≥ -50% |
| **FAIL** | 悲觀臂 Sharpe < 0.50 **或** MDD < -50% |

- 指標取悲觀臂 summary 的 `sharpe_ratio`（rf=1.5%，月化年化）與 `max_drawdown`。
- 臂 1（基準臂）只做滑價敏感度歸因；臂 3 只做銜接驗證——兩者**不參與裁決**。
- DSR / bootstrap CI 為附帶報告（缺陷 6 規則 1 的例行輸出），**不進裁決條件**，
  但 DSR p 值須如實報告：`n_trials = trial registry 行數 + 80`（引擎自動計算；
  registry 現 4 行，三臂依序執行時 n_trials 依次為 85 / 86 / 87）。
- PASS 亦不代表立刻全倉：依 2026-07-10 總體檢資金配置建議，首階段真金 ≤ 20-30 萬。

## 6. 已知限制與殘餘偏差（ex-ante 承認，不作為事後開脫）

1. **entry_delay=0 + T 日盤後籌碼**（point-in-time）：與 v2.2 headline 同口徑（為銜接保留）。
   D 重驗顯示此類 lookahead 在**日頻輪動**下價值巨大（Sharpe +0.87）；月頻全換倉影響遠小
   但非零、方向偏樂觀。個人口徑若 PASS，實盤前仍須跑 entry_delay=1 敏感度（後續另行預登記）。
2. **微型股 segment 資料品質最差**：官方 factor 覆蓋 1,997 檔，其餘（多為下市/資料缺）factor=1.0
   未還原；處置股/漲跌停不可成交/一張的最小下單粒度（部位 3.3 萬 vs 一張 33 元 × 1000 股
   的整數倍限制）均未建模。tiered slippage 小型股 1% 來回是近似，悲觀臂 ×1.5 是對此的緩衝。
3. **流動性 map 用還原 close × volume**（沿 v2.2 慣例，與臂 3 銜接一致）；個人門檻 0.0033 億
   極低，實際過濾力主要來自 max-price 33。
4. 月頻等權每檔 3.3 萬的「參與率 10%」假設進出各分散在一天內完成；未建模多日拆單。
5. DSR 的 n_trials 基數 80 為歷史估計下限，實際歷史試驗更多 → p 值偏樂觀端。

## 7. 執行紀律

- 三臂依序單線程執行（避免 DB/cache 競爭）；執行期間不跑 pipeline / 不重建 cache。
- 每臂檢查結果 JSON：`pnl_convention == "adj_from_db_official"`、`adj_db_snapshot` 三臂一致、
  `data_snapshot.cache_rebuilt_during_run == false`。
- 跑完後：結果與裁決追加至本文件「結果」段（判準段落不動），並更新 memory。

---

## 8. 結果（跑完後追加）

（預登記時此段為空。）

---

## 裁決（跑完後按判準機械適用）

| 臂 | Sharpe | 累積 | 超額 | MDD |
|----|-------:|-----:|-----:|----:|
| 1 基準（個人門檻 0.0033 億 + max-price 33 + tiered slip） | 0.537 | +215% | **-82.5pp** | -31.4% |
| 2 悲觀（滑價 ×1.5）——判準臂 | **0.305** | +81% | -217pp | -37.2% |
| 3 連續性（0.5 億） | 0.492 | +230% | -69.7pp | -39.8% |

（P&L=adj_from_db_official；臂 2 excess Sharpe 95% CI 上界仍為負）

**判定：FAIL**（悲觀臂 0.305 < 0.5）。個人可執行口徑不值得投入真金。

**關鍵推論**：無門檻臂的 +623pp 微型股 alpha **不在「買得起」的子集裡**——
加上 33 元價格上限與微型股滑價後 alpha 消失，等權大盤都跑不贏。
「個人吃得到微型股 alpha」的容量假設被證偽（至少在整股口徑下）。

**留下的唯一開放問題**（需另行預登記，不是本裁決的翻案管道）：
零股交易可解除 33 元上限（放寬 universe 至全部微型股），但零股的
spread/流動性成本需要新的滑價模型——若未來要驗，先建零股成本模型再預登記。
