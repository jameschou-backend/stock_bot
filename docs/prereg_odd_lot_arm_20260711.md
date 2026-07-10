# 預登記：odd-lot 零股口徑臂（personal-baseline 的零股延伸）

> **預登記時間戳：2026-07-10 22:47 CST（Asia/Taipei）**
> 本文件在跑任何回測臂**之前**寫完並固定。判準、配置、成本模型、裁決規則以本文件為準，
> 跑完後不得回頭修改判準（結果段落於跑完後追加，判準段落凍結）。
> 零股成本模型的校準（calibration.json，2026-07-10 22:39 完成）先於本預登記、
> 且不依賴任何回測結果——校準只用市場微結構資料，估計量在看到回測數字前已固定。

---

## 1. 背景與動機

- personal-baseline 三臂 **FAIL**（docs/prereg_personal_baseline_20260710.md，2026-07-11 裁決）：
  悲觀臂 Sharpe 0.305 < 0.50。關鍵推論：無門檻臂的 +623pp 微型股 alpha 不在
  「買得起（≤33 元整股）」的子集——**33 元價格上限砍掉了微型股 alpha**。
- 該裁決留下的唯一開放問題（原文）：「零股交易可解除 33 元上限（放寬 universe 至全部
  微型股），但零股的 spread/流動性成本需要新的滑價模型——若未來要驗，先建零股成本模型
  再預登記。」**本文件就是那個預登記**；成本模型已實證校準完成。
- 本臂回答唯一問題：**用零股交易解除價格上限後，個人可執行口徑（100 萬 / 30 檔）
  是否值得實盤。**

## 2. 零股成本模型（實證校準，ex-ante 固定）

### 2.1 資料源與校準方法（scripts/calibrate_odd_lot_costs.py）

- **TWSE 盤中零股交易行情單（TWTC7U）** + **TPEx 盤中零股每日收盤行情（oddQuote）**，
  官方免費日資料，含每檔收盤**最後揭示買/賣價**。
- 樣本：**2025-07-01 ~ 2026-06-30（241 交易日，465,537 檔-日，僅上市/上櫃四碼普通股）**，
  與 DB 整股行情 join。原始資料 `artifacts/odd_lot/odd_lot_daily.parquet`，
  校準報告 `artifacts/odd_lot/calibration.json`（generated_at 2026-07-10T22:39 CST）。
- **估計量（先寫死）**：per-side premium = 各 amt_20 流動性層
  「收盤零股報價半價差 (ask−bid)/2/mid」的 **P75**（取 P75 偏保守，涵蓋較差簿況）。
  VWAP 偏離與零股/整股收盤基差為佐證分佈（量級一致），不進模型。
- 分層門檻（amt_20 = 整股 20 日均成交金額，與既有 tiered slippage 同語義）：
  0.1 / 0.3 / 1 / 5 億元。

### 2.2 校準結果（模型採用值，鎖定於 skills/odd_lot_costs.py + tests/test_odd_lot_costs.py）

| amt_20 層 | n（檔-日） | 無零股成交率 | half-spread median | **P75（採用）** | P90 | \|basis\| median |
|-----------|---------:|-----------:|------:|------:|------:|------:|
| < 0.1 億 | 204,549 | 3.4% | 0.45% | **0.99%** | 2.15% | 1.05% |
| 0.1~0.3 億 | 73,803 | 0.1% | 0.23% | **0.44%** | 0.83% | 0.54% |
| 0.3~1 億 | 63,875 | ~0% | 0.18% | **0.33%** | 0.60% | 0.45% |
| 1~5 億 | 65,427 | 0% | 0.15% | **0.23%** | 0.40% | 0.42% |
| ≥ 5 億 | 57,883 | 0% | 0.12% | **0.19%** | 0.25% | 0.41% |

### 2.3 成本組成（round trip，`skills/odd_lot_costs.odd_lot_round_trip_cost`）

```
2 × premium_per_side(amt_20) × premium_mult × era_mult(rb_date)
+ 2 × min_fee / position_size
```

- **premium_mult**：悲觀臂 1.5（只縮放 premium，不縮放低消）。
- **era_mult**：盤中零股 **2020-10-26** 上線；之前只有盤後零股（一天一撮、流動性更差），
  該時代 premium **×2** 近似——只有全窗參考臂會觸發。
- **min_fee**：券商零股低消保守取 **20 元/邊**；對 33,333 元部位 = **0.06%/邊**
  （0.12% 來回）。註：引擎 transaction_cost（0.585% 來回）已含整股 0.1425% 手續費，
  低消為**額外疊加的保守 buffer**（偏悲觀方向，ex-ante 接受）。
- 微型股層來回成本 = 0.99%×2 + 0.12% ≈ **2.10%**（悲觀臂 ≈ 3.09%）——
  約為既有整股 tiered slippage 小型股假設（1.0% 來回）的 2~3 倍。
- 取代（非疊加）tiered slippage：`--odd-lot-costs` 與 `--tiered-slippage` 互斥，前者優先。

## 3. 配置（機械推導，ex-ante 固定，不掃描）

生產 preset（`--production-baseline`）為基底，僅改口徑相關參數：

| 參數 | 值 | 推導 |
|------|-----|------|
| 資金 C / 持股數 N / 每檔部位 | 1,000,000 TWD / 30 / 33,333 TWD | 與 personal-baseline 相同 |
| **min_avg_turnover** | **0.0033 億（≈333,333 TWD）** | (C/N)/參與率 10%，沿 personal-baseline，不掃描 |
| **max_stock_price** | **無（不設）** | 零股以「股」為單位，33,333 元部位可買任何價位股票——價格上限的存在理由消失 |
| **執行成本** | **零股成本模型（§2）** | 取代 tiered slippage |
| 其餘 | 生產 preset | topn 30、seasonal filter、no-stoploss、market-filter-tiers、min-pos 2、liq-weighted、pruned 58 特徵、entry_delay=0、retrain 3 個月、月頻、clip -50% |

### P&L 口徑

- `BACKTEST_ADJ_FROM_DB=1`（DB 官方 factor 還原，與 label/特徵同源；
  summary `pnl_convention == "adj_from_db_official"`）。
- **ex-ante factor 表快照（2026-07-10 22:45）**：4,512,766 列 / 1,997 檔 /
  2016-02-15 ~ 2026-07-09（與 personal-baseline 三臂同快照）。三臂 `adj_db_snapshot` 必須一致。
- 三臂一律 `DATA_STORE_FREEZE=1`；`data_snapshot.cache_rebuilt_during_run` 必須 false，
  否則該臂作廢重跑一次並記錄。

## 4. 時代限制與三臂（各跑一次，不因結果重跑調參）

盤中零股 2020-10-26 才上線（之前盤後零股一天一撮）→ **誠實窗口 = 2020-11 起（~68 個月）**。

```bash
# 臂 1：主窗基準臂（2020-11-01 起，盤中零股時代）
BACKTEST_ADJ_FROM_DB=1 DATA_STORE_FREEZE=1 python scripts/run_backtest.py \
  --months 120 --production-baseline --min-avg-turnover 0.0033 \
  --odd-lot-costs --eval-start 2020-11-01 \
  --output artifacts/backtest/odd_lot_20260711_arm1_main_base.json

# 臂 2：主窗悲觀臂（premium ×1.5）——判準只看這一臂
BACKTEST_ADJ_FROM_DB=1 DATA_STORE_FREEZE=1 python scripts/run_backtest.py \
  --months 120 --production-baseline --min-avg-turnover 0.0033 \
  --odd-lot-costs --odd-lot-premium-mult 1.5 --eval-start 2020-11-01 \
  --output artifacts/backtest/odd_lot_20260711_arm2_main_pessimistic.json

# 臂 3：全窗參考臂（10 年；2016~2020-10 段 premium ×2 盤後零股近似）——僅供參考，不進裁決
BACKTEST_ADJ_FROM_DB=1 DATA_STORE_FREEZE=1 python scripts/run_backtest.py \
  --months 120 --production-baseline --min-avg-turnover 0.0033 \
  --odd-lot-costs \
  --output artifacts/backtest/odd_lot_20260711_arm3_fullwindow_ref.json
```

## 5. 判準（單一、先寫死，只適用**主窗悲觀臂（臂 2）**）

| 裁決 | 條件 |
|------|------|
| **PASS**（值得零股口徑實盤） | 悲觀臂 Sharpe ≥ 0.70 **且** MDD ≥ -50%（即 max_drawdown > -0.50） |
| **GRAY**（paper 半年再議） | 悲觀臂 Sharpe 在 [0.50, 0.70) **且** MDD ≥ -50% |
| **FAIL** | 悲觀臂 Sharpe < 0.50 **或** MDD < -50% |

- 指標取悲觀臂 summary 的 `sharpe_ratio`（rf=1.5%，月化年化）與 `max_drawdown`。
- 臂 1 只做滑價敏感度歸因；臂 3 只做長窗參考——兩者**不參與裁決**。
- **主臂窗口只有 ~68 個月，統計效力比 120 期更弱**：DSR / bootstrap CI **必附**、如實報告，
  但不進裁決條件。`n_trials = trial registry 行數 + 80`；registry 現 8 行（含本次 odd-lot
  smoke run），三臂依序執行時 n_trials 依次為 **89 / 90 / 91**。
- PASS 亦不代表立刻全倉：依 2026-07-10 總體檢資金配置建議，首階段真金 ≤ 20-30 萬。

## 6. 已知限制與殘餘偏差（ex-ante 承認，不作為事後開脫）

1. **entry_delay=0 + T 日盤後籌碼**（point-in-time lookahead）：與 v2.2 / personal-baseline
   同口徑（為銜接保留）。月頻影響遠小於日頻但非零、方向偏樂觀。若 PASS，實盤前仍須跑
   entry_delay=1 敏感度（另行預登記）。
2. **校準窗（2025-07~2026-06）套用到整個 2020-11 起的窗口**：假設零股價差 regime 穩定。
   早年零股滲透率較低 → 價差可能更寬 → 主臂偏樂觀端；悲觀臂 ×1.5 是對此的緩衝。
3. **premium 模型 = 跨一次收盤報價半價差**：未建模「單日簿深不足 3.3 萬」的多日拆單/
   衝擊成本。微型股層 3.4% 檔-日**完全無零股成交**（該日可能根本買不到）——未建模
   miss-fill/替代效應。P75（而非 median）+ 悲觀 ×1.5 為緩衝。
4. **微型股 segment 資料品質最差**（與 personal-baseline 同）：官方 factor 覆蓋 1,997 檔，
   其餘（多為下市股）factor=1.0 未還原；處置股/漲跌停不可成交未建模。
   零股不能下市價單掛漲跌停鎖死簿——漲跌停日實際成交概率低於整股，未建模。
5. **era ×2（2016~2020-10 盤後零股近似）是粗近似**：該時代一天一撮、
   簿更薄，×2 無實證校準（無法取得當年零股報價資料）——故全窗臂**僅供參考**。
6. **主臂窗口涵蓋 2020-11~2021 微型股大多頭**（歸因臂該段年報酬 +58%/+70%）：
   窗口短且起點偏多頭，Sharpe 可能被抬高；2022 熊市在窗內，MDD 判準仍有效。
7. **benchmark = 等權大盤（同 0.0033 門檻）零成本**（v2.1 起口徑）：benchmark 本身
   假設零成本執行（不可實際達成），超額判讀偏嚴——方向與悲觀原則一致。
8. 低消 20 元/邊與 transaction_cost 內含手續費部分重複計算（保守）；
   股利/除權息由 adj factor P&L 涵蓋（零股享有同等配息權利）。
9. DSR 的 n_trials 基數 80 為歷史估計下限，實際歷史試驗更多 → p 值偏樂觀端。

## 7. 執行紀律

- 三臂依序單線程執行（避免 DB/cache 競爭）；執行期間不跑 pipeline / 不重建 cache。
- 每臂檢查結果 JSON：`pnl_convention == "adj_from_db_official"`、`adj_db_snapshot` 三臂一致
  且與 §3 快照一致、`data_snapshot.cache_rebuilt_during_run == false`、
  `config.odd_lot_costs.premium_mult` 分別為 1.0 / 1.5 / 1.0。
- 跑完後：結果與裁決追加至本文件「結果」段（判準段落不動），並更新 memory。

---

## 8. 結果（2026-07-10 23:20 CST 追加；判準段落未動）

三臂依序執行完畢（trial registry #9/#10/#11，n_trials 89/90/91）。執行紀律全過：
三臂 `pnl_convention == "adj_from_db_official"`、`adj_db_snapshot` 一致
（window 內 4,442,158 列 / 1,997 檔 / 2016-03-15 ~ 2026-06-10；全表快照 4,512,766 列與
§3 一致）、`cache_rebuilt_during_run == false`、premium_mult 1.0 / 1.5 / 1.0。

| 臂 | 窗口 | Sharpe | 累積 | 大盤（等權零成本） | 超額 | MDD | 年化 |
|----|------|-------:|-----:|-----:|-----:|----:|----:|
| 1 主窗基準 | 2020-11-02 ~ 2026-06-10（68 期） | 0.310 | +43.8% | +160.7% | **-117pp** | **-54.4%** | +6.7% |
| **2 主窗悲觀（判準臂）** | 同上 | **0.083** | **-1.3%** | +160.7% | **-162pp** | **-58.4%** | -0.2% |
| 3 全窗參考（era ×2 近似） | 2016-08-01 ~ 2026-06-10（119 期） | -0.063 | -34.9% | +297.5% | -332pp | -72.5% | -4.3% |

統計紀律（block-bootstrap 95% CI + DSR，如實附上）：

| 臂 | Sharpe 95% CI | Excess Sharpe 95% CI | DSR p（n_trials） |
|----|--------------|---------------------|------------------|
| 1 | [-0.774, 1.282] | [-1.571, +0.511] | 0.223（89） |
| 2 | [-1.038, 1.037] | [-1.928, +0.205] | 0.099（90） |
| 3 | [-0.756, 0.624] | [-1.673, **-0.118**] | 0.016（91） |

主窗逐年（策略 / 大盤）：臂 1：2021 +38.0%/+30.7%、**2022 -48.2%/-8.1%**、2023 +32.2%/+34.8%、
2024 +44.7%/+12.7%、**2025 -24.5%/+4.6%**、2026H1 +18.9%/+20.1%。
臂 2 同型更差（2022 **-51.8%**）。

---

## 裁決（按 §5 判準機械適用）

**FAIL**（雙重觸發）：悲觀臂 Sharpe **0.083 < 0.50**，且 MDD **-58.4% < -50%**。
基準臂（premium ×1.0）也自行擊穿兩條線（0.310 / -54.4%），FAIL 不依賴悲觀倍率。
全窗參考臂 excess Sharpe 的 95% CI 上界為**負**——零成本等權大盤在統計上顯著跑贏本策略。

**關鍵推論**：
1. **微型股 alpha 付不起零股的價差**。歸因臂（無門檻、無執行成本模型）+623pp 的紙上超額，
   在「校準過的零股 spread（微型股層 0.99%/邊 = 來回 ~2.1%）× 月頻全換倉」下不但歸零、
   還深度轉負——比整股 33 元上限口徑（personal-baseline 臂 1 Sharpe 0.537）更差。
   解除價格上限所引入的微型股，其成交成本吃掉的比其 alpha 貢獻的更多。
2. personal-baseline 裁決留下的「零股翻案管道」**已關閉**：整股買不起、零股付不起。
   **個人 100 萬資金在任何已驗口徑（機構 0.5 億門檻 / 個人整股 / 個人零股）下，
   現行 A 線模型皆不值得投入真金。**
3. 後續若再驗個人口徑，必須先有新的 alpha 來源（如 PEAD 事件臂），
   不是換執行通路——執行通路的自由度已窮盡。

