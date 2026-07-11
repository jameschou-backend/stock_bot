# 預登記：月營收 PEAD 事件臂（新 alpha 來源，非換執行通路）

> **預登記時間戳：2026-07-11 10:33 CST（Asia/Taipei）**
> 本文件在跑任何回測臂**之前**寫完並固定。判準、訊號定義、時序、成本模型、裁決規則
> 以本文件為準；跑完後不得回頭修改判準（結果段落於跑完後追加，判準段落凍結）。
>
> 引擎：`scripts/backtest_pead.py`（獨立事件錨定回測，**不碰 skills/backtest.py**）。

---

## 0. 為什麼是這個臂

ML A 線三口徑全 **FAIL**（機構 0.5 億門檻 v2.2 超額 -41pp／個人整股 personal-baseline 悲觀
0.305／個人零股 odd-lot 悲觀 0.083）。三次裁決的共同終局結論：**「執行通路的自由度已窮盡——
再驗個人口徑必須有新的 alpha 來源，不是換通路」**。

月營收 PEAD（post-earnings-announcement drift 的營收版）是**事件驅動**、與現行 ML 動能/籌碼
截然不同的訊號家族，是文獻上台股最強異常之一（微型股、零分析師覆蓋段 drift 最強）。
本臂回答唯一問題：**截面營收驚喜（YoY）在營收公告後 20 交易日是否存在可辨識的 drift，
且該 drift 在個人可執行口徑下是否值得實盤。**

## 1. 訊號定義（ex-ante 固定）

- **主訊號（唯一進裁決）= `revenue_yoy`**：`revenue_current_month(M) / revenue_current_month(M−12mo) − 1`
  （同月比去年，去季節性）。
  - **重要資料事實**：DB `raw_fundamentals.revenue_yoy` 欄位**全表 NULL**
    （FinMind `TaiwanStockMonthRevenue` 未提供 last-year 欄位，ingest denom 為 NaN）。
    故本引擎**自 `revenue_current_month` 時間序列自算 YoY**——以「同一 stock_id、
    trading_date 減 1 年」的同月營收為分母。這是 YoY 的定義本身，非替代品。
  - 需要同月去年營收存在且 ≠ 0；否則該 stock-月無訊號（不進 cohort）。
    全表 224,749 stock-月中 **198,917（88%）** 有同月去年匹配。
- **次要訊號（僅 robustness，不進裁決）= `revenue_yoy_accel`**：`yoy(M) − yoy(M−1mo)`
  （本期 YoY 減前一自然月 YoY，營收動能加速度）。只在結果段附穩健性對照，不改變任何裁決。

## 2. 時序（deadline 口徑 = 誠實下界，ex-ante 明文）

台股法規：每月營收須於**次月 10 日前**公告（`TWSE 上市公司資訊申報作業辦法`）。

- 營收月 M（DB 存於 M 月 1 日）→ **法定申報截止 deadline = (M+1) 月 10 日**。
- **進場日 = 「≥ deadline 的第一個交易日」再 +1 個交易日（deadline+1，實盤最早可執行）。**
  - 機制：`entry_idx = searchsorted(trading_days, deadline, "left") + 1`；
    `entry_date = trading_days[entry_idx]`。交易日序列 = `raw_prices` 全市場 distinct 交易日
    （與 skills/backtest.py 的交易日 searchsorted 同源）。
- **出場日 = 進場日 + 20 個交易日**（= 現有 label horizon，無 4:1 mismatch）。

> ⚠️ **這個時序系統性截掉「早公告」（多為好消息）的前段 drift**：實務上約半數公司在
> 月初 1~8 日就公告，本引擎一律等到次月 ~11 日才進場 → 對早公告股，前 6~10 交易日的
> drift 完全吃不到。**因此本臂量到的效果量是 drift 的下界（lower bound）**，不是全貌。
> 若下界都顯著，真實效果只會更強；若下界不顯著，仍不能排除早段 drift 存在——
> 但那需要逐股公告日資料（未建，見 §6）。此為 ex-ante 承認的方向性偏差（偏保守）。

## 3. 持有期報酬與 P&L 口徑（ex-ante 固定）

- 每檔持有期報酬 = `adj_close(exit) / adj_close(entry) − 1`，**單筆 clip −50%**
  （`max(ret, −0.50)`，退市/地雷股保護，與生產同）。
- **adj_close = raw close × 官方 adj_factor**（DB `price_adjust_factors`），per-stock
  ffill → bfill → 1.0 缺日語義（`skills.build_features.apply_adj_factors` 同源，含息）。
  整檔無 factor（無 adj 下市股）→ factor=1.0 未還原。
- **同一 cohort（營收月 M）所有股票共享同一 deadline → 同一 entry/exit 日**
  （時序只由營收月決定，與個股無關）→ cohort 內橫截面在同區間比較，乾淨。

## 4. 無 lookahead 鐵律（invariant，測試鎖定）

1. 營收月 M 的訊號在 **deadline（(M+1)-10）之前不可用**——進場日嚴格 ≥ deadline 之後
   第一個交易日再 +1，故 `entry_date > deadline`（不早於申報截止）。
2. 進場價取 entry_date（deadline+1）當日 adj_close，**不早於 deadline**。
3. `tests/test_backtest_pead.py` 鎖定：deadline 公式、`entry_date > deadline`、
   `exit_idx = entry_idx + 20`、YoY 同月去年匹配、成本套用方向。
4. **revision 偏差（已知輕微 caveat）**：FinMind 提供的是**最終修正值**（非首刊值）。
   月營收極少事後大幅修正（不同於財報），量級偏差輕微、方向不定；ex-ante 記為已知限制，
   無法在現有資料修正。

## 5. 兩臂與判準（判準先寫死；Arm A FAIL → 不建完整引擎、停在 A）

### Arm A — 訊號存在性（gate，gross、無成本、無門檻 universe）

- **universe**：四碼普通股（`^\d{4}$`）、排除 EMERGING、進出場價齊備（基本 tradability）。
  **無流動性門檻**（含微型股，drift 理論最強段）。
- **選股**：每 cohort 依 `revenue_yoy` 降冪取 **top 30 等權**。
  cohort 報酬 = top-30 持有期報酬均值。
- **benchmark**：同 cohort **等權零成本 universe**（該 cohort 全 universe 股票持有期報酬均值，
  同 entry/exit 日、零成本）——v2.1 起零成本 benchmark 口徑。
- **long-short 診斷**（附註，不進裁決）：每 cohort top decile 均值 − bottom decile 均值 →
  月序列 Sharpe；rank IC = 每 cohort Spearman(`revenue_yoy`, 持有報酬) 的均值、ICIR = 均值/標準差。
- **判準（單一，先寫死）**：

  | 裁決 | 條件 |
  |------|------|
  | **PASS**（訊號存在，續建 Arm B） | 超額（strategy − benchmark）Sharpe 的 paired block-bootstrap **95% CI 下界 > 0** |
  | **FAIL**（PEAD 便宜死，停在 A） | 超額 Sharpe 95% CI 下界 ≤ 0 |

  - 超額 Sharpe = information-ratio 口徑（strategy 月報酬 − benchmark 月報酬，不扣 rf），
    與 `skills.statistics.paired_block_bootstrap_sharpe_ci` 的 `excess_*` 同口徑。
  - bootstrap：block=6、n=1000、seed=42、periods_per_year=12（與 run_backtest 同常數）。

### Arm B — 可執行性（僅 Arm A PASS 才跑）

- **口徑（個人，機械推導、不掃描）**：與 personal-baseline / odd-lot 同——
  資金 100 萬 / 30 檔 / 每檔 33,333 元；**min_avg_turnover = 0.0033 億（≈333,333 元）**；
  **無價格上限**（零股可買任何價位）。
- **執行成本（每檔來回，套在持有報酬上）**：
  `net = gross − 0.00585（整股稅費來回）− odd_lot_round_trip_cost(amt_20, entry, premium_mult=1.5)`
  （`skills.odd_lot_costs`，**悲觀 premium ×1.5**；微型股層來回 spread ≈ 3.09%）。
  benchmark 仍為零成本等權 universe（同 0.0033 門檻）——超額判讀偏嚴，與悲觀原則一致。
- **判準（單一，先寫死，取悲觀臂 summary）**：

  | 裁決 | 條件 |
  |------|------|
  | **PASS**（值得個人實盤） | 悲觀 Sharpe ≥ 0.70 **且** MDD ≥ −50% |
  | **GRAY**（paper 半年再議） | 悲觀 Sharpe ∈ [0.50, 0.70) **且** MDD ≥ −50% |
  | **FAIL** | 悲觀 Sharpe < 0.50 **或** MDD < −50% |

## 6. 與 A 線相關性檢查（獨立來源驗證，附註）

- PEAD cohort 月報酬（Arm A 主臂，按 entry 年-月）vs **生產 ML 主臂月報酬**
  （`artifacts/backtest/backtest_20260710_190850.json`，v2.2 機構口徑 Sharpe 0.520，119 期，
  按 rebalance 年-月）→ 年-月對齊、重疊期 Pearson 相關。
- **判讀**：營收 YoY 特徵（`fund_revenue_yoy_accel` 等）已在 ML 模型特徵集內，
  須確認 PEAD 是**新來源**而非重複已被 ML 吃掉的訊號。**相關性 < 0.5 才算獨立來源**。
  （此為附註判讀，不改變 Arm A/B 的機械裁決。）

## 7. 統計紀律與 n_trials（附上，Arm A 判準為 CI 下界，DSR 僅附）

- 每臂附：paired block-bootstrap Sharpe 95% CI（含 excess）+ DSR p-value。
- **DSR n_trials = trial_registry 行數（跑該臂時）+ HISTORICAL_TRIALS_BASE(80)**。
  registry 現 **11 行** → Arm A 跑完記入為第 12 行，n_trials = **92**；
  若 Arm B 跑，記入第 13 行，n_trials = **93**。
- **樣本為月頻事件（~120 個 cohort）**；deadline 口徑截掉早段 drift → 效果量為下界。
  這兩點使本臂即便 PASS 也偏保守，即便 FAIL 也不能完全排除 PEAD（只排除「deadline+1 進場
  的可執行 PEAD」）。

## 8. 已知限制與殘餘偏差（ex-ante 承認，不作事後開脫）

1. **deadline 口徑截早段 drift** → 效果量下界（§2）。
2. **revision 偏差**：FinMind 為最終修正值，非首刊（§4.4）。
3. **微型股 segment 資料品質最差**（與 personal-baseline / odd-lot 同）：官方 factor 覆蓋
   ~1,997 檔，其餘（多下市股）factor=1.0 未還原；處置股/漲跌停不可成交未建模；
   零股簿深不足、漲跌停日成交概率低——未建模（Arm B odd-lot 成本 P75+×1.5 為部分緩衝）。
4. **benchmark 零成本**（不可實際達成）→ 超額判讀偏嚴（方向與悲觀一致）。
5. **YoY 自算需同月去年營收**：新上市股前 12 個月無訊號（88% 可算，survivorship 中性——
   缺的是每檔自身歷史起點，非選擇性刪除）。
6. **cohort 內 top-30 對微型股極端 YoY 敏感**：極小基期股（去年營收接近 0）YoY 可爆量。
   引擎以 `revenue_last_year ≠ 0` 過濾，並在 Arm A 附 winsorize-free 的 raw 結果；
   若擔心離群，long-short decile 診斷（rank-based）不受極值影響，作為交叉驗證。

## 9. 執行紀律

- Arm A 先跑；**FAIL 就停在 A**（記錄裁決，不跑 Arm B）；PASS 才跑 Arm B。
- 結果 JSON：`artifacts/backtest/pead_20260711_armA.json`（及 `_armB` 若跑）；寫 trial registry。
- 跑完後：結果與裁決追加至本文件「結果」段（判準段落不動），更新 memory，`make test`（基線 808 passed）。

---

## 10. 結果（2026-07-11 10:40 CST 追加；判準段落未動）

Arm A 跑完（`artifacts/backtest/pead_20260711_armA.json`，trial registry 第 12 筆，n_trials=92）。
執行紀律：`pnl_convention == "adj_from_db_official"`；訊號 198,917 stock-月（113 營收月）；
universe 排除 941 檔（EMERGING+非普通股）；價格 4,680,298 列 / 2,281 交易日；
110 個有效 cohort，窗口 **2017-04-11 ~ 2026-06-11**（top-30 需 ≥30 檔且進出場價齊備）。

### Arm A 主臂（top-30 by revenue_yoy 等權，gross，無門檻 universe）

| 指標 | 值 |
|------|----:|
| 策略累積 | **+38.97%** |
| 大盤（等權零成本 universe，中位 1,750 檔/cohort） | **+202.04%** |
| **超額** | **−163.06%** |
| Sharpe | 0.2018 |
| **超額 Sharpe** | **−0.5407** |
| MDD | −36.16% |
| 年化 | +3.66% |
| 勝率 | 50.0% |

**統計紀律（bootstrap 95% CI + DSR，如實附上）**：

| 量 | 值 |
|----|----|
| Sharpe 95% CI | [−0.545, 0.865] |
| **超額 Sharpe 95% CI**（判準看下界） | **[−1.6188, 0.2889]** → 下界 **−1.6188 ≤ 0** |
| DSR p-value（n_trials=92） | 0.0006（**不顯著**，遠 < 0.95） |

**long-short 診斷（附註，rank-based，不進裁決）**：

| 量 | 值 |
|----|----|
| long-short（top decile − bottom decile）Sharpe | **+1.8056** |
| long-short 累積 | +371.1% |
| rank IC 均值 / ICIR | **+0.0226 / +0.329** |

**與 A 線 ML 主臂相關性**：Pearson **r = 0.3596**（n=110 重疊月）→ **|r| < 0.5 = 獨立來源**
（PEAD 確為與現行 ML 動能/籌碼不同的回報流）。

### 逐年超額（策略 / 大盤）

7/10 年超額為負（−13~−24pp），僅 2018/2023/2024 正——**廣泛且持續的落後**，非單年事件。

### 機制診斷（§8.6 ex-ante 已預期，實證確認）

抽 2023-06 cohort（n=1,789）：**top-30 by 絕對 yoy 的 yoy 中位數 = +1,168%、最大 +71,252%、
16/30 為 <0.5 億微型股**——選到的是「去年基期接近 0 → yoy 爆量」的極端微型股（pump/雞蛋水餃股），
持有 20 日大幅 mean-revert。相對地，top decile（178 檔）yoy 中位僅 +108%、ranks 31-175 中位 +92%
（健康成長）。**極端右尾有毒、廣義高成長段有效**——這正是 top-30 深度落後、
而 rank-based long-short（Sharpe 1.81、IC +0.0226）為正的矛盾解。

## 11. 裁決（按 §5 判準機械適用；判準段落未動）

**Arm A：FAIL** — 超額 Sharpe 的 paired block-bootstrap 95% CI 下界 = **−1.6188 ≤ 0**
（§5 PASS 條件：下界 > 0；未達）。**停在 Arm A，不跑 Arm B、不建完整可執行引擎**（§9 執行紀律）。

### 關鍵推論（誠實、含 nuance）

1. **裁決按預登記機械適用 = FAIL**：預登記的執行規則（raw top-30 by 絕對 yoy、無訊號轉換）
   在無門檻 universe 上**深度輸給等權大盤**（−163pp），超額 Sharpe CI 下界顯著為負。
   判準在看數字前已凍結，不得事後改用 long-short Sharpe 當 gate（會移動球門）。
2. **但這不是「PEAD 訊號不存在」**：rank-based long-short decile Sharpe **1.81**、
   rank IC **+0.0226 / ICIR 0.329** 顯示營收 YoY 在**排序意義上**確實預測 forward 20 日報酬
   （高成長段勝低成長段）。失敗的是**「raw 絕對 yoy top-N 選股」這個 execution**，
   不是訊號家族本身——極端 yoy（tiny-base 微型股）有毒，naive top-N 系統性選中毒尾。
3. **後續若續驗 PEAD，須新預登記**（不是換通路、也不是本臂的延伸）：候選轉換 =
   截面 rank / winsorize yoy、剔除去年基期過小（revenue_last_year 下限）、
   或改用 decile-neutral 的中段高成長組；並須併入執行成本（Arm B 口徑）後才有意義。
   **本次不做**（會是看到數字後的新自由度，須另立判準）。
4. **相關性 0.36 < 0.5**：PEAD 是獨立回報流——若未來 rank 版 Arm A' 通過，它與現行 ML
   有分散價值。但那是**未來、須先過新預登記的 gate** 才成立的假設。

**與三次 A 線 FAIL 的關係**：本臂是「新 alpha 來源」路線的第一次嘗試。裁決 FAIL 於預登記的
naive execution，但留下一條**明確、受紀律約束**的後續：rank-transformed PEAD（新預登記）。
執行通路仍窮盡（本臂無關通路，是訊號/選股法問題）。
