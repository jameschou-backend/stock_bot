# 預登記：月營收 PEAD rank/winsorize 訊號臂（可執行性裁決）

> **預登記時間戳：2026-07-11 13:40 CST（Asia/Taipei）**
> 本文件在跑任何回測臂**之前**寫完並固定。判準、訊號 transform 定義、時序、成本模型、
> 裁決規則以本文件為準；跑完後**不得回頭修改判準**（結果段落於跑完後追加，判準段落凍結）。
> 檔案由 AI 助手撰寫、由使用者提交 git（時間戳與 diff 由 git 見證，事後無法竄改判準）。
>
> 引擎：`scripts/backtest_pead.py`（既有事件錨定回測引擎，本輪**擴充不重寫**，
> 新增 signal-transform 與 cost-mode 兩軸；不碰 skills/backtest.py）。
> 前一輪預登記：`docs/prereg_pead_arm_20260711.md`（Arm A naive top-N → FAIL）。

---

## 0. 為什麼是這個臂（訊號存在性已答，本輪只問可執行性）

前一輪（`docs/prereg_pead_arm_20260711.md`）預登記的 **naive top-30 by 絕對 revenue_yoy**
在無門檻 universe 上 **FAIL**（超額 Sharpe 95% CI 下界 −1.62）。**但同批預登記的診斷已確定
兩件事，不需再問**：

1. **訊號在 rank 空間存在**：long-short（top decile − bottom decile）Sharpe **1.81**、
   rank IC **+0.0226 / ICIR 0.329**——營收 YoY 在**排序意義上**確實預測 forward 20 日報酬。
2. **失敗機制已定位**：絕對 top-30 系統性選中「去年基期 ≈ 0 → yoy 爆量（+1,168% 中位、
   最大 +71,252%）」的極端微型股（pump/雞蛋水餃股），20 日大幅 mean-revert。
   廣義高成長段（top decile yoy 中位 +108%）才健康。

因此本輪**不再驗訊號是否存在**（已知存在），也**不作 Arm A gate**。本輪回答唯一問題：

> **針對已定位的失敗機制做一個受紀律約束的訊號 transform（排除基期效應假 YoY），
> 在個人零股可執行口徑（含悲觀零股價差成本）下，這個 PEAD 策略是否值得投真金？**

Arm B（個人零股可執行口徑）是**唯一裁決臂**。long-only（微型股不能放空）。

---

## 1. 訊號 transform（ex-ante 凍結，先寫死機制、後跑）

主訊號 = `revenue_yoy`（= `revenue_current_month(M) / revenue_current_month(M−12mo) − 1`，
自算同月去年，去季節性；DB `raw_fundamentals.revenue_yoy` 全表 NULL，引擎自算，見前一輪 §1）。

transform pipeline（三步，**參數在此凍結**）：

### (a) 基期效應剔除（**本 transform 的操作性槓桿**）

**剔除規則**：只保留同月去年營收 `revenue_last_year ≥ NT$10,000,000`（1,000 萬元）的 observation。

- `revenue_last_year` = 自算的同月去年 `revenue_current_month`（YoY 的分母本身，
  引擎 `compute_yoy_columns` 產出的 `rev_prior_year` 欄，單位 = 元；
  已實證確認 DB `revenue_current_month` 單位為 NTD 元，例：2330 2026-06 = 416,975,163,000）。
- **雙重正當性（ex-ante）**：
  1. **直接打擊已定位的失敗機制**：基期 ≈ 0 的極端微型股（去年月營收數十萬～數百萬元）
     產生的爆量 YoY（+1,000%～+71,000%）是雜訊不是訊號；1,000 萬元/月（≈1.2 億元/年）
     的下限剔除這些退化基期，同時**保留絕大多數真實微型股**（PEAD 理論最強段）。
  2. **同時清掉負營收金控雜訊**：DB 內金控/金融股月營收有負值
     （例：2888 2025-07 = −44,427,526,000；FinMind 對金控報「淨」營收含投資損失），
     負分母使 YoY 符號無意義。正值下限 10,000,000 一併剔除（`rev_prior_year < floor`
     或 NaN 或負值皆剔），使 YoY 只在「同月去年為實質正營運」時成立。
- **floor 值來源（防 p-hacking 聲明）**：10,000,000 元 = 前一輪預登記 §8.6 與本輪任務
  明文列舉的候選值（「絕對值如 1000 萬元」），**非看數字後挑選、非掃描最佳化**。
  只此一個值進裁決；不試多個 floor 取最好。

### (b) winsorize（防禦性 hygiene，**對 rank 選股不改變成員**）

對 (a) 剔除後、各 cohort 的存活 `revenue_yoy` 截面，clip 到 **[P1, P99]**
（該 cohort 存活集合的第 1 / 99 百分位），產生 `revenue_yoy_wins`。

> **誠實聲明（不隱藏自由度）**：winsorize 是**單調轉換**，對「top-30 by rank」的
> **選股成員無影響**（存活集合內排序不變 → top-30 = winsorize 前的 top-30）。
> 保留此步僅為 (i) 標準防禦性 hygiene、(ii) 讓輸出的 YoY 分佈統計不被離群主導、
> (iii) 若未來改用 value-based 加權已就位。**本 transform 改變選股的操作性槓桿是 (a) 的
> 基期 floor**，winsorize 不是。此聲明使本臂不落入「看數字後移動球門」的批評。

### (c) 選股（rank，非絕對值 top-N）

對 (a)(b) 後的存活 pool，依 `revenue_yoy` **降冪排名（rank）**取 **top 30 等權**。
（rank 選股 = 按存活集合內名次選，非按絕對 yoy 值門檻選；因 (a)(b) 皆單調，
等價於按 (a) 剔除後的原始 yoy 降冪 top-30。）

### sanity gate（健全性檢查，**非裁決判準**）

transform 後，在**實際可選 pool（(a) 基期 floor + Arm B 流動性門檻後）**上計算
long-short decile Spearman rank IC，**須仍 > 0**——確認 transform 沒把訊號洗掉。
IC ≤ 0 = 紅旗（transform 破壞訊號），但**裁決仍以 Arm B 的 Sharpe/MDD 為準**（見 §3）。

### 時序 / P&L（與前一輪相同，不變）

- 時序 = **deadline+1**（營收月 M → 法定申報截止 (M+1) 月 10 日 → 進場 = ≥deadline
  第一個交易日再 +1 交易日 → 出場 +20 交易日）。系統性截早段 drift = **效果量下界**。
- P&L = `adj_close(exit)/adj_close(entry) − 1`，官方 adj_factor（DB `price_adjust_factors`，
  `apply_adj_factors` 同源含息），**單筆 clip −50%**。同 cohort 全股共享 entry/exit 日。

---

## 2. universe / benchmark（ex-ante 固定）

- **universe**：四碼普通股（`^\d{4}$`）、排除 EMERGING 興櫃 + 非普通股（etf/warrant/...）、
  進出場價齊備（tradability）。survivorship 中性（point-in-time 由價格齊備決定）。
- **benchmark（唯一，兩臂共用）= 等權零成本 universe**：本 cohort **全 universe**
  （價格齊備、未套 (a) floor、未套流動性門檻）持有期報酬均值，同 entry/exit 日、零成本。
  - 與前一輪、odd-lot、personal-baseline 同口徑（vs 廣義等權大盤）。
  - **超額判讀偏嚴**：benchmark 零成本不可實際達成，且不受 floor/門檻限制（是最寬的
    對照）→ 策略必須贏「免費持有整個等權市場」才算超額，方向與悲觀原則一致。
- 另附 **vs TAIEX TR**（發行量加權股價報酬指數，含息真實大盤，`skills/taiex_tr.py`
  快取 `artifacts/benchmark/taiex_tr.parquet`，`allow_fetch=False` 用既有快取，確定性可重現）。

---

## 3. Arm B — 可執行性（**唯一裁決臂**，long-only，判準先寫死）

- **口徑（個人，機械推導、不掃描）**：資金 100 萬 / 30 檔 / 每檔 33,333 元；
  `min_avg_turnover = 0.0033 億`（≈333,333 元）；**無價格上限**（零股可買任何價位）。
- **執行成本（每檔來回，套在持有報酬上）**：
  `net = gross − 0.00585（整股稅費來回）− odd_lot_round_trip_cost(amt_20, entry, premium_mult=1.5)`
  （`skills.odd_lot_costs`，**悲觀 premium ×1.5**；微型股層來回 spread ≈ 3.09%，
  實證校準 TWSE TWTC7U + TPEx oddQuote P75，見 `docs/prereg_odd_lot_arm_20260711.md`）。
- **判準（單一，先寫死；取悲觀臂 summary）**：

  | 裁決 | 條件 |
  |------|------|
  | **PASS**（值得零股實盤） | 悲觀 Sharpe **≥ 0.70** 且 MDD **≥ −50%** |
  | **GRAY**（paper 半年再議） | 悲觀 Sharpe ∈ **[0.50, 0.70)** 且 MDD ≥ −50% |
  | **FAIL** | 悲觀 Sharpe **< 0.50** 或 MDD **< −50%** |

  Sharpe = backtest summary 口徑（`(mean − rf_month)/std(ddof=0)×√12`，rf=1.5%/年）；
  MDD 由 cohort 月序列 equity 計。與 odd-lot / personal-baseline 臂完全同判準（可比）。

## 4. 參考臂（僅參考，**不進裁決**）

**同 transform（(a)(b)(c) 相同）、同 0.0033 流動性門檻，但成本改整股 tiered slippage
（無零股價差）**：`net = gross − 0.00585 − tiered_slippage(amt_20)`，
其中 `tiered_slippage`：amt_20 ≥ 5 億 → 0.2%、≥ 1 億 → 0.6%、< 1 億 → 1.0%（來回；
與 `skills/backtest.py` 生產 `--tiered-slippage` 分層值逐字相同）。

用途：量化「零股價差吃掉多少」= Arm B（零股 ×1.5）與參考臂（整股 tiered）的
Sharpe/累積差。**僅診斷、不改變 Arm B 裁決。**

## 5. 附帶統計與獨立性（不進裁決，如實附上）

- **paired block-bootstrap 超額 Sharpe 95% CI**（block=6、n=1000、seed=42、ppy=12，
  `skills.statistics.paired_block_bootstrap_sharpe_ci`）——附 Arm B 的超額（strategy − benchmark）
  Sharpe CI；**本輪 Arm B 裁決以 §3 的原始 Sharpe/MDD 為準**（不是 CI 下界），CI 為佐證。
- **DSR p-value**：`n_trials = trial_registry 行數（跑該臂時）+ HISTORICAL_TRIALS_BASE(80)`。
  registry 現 **12 行**；Arm B 跑完記入為第 13 行 → n_trials = **93**；
  參考臂記入第 14 行 → n_trials = **94**（本輪讀 2023-2026 診斷計入既有 registry 機制）。
- **與 A 線 ML 主臂相關性**：Arm B cohort 月報酬 vs 生產 ML 主臂月報酬
  （`artifacts/backtest/backtest_20260710_190850.json`，v2.2 機構口徑）按年-月對齊 Pearson。
  **|r| < 0.5 = 獨立回報流**（再確認 PEAD rank 版與現行 ML 動能/籌碼不同源；附註）。

## 6. 已知限制與殘餘偏差（ex-ante 承認，不作事後開脫）

1. **deadline 口徑截早段 drift** → 效果量為下界（前一輪 §2）。
2. **revision 偏差**：FinMind 為最終修正值非首刊（前一輪 §4.4）。
3. **微型股 segment 資料品質最差**：官方 factor 覆蓋 ~1,997 檔，其餘 factor=1.0 未還原；
   處置股/漲跌停不可成交、零股簿深不足、漲跌停日成交率低——未建模
   （Arm B odd-lot P75 + ×1.5 為部分緩衝）。
4. **benchmark 零成本 + 不受 floor/門檻限制** → 超額判讀偏嚴（與悲觀一致）。
5. **base floor 10M 是單一 ex-ante 值**：只此一值進裁決，不掃描 floor 取最佳；
   若 Arm B FAIL，不得事後調 floor 再跑當作翻案（那是新自由度，須另立預登記）。
6. **winsorize 對 rank 選股為單調 no-op**（§1b 已明文）——本臂改變選股的實質只有 (a)。

## 7. 執行紀律

- 本輪 Arm B 是**唯一裁決臂**，直接跑（訊號存在性前一輪已答，不再 gate）。
- 結果 JSON：
  - `artifacts/backtest/pead_rank_20260711_armB.json`（Arm B，裁決）
  - `artifacts/backtest/pead_rank_20260711_ref_tiered.json`（參考臂，不裁決）
  - （另附 `artifacts/backtest/pead_rank_20260711_gross.json`：transform 後 gross 無成本，
    給 sanity gate + 訊號存在確認的乾淨對照，不裁決）
- 每臂寫 trial registry；跑完 `make test`（基線 822 passed，含新增 transform 測試）。
- 跑完後結果與裁決追加至本文件「§8 結果」（判準段落 §1–§7 不動），更新 memory。

---

## 8. 結果（2026-07-11 13:53 CST 追加；§1–§7 判準段落未動）

執行紀律：三臂 `pnl_convention == "adj_from_db_official"`；訊號 198,917 stock-月（113 營收月）；
universe 排除 941 檔（EMERGING+非普通股）；價格 4,680,298 列 / 2,281 交易日；
110 個有效 cohort，窗口 **2017-04-11 ~ 2026-06-11**。三臂共享同一選股 pool（transform +
0.0033 門檻），只成本不同。trial registry：Arm B #13（n_trials=93）、參考臂 #14（94）、gross #15（95）。

### sanity gate（§1，非裁決）— **PASS**

transform 後 pool（基期 floor + 0.0033 門檻）rank IC = **+0.0238 > 0**、long-short Sharpe **1.7361**
（full-universe 對照 IC +0.0226 / LS Sharpe 1.8056，與前一輪一致）。
→ **transform 沒有洗掉訊號**；後續 FAIL 不是「訊號被 transform 破壞」造成，是純執行成本。

### 三臂解構（同 pool、同 transform，只成本不同）

| 臂 | 成本 | 累積 | 超額 vs 等權零成本 universe(+202.04%) | Sharpe | MDD | 超額 Sharpe | DSR p | vs TAIEX TR(+508.15%) |
|----|------|-----:|-----:|-----:|----:|-----:|-----:|-----:|
| **gross**（無成本，對照） | — | **+370.07%** | **+168.03pp** | **0.8318** | −32.12% | 0.5087 | 0.1067 | −138.08pp |
| **參考**（整股 tiered，不裁決） | 稅費+整股滑價 | −1.68% | −203.71pp | 0.0294 | −40.28% | −1.0001 | 0.0001 | −509.83pp |
| **Arm B**（零股 ×1.5，**裁決**） | 稅費+零股 spread ×1.5 | **−84.67%** | **−286.71pp** | **−0.8877** | **−84.77%** | −2.658 | 0.0 | −592.82pp |

- gross 超額 Sharpe 95% CI [−0.155, +1.189]；Arm B 超額 Sharpe 95% CI [−4.275, −1.554]。
- 與 A 線 ML 主臂相關性：Arm B Pearson r = **0.4608**（gross 0.4559）→ |r|<0.5，仍屬獨立來源
  （惟較前一輪 naive 0.36 上升——基期 floor 剔除微型 mean-revert 名單後，選股更靠近 ML 動能段）。

### 「零股價差吃掉多少」（參考臂 vs Arm B，§4 目的）

| 對照 | 累積差 | Sharpe 差 |
|------|-----:|-----:|
| gross → 整股 tiered（月頻微型股整股成本） | **−372pp**（+370% → −1.68%） | −0.80（0.83 → 0.03） |
| 整股 tiered → 零股 ×1.5（**零股價差淨增量**） | **−83pp**（−1.68% → −84.67%） | **−0.92**（0.03 → −0.89） |

零股價差（悲觀 ×1.5）在整股成本之上再吃掉 **~83pp 累積 / ~0.92 Sharpe**；
而整股成本本身已把 +370% gross edge 吃到 −1.7%。**月頻微型股輪動的成交成本，
無論整股或零股，都已超過 alpha。**

## 9. 裁決（按 §3 判準機械適用；判準段落未動）

**Arm B：FAIL** — 悲觀 Sharpe = **−0.8877 < 0.50** 且 MDD = **−84.77% < −50%**（§3 FAIL 雙重觸發）。
不值得零股實盤。

### 關鍵推論（誠實、含 nuance）

1. **transform 在 gross 層確實修好了 naive 的失敗**：預登記的基期 floor（+0.0033 門檻）把
   naive top-30 的「毒尾微型股」剔除後，gross 超額由前一輪 naive 的 **−163pp（Sharpe 0.20）**
   翻成 **+168pp（Sharpe 0.83）**，sanity IC +0.0238 > 0。**營收 YoY 排序訊號 + 受紀律約束的
   transform 有真實 gross edge**——這是整個 A 線/執行通路系列中，第一個「訊號存在 **且**
   有可辨識 gross 超額」的臂（前面都只到 rank IC/long-short，未見組合層 gross 超額）。
   ⚠️ 但 gross 的超額 Sharpe CI 下界 −0.155、DSR p=0.107（<0.95）——point estimate 正、
   統計上未過多重檢定門檻；且 vs 含息 TAIEX TR 仍 −138pp（等權策略 vs 市值加權含息大盤）。
2. **裁決臂 Arm B = FAIL，且是史上最深的一次**（Sharpe −0.89、MDD −85%）：純由**月頻
   微型股輪動的成交成本**造成，不是訊號問題（sanity gate PASS 已排除）。參考臂證明**連整股
   tiered 成本**都已把 +370% gross 吃到 −1.7%（Sharpe 0.03）；零股 spread ×1.5 再加碼到 −85%。
3. **與 odd-lot 臂終局一致、且更強**：前一輪 odd-lot（ML 模型口徑）悲觀 −1.3%/Sharpe 0.083；
   本臂即使換上一個**有真實 gross alpha 的新訊號源**，個人零股執行仍 FAIL——**「執行通路
   自由度窮盡」不只對現行 ML 成立，對一個 gross 層明確有超額的新 alpha 也成立**：
   微型股 alpha 的容量/成交成本結構，讓它在買得起的口徑（零股、月頻）下無法被捕獲。
4. **base floor 10M 是單一 ex-ante 值，不翻案**（§6.5）：本 FAIL 不得事後調 floor 再跑。
   若未來要救 PEAD 執行，方向不是調參，是**降低換手成本結構**（更長持有期、
   更大市值/更高流動性子集、或整股買得起的高價股子集）——那是**新策略、須另立預登記**，
   不是本臂延伸。

**與四次先前 A 線/執行 FAIL 的關係**：v2.2 機構口徑（−41pp）、personal 整股（0.305）、
odd-lot（0.083）、PEAD naive（−163pp）皆 FAIL。本臂是「新 alpha 來源 + 受紀律 transform」
路線的收尾：**gross 層第一次看到真實組合超額（+168pp/Sharpe 0.83），但可執行口徑
（零股 ×1.5）仍 FAIL**。結論定調：**訊號家族（營收 YoY PEAD）已被反覆證明存在且獨立，
但在個人可執行的微型股月頻口徑下，成交成本結構性地大於 alpha——這不是換通路或換訊號能解的，
是月頻微型股輪動的容量天花板。**

### 工件

- `artifacts/backtest/pead_rank_20260711_armB.json`（裁決）、
  `..._ref_tiered.json`（參考）、`..._gross.json`（gross 對照）。
- 引擎擴充：`scripts/backtest_pead.py`（signal-transform + cost-mode 兩軸，不重寫）。
- 測試：`tests/test_backtest_pead_rank.py`（9，鎖基期剔除/winsorize 單調/rank 不選毒尾/
  tiered 分層/cost_mode 單調遞減）。
