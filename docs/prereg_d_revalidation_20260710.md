# 預登記：Strategy D 修復後 10y 重驗（2026-07-10）

> **本文件必須在重驗結果產生之前 commit**（以 git history 為證）。
> 依據：2026-07-10 系統總體檢報告缺陷 1——D 是每日實際推播的策略，
> 其唯一績效依據（+976,990% / Sharpe 1.925，2026-04-16）產生於損壞特徵
> （OHLC 混用）+ rotation buffer 日曆天洩漏 + raw close label 之上，
> 三者已於 2026-07-03 修復（commit b5ed7f1 / c475b78），修復後從未重驗。

## 重驗配置（口徑 = 實盤 daily_run.sh 推播的東西）

```bash
DATA_STORE_FREEZE=1 python scripts/backtest_rotation.py --months 120 \
  --train-label-horizon 5 --max-positions 4 --rank-threshold 0.20 \
  --trailing-stop -0.25 --min-avg-turnover 1.0 --max-price 250 \
  --excess-label --tiered-slippage --transaction-cost 0.00585 \
  --output artifacts/backtest/d_revalidation_20260710.json
```

- 對齊 `strategy_d_pick.py` 生產常數：label-5、pos=4、rank 0.20、trailing -25%、
  1 億流動性門檻、excess-label；`--max-price 250` 對齊 daily_run.sh（本次新增
  回測支援，排名用全宇宙、過濾只套進場候選）。
- 資料：凍結快照（DATA_STORE_FREEZE=1），P&L/label 均為還原價口徑（2026-07-03 修復後）。
- 成本：來回 0.585% + 分級滑價（最保守口徑）。

## 判準（單一主判準，先寫後跑）

主判準 = **10 年淨 Sharpe**（上述成本口徑）。輔助約束 = MDD。

| 結果 | 裁決 | 後續動作 |
|------|------|----------|
| Sharpe ≥ 0.85 且 MDD ≥ -50% | **PASS** | D 保留 live（真金仍受階段一 ≤20-30 萬上限）；可進入優化，但調參只用 2016-2022、2023-2026 每候選讀一次 |
| 0.5 ≤ Sharpe < 0.85 | **GRAY** | D 降 paper-only，持續每日對帳 6 個月再議；**灰區禁止調參**（在雜訊裡優化=meta-overfitting） |
| Sharpe < 0.5 或 MDD < -50% | **FAIL** | D 退役；實盤訊號改跟 A 線（等 personal-baseline 出來後換）|

門檻理由：0.85 = 基準 v2 歸因臂（無門檻 A 線）的 Sharpe——D 日頻換股 + 高成本 +
每日人工注意力，必須明確優於「同樣不受 0.5 億門檻限制的月頻 A 線」才有存在價值；
0.5 = 低於它時連「比大盤好」都不穩固，不值得日頻維運。

## 誠實註記（先承認再跑）

1. **in-sample 上界**：trailing -25%、rank 0.20、pos=4、label-5 是在髒資料上
   經大量搜尋選出的參數；即使資料修好，本次重驗對「參數選擇」仍是 in-sample，
   結果應視為上界。歷史 C/D 線搜尋次數以 n_trials≈80 計入 DSR 折扣
   （沿用 scripts/honest_baseline.py 慣例）。
2. **讀取預算**：本次讀取 2023-2026 段，計入 trial registry（memory 記錄）。
3. **中途不改判準**：跑完後無論數字多誘人/多難看，按上表裁決。任何
   「再跑一個變體看看」都是新實驗，需另行預登記。
4. 預期管理：A 線同型修正的先例是 +10004% → +205%（去洩漏）、0.99 → 0.65
   （誠實化）——D 從 1.925 大幅回落是預期內，不構成「修壞了」的證據。

---

## 修正案 1（2026-07-10，臂 1 結果已出但裁決前）：signal-lag 誠實臂

**臂 1 結果**（上述指令，`d_revalidation_20260710.json`）：Sharpe 1.823 /
+594,556% / MDD -42.7%——表面落在 PASS 區，但**發現預登記設計疏失**：
rotation 引擎不支援 entry delay，臂 1 每筆交易「用 T 日特徵（含 T 日 16-17 點
才公佈的法人資料）在 T 日收盤進場」。此口徑已於同日總體檢報告（缺陷 4，
commit 早於本重驗）**無條件裁定為不誠實**（delay 口徑是先驗論證，不掛效果量）；
D 平均持倉僅 5.5 天，隔夜跳空 lookahead 占比遠大於月頻 A 線——臂 1 的 1.82
與髒資料時代 1.925 幾乎無差，與 A 線修復經驗（0.99→0.65）矛盾，即為此故。

**處置（先驗裁定的機械適用，非對好數字的反應）**：
- rotation 新增 `--signal-lag 1`（T-1 特徵決策、T 日收盤執行 = 實盤時序：
  T-1 晚間 18:00 pipeline 產訊號 → T 日下單）。
- **D 的最終裁決以臂 2（signal-lag 1）為準**，判準表不變（PASS ≥0.85 / GRAY / FAIL）。
- 臂 1 保留為「lookahead 溢價」的歸因記錄（臂1 − 臂2 = 隔夜資訊優勢的價值）。

臂 2 指令（其餘與臂 1 完全相同）：

```bash
DATA_STORE_FREEZE=1 python scripts/backtest_rotation.py --months 120 \
  --train-label-horizon 5 --max-positions 4 --rank-threshold 0.20 \
  --trailing-stop -0.25 --min-avg-turnover 1.0 --max-price 250 \
  --excess-label --tiered-slippage --transaction-cost 0.00585 \
  --signal-lag 1 \
  --output artifacts/backtest/d_revalidation_lag1_20260710.json
```

---

## 裁決（2026-07-10，臂 2 完成後按判準表機械適用）

**臂 2（誠實時序）**：Sharpe **0.952** / 累積 **+4,800%** / MDD **-60.89%** /
年化 +50.5% / 勝率 47.1% / 1,531 筆 / 平均持倉 5.4 天 / 年化成本拖累 94%
（2016-08-01 ~ 2026-06-10，`d_revalidation_lag1_20260710.json`）。

**判定：FAIL**——Sharpe 0.952 ≥ 0.85 通過主判準，但 **MDD -60.89% 觸發
FAIL 條件（MDD < -50%）**。依預登記處置：**D 退役（真金），訊號降級紙上追蹤**；
實盤配置回到 A 線／等 personal-baseline。逐年亮點：2018 **-36.9%**、2019 -2.6%
——報酬幾乎全部由 2020(+303%)/2021(+198%) 兩年貢獻，行為上不可持有。

**副產品（本次最重要的方法論發現）**：臂 1 − 臂 2 = 「T 日盤後籌碼 lookahead」
的價值 = Sharpe +0.87、累積 123 倍、MDD 假象改善 18pp。**歷史 C/D 線全系列
數字（C2 1.62、Label-10 1.85/2.31、excess+pos4 1.48、D 1.925…）全部含此偏差，
全部退役**——它們共享 lag=0 時序。此發現同時解釋了「D 紙上數字永遠追不上
實盤體感」的長期疑問。

**讀取記錄**：本重驗兩臂各讀取 2023-2026 一次，計入 trial registry。
後續任何 D 變體實驗（如改 trailing、改 pos、lag=1 下重調參）屬**新策略研發**，
需另行預登記且尊重「同時最多一條研究臂」（現排 PEAD 之後）。
