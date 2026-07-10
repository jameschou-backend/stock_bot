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
