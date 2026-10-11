# 31 組進場條件的共同帳戶比較

開啟 `http://127.0.0.1:8000/terminal/#backtest`，最上方為「31 組進場條件，同一套資金規則」。這張表比較 29 個已實作進場規則與兩個情境條件，另列 0050 含息基準。

1. 先看完整期間已完成幾組。未完成組別有缺件原因，不顯示完整報酬或名次。
2. 預設依扣費累積報酬排列；可改看較小回撤或分年報酬。固定名次仍以全期間淨報酬為準。
3. 搜尋策略名稱，點「曲線與明細」，查看每日資產、交易、已結清批次、期末持股市值與應收權益。勝率只計已結清批次。
4. 展開「共同規格與實戰缺件」。歷史最高報酬不等於已取得實戰資格。

這張比較表的期間是 2024-01-02～2026-10-02，共 666 個市場交易日。本金每組 100 萬元、最多三個活躍股票名額，同日候選用訊號當日已知的 20 日均成交值排序。採 T+1 日區間成交代理、共同出場及成本；規格詳見 [事前固定規格](prereg_all_entry_accounts_20261011.md)。頁面下方「建立實驗」仍是單筆訊號研究，不能直接與這張資金帳戶表混比。

## 重現與資料補件

候選清單由 `skills/all_entry_accounts.py` 依掃描器產生並封存；v3 補齊原先未計算的同儕座標，詳見 [補件紀錄](all_entry_peer_completion_20261011.md)。載入時核對全部來源與候選指紋；結果不符合原來源時停止。一般規則保留每天成立的候選，兩個情境條件依原定義取母訊號首日。

```bash
python scripts/run_all_entry_batches.py \
  --candidate-bundle .cache/all-entry-accounts-20261011/inputs-v3 \
  --terms docs/all_entry_corporate_terms_v3_20261011.json \
  --settlement-completion docs/all_entry_6669_delivery_20261011.json \
  --output .cache/all-entry-accounts-20261011/new-offline-batch
```

輸出目錄必須是新的。本次使用者明確要求補資料，線上執行可加 `--fetch`；新請求使用持久的分類預算、共用 FinMind 限額與既有來源限制。命令不會下單。既有失敗請求、公司行動不明、來源不可取得都會保留原因，不能用零成交或另一套模型隱藏。

每組帳戶使用獨立程序，結束後釋放記憶體。`progress.json` 是可更新的操作進度；各組財務帳本、報告與指紋封存後不覆寫。若只有批次管理程序中斷，可加 `--resume` 核對並沿用已完整寫出的單組報告。未完整結束的目錄會明確拒絕續寫；必須另開目錄重播，不能拼接部分帳本。

加速版使用公司行動索引、等價前日收盤查詢、獨立稽核行情索引及每次仍驗證來源的零股解析快取。封存財務腳本不改寫；查詢加速須核對完整帳本一致。另有一項明確的零買單修正：當日舊持股賣出後，新訊號原本已因名額鎖定而拒絕的零股數買單仍記為拒絕，不得再次使用成交量。所有比較組別使用同一版修正。

4763、5871 與 2887 的普通股交付補件各使用三份 MOPS 原始公告；未確認的畸零股淨額與支付日期不列入可用現金。6669 配股於 2026-10-14 才能交易，帳戶終點 10 月 2 日仍列不可交易應收股票。詳見各公司補件紀錄與 [等價查詢驗證](all_entry_replay_acceleration_20261011.md)。

2352 減資另外核對發行公司議事錄、已執行年報及公告轉載，使用既有歷史收盤價結算器。原 8,281 股換為 6,790 股，退款 14,905 元；12 元畸零毛額保留為不可動用應收款。詳見 [佳世達補件](all_entry_2352_completion_20261011.md)。正式停牌與名額拒單的核對詳見 [補件稽核](all_entry_resource_audit_20261011.md)。只有受此事件影響的組別需要補件重播；相同候選與模型下，以明確較晚的嘗試替換，不能按報酬高低挑版本。

```bash
python scripts/research_all_entry_accounts_resource_audited.py \
  --arms bollinger_reclaim \
  --candidate-bundle .cache/all-entry-accounts-20261011/inputs-v3 \
  --terms docs/all_entry_corporate_terms_v3_20261011.json \
  --settlement-completion docs/all_entry_6669_delivery_20261011.json \
  --capital-completion docs/all_entry_2352_capital_20261011.json \
  --output .cache/all-entry-accounts-20261011/new-capital-replay

python scripts/combine_all_entry_accounts.py \
  --report .cache/all-entry-accounts-20261011/new-offline-batch/report.json \
  --report .cache/all-entry-accounts-20261011/new-capital-replay/report.json \
  --output .cache/all-entry-accounts-20261011/new-combined-report.json
```

主批次與補件重播共用互斥鎖，依序執行；不要同時啟動。離線缺少必要來源會停止，取得新來源需要本次已授權的 `--fetch`。權利證書的法律交付、普通股換發、可交易庫存及成交帶必須分別核對，不能只將交付日期延後來繞過問題。

6914／2024-06-06、8045／2025-02-06 與 8045／2025-03-10 的逐筆空值，另以完整官方通路分解補齊日區間代理所需資料，詳見 [官方日彙總補件第二版](all_entry_official_execution_v2_20261011.md)。第二版核對非空鉅額組合交易的全部成分，不能將非空母表直接當作該股票零成交。[第三版](all_entry_official_execution_v3_20261011.md) 再補齊 6914／2024-06-14 的必要成交座標。這不產生逐筆成交或 POC 分布。

聚陽及臺企銀的配股比例、普通股交付與畸零款處理，詳見 [配股補件](all_entry_distribution_completion_20261011.md)；青雲另由公司議事錄、年報及明確標示為第二手的掛牌公告交叉核對，詳見 [青雲交付補件](all_entry_5386_delivery_20261011.md)。臺企銀權利證書列為獨立的不可交易股票應收權益，普通股 9/10 交付後才可能賣出，證書交付日 9/2 不提早增加普通股庫存。儀表板明細會揭露這項限制與未確認淨額應收款的期末估值差異。

上海商銀另有獨立的 [證書單位核對](all_entry_certificate_completion_20261011.md)。[裕融證書](all_entry_9941_certificate_20261011.md) 在期末仍未換發普通股，保留獨立不可交易單位；[兆豐金與玉山金](all_entry_bank_distributions_20261011.md) 則核對普通股直接上市、最終配股率及員工酬勞股數。不得以未驗證的同代碼成交假設將證書提早當成普通股。最新完整補件入口每次執行一組策略，例如：

```bash
python scripts/research_all_entry_accounts_tcfhc_completion.py \
  --arms ma20_60_cross \
  --candidate-bundle .cache/all-entry-accounts-20261011/inputs-v3 \
  --terms docs/all_entry_corporate_terms_v3_20261011.json \
  --settlement-completion docs/all_entry_6669_delivery_20261011.json \
  --capital-completion docs/all_entry_2352_capital_20261011.json \
  --official-execution docs/all_entry_official_execution_v3_20261011.json \
  --distribution-completion docs/all_entry_distribution_completion_20261011.json \
  --albatron-completion docs/all_entry_5386_delivery_20261011.json \
  --certificate-completion docs/all_entry_certificate_completion_20261011.json \
  --late-certificate-completion docs/all_entry_9941_certificate_20261011.json \
  --bank-distributions docs/all_entry_bank_distributions_20261011.json \
  --issuer-distributions docs/all_entry_issuer_distributions_20261011.json \
  --northstar-completion docs/all_entry_8927_delivery_20261011.json \
  --context-distributions docs/all_entry_context_distributions_20261011.json \
  --poya-completion docs/all_entry_5904_certificate_20261012.json \
  --merger-halt docs/all_entry_2888_halt_20261012.json \
  --yageo-split docs/all_entry_2327_split_20261012.json \
  --tcfhc-completion docs/all_entry_5880_certificate_20261012.json \
  --output .cache/all-entry-accounts-20261011/new-completed-data-replay
```

華城與和潤企業的比例及交付補件見 [公司配股補件](all_entry_issuer_distributions_20261011.md)。華城官方議事錄只留存先前工具擷取文字，原始 PDF 未留存、尚未目視核對。[北基配股](all_entry_8927_delivery_20261011.md) 的議事錄與財報原檔已核對，交付及掛牌日仍依公告轉載交叉核對。這些限制會跟隨實際持有該事件的帳戶顯示，不能因回放完成而消失。

[情境帳戶補件](all_entry_context_distributions_20261011.md) 保留彰銀 2026 年證書與普通股的不同交付日，並補台新金 2024 年普通股交付。台新金議事錄為官方工具擷取文字，原始 PDF 未留存；這項限制同樣顯示在實際持有該事件的帳戶。

[休市順延補件](all_entry_rescheduled_distributions_20261011.md) 讓玉山金與華城的 2024 年原訂除權日取得同一份已驗證條款，再由既有交易日曆順延；只產生一筆權益。

[寶雅證書補件](all_entry_5904_certificate_20261012.md) 核對 2025 年配股的證書交付、普通股轉換及畸零款淨額；日期轉載來源限制另列。

[新光金停牌補件](all_entry_2888_halt_20261012.md) 依事前公告拒絕停牌期間的新買單，保持預留資金，不假造當日行情。

[國巨面額變更補件](all_entry_2327_split_20261012.md) 核對 2025 年 1 拆 4 及停牌期間，重用既有拆股結算，不把股價除以 4 誤認為損失。

[合庫金證書補件](all_entry_5880_certificate_20261012.md) 核對 2026 年證書及普通股日期，保留比例與畸零條款採公告轉載的來源限制。

每個新嘗試使用不同輸出目錄；用 `combine_all_entry_accounts.py --report` 依嘗試順序列入各組報告，完整覆蓋 31 個策略與基準後才發布。後半段尚未執行的組別直接使用最新補件，已完成且未受補件事件影響者保留原始帳本。補件組另核對事件前的共同帳本及新股實際交付紀錄。

所有組別都有結果紀錄後，以獨立的帳本核對工具產生發布資料：

```bash
python scripts/summarize_all_entry_accounts.py \
  --report .cache/all-entry-accounts-20261011/REPLAY_VERSION/report.json \
  --output artifacts/forward_simulation/all_entry_accounts_20261011.json
```

工具核對來源閉包、666 日 NAV、現金、市值、應收、費稅、分年報酬與已結清批次；也禁止未建模的新股權利證書混入普通股績效。發布紀錄及同名 `.sha256` 必須一起保留。每日曲線與完整財務帳本仍引用封存回測目錄，因此不要刪除對應 `.cache` 證據。

## 分享入口

```bash
make api
python scripts/serve_research_terminal_share.py --port 18125
```

分享代理只監聽本機，由既有公開 tunnel 指向此埠。僅允許介面、研究訊號、封存研究與帳戶明細的 GET／HEAD；不開放回測工作建立、成交帳本、私人持倉或寫入操作。此命令本身不建立 tunnel 或新增排程。
