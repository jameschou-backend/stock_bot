# 整股五檔補來源與執行audit：仍被配股資料阻擋

本輪補足3687漲跌停資料並加嚴執行audit，兩個新五檔帳戶仍未完成全期。**沒有新的整股五檔完整收益，也不能判斷它是否改善原策略**。原blocked版本與本輪結果均保留，未因結果不完整改參數或放寬公司行動證據。

依[事前規格](prereg_board_only_supplement_20260925.md)，固定458訊號、前20日成交金額排序、5檔、前日NAV分五份、開盤前現金／名額及未用預算／失敗名額鎖定、12%收盤停損、63日到期、閒錢現金。目標期間2022-01-03～2026-09-09，100萬元複利。執行仍只允許整張成交，配股殘股留帳、估值、占名額。

| 新五檔帳戶 | 最後完整日期 | 已完成市場日 | 阻擋原因 | 完整日前成交 |
|---|---|---:|---|---:|
| 正常 | 2025-09-24 | 905 | 2881於2025-09-25配股，缺`shares_per_share`、`pay_date` | 115筆整股、0零股 |
| combined | 2025-08-12 | 874 | 2880於2025-08-13配股，缺`shares_per_share`、`pay_date` | 89筆整股、0零股 |

正常案與v1完整partial account一致；combined與v1原先完成的255日每日帳戶逐欄一致。新增3687來源使combined越過先前2023-01-13的進度，隨後在另一配股事件停止。缺資料是blocked，沒有被當成普通未成交或填入推測交付日。部分期間不公布可被誤當全期的收益、年化、回撤或勝率。

四個原mixed控制帳戶全帳戶逐欄精確重現，兩個整股0050也完整重現v1；故基準收益未變。新策略原定主要基準仍為同policy整股0050，原mixed0050另行保留，沒有偷換基準。

| 帳戶 | 正常累積淨報酬 | 正常最大回撤 | combined累積淨報酬 | combined最大回撤 |
|---|---:|---:|---:|---:|
| 原mixed五檔控制 | 552.49% | -29.20% | 200.57% | -28.90% |
| 整股五檔 | blocked | — | blocked | — |
| 同policy整股0050 | 233.87% | -32.60% | 230.99% | -32.59% |
| 原mixed0050參考 | 242.34% | -33.89% | 239.83% | -33.90% |

正常整股0050期末3,338,745元、現金49,245元、年化29.37%、成本6,595元；combined期末3,309,851元、現金20,351元、年化29.13%、成本11,839元。各4筆整股買入、0零股成交。年度、成本與逐日帳戶見cases及[前版報告](research_board_only_20260925.md)，數值完全不變。

殘股會實際占用名額：正常案截至阻擋前有683個低於一張的持股日，最後4123有396股、2881有150股；combined有1,253個低於一張的持股日，最後4123有297股、2885有122股、2880有70股、2881有100股。這是逐股逐日加總，不是独立事件數。殘股維持估值與原退出指令，不假造清倉或現金；以0零股成交描述本政策，不代表盤中整股成交已得到驗證。

本輪來源準備共1次FinMind請求，取得3687的`TaiwanStockPriceLimit`並以原parser封存raw／normalized／index。另有10次官方直接讀取、8次websearch呼叫（16個query）與1次二手來源HTTP讀取。2881公司行動12個來源／嘗試紀錄逐檔hash核對，但沒有取得符合本輪要求的官方完整交付證據，因此接受的新增override為0；搜尋索引與二手轉載僅保留為研究證據，沒有注入帳戶。詳見[來源接納紀錄](board_only_corporate_additions_20260925.json)。績效執行及離線重播本身皆0網路、0 DB writes、0重訓。

新增子類修正v1兩個邊界：有價量及整張請求但缺當日限價立即blocked；負淨收入賣單補做結算後，最外order才按實際trade sequence記錄最終成交量。Audit精確檢查decision與每筆成交身份／數量／序號，並拒絕重用成交。15項合成測試通過，含配股殘股、資金／名額鎖定、缺單日來源、負淨收入結算及竄改decision檢出。完成帳戶與兩個blocked帳戶的完整日前綴均通過每日NAV、金流、股數、費用、容量、資源、名額與最終fill audit；失敗當日未完成記錄保留但不冒充已對帳日。

有效結果在`.cache/board-only-supplement-verified-r2-20260925`，包含八個case、`summary.json`、`identity.json`、`manifest.json`及`offline.json`。來源／程式inventory共24,094筆，輸入／結果manifest共3,848檔。主要manifest SHA256為`789b2e0e1fadf8b9dcaad92713e6a0ce8ba36a0d975bcc8054f94bff9cfa1500`，事前規格SHA256為`ddd57c3f342836e49e09bbd45f8cf876ff7f22ece3d57e6e6825b976f3694ae6`。首次執行62.023秒，完整離線重播65.274秒；八個case（含兩個blocked、partial ledger與audit）逐欄一致，四個父控制及兩個完整整股0050亦全部精確一致。

首次CLI嘗試使用相對`--output`，在只寫入第一個mixed控制帳戶後因`Path.relative_to`相對／絕對路徑不一致停止，沒有執行新策略。該嘗試保留於`.cache/board-only-supplement-verified-20260925/failure.json`；同程式與來源改以絕對輸出路徑執行r2，沒有增加新因子或改輸入。此wrapper的自訂`--output`／`--sources`須使用絕對路徑。

可重現命令：

```sh
python /Users/james.chou/JamesProject/stock_bot/scripts/research_board_only_supplement.py --output /Users/james.chou/JamesProject/stock_bot/.cache/board-only-supplement-verified-r2-20260925 --offline-replay
python -m pytest -q /Users/james.chou/JamesProject/stock_bot/tests/test_board_only_replay.py /Users/james.chou/JamesProject/stock_bot/tests/test_board_only_verified_replay.py
```

日級整股成交依然是假設，不等於盤中逐筆、排隊順序或實際下單證據；市場歷史身分、公告修訂與已見樣本限制仍在。`live_qualified=false`、`unseen_validation=false`，不更換正式策略或恢復排程。
