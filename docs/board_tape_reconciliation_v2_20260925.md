# TWSE 普通盤日總量補件：2026-09-25

本輪已執行公開來源批次準備，但官方同站安全拒絕使批次中止。**681 個 TWSE 必要股日仍未補齊；不可因成功收到部分日表就升級回測資格。** 舊 v1 報告與來源沒有改寫。

## 範圍、取得結果與剩餘工作

- 固定樣本共 917 股日，TWSE 686 股日、407 日期；TPEX 231 股日沿用 v1。
- TWSE 六種日表共 2,442 個來源 identity，先重用 403 份既有來源，最初需要 2,039 個新 base 請求，再加有交易的組合鉅額明細。
- 本輪 ledger 記 398 次 transport 呼叫：組合母表 394 次、明細 prototype 1 次、官方明細頁 3 次。367 次回應通過該階段回應檢查、29 次官方 307 拒絕、2 次 sandbox ConnectionError。
- 367 次回應包含 365 次組合母表、1 次錯帶 `selectType=M` 而返回母表的 prototype、1 次官方 HTML 文件。**這些不是 367 個已驗收股日。** 歷史 fetcher 曾使用 requests 預設 redirect；當時未記錄所有 redirect hops，因此 398 是 transport 呼叫數，不能宣稱精確 wire HTTP 次數。
- 組合母表目前涵蓋 366/407 日期；其中 113 日期存在 178 個需要明細的組合交易。尚未取得任何真正組合明細。
- base 尚缺 1,674 個 identity，其中 28 個已請求但未成功、1,646 個未派送；另有已知 178 個明細，**最少還缺 1,852 個取得請求**。尚缺的 41 日母表可能再增加明細，不假設零。
- 最終對帳仍是 TWSE 5 日總量匹配、681 缺來源；TPEX 沿用 189 匹配、42 衝突。合計 194 匹配、42 衝突、681 缺來源。新增可完整分解 TWSE 日期為零。

## 普通盤同口徑公式與來源

`MI_INDEX（一般＋零股＋盤後定價＋鉅額） − TWTC7U（盤中零股） − TWT53U（盤後零股） − BFT41U（盤後定價） − BFIAUU/S（單一鉅額） − BFIAUU 明細（組合鉅額）`。

固定價格數量依官方欄註以千股換算成股；兩種零股必須是 `type=ALL`，盤後定價必須 `selectType=ALL`。缺少來源不能當作 0；組合母表即使只給總額，也不能按股價或平均數猜個股分配。

官方 [組合鉅額明細頁](https://www.twse.com.tw/zh/trading/block/bfiauu-detail.html?MQkyCTg0NDBMNTUwMQkyMDIyMDEwNg) 已實際取得。其 `form data-api="/block/BFIAUU"` 與 `onload-argv` 表明，明細只傳來源給的 `sub, stockType, buyNo, date`（另加 JSON response）；不應帶母表的 `selectType=M`。本地原文保存為 `.cache/twse-board-reconciliation-v2-20260925/raw/44a164810932902fbcae1004fa8ec3c6d320ba09c5591efb138140875b5be0be-3.json`。

Parser 已加入母表身分、日期、證券數、股數、金額逐項一致性檢查及負殘差拒絕。**真正明細 JSON 尚未取得；預期欄位與 echo 仍須以真實回應驗證。** 目前僅 fixture 測試通過，不宣稱真實明細已成功解析。

## 安全停止、效能與續跑

官方來源後續在另一項市場身分讀取中返回 HTTP 428 安全頁。實際 raw 與 UTC/hash receipt 綁進 repo 共用 `.cache/official-origin-holds/www.twse.com.tw.json`；本輪不再冷卻後探測、換 host、proxy 或替代路徑。舊單次 recovery 成功不會覆蓋稍後的同站拒絕。

Fetcher 新增：所有 3xx/401/403/428 或安全頁立即斷路；`allow_redirects=False`；先落地 rejection receipt 與 raw hash，再以 file lock、fsync、atomic replace 建立共用 origin hold，然後停止。換 `--cache` 或啟用 recovery 旗標也不能繞過 hold。既有成功快取可以離線讀取。

不可變 `plan.json` 沒有修改。後續有合法來源恢復證據後，續跑程式採日期順序，每日六分項與所有組合明細完成才換下一日；每個新開始至少間隔 3.1 秒（硬下限 1.5 秒），每 identity 最多三次，transport ledger 硬上限 3,000，不重抓已成功資料，沒有使用 FinMind。本輪目前仍受共享 hold 阻止，以下取得命令**不能自動恢復網路**：

```sh
python scripts/audit_twse_board_reconciliation_v2.py --fetch-complete-days
```

本輪報告定版後，原 cache 的 ledger、plan、local hold 與已引用 raw 都是封存輸入，不能原地續寫。未來取得作業須使用新的 cache，複製完整 plan／ledger 以承接既有次數与 receipts，另存新報告；不能藉新 cache 清零 3,000 次預算。報告綁定本地 hold 的歷史快照與真實拒絕來源，共享 runtime guard 的日後合法變更不改寫這次歷史結論。

403 份重用來源有既有 metadata；部分舊 metadata 沒有取得時間或 HTTP status。其 hash 與資料內日期可核對，但不能替它補寫未記錄的取得時間或 HTTP 證據。

## 報告、重播與資格

新增報告 `artifacts/forward_simulation/board_tape_reconciliation_v2_20260925.json` / `.sha256`，schema `board_tape_reconciliation_v2`。`summary`、`markets` 保存日總量匹配/衝突/缺件；`official_acquisition` 分開記錄來源準備狀態、已取得回應、未派送 identity、已知明細與共用 hold。所有原始回應（含拒絕頁）、plan、ledger、舊輸入与程式 hash 閉包均保留。

```sh
python -m pytest -q tests/test_twse_board_preparation_v2.py tests/test_board_tape_reconciliation_v2.py
python scripts/audit_twse_board_reconciliation_v2.py --verify
```

Callable 為 `skills.board_tape_reconciliation_v2.verify_report(path, root)`。離線 CLI 逐原始 tick、日表與來源重建，檢查結果完全相同，不打 HTTP。日總量匹配仍無法證明逐筆序列完整、撮合佇列、自己的成交，strict/live/fill 資格全部維持 false；訊息列數也不等於交易所成交筆數。

原封存成交引擎本來已限定 13:25 以前的委託視窗；本輪普通盤日表核對用 09:00 至 13:34 前（含延後收盤），排除 14:30 盤後定價。沒有證據說舊策略把 14:30 當普通盤成交。TPEX 42 衝突的進一步診斷由獨立工項處理，這份報告保持 v1 分類，不重寫來源。
