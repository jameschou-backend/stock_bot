# 2026-09-14 前向來源時序核對及逐筆守門

## 核對結果

11:01:45 的 latest-report.json 三帳本顯示公司行動來源已過期；該輪開始時來源仍在 3600 秒期限內，但最早來源 10:01:43 取得，20 秒後報告已跨過期限。當輪沒有新增成交。這不表示較早成交使用了過期資料。

讀取原始雜湊鏈及 corporate-audit.sqlite3，按每筆成交 executed_at 重建當時已 recorded 的來源，沒有使用後來取得的資料追認。啟用守門前 4 筆模擬成交（策略 1、對照 1、0050 基準 2）的 5 項必要來源期限均通過。這僅驗證已取得且未過期，不保證供應商完整性、人工核對或真實成交。

| 帳本 | 成交時間（台北） | 股票／股數 | 來源取得時間（台北） |
|---|---|---|---|
| 策略 | 10:17:39 | 1560／4 | 10:01:43 |
| 原排序對照 | 10:17:39 | 1560／4 | 10:01:43 |
| 0050 基準 | 10:02:14 | 0050／271 | 10:01:43–44 |
| 0050 基準 | 10:32:09 | 0050／4000 | 10:01:43–44 |

完整本機初始核對：`.cache/source-guard-audit-before-20260914.json`。零網路請求、原帳本不變。

## 修正範圍

原 observer 只在整輪開始前做公司行動檢查。若第一輪有效而第二轮跨過 TTL，原 matcher 本身沒有該來源門檻。合成測試已證明此邊界會讓原 matcher 接受可撮合報價；同樣報價經新守門會被阻擋。

新增 `capacity_source_guard` 及 `capacity_guard_runner`，在原 matcher 外逐筆檢查，並透過受限時鐘檢查交易內到期；成交已插入但尚未提交時若過期，整個該次觀察交易回滾。新指紋独立封存，不改寫任何父版本指紋。未完成的檢查會阻擋下一輪，不能用重跑掩蓋結果不明。非盤中沿用原版結算及人工核對門檻，並防止開盤邊界落入未受保護 observer。

工作台新增「成交當時的來源期限」表格、目前資料狀態、守門生效時間及下載，避免把當時通過與現在過期混為一談。API 能連線、日價資料齊全、公司行動期限及實盤資格仍是不同狀態。

## 操作與限制

```bash
python scripts/audit_capacity_sources.py --output .cache/source-audit-new.json
python scripts/run_capacity_guarded.py activate
REQUESTS_CA_BUNDLE="$PWD/.cache/capacity-tls-20260913/bundle.pem" python scripts/run_capacity_guarded.py run
```

執行網路命令前核對 bundle 與 verification.json 的 SHA256。已存在的稽核輸出不覆寫。`activate` 不抓資料、不建立新委託、不成交，只記錄往後適用的檢查版本；既有排程改用新入口，不新增第二份排程或併跑舊入口。

每日結算仍需使用者核對持股公司行動，不能由 agent 代勾。完整歷史跨日回測仍缺盤中零股來源，報酬尚未驗證；此修正不增加任何策略報酬或實盤資格。

## 驗收

包含來源過期／未來時間、20 秒前後跨期限、插入成交後到期及倒退時鐘的整筆回滾、重複觀察不重算成交、指紋不符、檢查中斷、預算共用、休市無請求、開盤邊界及舊帳本不變。交易提交後若驗證失敗，保留未完成檢查並阻擋下一輪，不虛報零成交。

- 最終 `make test`：2524 passed、28 warnings，61.83 秒；日誌 `.cache/source-guard-make-test-final.log`。
- `INGEST_PRICES_SOURCE=finmind SPONSOR_INGEST=off` 加已核對 TLS bundle 的 `make pipeline` 成功；日誌 `.cache/source-guard-pipeline.log`。既有退休設定警告與 labels 查詢 8.521 秒仍存在，未宣稱本修正解決查詢效能。
- `make api` 後 `/health`、`/picks`、`/models`、`/jobs?limit=10` 均 HTTP 200 且 JSON 可解析。
- AppTest 驗證真實元件呈現「當時 1/1 通過、目前過期」且不改帳本；瀏覽器實際看到 1560 的 10:17:39、4 股、来源年齡 956.37 秒與下載按鈕。
- 2026/9/14 台北 11:23:52 啟用附加守門，逐筆比對三帳本事件與啟用前完全相同，父版本指紋全部通過。既有 automation 排程改用 guarded 入口，頻率及帳本不變。
- 11:25:30 首輪完成，9 次公司行動查詢、6 筆重用、4 次行情查詢；12 筆守門檢查各有結果，無未完成項目、無新增成交。三帳本 `source_blocked=false`；資料期限僅對該觀察時間有效，並非永久有效或完整覆蓋保證。
- 啟用後稽核 `.cache/source-guard-audit-after-20260914.json`；逐筆守門紀錄 `.cache/forward-simulation/capacity-v1/source-guard-v1.sqlite3`。來源更新不改寫既有成交，結算未完成仍保持 `nav=null`、`live_qualified=false`。
